"""Run the permanent tactical regressions used by the promotion gate.

These fixtures come from concrete ladder mistakes, not generic style preferences.
They cover the complete deployed decision stack around the learned leaf value: exact
speed order, future-turn setup, factual guards, sacrifice coordination, switching,
weather/Mega conservation, and hidden-set downside.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FIXTURES = (
    "unit_tests/test_tempo_reranker.py::test_tailwind_changes_speed_before_trick_room_inverts_order",
    "unit_tests/test_tempo_reranker.py::test_turn_five_good_trick_room_penalizes_double_protect",
    "unit_tests/test_tempo_reranker.py::test_protect_plus_earthquake_is_recognized_as_coordinated_progress",
    "unit_tests/test_tempo_reranker.py::test_encore_can_force_trick_room_toggle_when_it_really_moves_first",
    "unit_tests/test_guard_repairs.py::test_solo_guaranteed_earthquake_beats_resisted_dragon_claw",
    "unit_tests/test_guard_repairs.py::test_last_respects_ko_beats_rain_aqua_jet_at_observed_policy_gap",
    "unit_tests/test_guard_repairs.py::test_two_on_one_focus_attacks_with_both_slots",
    "unit_tests/test_guard_repairs.py::test_yawn_switch_values_the_next_turn",
    "unit_tests/test_guard_repairs.py::test_minus_two_physical_attacker_gets_a_switch_candidate",
    "unit_tests/test_guard_repairs.py::test_rain_reserves_mega_charizard_y",
    "unit_tests/test_guard_repairs.py::test_final_two_on_two_double_protect_makes_progress",
    "unit_tests/test_guard_repairs.py::test_double_protect_is_demoted_into_likely_shell_smash",
    "unit_tests/test_guard_repairs.py::test_last_pokemon_does_not_repeat_protect_without_a_stall_goal",
    "unit_tests/test_guard_repairs.py::test_known_prankster_encore_demotes_first_protect",
    "unit_tests/test_guard_repairs.py::test_no_weather_weather_ball_is_demoted_when_heat_wave_dominates",
    "unit_tests/test_guard_repairs.py::test_sun_weather_ball_beats_heat_wave_with_one_delphox",
    "unit_tests/test_guard_repairs.py::test_delphox_two_on_one_near_tie_attacks_with_both_slots",
    "unit_tests/test_exact_planner.py::test_determinization_aggregation_penalizes_hidden_set_downside",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results_tactical_gate.json")
    )
    parser.add_argument("--minimum", type=float, default=0.90)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="vgc-tactical-") as directory:
        report = Path(directory) / "report.xml"
        completed = subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                "-m",
                "pytest",
                "-q",
                *FIXTURES,
                f"--junitxml={report}",
            ],
            cwd=ROOT,
            check=False,
        )
        root = ET.parse(report).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise RuntimeError("pytest did not produce a tactical test suite")
    tests = int(suite.attrib.get("tests", 0))
    failures = int(suite.attrib.get("failures", 0))
    errors = int(suite.attrib.get("errors", 0))
    skipped = int(suite.attrib.get("skipped", 0))
    passed = tests - failures - errors - skipped
    accuracy = passed / max(1, tests)
    payload = {
        "schema": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fixtures": list(FIXTURES),
        "tests": tests,
        "passed": passed,
        "failures": failures,
        "errors": errors,
        "skipped": skipped,
        "accuracy": accuracy,
        "minimum": args.minimum,
        "accepted": completed.returncode == 0 and accuracy >= args.minimum,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"tactical ordering: {passed}/{tests} = {accuracy * 100:.1f}% "
        f"(required {args.minimum * 100:.0f}%)",
        flush=True,
    )
    if not payload["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
