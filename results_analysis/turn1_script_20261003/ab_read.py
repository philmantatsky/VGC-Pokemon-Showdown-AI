"""The turn-1 script ladder A/B (pre-registered in PROJECT_STATUS 2026-10-03 10:40):
script arm vs control arm, alternating 10-game blocks.

Per arm: the record without free wins (the opponent quit by turn 1, or at team
preview), the Newcombe 95% CI of the win-rate difference (script minus control),
and the mechanism -- in games we led Blastoise + Farigiraf, the share where we lost
a Pokemon on turns 1-3 -- plus, for the script arm, how often the script changed
turn 1 (decision-log stage playbook_opening). A read, not proof. Read-only.
Run from the repo root:
    .venv/bin/python results_analysis/turn1_script_20261003/ab_read.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results_analysis/openings_20260927"))
from openings_parse import base, parse, sid  # noqa: E402

from tools.ladder_loss_profile import extract_log  # noqa: E402

OUR = "antonius1"
ARMS = {
    "script": ROOT / "ladder_replays_mc_T6tac_turn1_script",
    "control": ROOT / "ladder_replays_mc_T6tac_turn1_control",
}
QUIT = re.compile(r"\|-message\|(.+?) (forfeited|lost due to inactivity)")
WATER = {"Blastoise", "Farigiraf"}


def quit_turn(log: str, opponent: str) -> int | None:
    """The turn on which the opponent forfeited / timed out, if they did."""
    turn = 0
    for line in log.split("\n"):
        parts = line.split("|")
        if len(parts) > 2 and parts[1] == "turn":
            turn = int(parts[2])
        match = QUIT.match(line)
        if match and sid(match.group(1)) == sid(opponent):
            return turn
    return None


def first_faint_side(log: str) -> str | None:
    """The side (p1 / p2) whose Pokemon fainted first, in log order."""
    for line in log.split("\n"):
        if line.startswith("|faint|"):
            return line.split("|")[2][:2]
    return None


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def newcombe(w1: int, n1: int, w2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 with Newcombe's hybrid-score 95% interval."""
    p1, p2 = w1 / max(1, n1), w2 / max(1, n2)
    l1, u1 = wilson(w1, n1)
    l2, u2 = wilson(w2, n2)
    d = p1 - p2
    return d, d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def games(folder: Path) -> list[dict]:
    rows = []
    for path in sorted(folder.glob("*.html")):
        log = extract_log(path) or ""
        me = path.name.split(" - battle-")[0]
        rec = parse(log) if log else None
        if rec is None:  # no leads: a quit at team preview, or unreadable
            rows.append(
                {
                    "file": path.name,
                    "win": f"|win|{me}" in log,
                    "free": bool(QUIT.search(log)),
                    "parsed": False,
                }
            )
            continue
        side = next((s for s, n in rec["players"].items() if sid(n) == sid(OUR)), None)
        if side is None:
            continue
        other = "p2" if side == "p1" else "p1"
        quit_at = quit_turn(log, rec["players"].get(other, ""))
        ours = rec["faints"][side]
        battle = re.search(r"regmc-(\d+)", path.name)
        rows.append(
            {
                "file": path.name,
                "battle": battle.group(1) if battle else "",
                "parsed": True,
                "win": rec["winner_side"] == side,
                "free": quit_at is not None and quit_at <= 1,
                "water_lead": {base(s) for s in rec["leads"][side]} == WATER,
                "early_loss": any(turn <= 3 for turn, _ in ours),
                "lost_first": first_faint_side(log) == side,
            }
        )
    return rows


def script_changes(folder: Path) -> tuple[int, set[str]]:
    """(turn-1 decisions the script changed, battle ids where it did)."""
    path = folder / "decisions.jsonl"
    changed: set[str] = set()
    total = 0
    if not path.exists():
        return 0, changed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("turn") != 1:
            continue
        if "playbook_opening" in ((row.get("guards") or {}).get("stages") or []):
            total += 1
            match = re.search(r"regmc-(\d+)", row.get("battle", ""))
            if match:
                changed.add(match.group(1))
    return total, changed


def pct(a: int, n: int) -> str:
    return f"{a}/{n} ({100 * a / n:.0f}%)" if n else "0/0"


def count(rows: list[dict], key: str) -> str:
    return pct(sum(bool(r.get(key)) for r in rows), len(rows))


def main() -> None:
    summary = {}
    for arm, folder in ARMS.items():
        rows = games(folder)
        played = [r for r in rows if not r["free"]]
        wins = sum(r["win"] for r in played)
        water = [r for r in played if r.get("water_lead")]
        summary[arm] = (wins, len(played))
        free = len(rows) - len(played)
        print(f"== {arm} ({folder.name}): {len(rows)} games, {free} free wins")
        print(f"   record without free wins: {wins}-{len(played) - wins}")
        print(f"   lost a Pokemon on turns 1-3: {count(played, 'early_loss')}")
        print(f"   lost a Pokemon first: {count(played, 'lost_first')}")
        print(
            f"   led Blastoise + Farigiraf: {len(water)}; won {count(water, 'win')}; "
            f"lost a Pokemon on turns 1-3 in {count(water, 'early_loss')}"
        )
        if arm == "script":
            total, changed = script_changes(folder)
            hit = [r for r in played if r.get("battle") in changed]
            print(
                f"   the script changed turn 1 in {total} decisions "
                f"({len(changed)} games); those games won {count(hit, 'win')}, "
                f"lost a Pokemon on turns 1-3 in {count(hit, 'early_loss')}"
            )
    (ws, ns), (wc, nc) = summary["script"], summary["control"]
    if ns and nc:
        d, low, high = newcombe(ws, ns, wc, nc)
        print(
            f"== script minus control: {100 * d:+.1f}pp "
            f"[{100 * low:+.1f}, {100 * high:+.1f}] ({ws}/{ns} vs {wc}/{nc})"
        )


if __name__ == "__main__":
    main()
