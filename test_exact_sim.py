"""Smoke-test exact Showdown cloning on the Champions doubles format."""

from pathlib import Path

from vgc_bench.src.exact_sim import ExactShowdownBridge

ROOT = Path(__file__).resolve().parent


def first_legal_pair(request) -> str:
    choices = []
    for active in request["active"]:
        slot, move = next(
            (slot, move)
            for slot, move in enumerate(active["moves"], start=1)
            if not move.get("disabled")
        )
        target = ""
        if move.get("target") in {"normal", "any", "adjacentFoe"}:
            target = " 1"
        choices.append(f"move {slot}{target}")
    return ", ".join(choices)


with ExactShowdownBridge() as bridge:
    ping = bridge.ping()
    assert ping == {"ok": True, "backend": "pokemon-showdown", "exact": True}
    created = bridge.create(
        formatid="gen9championsvgc2026regmb",
        seed=[1, 2, 3, 4],
        p1_team_text=(ROOT / "teams/reg_mb/our_team.txt").read_text(),
        p2_team_text=(ROOT / "teams/reg_mb/MB11.txt").read_text(),
        p1_preview="team 1234",
        p2_preview="team 1234",
    )
    assert created["request_state"] == "move"
    assert created["turn"] == 1
    state = created["state"]
    advanced = bridge.simulate(
        state,
        first_legal_pair(created["requests"][0]),
        first_legal_pair(created["requests"][1]),
    )
    repeated = bridge.simulate(
        state,
        first_legal_pair(created["requests"][0]),
        first_legal_pair(created["requests"][1]),
    )
    assert advanced["turn"] >= 1
    assert advanced["state"]["formatid"] == "gen9championsvgc2026regmb"
    assert advanced["log"], "an exact turn should emit protocol events"
    assert repeated["state"] == advanced["state"]
    assert repeated["log"] == advanced["log"]

print("PASS - exact Showdown state cloned and advanced")
