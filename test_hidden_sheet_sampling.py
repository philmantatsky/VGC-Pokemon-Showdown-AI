"""Hidden-sheet training must sample the requested mode before each battle."""

from vgc_bench.src.env import ShowdownEnv


def make(probability: float) -> ShowdownEnv:
    return ShowdownEnv(
        battle_format="gen9championsvgc2026regmb",
        start_listening=False,
        hidden_sheet_prob=probability,
        sheet_seed=7,
    )


hidden = make(1.0)
assert hidden._configure_open_team_sheets()
assert not hidden.agent1._accept_open_team_sheet
assert not hidden.agent2._accept_open_team_sheet

open_sheets = make(0.0)
assert not open_sheets._configure_open_team_sheets()
assert open_sheets.agent1._accept_open_team_sheet
assert open_sheets.agent2._accept_open_team_sheet

mixed = make(0.5)
hidden_count = sum(mixed._configure_open_team_sheets() for _ in range(1000))
assert 450 <= hidden_count <= 550, hidden_count
assert mixed._hidden_sheet_battles == hidden_count
assert mixed._sheet_battles == 1000 - hidden_count

print(f"PASS - mixed run hid sheets in {hidden_count}/1000 sampled battles")
