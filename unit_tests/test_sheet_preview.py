"""vgc_bench/src/sheet_preview.py (2026-10-03): an open sheet moves our preview only
by how the opponent's sets differ from the usual ones."""

from __future__ import annotations

from poke_env.teambuilder import Teambuilder

from unit_tests.ladder_position import position
from vgc_bench.src import sheet_preview as S
from vgc_bench.src.opponent_preview import PreviewPlan

# Our team-file order: Blastoise, Farigiraf, Charizard, Venusaur, Torkoal, Incineroar
OURS = ["blastoise", "farigiraf", "charizard", "venusaur", "torkoal", "incineroar"]

USUAL = """
Incineroar @ Sitrus Berry
Ability: Intimidate
- Fake Out
- Flare Blitz
- Parting Shot
- Throat Chop

Pelipper @ Sitrus Berry
Ability: Drizzle
- Hurricane
- Tailwind
- Weather Ball
- Wide Guard

Rillaboom @ Miracle Seed
Ability: Grassy Surge
- Fake Out
- Grassy Glide
- High Horsepower
- Wood Hammer

Sylveon @ Fairy Feather
Ability: Pixilate
- Detect
- Hyper Beam
- Hyper Voice
- Quick Attack

Kingambit @ Chople Berry
Ability: Defiant
- Iron Head
- Kowtow Cleave
- Low Kick
- Sucker Punch

Garchomp @ Life Orb
Ability: Rough Skin
- Dragon Claw
- Protect
- Rock Slide
- Stomping Tantrum
"""


def _preview(sheet: str = USUAL, shown: bool = True):
    battle = position([])
    for tb in Teambuilder.parse_showdown_team(sheet):
        species = tb.species or tb.nickname
        mon = battle.get_pokemon(f"p2: {species}", details=f"{species}, L50")
        if shown:
            mon._update_from_teambuilder(tb)
    return battle


def _plan(lead: tuple[str, str], back: tuple[str, str], p: float) -> PreviewPlan:
    index = {s: i for i, s in enumerate(OURS)}
    leads = (index[lead[0]], index[lead[1]])
    return PreviewPlan(leads, (*leads, index[back[0]], index[back[1]]), p)  # type: ignore[arg-type]


PLANS = [
    _plan(("blastoise", "farigiraf"), ("incineroar", "torkoal"), 0.40),
    _plan(("charizard", "venusaur"), ("farigiraf", "torkoal"), 0.35),
]
THEIR_LEAD = [PreviewPlan((0, 2), (0, 2, 1, 3), 1.0)]  # Incineroar + Rillaboom


def test_the_sheet_is_open_only_with_every_set_shown():
    assert S.sheet_open(_preview())
    assert not S.sheet_open(_preview(shown=False))


def test_a_sheet_of_usual_sets_keeps_the_model_plan():
    battle = _preview()
    plan, audit = S.rerank(battle, PLANS, THEIR_LEAD)
    assert plan is PLANS[0] and audit["chosen_rank"] == 1
    assert all(abs(row["sheet_delta"]) < 1e-9 for row in audit["plans"])


def test_their_usual_sets_are_restored():
    battle = _preview(
        USUAL.replace("- Fake Out\n- Flare Blitz", "- Will-O-Wisp\n- Flare Blitz")
    )
    incineroar = battle.opponent_team["p2: Incineroar"]
    with S.usual_sets(battle):
        assert "fakeout" in incineroar.moves
    assert "fakeout" not in incineroar.moves and "willowisp" in incineroar.moves


def test_no_fake_out_on_their_sheet_lifts_the_plan_it_would_have_hurt():
    """Their leads usually carry two Fake Outs; this sheet has none. Our
    Farigiraf lead blocks Fake Out with Armor Tail anyway (no change), the sun
    lead does not -- so the sheet favours it."""
    sheet = USUAL.replace(
        "- Fake Out\n- Flare Blitz", "- Will-O-Wisp\n- Flare Blitz"
    ).replace("- Fake Out\n- Grassy Glide", "- Protect\n- Grassy Glide")
    battle = _preview(sheet)
    plan, audit = S.rerank(battle, PLANS, THEIR_LEAD)
    deltas = {tuple(row["lead"]): row["sheet_delta"] for row in audit["plans"]}
    assert abs(deltas[("blastoise", "farigiraf")]) < 0.05
    assert deltas[("charizard", "venusaur")] > 0.1
    assert plan is PLANS[1]
