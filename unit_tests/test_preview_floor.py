from types import SimpleNamespace

from poke_env.battle import Pokemon

from vgc_bench.src.policy_player import PolicyPlayer


def test_forced_lead_resolves_team_slots_and_preserves_order():
    player = PolicyPlayer.__new__(PolicyPlayer)
    player.forced_lead_species = ("whimsicott", "basculegion")
    team = [
        Pokemon(gen=9, species="floetteeternal"),
        Pokemon(gen=9, species="charizard"),
        Pokemon(gen=9, species="whimsicott"),
        Pokemon(gen=9, species="garchomp"),
        Pokemon(gen=9, species="basculegion"),
        Pokemon(gen=9, species="kingambit"),
    ]
    battle = SimpleNamespace(team={str(index): mon for index, mon in enumerate(team)})

    actions = player._forced_lead_actions(battle)

    assert actions is not None
    assert actions.tolist() == [3, 5]


def test_forced_lead_stands_down_when_a_species_is_missing():
    player = PolicyPlayer.__new__(PolicyPlayer)
    player.forced_lead_species = ("whimsicott", "basculegion")
    battle = SimpleNamespace(
        team={"0": Pokemon(gen=9, species="whimsicott")}
    )

    assert player._forced_lead_actions(battle) is None
