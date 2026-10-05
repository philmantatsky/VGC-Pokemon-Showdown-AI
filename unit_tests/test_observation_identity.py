"""The observation encoder may get faster, never different (2026-10-04).

``embed_move`` and ``embed_pokemon`` re-read a Pokemon's or a move's properties once
per enum member (38 reads a move); reading each once halves the cost of a position,
which is most of what the exact search pays for on cpu. The encoding as it stood
before that change is kept here, verbatim, as the reference: every move in the
vocabulary and every Pokemon of a few hundred played-out positions must encode to
the same bytes.

The two per-state caches (knowledge and threat blocks) are keyed by ``id(pokemon)``:
a second battle object in the same state -- a search copy, the mirror's other side --
used to hit the first object's entry and read zeros for its own Pokemon.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from poke_env.battle import Move

import vgc_bench.src.policy_player as policy_player
from vgc_bench.src import utils
from vgc_bench.src.exact_observation import state_to_battle
from vgc_bench.src.exact_planner import ExactNode
from vgc_bench.src.exact_sim import ExactShowdownBridge
from vgc_bench.src.policy_player import PolicyPlayer

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "gen9championsvgc2026regmc"

# vgc_bench/src/policy_player.py as of commit 0cb9ed91 (2026-10-04), unchanged
ENCODING_BEFORE = r'''
def embed_move(move: Move) -> npt.NDArray[np.float32]:
    """Embed a move's power, accuracy, type, and special properties."""
    power = move.base_power / 250
    # Legacy feature retained for checkpoint compatibility. The correctly scaled
    # value is appended by embed_move_accuracies at the end of each token.
    acc = move.accuracy / 100
    category = [float(c == move.category) for c in MoveCategory]
    target = [float(t == move.target) for t in Target]
    priority = (move.priority + 7) / 12
    crit_ratio = move.crit_ratio
    drain = move.drain
    force_switch = float(move.force_switch)
    recoil = move.recoil
    self_destruct = float(move.self_destruct is not None)
    self_switch = float(move.self_switch is not False)
    pp = move.max_pp / 64
    pp_frac = move.current_pp / move.max_pp
    is_last_used = float(move.is_last_used)
    move_type = [float(t == move.type) for t in PokemonType]
    return np.array(
        [
            power,
            acc,
            *category,
            *target,
            priority,
            crit_ratio,
            drain,
            force_switch,
            recoil,
            self_destruct,
            self_switch,
            pp,
            pp_frac,
            is_last_used,
            *move_type,
        ]
    )


def embed_pokemon(
    pokemon: Pokemon,
    pos: int,
    from_opponent: bool,
    active_a: bool,
    active_b: bool,
    knowledge: npt.NDArray[np.float32] | None = None,
    formatid: str | None = None,
) -> npt.NDArray[np.float32]:
    """Embed a Pokemon's stats, moves, status, and effects."""
    # Invariant: one of OUR revealed Pokemon must have been drafted at teampreview.
    # It is violated by a race -- with Open Team Sheets the server can reveal our
    # side before _teampreview() has recorded its picks -- and a bare assert here
    # is NOT survivable. poke-env catches the exception inside its battle-message
    # handler, so the battle never receives an order, never finishes, and the
    # evaluation blocks on it forever. That is what stalled the 200-battle search
    # eval: 24 assertions in the opening milliseconds, then a permanent hang.
    #
    # A revealed Pokemon is on the field, so it was self-evidently drafted. Repair
    # the flag (which is what `in_draft` below reads), count it, and carry on.
    if (
        not from_opponent
        and pokemon.revealed
        and not pokemon.selected_in_teampreview
    ):
        pokemon._selected_in_teampreview = True
        PolicyPlayer.guard_fire_counts["teampreview_flag_repaired"] += 1
    # Reg M-B's Open Team Sheets rule is opt-in, so an opponent can deny it.
    # Training always had sheets, so opponent slots were populated; without them
    # the sentinels below are near-unseen inputs. When the prior is enabled we fill
    # unknown opponent fields with the most common competitive set instead, keeping
    # the observation nearer the training distribution. Shape is unchanged, so
    # existing checkpoints stay valid. Off by default -> training is unaffected.
    prior = (
        PolicyPlayer._moveset_prior(pokemon, formatid) if from_opponent else None
    )
    # (mostly) stable fields
    ability = pokemon.ability
    if ability is None and prior and prior.get("ability"):
        ability = prior["ability"]
    ability_id = abilities.index(ability if ability in abilities else "null")
    item = pokemon.item
    if item in (None, "unknown_item") and prior and prior.get("item"):
        item = prior["item"]
    item_id = items.index(item if item in items else "null")
    # [:4] to match poke-env's action decoder (doubles_env.py:200,342,393);
    # [-4:] silently shifted which move each action index referred to whenever
    # more than four were stored.
    move_list = PolicyPlayer._resolved_moves(pokemon, from_opponent, formatid)
    move_ids = [moves.index(move.id) for move in move_list]
    move_ids += [0] * (4 - len(move_ids))
    move_embeds = [embed_move(move) for move in move_list]
    move_embeds += [np.zeros(move_obs_len, dtype=np.float32)] * (
        4 - len(move_embeds)
    )
    move_embeds = np.concatenate(move_embeds)
    move_sem = np.concatenate(
        [move_semantics(m.id) for m in move_list]
        + [_ZERO_MOVE_SEM] * (4 - len(move_list))
    )
    types = [float(t in pokemon.base_types) for t in PokemonType]
    tera_type = [float(t == pokemon.tera_type) for t in PokemonType]
    base_stats = [s / 255 for s in pokemon.base_stats.values()]
    if from_opponent:
        stats = [-1] * 6
    else:
        stat_names = ("hp", "atk", "def", "spa", "spd", "spe")
        raw_stats = pokemon.stats or {}
        if any(raw_stats.get(name) is None for name in stat_names):
            # poke-env can temporarily expose an unselected or newly restored
            # teammate without its request-side stats during a forced-switch
            # message.  Crashing here leaves the battle waiting forever.  Use
            # the same conservative Champions stat completion as the mechanics
            # layer; complete request stats remain bit-for-bit unchanged.
            from vgc_bench.src.vgc_knowledge import ensure_stats

            if ensure_stats(pokemon):
                PolicyPlayer.guard_fire_counts["own_stats_imputed"] += 1
            raw_stats = pokemon.stats or {}
        stats = [float(raw_stats[name]) / 255 for name in stat_names]
    gender = [float(g == pokemon.gender) for g in PokemonGender]
    weight = pokemon.weight / 1000
    # volatile fields
    hp_frac = pokemon.current_hp_fraction
    revealed = float(pokemon.revealed)
    in_draft = float(pokemon.selected_in_teampreview)
    status = [float(s == pokemon.status) for s in Status]
    status_counter = pokemon.status_counter / 16
    boosts = [b / 6 for b in pokemon.boosts.values()]
    effects = [
        (min(pokemon.effects[e], 8) / 8 if e in pokemon.effects else 0)
        for e in Effect
    ]
    first_turn = float(pokemon.first_turn)
    protect_counter = pokemon.protect_counter / 5
    must_recharge = float(pokemon.must_recharge)
    preparing = float(pokemon.preparing)
    gimmicks = [float(s) for s in [pokemon.is_dynamaxed, pokemon.is_terastallized]]
    pos_onehot = [float(pos == i) for i in range(6)]
    return np.array(
        [
            ability_id,
            item_id,
            *move_ids,
            *move_embeds,
            *types,
            *tera_type,
            *base_stats,
            *stats,
            *gender,
            weight,
            hp_frac,
            revealed,
            in_draft,
            *status,
            status_counter,
            *boosts,
            *effects,
            first_turn,
            protect_counter,
            must_recharge,
            preparing,
            *gimmicks,
            float(active_a),
            float(active_b),
            *pos_onehot,
            float(from_opponent),
            # knowledge block LAST -- see utils.knowledge_obs_len for why order
            # matters for checkpoint compatibility
            *(knowledge if knowledge is not None else _ZERO_KNOWLEDGE),
            # what each move / this ability actually does
            *move_sem,
            *ability_semantics(ability),
        ],
        dtype=np.float32,
    )
'''


def _reference() -> dict[str, Any]:
    namespace = dict(vars(policy_player))
    exec(ENCODING_BEFORE, namespace)  # noqa: S102 - our own frozen source, above
    return namespace


def _same(first: np.ndarray, second: np.ndarray) -> bool:
    return first.dtype == second.dtype and first.tobytes() == second.tobytes()


def test_every_move_in_the_vocabulary_encodes_as_before():
    before = _reference()["embed_move"]
    checked = 0
    for move_id in utils.moves:
        try:
            move = Move(move_id, gen=9)
            expected = before(move)
        except Exception as error:  # not a move, or one the old code refused too
            with pytest.raises(type(error)):
                PolicyPlayer.embed_move(Move(move_id, gen=9))
            continue
        assert _same(expected, PolicyPlayer.embed_move(move)), move_id
        # a used move: the two fields that change during a battle
        move._current_pp = max(0, move.max_pp - 3)
        move._is_last_used = True
        assert _same(before(move), PolicyPlayer.embed_move(move)), move_id
        checked += 1
    assert checked > 600


def _positions(seed: int, games: int):
    """Battle objects from played-out games: both seats, sheets open and hidden."""
    rng = random.Random(seed)
    teams = [
        (ROOT / f"teams/candidates_mc/{name}.txt").read_text() for name in ("T6e", "T4")
    ]
    with ExactShowdownBridge() as bridge:
        for _game in range(games):
            node = ExactNode.from_result(
                bridge.create(
                    formatid=FORMAT,
                    seed=[rng.randrange(1, 65536) for _ in range(4)],
                    p1_team_text=rng.choice(teams),
                    p2_team_text=rng.choice(teams),
                    p1_preview=rng.choice(["team 1234", "team 3412", "team 2156"]),
                    p2_preview=rng.choice(["team 1234", "team 3412", "team 6521"]),
                )
            )
            for _turn in range(14):
                if node.ended:
                    break
                for role in ("p1", "p2"):
                    for reveal in (True, False):
                        yield state_to_battle(node.state, node.requests, role, reveal)
                picks = {}
                for role in ("p1", "p2"):
                    choices = bridge.choices(node.state, role)
                    attacks = [c for c in choices if "move" in c] or choices
                    picks[role] = rng.choice(attacks if rng.random() < 0.8 else choices)
                node = ExactNode.from_result(
                    bridge.simulate_batch(
                        node.state,
                        [{"p1_choice": picks["p1"], "p2_choice": picks["p2"]}],
                    )[0]
                )


def test_every_pokemon_of_played_out_positions_encodes_as_before(monkeypatch):
    monkeypatch.setattr(PolicyPlayer, "use_moveset_prior", True)
    before = _reference()["embed_pokemon"]
    tokens = with_effects = without_effects = with_status = 0
    for battle in _positions(seed=20261004, games=6):
        actives = [battle.active_pokemon, battle.opponent_active_pokemon]
        sides = [list(battle.team.values()), list(battle.opponent_team.values())]
        for from_opponent, (mons, (first, second)) in enumerate(zip(sides, actives)):
            for position, pokemon in enumerate(mons):
                arguments: Any = dict(
                    from_opponent=bool(from_opponent),
                    active_a=first is not None and pokemon.name == first.name,
                    active_b=second is not None and pokemon.name == second.name,
                    formatid=battle.format,
                )
                expected = before(pokemon, position, **arguments)
                actual = PolicyPlayer.embed_pokemon(pokemon, position, **arguments)
                assert _same(expected, actual), (battle.turn, pokemon.species)
                tokens += 1
                with_effects += bool(pokemon.effects)
                without_effects += not pokemon.effects
                with_status += pokemon.status is not None
    # both branches of the effects shortcut, and a status, were really exercised
    assert tokens > 2000 and with_effects > 20 and without_effects > 1000
    assert with_status > 0


@pytest.fixture
def knowledge_on(monkeypatch):
    """The deployed bot's observation switches, and empty caches of our own."""
    monkeypatch.setattr(PolicyPlayer, "use_knowledge_obs", True)
    monkeypatch.setattr(PolicyPlayer, "use_moveset_prior", True)
    monkeypatch.setattr(PolicyPlayer, "_knowledge_cache", {})
    monkeypatch.setattr(PolicyPlayer, "_threat_cache", {})


def _midgame_battle():
    """A position a few turns in, both sides with two Pokemon on the field."""
    for battle in _positions(seed=7, games=1):
        ours = [p for p in battle.active_pokemon if p is not None and not p.fainted]
        theirs = [
            p for p in battle.opponent_active_pokemon if p is not None and not p.fainted
        ]
        if battle.turn >= 2 and len(ours) == 2 and len(theirs) == 2:
            PolicyPlayer.embed_battle(battle, fake_rating=2000)  # settle side effects
            return battle
    raise AssertionError("no mid-game position was reached")


def _blocks(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(threat block, knowledge block) of every token."""
    tokens = observation.reshape(12, utils.chunk_obs_len)
    pokemon_start = utils.glob_obs_len + utils.side_obs_len
    threat = (
        pokemon_start
        + utils.pokemon_obs_len
        + utils.presence_obs_len
        + utils.global_presence_obs_len
        + utils.correct_accuracy_obs_len
    )
    knowledge = (
        pokemon_start
        + utils.pokemon_obs_len
        - utils.semantics_obs_len
        - utils.knowledge_obs_len
    )
    return (
        tokens[:, threat : threat + utils.threat_obs_len],
        tokens[:, knowledge : knowledge + utils.knowledge_obs_len],
    )


def test_a_copy_of_a_battle_reads_its_own_threat_and_knowledge_blocks(knowledge_on):
    battle = _midgame_battle()
    PolicyPlayer._threat_cache.clear()
    PolicyPlayer._knowledge_cache.clear()
    original = PolicyPlayer.embed_battle(battle, fake_rating=2000)
    threat, knowledge = _blocks(original)
    assert np.abs(threat).sum() > 0 and np.abs(knowledge).sum() > 0
    # the exact search values a copy of the live battle: same state, new objects
    clone = copy.deepcopy(battle)
    assert _same(original, PolicyPlayer.embed_battle(clone, fake_rating=2000))
    # ... and the original still reads its own entry afterwards
    assert _same(original, PolicyPlayer.embed_battle(battle, fake_rating=2000))


def test_the_same_battle_object_still_reuses_its_entry(knowledge_on):
    battle = _midgame_battle()
    PolicyPlayer._threat_cache.clear()
    counts = PolicyPlayer.guard_fire_counts
    start = counts["threat_obs_computed"]
    first = PolicyPlayer.embed_battle(battle, fake_rating=2000)
    assert counts["threat_obs_computed"] == start + 1
    assert _same(first, PolicyPlayer.embed_battle(battle, fake_rating=2000))
    assert counts["threat_obs_computed"] == start + 1  # a hit: nothing recomputed
    PolicyPlayer.embed_battle(copy.deepcopy(battle), fake_rating=2000)
    assert counts["threat_obs_computed"] == start + 2  # another object: its own


def test_an_entry_left_at_a_reused_address_is_not_served(knowledge_on):
    """A copy that is gone can leave its entry under an address the next copy gets."""
    battle = _midgame_battle()
    PolicyPlayer._threat_cache.clear()
    PolicyPlayer._knowledge_cache.clear()
    expected = PolicyPlayer.embed_battle(battle, fake_rating=2000)
    for cache in (PolicyPlayer._threat_cache, PolicyPlayer._knowledge_cache):
        assert len(cache) == 1
        (key,) = cache
        stale = {object_id + 8: vector for object_id, vector in cache[key].items()}
        cache[key] = stale  # this state, this address, somebody else's Pokemon
    assert _same(expected, PolicyPlayer.embed_battle(battle, fake_rating=2000))
