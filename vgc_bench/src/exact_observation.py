"""Adapt exact Pokemon Showdown states to the existing policy network.

The exact simulator owns the forward state.  poke-env remains useful as the policy's
observation/action vocabulary, so this module reconstructs a read-only ``DoubleBattle``
from Showdown's public log and side request.  It never tries to turn that reconstruction
back into simulator state; directionality is what keeps the planner honest.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.battle.move import SPECIAL_MOVES, Move, MoveSet
from poke_env.data import to_id_str
from poke_env.environment import DoublesEnv
from poke_env.teambuilder import TeambuilderPokemon

from vgc_bench.src.opponent_preview import PreviewPredictor, plan_to_showdown_order
from vgc_bench.src.opponent_tactics import (
    MovePrediction,
    MovePredictor,
    SwitchPrediction,
    SwitchPredictor,
)
from vgc_bench.src.policy_player import PolicyPlayer

_IGNORED_PROTOCOL = {"t:", "uhtml", "uhtmlchange", "win", "tie", "vgcsnapshot"}
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_LOGGER = logging.getLogger("vgc_bench.exact_observation")
_LOGGER.addHandler(logging.NullHandler())
_LOGGER.propagate = False


def _stale_branch_move_conflict(event: list[str], exc: AssertionError) -> bool:
    """Identify a reconciled hidden particle whose old log learned a fifth move."""
    return (
        len(event) > 1
        and event[1] == "move"
        and "Expected self.moves to contain" in str(exc)
    )


def perspective_log(lines: Iterable[str], role: str) -> list[str]:
    """Resolve Showdown ``|split|`` blocks for one player's perspective."""
    source = list(lines)
    out: list[str] = []
    index = 0
    while index < len(source):
        line = source[index]
        if line.startswith("|split|"):
            owner = line.split("|", 2)[2]
            private = source[index + 1] if index + 1 < len(source) else ""
            public = source[index + 2] if index + 2 < len(source) else ""
            out.append(private if owner == role else public)
            index += 3
        else:
            out.append(line)
            index += 1
    return out


def _as_stat_list(values: dict[str, int] | None, default: int) -> list[int]:
    values = values or {}
    return [int(values.get(stat, default)) for stat in _STATS]


def _state_species_id(value: object) -> str:
    """``[Species:blastoisemega]``, how Showdown serializes a species -> its id."""
    text = str(value or "")
    if text.startswith("[Species:") and text.endswith("]"):
        text = text[len("[Species:") : -1]
    return to_id_str(text)


def _restore_battle_state(pokemon: Any, exact: dict[str, Any]) -> None:
    """Put back what ``_update_from_teambuilder`` just overwrote with the sheet.

    That call resets a Pokemon to its team-sheet self: a Mega returns to its base
    forme, a spent or knocked-off item comes back, a changed ability reverts. The
    exact state knows the current forme, item and ability. (Our own side gets its
    forme from the request afterwards; an open-sheet opponent got nothing, so it was
    shown un-evolved and still holding its eaten berry until 2026-10-04.)
    """
    forme = _state_species_id(exact.get("species"))
    if forme and forme != to_id_str(pokemon.species):
        # As poke-env itself applies -mega / detailschange / -formechange: the
        # forme's stats, types and ability, under the name it already had (our own
        # side's name then follows the request).
        try:
            pokemon._update_from_pokedex(forme, store_species=False)
        except KeyError:
            pass
    if isinstance(exact.get("item"), str):
        pokemon._item = to_id_str(exact["item"]) or None
    if exact.get("ability"):
        pokemon._ability = to_id_str(exact["ability"])


def _enrich_team(
    battle: DoubleBattle, state: dict[str, Any], role: str, own: bool
) -> None:
    side_index = int(role[1]) - 1
    state_side = state["sides"][side_index]
    table = battle.team if own else battle.opponent_team
    for exact in state_side.get("pokemon", []):
        pokemon_set = exact.get("set") or {}
        name = str(pokemon_set.get("name") or pokemon_set.get("species") or "Pokemon")
        species = str(pokemon_set.get("species") or name)
        ident = f"{role}: {name}"
        details = f"{species}, L{int(pokemon_set.get('level') or 50)}"
        pokemon = table.get(ident)
        if pokemon is None:
            pokemon = battle.get_pokemon(ident, details=details, force_self_team=own)
        tb = TeambuilderPokemon(
            nickname=name,
            species=species,
            item=pokemon_set.get("item"),
            ability=pokemon_set.get("ability"),
            moves=list(pokemon_set.get("moves") or []),
            nature=pokemon_set.get("nature"),
            evs=_as_stat_list(pokemon_set.get("evs"), 0),
            ivs=_as_stat_list(pokemon_set.get("ivs"), 31),
            gender=pokemon_set.get("gender") or None,
            level=int(pokemon_set.get("level") or 50),
            tera_type=pokemon_set.get("teraType"),
        )
        pokemon._update_from_teambuilder(tb)
        _restore_battle_state(pokemon, exact)
        pokemon._selected_in_teampreview = True


def state_to_battle(
    state: dict[str, Any],
    requests: list[dict[str, Any] | None],
    role: str = "p1",
    reveal_opponent_sets: bool = True,
) -> DoubleBattle:
    """Build the policy's observation view of an exact Showdown state."""
    if role not in {"p1", "p2"}:
        raise ValueError(f"role must be p1 or p2, got {role!r}")
    request = requests[int(role[1]) - 1]
    if request is None:
        raise ValueError(f"exact state has no active request for {role}")
    battle = DoubleBattle(
        f"battle-exact-{state.get('turn', 0)}-{role}",
        str(state["sides"][int(role[1]) - 1].get("name") or role),
        _LOGGER,
        9,
    )
    perspective_lines = perspective_log(state.get("log", []), role)
    latest_snapshot = None
    # A reconciled root ends with the private marker. Exact child simulations inherit
    # the historical line and append new protocol events; reapplying the old root
    # snapshot there would erase future switches, KOs, and active slots.
    for line in reversed(perspective_lines):
        if not line:
            continue
        if line.startswith("|vgcsnapshot|"):
            try:
                latest_snapshot = json.loads(line.split("|", 2)[2])
            except (IndexError, json.JSONDecodeError):
                latest_snapshot = None
        break
    _replay(battle, perspective_lines)
    _enrich_team(battle, state, role, own=True)
    if reveal_opponent_sets:
        _enrich_team(battle, state, "p2" if role == "p1" else "p1", own=False)
    # The private request is authoritative for the active Pokemon's *current* move
    # slots. Apply it after base-team enrichment so Transform/Imposter is not reset to
    # Ditto's original Transform-only set.
    try:
        battle.parse_request(request)
    except AssertionError as exc:
        # poke-env asserts on a request it cannot square with the Pokemon it holds (a
        # fifth move: "Error with move chillyreception. Expected self.moves to
        # contain copycat, metronome ..."). That is one world failing, and the
        # planner sets a failed world aside; an AssertionError it does not catch, so
        # this ended the whole decision (2026-10-10, one of 595).
        raise ActionEncodingError(
            f"the {role} request does not fit the battle rebuilt from the state: {exc}"
        ) from exc
    if latest_snapshot is not None:
        from vgc_bench.src.live_snapshot import apply_public_snapshot

        apply_public_snapshot(battle, latest_snapshot, perspective=role)
    _apply_transformed_moves(battle, state, request, role)
    return battle


def _replay(battle: DoubleBattle, lines: Iterable[str]) -> None:
    """Feed protocol lines to a poke-env battle, tolerating what an exact branch of
    a sampled hidden world can legitimately get wrong."""
    for line in lines:
        if not line or "|" not in line:
            continue
        event = line.split("|")
        if len(event) < 2 or event[1] in _IGNORED_PROTOCOL:
            continue
        try:
            battle.parse_message(event)
        except NotImplementedError:
            # Pure UI/timestamp messages can vary with the server version and do not
            # describe battle state. Unknown mechanics are not swallowed elsewhere.
            if event[1] not in _IGNORED_PROTOCOL:
                raise
        except KeyError:
            # poke-env's end handlers use ``dict.pop`` without a default. In an
            # exact cloned branch, private/public split filtering can expose an end
            # marker after poke-env has already cleared that condition. The final
            # state is still correctly absent, so duplicate removals are idempotent.
            # ``move`` lines with a ``[from]move:`` override additionally assume the
            # overridden move exists in the mon's tracked set; Transform copies and
            # echo mechanics (e.g. Round) in sampled hidden worlds violate that, and
            # skipping the line loses only that move's PP bookkeeping.
            if event[1] not in {"-fieldend", "-sideend", "-weather", "-end", "move"}:
                raise
        except AssertionError as exc:
            # A low-mass hidden-set placeholder can survive bring-four conditioning
            # after its moves are disproven. Reconciliation repairs its current
            # public state, but its inherited branch log may then show a fifth move.
            # The concrete set is enriched below and remains authoritative; refusing
            # only this stale log-learning assertion keeps the particle evaluable.
            if not _stale_branch_move_conflict(event, exc):
                raise


def _apply_transformed_moves(
    battle: DoubleBattle, state: dict[str, Any], request: dict[str, Any], role: str
) -> None:
    state_side = state["sides"][int(role[1]) - 1]
    exact_active = {
        int(pokemon.get("position") or 0): pokemon
        for pokemon in state_side.get("pokemon", [])
        if pokemon.get("isActive")
    }
    for slot, active_request in enumerate(request.get("active") or []):
        exact_pokemon = exact_active.get(slot)
        if exact_pokemon is None or not exact_pokemon.get("transformed"):
            continue
        pokemon = battle.active_pokemon[slot]
        if pokemon is None:
            continue
        transformed = {
            move_id: Move(move_id, 9, from_transform=True)
            for raw in active_request.get("moves", [])
            if (move_id := to_id_str(raw.get("id") or raw.get("move")))
            not in SPECIAL_MOVES
        }
        pokemon._moves._transform_moves = MoveSet(transformed)


@dataclass
class LiveAnchor:
    """The live battle a session's shadows were just reconciled to.

    With it, our side's view of a shadow is the live battle itself (at the root) or
    a copy of it that has watched the shadow's new protocol (a child), instead of a
    battle rebuilt from the shadow's own log. The rebuilt view never matched the
    live one (2026-10-04: 0 of 77 decisions; a median 642 of 12,276 observation
    cells off): it holds four of our six Pokemon in a different order, forgets which
    opponents have appeared, and in a shadow recreated mid-battle starts every
    Pokemon on its first turn in its base forme.

    ``opponent_names`` maps the shadow's name for an opposing Pokemon (its species,
    as an id) to the nickname the live protocol uses.
    """

    battle: DoubleBattle
    opponent_names: dict[str, str] = field(default_factory=dict)


_IDENT_RE = re.compile(r"(?<![A-Za-z0-9])p([12])([abc]?): ([^|]*)")
_SNAPSHOT_MARK = "|vgcsnapshot|"


def lines_since_reconcile(log: Sequence[str]) -> list[str] | None:
    """The protocol a shadow has produced since it was last reconciled to the live
    battle (None: it never was)."""
    for index in range(len(log) - 1, -1, -1):
        if log[index].startswith(_SNAPSHOT_MARK):
            return list(log[index + 1 :])
    return None


def to_live_line(line: str, anchor: LiveAnchor) -> str:
    """A shadow's protocol line as the live battle would have received it.

    Shadows seat us as p1 and name every opposing Pokemon by its species; the server
    may have seated us as p2 and the opponent may use nicknames.
    """
    flip = getattr(anchor.battle, "player_role", None) == "p2"

    def rewrite(match: re.Match[str]) -> str:
        seat, slot, name = match.group(1), match.group(2), match.group(3)
        if seat == "2":
            name = anchor.opponent_names.get(to_id_str(name), name)
        if flip:
            seat = "1" if seat == "2" else "2"
        return f"p{seat}{slot}: {name}"

    return _IDENT_RE.sub(rewrite, line)


def _to_live_request(request: dict[str, Any], anchor: LiveAnchor) -> dict[str, Any]:
    if getattr(anchor.battle, "player_role", None) != "p2":
        return request
    request = copy.deepcopy(request)
    side = request.get("side") or {}
    if side.get("id") == "p1":
        side["id"] = "p2"
    for pokemon in side.get("pokemon") or []:
        ident = str(pokemon.get("ident") or "")
        if ident.startswith("p1"):
            pokemon["ident"] = "p2" + ident[2:]
    return request


def live_view(
    anchor: LiveAnchor, state: dict[str, Any], request: dict[str, Any] | None
) -> DoubleBattle | None:
    """Our view of a reconciled shadow, grown from the live battle (None: this state
    was never reconciled, or has no request for us)."""
    new_lines = lines_since_reconcile(state.get("log", []))
    if new_lines is None or request is None:
        return None
    if not any(line for line in new_lines):
        return anchor.battle
    battle = copy.deepcopy(anchor.battle)
    ours = perspective_log(new_lines, "p1")
    _replay(battle, (to_live_line(line, anchor) for line in ours))
    battle.parse_request(_to_live_request(request, anchor))
    _apply_transformed_moves(battle, state, request, "p1")
    return battle


_TARGET_RE = re.compile(r"^[+-]?[12]$")


class ActionEncodingError(ValueError):
    """An exact legal Showdown choice cannot be represented by the policy."""


def _choice_atoms(
    choice: str,
    request: dict[str, Any],
    battle: DoubleBattle | None = None,
    state: dict[str, Any] | None = None,
    role: str = "p1",
) -> list[str]:
    """Restore a canonical choice's omitted leading pass in one-Pokemon endgames.

    Showdown's ``Side.getChoice()`` serializes ``pass, move moonblast`` as only
    ``move moonblast`` when slot 0 is empty. The policy action space still has two
    fixed slots, so blindly appending a pass assigns that move to the fainted slot.
    """
    atoms = [atom.strip() for atom in choice.split(",") if atom.strip()]
    if len(atoms) == 1:
        forced = list(request.get("forceSwitch") or [])
        if len(forced) >= 2 and sum(bool(value) for value in forced[:2]) == 1:
            required_slot = next(index for index, value in enumerate(forced[:2]) if value)
            if required_slot == 1:
                atoms.insert(0, "pass")
        elif len(request.get("active") or []) >= 2:
            request_slots = [
                index
                for index, active in enumerate((request.get("active") or [])[:2])
                if active is not None
            ]
            if request_slots == [1]:
                atoms.insert(0, "pass")
        elif state is not None:
            try:
                exact_slots = sorted(
                    int(pokemon.get("position") or 0)
                    for pokemon in state["sides"][int(role[1]) - 1].get(
                        "pokemon", []
                    )
                    if pokemon.get("isActive")
                    and not pokemon.get("fainted")
                    and float(pokemon.get("hp") or 0) > 0
                )
            except (IndexError, KeyError, TypeError, ValueError):
                exact_slots = []
            if exact_slots == [1]:
                atoms.insert(0, "pass")
        elif battle is not None:
            live_slots = [
                index
                for index, pokemon in enumerate(battle.active_pokemon[:2])
                if pokemon is not None and not getattr(pokemon, "fainted", False)
            ]
            if live_slots == [1]:
                atoms.insert(0, "pass")
    while len(atoms) < 2:
        atoms.append("pass")
    return atoms[:2]


def _choice_targets_empty_slot(atoms: Iterable[str], battle: DoubleBattle) -> bool:
    """True for Showdown's legal but no-effect target variants on empty slots.

    Showdown enumerates commands such as ``pass, move weatherball -1`` after the
    left ally has fainted. poke-env intentionally removes empty coordinates from its
    action mask. These are not distinct strategic actions—the move simply has no
    target—so omit only this known representational surplus while continuing to
    raise on every genuine action-mapping incompatibility.
    """
    ours = getattr(battle, "active_pokemon", ()) or ()
    foes = getattr(battle, "opponent_active_pokemon", ()) or ()
    for atom in atoms:
        parts = atom.split()
        if not parts or parts[0] != "move":
            continue
        targets = [int(part) for part in parts[2:] if _TARGET_RE.match(part)]
        for target in targets:
            slots = ours if target < 0 else foes
            index = abs(target) - 1
            pokemon = slots[index] if 0 <= index < len(slots) else None
            if pokemon is None or getattr(pokemon, "fainted", False):
                return True
    return False


def _switch_action(
    exact_slot: int,
    *,
    state: dict[str, Any] | None,
    role: str,
    battle: DoubleBattle | None,
) -> int:
    """Map Showdown's mutable party slot to poke-env's stable team action ID.

    Showdown swaps entries in ``side.pokemon`` as Pokemon enter and leave the field,
    so a later ``switch 3`` does not necessarily mean the third Pokemon in
    ``battle.team``.  poke-env action IDs, however, always index that stable mapping.
    Resolve through the Pokemon nickname/identity instead of copying the integer.

    ``state`` and ``battle`` remain optional for the small request-only unit tests and
    for backwards-compatible callers that contain no switch action.
    """
    if state is None or battle is None:
        return exact_slot
    try:
        exact = state["sides"][int(role[1]) - 1]["pokemon"][exact_slot - 1]
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise ActionEncodingError(
            f"Showdown switch slot {exact_slot} is absent for {role}"
        ) from exc
    pokemon_set = exact.get("set") or {}
    nickname = to_id_str(
        pokemon_set.get("name")
        or pokemon_set.get("species")
        or exact.get("name")
        or ""
    )
    species = to_id_str(
        pokemon_set.get("species")
        or exact.get("baseSpecies")
        or exact.get("species")
        or ""
    )
    matches: list[int] = []
    for action, (ident, pokemon) in enumerate(battle.team.items(), start=1):
        ident_name = to_id_str(ident.split(":", 1)[-1])
        if nickname and ident_name == nickname:
            return action
        if species and to_id_str(pokemon.species) == species:
            matches.append(action)
        elif species and to_id_str(pokemon.base_species) == species:
            matches.append(action)
    if len(matches) == 1:
        return matches[0]
    raise ActionEncodingError(
        f"cannot map Showdown switch slot {exact_slot} ({nickname or species}) "
        f"to the {role} policy team"
    )


def choice_to_actions(
    choice: str,
    request: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
    role: str = "p1",
    battle: DoubleBattle | None = None,
) -> tuple[int, int]:
    """Convert one canonical Showdown doubles choice to vgc-bench action IDs."""
    atoms = _choice_atoms(choice, request, battle, state, role)
    actions: list[int] = []
    for slot, atom in enumerate(atoms[:2]):
        parts = atom.split()
        if not parts or parts[0] in {"pass", "skip"}:
            actions.append(0)
            continue
        if parts[0] == "switch":
            actions.append(
                _switch_action(
                    int(parts[1]), state=state, role=role, battle=battle
                )
            )
            continue
        if parts[0] != "move" or len(parts) < 2:
            raise ActionEncodingError(f"unsupported exact choice atom: {atom!r}")
        active = (request.get("active") or [])[slot]
        move_id = to_id_str(parts[1])
        request_moves = active.get("moves") or []
        move_slot = next(
            i
            for i, move in enumerate(request_moves)
            if to_id_str(move.get("id") or move.get("move")) == move_id
        )
        if battle is not None:
            active_mon = battle.active_pokemon[slot]
            if active_mon is None:
                raise ActionEncodingError(
                    f"exact move {move_id!r} has no active Pokemon in slot {slot}"
                )
            available = list(battle.available_moves[slot])
            known = list(active_mon.moves.values())[:4]
            available_ids = [move.id for move in available]
            known_ids = [move.id for move in known]
            policy_moves = known if move_id in known_ids else available
            try:
                move_slot = [move.id for move in policy_moves].index(move_id)
            except ValueError as exc:
                raise ActionEncodingError(
                    f"exact move {move_id!r} is absent from the slot {slot} policy "
                    f"vocabulary {known_ids} with available {available_ids}"
                ) from exc
        target = next((int(p) for p in parts[2:] if _TARGET_RE.match(p)), 0)
        # Recharge/Struggle/Fight are represented by poke-env as the generic first
        # move with target zero even if Showdown's canonical choice includes the
        # automatically selected foe coordinate.
        if move_id in SPECIAL_MOVES:
            target = 0
        action = 7 + move_slot * 5 + target + 2
        if "mega" in parts or "megax" in parts or "megay" in parts:
            action += 20
        elif "zmove" in parts or "ultra" in parts:
            action += 40
        elif "dynamax" in parts:
            action += 60
        elif "terastallize" in parts or "terastal" in parts:
            action += 80
        actions.append(action)
    return actions[0], actions[1]


@dataclass(frozen=True)
class RankedChoice:
    choice: str
    actions: tuple[int, int] | None
    probability: float


class ExactPolicyAdapter:
    """Policy priors and critic values for exact simulator states."""

    def __init__(
        self,
        policy,
        preview_predictor: PreviewPredictor | None = None,
        reveal_opponent_sets: bool = True,
        residual_ranker=None,
        inference_lock: threading.RLock | None = None,
    ):
        self.policy = policy
        self.preview_predictor = preview_predictor
        self.residual_ranker = residual_ranker
        self.inference_lock = inference_lock or threading.RLock()
        # The exact simulator always needs both true sets, but the student and its
        # opponent prior must only see what a real player would see. Generation
        # toggles this per game to produce both open- and hidden-sheet examples.
        self.reveal_opponent_sets = reveal_opponent_sets
        # set by a live session before each decision: our side's views then grow
        # from the live battle (LiveAnchor) instead of being rebuilt from the log
        self.live_anchor: LiveAnchor | None = None
        self.live_masked_choices = 0

    @staticmethod
    def _roster(state, role: str) -> tuple[str, ...]:
        side = state["sides"][int(role[1]) - 1]
        return tuple(
            to_id_str(
                (pokemon.get("set") or {}).get("species") or pokemon.get("species")
            )
            for pokemon in side.get("pokemon", [])
        )

    def _rank_preview(self, state, role: str, choices: list[str]) -> list[RankedChoice]:
        if self.preview_predictor is None:
            probability = 1.0 / max(1, len(choices))
            return [RankedChoice(choice, None, probability) for choice in choices]
        ours = self._roster(state, role)
        theirs = self._roster(state, "p2" if role == "p1" else "p1")
        legal = {choice.replace(", ", ","): choice for choice in choices}
        ranked = []
        for plan in self.preview_predictor.predict_plans(ours, theirs, top_k=90):
            order = plan_to_showdown_order(plan)
            key = ("team " + ",".join(str(index) for index in order)).replace(", ", ",")
            if key in legal:
                ranked.append(RankedChoice(legal[key], None, plan.probability))
        return ranked

    def _inputs(self, state, requests, role):
        battle = None
        if role == "p1" and self.live_anchor is not None:
            battle = live_view(self.live_anchor, state, requests[0])
        if battle is None:
            battle = state_to_battle(
                state, requests, role, reveal_opponent_sets=self.reveal_opponent_sets
            )
        obs = PolicyPlayer.embed_battle(battle, fake_rating=2000)
        mask = np.asarray(DoublesEnv.get_action_mask(battle), dtype=np.float32)
        obs_dict = {
            "observation": torch.as_tensor(obs, device=self.policy.device).unsqueeze(0),
            "action_mask": torch.as_tensor(mask, device=self.policy.device).unsqueeze(
                0
            ),
        }
        return battle, obs, mask, obs_dict

    def observation(self, state, requests, role: str = "p1"):
        """Return the exact state's policy observation and legal action mask."""
        _battle, obs, mask, _obs_dict = self._inputs(state, requests, role)
        return obs, mask

    def rank(
        self, state, requests, role: str, choices: list[str]
    ) -> list[RankedChoice]:
        """Return legal choices ranked by the policy's joint probability."""
        if choices == [""]:
            return [RankedChoice("", (0, 0), 1.0)]
        if choices and choices[0].startswith("team "):
            return self._rank_preview(state, role, choices)
        battle, _obs, mask, obs_dict = self._inputs(state, requests, role)
        # A view grown from the live battle carries the LIVE action mask. A shadow
        # can allow what the live request does not (a switch the server has trapped,
        # a move it has disabled); with a rebuilt view the two masks were the same
        # object and a mismatch meant an encoding bug.
        live = role == "p1" and self.live_anchor is not None
        request = requests[int(role[1]) - 1]
        assert request is not None
        encoded: list[
            tuple[str, tuple[int, int], tuple[bool, bool], int]
        ] = []
        # the joint mask depends on the first slot's action alone: one update per
        # distinct first action, not one per choice (174 choices share about a dozen)
        joint_masks: dict[int, Any] = {}
        for choice in choices:
            atoms = _choice_atoms(choice, request, battle, state, role)
            if _choice_targets_empty_slot(atoms, battle):
                continue
            actions = choice_to_actions(
                choice, request, state=state, role=role, battle=battle
            )
            pass_slots = tuple(
                not atom.split() or atom.split()[0] in {"pass", "skip"}
                for atom in atoms[:2]
            )
            act_len = len(mask) // 2
            valid_first = np.flatnonzero(mask[:act_len])
            if not len(valid_first):
                raise ActionEncodingError(
                    f"policy mask has no slot-0 action for {role}"
                )
            conditioning_first = (
                int(valid_first[0]) if pass_slots[0] else actions[0]
            )
            if not pass_slots[0] and not bool(mask[actions[0]]):
                if live:
                    # legal in this shadow, not in the live battle: not ours to play
                    self.live_masked_choices += 1
                    continue
                raise ActionEncodingError(
                    f"exact legal choice {choice!r} maps slot 0 to masked action "
                    f"{actions[0]} for {role}"
                )
            joint_mask = joint_masks.get(conditioning_first)
            if joint_mask is None:
                first = torch.tensor(
                    [[conditioning_first]], device=self.policy.device
                )
                joint_mask = self.policy._update_mask(
                    obs_dict["action_mask"], first
                )[0]
                joint_masks[conditioning_first] = joint_mask
            if not pass_slots[1] and not bool(joint_mask[act_len + actions[1]]):
                if live:
                    self.live_masked_choices += 1
                    continue
                raise ActionEncodingError(
                    f"exact legal choice {choice!r} maps slot 1 to masked action "
                    f"{actions[1]} for {role}"
                )
            encoded.append((choice, actions, pass_slots, conditioning_first))
        if not encoded:
            return []
        with self.inference_lock:
            with torch.no_grad():
                if hasattr(self.policy, "logits_with_latent"):
                    logits, _value, latent = self.policy.logits_with_latent(
                        obs_dict, actor_grad=False
                    )
                else:  # test stubs and older policy objects
                    logits, _value = self.policy.get_logits(obs_dict, actor_grad=False)
                    latent = None
                first = (
                    self.policy.get_dist_from_logits(logits, obs_dict["action_mask"])
                    .distribution[0]
                    .probs[0]
                )
                first_actions = torch.as_tensor(
                    [
                        [conditioning_first]
                        for _choice, _actions, _pass_slots, conditioning_first in encoded
                    ],
                    device=self.policy.device,
                )
                repeated_logits = logits.repeat(len(encoded), 1)
                repeated_latent = (
                    latent.repeat(len(encoded), 1) if latent is not None else None
                )
                repeated_mask = obs_dict["action_mask"].repeat(len(encoded), 1)
                non_pass_second = [
                    index
                    for index, (
                        _choice,
                        _actions,
                        pass_slots,
                        _conditioning,
                    ) in enumerate(encoded)
                    if not pass_slots[1]
                ]
                second_probabilities = torch.ones(
                    len(encoded), dtype=first.dtype, device=self.policy.device
                )
                if non_pass_second:
                    selected = torch.as_tensor(
                        non_pass_second, dtype=torch.long, device=self.policy.device
                    )
                    latent_args = (
                        (repeated_latent[selected],)
                        if repeated_latent is not None
                        else ()
                    )
                    second = (
                        self.policy.get_dist_from_logits(
                            repeated_logits[selected],
                            repeated_mask[selected],
                            first_actions[selected],
                            *latent_args,
                        )
                        .distribution[1]
                        .probs
                    )
                    for row, index in enumerate(non_pass_second):
                        second_probabilities[index] = second[
                            row, encoded[index][1][1]
                        ]
                probabilities = [
                    float(
                        (1.0 if pass_slots[0] else first[actions[0]])
                        * second_probabilities[index]
                    )
                    for index, (
                        _choice,
                        actions,
                        pass_slots,
                        _conditioning_first,
                    ) in enumerate(encoded)
                ]
        total = sum(probabilities)
        if total <= 0:
            probabilities = [1.0 / len(encoded)] * len(encoded)
        else:
            probabilities = [p / total for p in probabilities]
        ranked = [
            RankedChoice(choice, actions, probability)
            for (choice, actions, _pass_slots, _conditioning_first), probability in zip(
                encoded, probabilities
            )
        ]
        if self.residual_ranker is not None and len(ranked) >= 2:
            from vgc_bench.src.residual_ranker import (
                candidate_semantic_features,
                champion_features,
            )

            with self.inference_lock:
                residual_device = next(self.residual_ranker.parameters()).device
                features = champion_features(self.policy, obs_dict).to(residual_device)
                residual_actions = torch.as_tensor(
                    [[item.actions for item in ranked]],
                    dtype=torch.long,
                    device=residual_device,
                )
                champion_log = torch.as_tensor(
                    [[max(1e-12, item.probability) for item in ranked]],
                    dtype=torch.float32,
                    device=residual_device,
                ).log()
                with torch.no_grad():
                    # Contextual-confidence residuals require per-candidate
                    # semantic features; this exact-adapter path predated them
                    # and passed nothing, so --rollout-residual crashed on any
                    # ranker trained after the architecture change.
                    adjusted, confidence, _residual = self.residual_ranker(
                        features,
                        residual_actions,
                        champion_log,
                        (
                            candidate_semantic_features(
                                self.policy, obs_dict, residual_actions
                            ).to(residual_device)
                            if self.residual_ranker.config.candidate_feature_dim
                            else None
                        ),
                    )
                if (
                    float(confidence[0])
                    >= self.residual_ranker.config.confidence_threshold
                ):
                    adjusted_probabilities = adjusted[0].softmax(dim=0).cpu().tolist()
                    ranked = [
                        RankedChoice(item.choice, item.actions, float(probability))
                        for item, probability in zip(ranked, adjusted_probabilities)
                    ]
        return sorted(ranked, key=lambda item: item.probability, reverse=True)

    def value(self, state, requests, role: str = "p1") -> float:
        """Return the PPO critic value from ``role``'s perspective."""
        _battle, _obs, _mask, obs_dict = self._inputs(state, requests, role)
        with self.inference_lock, torch.no_grad():
            _logits, value = self.policy.get_logits(obs_dict, actor_grad=False)
        return float(value.item())


def _padded_roster(names: Iterable[str], width: int = 6) -> tuple[str, ...]:
    """Opponent models were trained on six-slot previews, while exact post-preview
    states retain only the four brought Pokemon. Padding preserves the model's input
    shape without inventing species that are no longer relevant to the battle.
    """
    roster = tuple(to_id_str(name) for name in names)
    return (roster + ("",) * width)[:width]


def _target_class(atom: str, actor_slot: int) -> str | None:
    parts = atom.split()
    target = next((int(part) for part in parts[2:] if _TARGET_RE.match(part)), None)
    if target is None:
        return None
    if target > 0:
        return "foe_a" if target == 1 else "foe_b"
    own_slot = abs(target) - 1
    return "self" if own_slot == actor_slot else "ally"


def opponent_choice_likelihood(
    choice: str,
    move_predictions: tuple[MovePrediction, MovePrediction] | None,
    switch_predictions: tuple[SwitchPrediction, SwitchPrediction] | None,
    roster: tuple[str, ...],
    *,
    forced_switch: bool = False,
) -> float:
    """Likelihood of one canonical joint choice under the opponent models.

    This function is intentionally pure so target-coordinate handling and joint
    probabilities can be regression tested without launching a simulator.
    """
    atoms = [atom.strip() for atom in choice.split(",")]
    while len(atoms) < 2:
        atoms.append("pass")
    likelihood = 1.0
    for slot, atom in enumerate(atoms[:2]):
        parts = atom.split()
        if not parts or parts[0] in {"pass", "skip"}:
            continue
        if parts[0] == "switch":
            if switch_predictions is None:
                continue
            prediction = switch_predictions[slot]
            try:
                species = roster[int(parts[1]) - 1]
            except (IndexError, ValueError):
                return 0.0
            target_probability = dict(prediction.targets).get(to_id_str(species), 0.0)
            switch_probability = 1.0 if forced_switch else prediction.switch_probability
            likelihood *= max(1e-5, switch_probability * target_probability)
            continue
        if parts[0] != "move" or move_predictions is None or len(parts) < 2:
            continue
        prediction = move_predictions[slot]
        move_id = to_id_str(parts[1])
        move_probability = dict(prediction.moves).get(move_id, 0.0)
        target_class = _target_class(atom, slot)
        if target_class is not None and move_probability > 0:
            joint = {
                (candidate_move, candidate_target): probability
                for candidate_move, candidate_target, probability in prediction.actions
            }.get((move_id, target_class), 0.0)
            # Target prediction is useful but materially noisier than move prediction.
            # A floor prevents a target-head miss from deleting an otherwise plausible
            # exact branch.
            conditional = joint / move_probability if move_probability else 0.0
            move_probability *= 0.35 + 0.65 * conditional
        reliability = max(0.0, min(1.0, prediction.reliability))
        uniform = 1.0 / max(1, len(prediction.moves))
        likelihood *= max(
            1e-5, reliability * move_probability + (1 - reliability) * uniform
        )
    return float(likelihood)


def _state_roster(state: dict[str, Any], role: str) -> tuple[str, ...]:
    """Base species of ``role``'s party in an exact state, by party position (what a
    "switch N" names). Empty where a record cannot be read."""
    try:
        party = state["sides"][int(role[1]) - 1]["pokemon"]
    except (IndexError, KeyError, TypeError, ValueError):
        return ()
    names = []
    for pokemon in party:
        if not isinstance(pokemon, dict):
            names.append("")
            continue
        pokemon_set = pokemon.get("set") or {}
        names.append(
            to_id_str(
                pokemon_set.get("species")
                or pokemon.get("baseSpecies")
                or pokemon.get("species")
                or ""
            )
        )
    return tuple(names)


def _mega_likelihood(choice: str, mega: Sequence[float | None] | None) -> float:
    """How well a joint choice agrees with "this slot Mega-evolves this turn".

    A move atom with the Mega Evolution is credited with the slot's Mega probability,
    one without it with the rest. Where a slot cannot Mega-evolve every atom of it
    carries the same factor, which the caller's normalisation removes. Clamped so one
    confident head cannot erase the other spelling.
    """
    if not mega:
        return 1.0
    atoms = [atom.strip() for atom in choice.split(",")]
    factor = 1.0
    for slot, atom in enumerate(atoms[:2]):
        probability = mega[slot] if slot < len(mega) else None
        if probability is None or not atom.startswith("move "):
            continue
        probability = min(0.98, max(0.02, float(probability)))
        # the event is a word of its own after the move ("move megahorn +1 mega")
        evolves = "mega" in atom.split()[2:]
        factor *= probability if evolves else 1.0 - probability
    return factor


FORECAST_MIXES = ("sum", "product")


def mixed_probability(pair: tuple[float, float], weight: float, mix: str) -> float:
    """One reply's (brain, forecast) probabilities as one number, unnormalised."""
    brain, forecast = pair
    if mix == "product":
        return max(1e-12, brain) ** (1.0 - weight) * max(1e-12, forecast) ** weight
    return (1.0 - weight) * brain + weight * forecast


def replies_by_mixture(
    entry: Mapping[str, tuple[float, float]], weight: float, mix: str = "sum"
) -> list[str]:
    """A world's replies, likeliest first, as a prior with this weight and mix
    would have ranked them (ties keep the order they were logged in: the brain's)."""
    return sorted(
        entry, key=lambda choice: -mixed_probability(entry[choice], weight, mix)
    )


@dataclass(frozen=True)
class RootForecast:
    """The opponent predictor's forecast of the reply to ONE live decision: the two
    slots' move and switch predictions in the shape the opponent models have always
    been read in, each slot's Mega probability, and the turn it was made for."""

    turn: int
    moves: tuple[MovePrediction, MovePrediction] | None
    switches: tuple[SwitchPrediction, SwitchPrediction] | None
    mega: tuple[float | None, float | None] | None = None


class OpponentModelPrior:
    """Use high-rated-player models for opponent branches, not our own policy.

    The policy remains a small smoothing prior so out-of-distribution model outputs
    cannot erase legal responses. For our side, ranking is unchanged.

    ``root_forecast`` (2026-10-09, opt-in): the opponent predictor's forecast for the
    live decision. While it is set, the opponent's replies to THAT decision -- a move
    request of the forecast's turn -- are ranked with it at ``forecast_weight``
    and ``forecast_mix`` (``_rank_with_forecast``) instead of the move / switch
    models; everything else (replacements, later turns) is ranked as before.
    """

    def __init__(
        self,
        base: ExactPolicyAdapter,
        move_predictor: MovePredictor | None = None,
        switch_predictor: SwitchPredictor | None = None,
        controlled_role: str = "p1",
        model_weight: float = 0.60,
        open_sheet_model_weight: float = 0.40,
        forecast_weight: float = 0.50,
        forecast_mix: str = "sum",
    ):
        if not 0 <= model_weight <= 1 or not 0 <= open_sheet_model_weight <= 1:
            raise ValueError("opponent model weights must be in [0, 1]")
        if not 0 <= forecast_weight <= 1:
            raise ValueError("forecast_weight must be in [0, 1]")
        if forecast_mix not in FORECAST_MIXES:
            raise ValueError(f"forecast_mix must be one of {FORECAST_MIXES}")
        self.base = base
        self.move_predictor = move_predictor
        self.switch_predictor = switch_predictor
        self.controlled_role = controlled_role
        self.model_weight = model_weight
        self.open_sheet_model_weight = open_sheet_model_weight
        self.forecast_weight = forecast_weight
        self.forecast_mix = forecast_mix
        self.root_forecast: RootForecast | None = None
        # rankings the forecast served since it was last set (the audit reads it)
        self.forecast_rankings = 0
        # per world (id of its state): reply -> (brain, forecast) probability, for
        # the decision being searched and for the one before it
        self.reply_log: dict[int, dict[str, tuple[float, float]]] = {}
        self.previous_reply_log: dict[int, dict[str, tuple[float, float]]] = {}

    def set_root_forecast(self, forecast: RootForecast | None) -> None:
        """The forecast for the decision about to be searched; None clears it. What
        the last decision logged moves to ``previous_reply_log``: by the next
        decision the opponent's reply to it is known."""
        self.previous_reply_log, self.reply_log = self.reply_log, {}
        self.root_forecast = forecast
        self.forecast_rankings = 0

    def _forecast_for(self, state, forced: bool) -> RootForecast | None:
        """The live forecast, when this is the opponent's reply to the decision it
        was made for: a move request of the forecast's own turn."""
        forecast = self.root_forecast
        if forecast is None or forced:
            return None
        if forecast.moves is None and forecast.switches is None:
            return None
        try:
            turn = int(state.get("turn") or 0)
        except (AttributeError, TypeError, ValueError):
            return None
        return forecast if turn == forecast.turn else None

    @staticmethod
    def _species(mon) -> str:
        return to_id_str(mon.base_species) if mon is not None else ""

    def _predictions(self, state, requests, role: str):
        battle = state_to_battle(
            state,
            requests,
            role,
            reveal_opponent_sets=self.base.reveal_opponent_sets,
        )
        active = tuple(self._species(mon) for mon in battle.active_pokemon)
        opponent_active = tuple(
            self._species(mon) for mon in battle.opponent_active_pokemon
        )
        if len(active) != 2 or len(opponent_active) != 2 or not all(active):
            return None, None, (), battle
        roster = _padded_roster(
            self._species(mon) for mon in battle.team.values()
        )
        opponent_roster = _padded_roster(
            self._species(mon) for mon in battle.opponent_team.values()
        )
        hp = tuple(
            float(mon.current_hp_fraction or 0.0)
            for mon in (*battle.active_pokemon, *battle.opponent_active_pokemon)
            if mon is not None
        )
        if len(hp) != 4:
            return None, None, roster, battle
        request = requests[int(role[1]) - 1] or {}
        active_requests = request.get("active") or []
        move_predictions = None
        if self.move_predictor is not None and len(active_requests) == 2:
            predicted = []
            for slot in range(2):
                available = tuple(
                    to_id_str(move.get("id") or move.get("move"))
                    for move in active_requests[slot].get("moves", [])
                    if not move.get("disabled")
                )
                result = self.move_predictor.predict(
                    roster,
                    opponent_roster,
                    active,
                    opponent_active,
                    hp,
                    actor_slot=slot,
                    turn=int(state.get("turn") or 0),
                    available_moves=available,
                )
                predicted.append(result)
            move_predictions = (predicted[0], predicted[1])

        switch_predictions = None
        if self.switch_predictor is not None:
            predicted_switches = tuple(
                self.switch_predictor.predict(
                    roster,
                    opponent_roster,
                    active,
                    opponent_active,
                    hp,
                    actor_slot=slot,
                    turn=int(state.get("turn") or 0),
                )
                for slot in range(2)
            )
            switch_predictions = (predicted_switches[0], predicted_switches[1])
        return move_predictions, switch_predictions, roster, battle

    def rank(self, state, requests, role: str, choices: list[str]):
        ranked = self.base.rank(state, requests, role, choices)
        if (
            role == self.controlled_role
            or not ranked
            or ranked[0].choice.startswith("team ")
        ):
            return ranked
        request = requests[int(role[1]) - 1] or {}
        forced = any(bool(value) for value in request.get("forceSwitch", []))
        # one legal reply (the opponent waits while we replace): nothing to rank
        forecast = self._forecast_for(state, forced) if len(ranked) > 1 else None
        if forecast is not None:
            return self._rank_with_forecast(state, role, ranked, forecast)
        if self.move_predictor is None and self.switch_predictor is None:
            return ranked
        try:
            moves, switches, roster, _battle = self._predictions(state, requests, role)
        except (AssertionError, KeyError, RuntimeError, ValueError):
            return ranked
        if moves is None and switches is None:
            return ranked
        model_weight = (
            self.open_sheet_model_weight
            if self.base.reveal_opponent_sets
            else self.model_weight
        )
        base_weight = 1.0 - model_weight
        rescored = []
        for item in ranked:
            model_probability = opponent_choice_likelihood(
                item.choice,
                moves,
                switches,
                roster,
                forced_switch=forced,
            )
            probability = max(1e-12, item.probability) ** base_weight
            probability *= max(1e-12, model_probability) ** model_weight
            rescored.append(RankedChoice(item.choice, item.actions, probability))
        total = sum(item.probability for item in rescored)
        if total <= 0:
            return ranked
        return sorted(
            (
                RankedChoice(item.choice, item.actions, item.probability / total)
                for item in rescored
            ),
            key=lambda item: item.probability,
            reverse=True,
        )

    def _rank_with_forecast(self, state, role: str, ranked, forecast: RootForecast):
        """The opponent's replies to the live decision, ranked by the brain's prior
        and the forecast together.

        Both are made distributions over the legal replies and mixed:
        ``forecast_mix`` "sum" is ``(1 - w) * brain + w * forecast``, a reply either
        of them expects stays near the top; "product" is ``brain^(1 - w) *
        forecast^w``, as the move / switch models are blended, which keeps only what
        both expect. On the ladder trial's first games the two lists missed
        different replies (hidden sheets: both held the real reply 6 times, only
        the forecast 4, only the brain 5, neither 8 -- of 23), which is the case
        for the sum. With ``w = 0`` the ranking is the brain's own, reply for
        reply, and the pair of probabilities is still kept (``reply_log``) for the
        session to hold against what the opponent then does.
        """
        roster = _state_roster(state, role)
        likelihoods = []
        for item in ranked:
            likelihood = opponent_choice_likelihood(
                item.choice, forecast.moves, forecast.switches, roster
            )
            if forecast.mega is not None:
                likelihood *= _mega_likelihood(item.choice, forecast.mega)
            likelihoods.append(max(0.0, float(likelihood)))
        forecast_total = sum(likelihoods)
        brain_total = sum(max(0.0, float(item.probability)) for item in ranked)
        if forecast_total <= 0 or brain_total <= 0:
            return ranked
        pairs = [
            (max(0.0, float(item.probability)) / brain_total, value / forecast_total)
            for item, value in zip(ranked, likelihoods)
        ]
        self.forecast_rankings += 1
        # Keyed by the world's state OBJECT, which the session still holds when the
        # opponent's reply is known. Nothing in a state tells two worlds apart: they
        # often give the Pokemon on the field the same moves, and once advanced on
        # shared random streams they carry the same seed too -- while the brain
        # ranks their replies differently (what is on the bench differs).
        self.reply_log[id(state)] = {
            item.choice: pair for item, pair in zip(ranked, pairs)
        }
        weight = self.forecast_weight
        if weight <= 0:
            return ranked
        mixed = [mixed_probability(pair, weight, self.forecast_mix) for pair in pairs]
        total = sum(mixed)
        if total <= 0:
            return ranked
        return sorted(
            (
                RankedChoice(item.choice, item.actions, value / total)
                for item, value in zip(ranked, mixed)
            ),
            key=lambda item: item.probability,
            reverse=True,
        )
