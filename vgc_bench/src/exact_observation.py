"""Adapt exact Pokemon Showdown states to the existing policy network.

The exact simulator owns the forward state.  poke-env remains useful as the policy's
observation/action vocabulary, so this module reconstructs a read-only ``DoubleBattle``
from Showdown's public log and side request.  It never tries to turn that reconstruction
back into simulator state; directionality is what keeps the planner honest.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from typing import Any, Iterable

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
    for line in perspective_lines:
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
    _enrich_team(battle, state, role, own=True)
    if reveal_opponent_sets:
        _enrich_team(battle, state, "p2" if role == "p1" else "p1", own=False)
    # The private request is authoritative for the active Pokemon's *current* move
    # slots. Apply it after base-team enrichment so Transform/Imposter is not reset to
    # Ditto's original Transform-only set.
    battle.parse_request(request)
    if latest_snapshot is not None and role == "p1":
        from vgc_bench.src.live_snapshot import apply_public_snapshot

        apply_public_snapshot(battle, latest_snapshot)
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
        request = requests[int(role[1]) - 1]
        assert request is not None
        encoded: list[
            tuple[str, tuple[int, int], tuple[bool, bool], int]
        ] = []
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
                raise ActionEncodingError(
                    f"exact legal choice {choice!r} maps slot 0 to masked action "
                    f"{actions[0]} for {role}"
                )
            first = torch.tensor([[conditioning_first]], device=self.policy.device)
            joint_mask = self.policy._update_mask(
                obs_dict["action_mask"], first
            )[0]
            if not pass_slots[1] and not bool(joint_mask[act_len + actions[1]]):
                raise ActionEncodingError(
                    f"exact legal choice {choice!r} maps slot 1 to masked action "
                    f"{actions[1]} for {role}"
                )
            encoded.append((choice, actions, pass_slots, conditioning_first))
        if not encoded:
            return []
        with self.inference_lock:
            with torch.no_grad():
                logits, _value = self.policy.get_logits(obs_dict, actor_grad=False)
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
                    second = (
                        self.policy.get_dist_from_logits(
                            repeated_logits[selected],
                            repeated_mask[selected],
                            first_actions[selected],
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


class OpponentModelPrior:
    """Use high-rated-player models for opponent branches, not our own policy.

    The policy remains a small smoothing prior so out-of-distribution model outputs
    cannot erase legal responses. For our side, ranking is unchanged.
    """

    def __init__(
        self,
        base: ExactPolicyAdapter,
        move_predictor: MovePredictor | None = None,
        switch_predictor: SwitchPredictor | None = None,
        controlled_role: str = "p1",
        model_weight: float = 0.60,
        open_sheet_model_weight: float = 0.40,
    ):
        if not 0 <= model_weight <= 1 or not 0 <= open_sheet_model_weight <= 1:
            raise ValueError("opponent model weights must be in [0, 1]")
        self.base = base
        self.move_predictor = move_predictor
        self.switch_predictor = switch_predictor
        self.controlled_role = controlled_role
        self.model_weight = model_weight
        self.open_sheet_model_weight = open_sheet_model_weight

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
            or (self.move_predictor is None and self.switch_predictor is None)
        ):
            return ranked
        try:
            moves, switches, roster, _battle = self._predictions(state, requests, role)
        except (AssertionError, KeyError, RuntimeError, ValueError):
            return ranked
        if moves is None and switches is None:
            return ranked
        request = requests[int(role[1]) - 1] or {}
        forced = any(bool(value) for value in request.get("forceSwitch", []))
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
