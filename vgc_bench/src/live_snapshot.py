"""Public poke-env snapshot used to reconcile a live exact shadow battle.

Pokemon Showdown does not expose its serialized server state on ladder. A safe live
planner therefore maintains concrete shadow battles and repairs each shadow from the
new public request before branching. This module contains only information available
to the client; hidden set details remain supplied by the selected determinization.
"""

from __future__ import annotations

from typing import Any, Mapping

from poke_env.battle import (
    AbstractBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import to_id_str


def _enum_id(value: object) -> str:
    return to_id_str(getattr(value, "name", value))


def _condition_duration(identifier: str, battle: AbstractBattle, started: object):
    age = 0
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        age = max(0, int(battle.turn) - int(started))
    defaults = {
        "tailwind": 4,
        "trickroom": 5,
        "magicroom": 5,
        "wonderroom": 5,
        "gravity": 5,
        "reflect": 5,
        "lightscreen": 5,
        "auroraveil": 5,
        "safeguard": 5,
        "mist": 5,
    }
    duration = defaults.get(identifier)
    return max(1, duration - age) if duration is not None else None


def _conditions(values: Mapping, battle: AbstractBattle) -> dict[str, dict]:
    result = {}
    for condition, started in values.items():
        identifier = _enum_id(condition)
        result[identifier] = {
            "duration": _condition_duration(identifier, battle, started)
        }
    return result


def _actor_matches(event: list[str], nickname: str, role: str | None = None) -> bool:
    if len(event) <= 2:
        return False
    actor = event[2]
    if role is not None and not actor.startswith(role):
        return False
    return to_id_str(actor.split(":", 1)[-1]) == to_id_str(nickname)


def _side_mechanic_usage(battle: AbstractBattle, role: str) -> dict[str, bool]:
    """Return public once-per-battle resources already spent by ``role``.

    A reconciled exact shadow can have followed a different earlier branch.  Repairing
    only the currently visible forme is insufficient: after Floette Mega Evolves, for
    example, every other Pokemon on that side must permanently lose its Mega option.
    All four facts below become public when used and are therefore safe to copy into
    every hidden-set determinization.
    """
    usage = {
        "mega_used": False,
        "z_move_used": False,
        "dynamax_used": False,
        "tera_used": False,
    }
    for event in getattr(battle, "_replay_data", []):
        if len(event) <= 2 or not str(event[2]).startswith(role):
            continue
        event_type = event[1]
        if event_type == "-mega":
            usage["mega_used"] = True
        elif event_type == "-zpower":
            usage["z_move_used"] = True
        elif event_type == "-terastallize":
            usage["tera_used"] = True
        elif (
            event_type == "-start"
            and len(event) > 3
            and to_id_str(event[3]) == "dynamax"
        ):
            usage["dynamax_used"] = True
    return usage


def _public_move_state(
    battle: AbstractBattle, nickname: str, role: str
) -> tuple[str | None, bool, str | None]:
    """Infer last move, recharge, and a pending charge from public protocol."""
    events = getattr(battle, "_replay_data", [])
    last_move = None
    must_recharge = False
    preparing_move = None
    for event in events:
        if not _actor_matches(event, nickname, role):
            continue
        if len(event) > 1 and event[1] in {"switch", "drag"}:
            last_move = None
            must_recharge = False
            preparing_move = None
            continue
        if len(event) > 3 and event[1] == "move":
            last_move = to_id_str(event[3])
            must_recharge = False
            # A completed second turn emits another move event but no subsequent
            # -prepare.  Clear first and let a following -prepare restore the lock.
            preparing_move = None
            continue
        if len(event) > 3 and event[1] == "-prepare":
            preparing_move = to_id_str(event[3])
        elif len(event) > 3 and event[1] == "-anim":
            # Solar Beam in sun and Electro Shot in rain emit -prepare followed
            # by an immediate animation instead of preserving the charge lock.
            if preparing_move == to_id_str(event[3]):
                preparing_move = None
        elif (
            event[1] == "cant"
            and len(event) > 3
            and to_id_str(event[3]) == "recharge"
        ):
            must_recharge = False
        elif event[1] == "-mustrecharge":
            must_recharge = True
    return last_move, must_recharge, preparing_move


def _public_effect_move(
    battle: AbstractBattle, nickname: str, role: str, effect_id: str
) -> str | None:
    """Recover the move captured when Encore/Disable began, not today's last move."""
    last_move = None
    captured = None
    for event in getattr(battle, "_replay_data", []):
        if not _actor_matches(event, nickname, role):
            continue
        event_type = event[1] if len(event) > 1 else ""
        if event_type in {"switch", "drag"}:
            last_move = None
            captured = None
        elif event_type == "move" and len(event) > 3:
            last_move = to_id_str(event[3])
        elif (
            event_type == "-start"
            and len(event) > 3
            and to_id_str(event[3]) == effect_id
        ):
            explicit = to_id_str(event[4]) if len(event) > 4 else ""
            captured = explicit or last_move
        elif (
            event_type == "-end"
            and len(event) > 3
            and to_id_str(event[3]) == effect_id
        ):
            captured = None
    return captured


def _public_item(
    battle: AbstractBattle,
    nickname: str,
    role: str,
    fallback: str | None,
    *,
    own: bool,
) -> str | None:
    """Repair stale poke-env item fields after Trick, Knock Off, and consumption."""
    item: str | None = to_id_str(fallback or "") if (own or fallback) else None
    for event in getattr(battle, "_replay_data", []):
        if not _actor_matches(event, nickname, role) or len(event) < 4:
            continue
        if event[1] == "-item":
            item = to_id_str(event[3])
        elif event[1] == "-enditem":
            item = ""
    return item


def _pokemon_snapshot(
    pokemon: Pokemon,
    *,
    ident: str,
    battle: AbstractBattle,
    active_slot: int | None,
    own: bool,
    request_moves: list[dict[str, Any]] | None,
    request_trapped: bool | None = None,
    previous_choice_atom: str | None = None,
) -> dict[str, Any]:
    nickname = ident.split(":", 1)[-1].strip() if ":" in ident else ident
    maximum = int(pokemon.max_hp or 0)
    current = int(pokemon.current_hp or 0)
    exact_hp = own and maximum > 0
    visible_fraction = not (
        not own and active_slot is None and maximum <= 0 and not pokemon.fainted
    )
    moves = []
    if request_moves is not None:
        moves = [
            {
                "id": to_id_str(move.get("id") or move.get("move")),
                "pp": move.get("pp"),
                "disabled": bool(move.get("disabled")),
            }
            for move in request_moves
        ]
    elif pokemon.moves:
        moves = [
            {
                "id": move.id,
                # Opponent PP is not present in the public request. poke-env's
                # inferred Move object starts from a generic maximum (and can even
                # exceed the set's true unboosted maximum), so copying it would
                # refill a determinization every turn. Keep each exact particle's
                # internally tracked PP instead. Our own PP remains public.
                **(
                    {"pp": getattr(move, "current_pp", None)}
                    if own
                    else {}
                ),
            }
            for move in pokemon.moves.values()
        ]
    role = "p1" if own else "p2"
    last_move, must_recharge, preparing_move = _public_move_state(
        battle, nickname, role
    )
    effects = {
        _enum_id(effect): {
            "duration": _condition_duration(_enum_id(effect), battle, value)
        }
        for effect, value in pokemon.effects.items()
    }
    forced_request_move = None
    if request_moves is not None and len(request_moves) == 1:
        only = request_moves[0]
        if only.get("pp") is None:
            forced_request_move = to_id_str(only.get("id") or only.get("move"))
    if must_recharge or forced_request_move == "recharge":
        effects["mustrecharge"] = {"duration": 1}
    if forced_request_move not in {None, "recharge"}:
        preparing_move = forced_request_move
    elif request_moves is not None:
        # Our request is authoritative.  A normal move list means no hard charge
        # lock even if an unusual public protocol sequence looked like one.
        preparing_move = None
    if preparing_move:
        charge = {
            "duration": 1,
            "move": preparing_move,
        }
        parts = (previous_choice_atom or "").split()
        if len(parts) >= 2 and parts[0] == "move" and to_id_str(parts[1]) == preparing_move:
            targets = [
                int(part)
                for part in parts[2:]
                if part.lstrip("+-").isdigit()
            ]
            if targets:
                # A charging move's public animation hides its selected target. We
                # still know our own submitted command and can preserve that private
                # client-side fact across exact-shadow reconciliation.
                charge["target_loc"] = targets[0]
        effects["twoturnmove"] = charge
    for lock in ("disable", "encore"):
        locked_move = _public_effect_move(battle, nickname, role, lock)
        if lock in effects and locked_move:
            effects[lock]["move"] = locked_move
        elif lock in effects and "move" not in effects[lock]:
            effects.pop(lock)
    if (
        active_slot is not None
        and to_id_str(pokemon.item or "")
        in {"choiceband", "choicescarf", "choicespecs"}
        and last_move
    ):
        effects["choicelock"] = {"duration": None, "move": last_move}
    for event in reversed(getattr(battle, "_replay_data", [])):
        if len(event) < 4 or not _actor_matches(event, nickname, role):
            continue
        if event[1] in {"switch", "drag"}:
            break
        if event[1] == "-start" and to_id_str(event[3]) == "disable":
            if len(event) > 4:
                effects.setdefault("disable", {})["move"] = to_id_str(event[4])
            break
        if event[1] == "-end" and to_id_str(event[3]) == "disable":
            break
    return {
        "nickname": to_id_str(nickname),
        "species": to_id_str(pokemon.species),
        "base_species": to_id_str(pokemon.base_species),
        "active_slot": active_slot,
        "hp": current if exact_hp else None,
        "maxhp": maximum if exact_hp else None,
        "hp_fraction": (
            float(pokemon.current_hp_fraction or 0.0) if visible_fraction else None
        ),
        "fainted": bool(pokemon.fainted),
        "status": _enum_id(pokemon.status) if pokemon.status is not None else "",
        "boosts": {key: int(value) for key, value in pokemon.boosts.items()},
        "ability": to_id_str(pokemon.ability) if pokemon.ability else None,
        "item": _public_item(
            battle, nickname, role, pokemon.item, own=own
        ),
        "effects": effects,
        "first_turn": bool(pokemon.first_turn),
        "last_move": last_move,
        "moves": moves,
        "trapped": request_trapped,
    }


def public_snapshot(
    battle: AbstractBattle,
    request: dict[str, Any] | None,
    *,
    request_state: str | None = None,
    side_requests: list[dict[str, Any] | None] | None = None,
    pending_our_choice: str | None = None,
) -> dict[str, Any]:
    """Serialize only client-visible state needed by exact forward simulation."""
    request = request or {}
    active_requests = request.get("active") or []
    pending_atoms = [
        atom.strip() for atom in (pending_our_choice or "").split(",")
    ]

    def side(own: bool) -> dict[str, Any]:
        side_index = 0 if own else 1
        table = battle.team if own else battle.opponent_team
        actives = battle.active_pokemon if own else battle.opponent_active_pokemon
        active_by_identity = {
            id(pokemon): slot
            for slot, pokemon in enumerate(actives)
            if pokemon is not None
        }
        pokemon_rows = []
        for ident, pokemon in table.items():
            slot = active_by_identity.get(id(pokemon))
            # Open Team Sheets populates all six opponent sets in poke-env, but the
            # server still hides which two back Pokemon were brought.  Reconciliation
            # must not demand that an unselected preview Pokemon exists in a concrete
            # four-Pokemon root.  Add a reserve only after it has entered battle.
            if not own and slot is None and not pokemon.revealed:
                continue
            if (
                own
                and slot is None
                and not pokemon.selected_in_teampreview
                and len(table) > 4
            ):
                continue
            request_moves = None
            request_trapped = None
            if own and slot is not None and slot < len(active_requests):
                active_request = active_requests[slot]
                request_moves = active_request.get("moves") or []
                # maybeTrapped means the client cannot safely assume switching is
                # legal. Search conservatively until the trapping uncertainty clears.
                request_trapped = bool(
                    active_request.get("trapped")
                    or active_request.get("maybeTrapped")
                )
            pokemon_rows.append(
                _pokemon_snapshot(
                    pokemon,
                    ident=ident,
                    battle=battle,
                    active_slot=slot,
                    own=own,
                    request_moves=request_moves,
                    request_trapped=request_trapped,
                    previous_choice_atom=(
                        pending_atoms[slot]
                        if own and slot is not None and slot < len(pending_atoms)
                        else None
                    ),
                )
            )
        role = "p1" if own else "p2"
        return {
            "pokemon": pokemon_rows,
            "side_conditions": _conditions(
                battle.side_conditions if own else battle.opponent_side_conditions,
                battle,
            ),
            "force_switch": (
                list((side_requests[side_index] or {}).get("forceSwitch") or [])
                if side_requests is not None
                else None
            ),
            "mechanic_usage": _side_mechanic_usage(battle, role),
        }

    if request_state is None:
        if request.get("teamPreview"):
            request_state = "teampreview"
        elif any(request.get("forceSwitch") or []):
            request_state = "switch"
        else:
            request_state = "move"
    weather = next(iter(battle.weather), None)
    terrain = next(
        (
            field
            for field in battle.fields
            if _enum_id(field).endswith("terrain")
        ),
        None,
    )
    pseudo_weather = {
        identifier: data
        for field, data in _conditions(battle.fields, battle).items()
        if (identifier := _enum_id(field))
        not in {"electricterrain", "grassyterrain", "mistyterrain", "psychicterrain"}
    }
    return {
        "schema": 2,
        "turn": int(battle.turn),
        "request_state": request_state,
        "weather": _enum_id(weather) if weather is not None else "",
        "terrain": _enum_id(terrain) if terrain is not None else "",
        "pseudo_weather": pseudo_weather,
        "sides": [side(True), side(False)],
    }


def _enum_member(enum_class, identifier: str):
    return next(
        (
            member
            for member in enum_class
            if to_id_str(member.name) == to_id_str(identifier)
        ),
        None,
    )


def _started_turn(turn: int, identifier: str, data: dict[str, Any]) -> int:
    duration = data.get("duration")
    maximum = {
        "tailwind": 4,
        "trickroom": 5,
        "magicroom": 5,
        "wonderroom": 5,
        "gravity": 5,
        "reflect": 5,
        "lightscreen": 5,
        "auroraveil": 5,
        "safeguard": 5,
        "mist": 5,
    }.get(identifier)
    if duration is None or maximum is None:
        return turn
    return max(0, turn - max(0, maximum - int(duration)))


def apply_public_snapshot(battle: AbstractBattle, snapshot: dict[str, Any]) -> None:
    """Make a reconstructed policy view agree with a reconciled public snapshot."""
    battle._turn = int(snapshot["turn"])

    def apply_side(index: int, own: bool) -> None:
        table = battle.team if own else battle.opponent_team
        rows = snapshot["sides"][index]["pokemon"]
        role = "p1" if own else "p2"
        # Historical shadow logs can leave poke-env's slot dictionary pointing at a
        # different Pokemon even after the public snapshot marks the right object
        # active. Rebuild both fixed doubles slots from the authoritative snapshot;
        # merely toggling ``pokemon._active`` lets p1b collapse into p1a in endgames.
        for suffix in ("a", "b"):
            battle._active_pokemon.pop(f"{role}{suffix}", None)
        for pokemon in table.values():
            pokemon._active = False
        stable_order = []
        stable_identities: set[str] = set()
        for row in rows:
            nickname = to_id_str(row.get("nickname"))
            species = to_id_str(row.get("base_species") or row.get("species"))
            matches = [
                (ident, pokemon)
                for ident, pokemon in table.items()
                if to_id_str(ident.split(":", 1)[-1]) == nickname
            ]
            if not matches:
                matches = [
                    (ident, pokemon)
                    for ident, pokemon in table.items()
                    if to_id_str(pokemon.base_species) == species
                ]
            if len(matches) != 1:
                continue
            ident, pokemon = matches[0]
            if ident not in stable_identities:
                stable_order.append((ident, pokemon))
                stable_identities.add(ident)
            active_slot = row.get("active_slot")
            pokemon._active = active_slot is not None
            if active_slot is not None and int(active_slot) in (0, 1):
                battle._active_pokemon[
                    f"{role}{'ab'[int(active_slot)]}"
                ] = pokemon
            if row.get("maxhp") is not None:
                pokemon._max_hp = int(row["maxhp"])
                pokemon._current_hp = int(row.get("hp") or 0)
            elif row.get("hp_fraction") is not None:
                # Opponent HP is public only as a percentage. Keep that precision
                # instead of leaking the concrete particle's internal integer HP.
                pokemon._max_hp = 1000
                pokemon._current_hp = round(float(row["hp_fraction"]) * 1000)
            status = row.get("status") or ""
            pokemon._status = _enum_member(Status, status) if status else None
            pokemon._boosts.update(
                {key: int(value) for key, value in row.get("boosts", {}).items()}
            )
            if row.get("ability"):
                pokemon._ability = to_id_str(row["ability"])
            if row.get("item"):
                pokemon._item = to_id_str(row["item"])
            effects = {}
            for identifier, data in row.get("effects", {}).items():
                effect = _enum_member(Effect, identifier)
                if effect is not None:
                    effects[effect] = _started_turn(battle.turn, identifier, data)
            pokemon._effects = effects
        # Showdown mutates ``side.pokemon`` as switches happen. Replaying that log
        # therefore builds a poke-env team dictionary in current party order, while
        # the live policy's action IDs and observation tokens remain in the original
        # Team Preview order. Snapshot rows are emitted in that stable live order;
        # restore it before exact policy scoring so switch priors stay attached to
        # the correct Pokemon identity.
        if own and stable_order:
            stable_order.extend(
                (ident, pokemon)
                for ident, pokemon in list(table.items())
                if ident not in stable_identities
            )
            table.clear()
            table.update(stable_order)
        target = battle._side_conditions if own else battle._opponent_side_conditions
        target.clear()
        for identifier, data in snapshot["sides"][index].get(
            "side_conditions", {}
        ).items():
            condition = _enum_member(SideCondition, identifier)
            if condition is not None:
                target[condition] = _started_turn(battle.turn, identifier, data)

    apply_side(0, True)
    apply_side(1, False)
    battle._weather.clear()
    weather = _enum_member(Weather, snapshot.get("weather") or "")
    if weather is not None:
        battle._weather[weather] = battle.turn
    battle._fields.clear()
    terrain = _enum_member(Field, snapshot.get("terrain") or "")
    if terrain is not None:
        battle._fields[terrain] = battle.turn
    for identifier, data in snapshot.get("pseudo_weather", {}).items():
        field = _enum_member(Field, identifier)
        if field is not None:
            battle._fields[field] = _started_turn(battle.turn, identifier, data)
