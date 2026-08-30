"""Defensive shims around poke-env's battle-message parsing.

An exception raised while parsing a battle message is NOT survivable. poke-env catches
it inside its own handler, so the battle never receives another order: it stalls
forever, the ladder clock runs out, and from the outside it looks like the bot is
"thinking". That failure mode has now cost this project twice -- once from an assert of
our own in embed_pokemon, and once from poke-env itself:

    abstract_battle.py:725  overridden = mon.moves[Move.retrieve_id(overridden_move)]
    KeyError: 'round'

It happens when a move is used through an effect that overrides it (Sleep Talk,
Copycat, Metronome, Dancer, Instruct) and the overridden move is not in the tracked
moveset. Real Showdown sends this; poke-env assumes it cannot.

A second unsurvivable case surfaced during league training (2026-08-29), four times
against one Zoroark-Hisui roster:

    abstract_battle.py:306  get_pokemon("p2: Zoroark")
    ValueError: p2's team already has 4 pokemons: cannot add p2: Zoroark to ...

Illusion reveals a species name that was never registered at Team Preview, and
poke-env's bring-4 bookkeeping refuses to add a "fifth" member mid-battle. The
message reaching that path is minor bookkeeping (Pressure PP accounting on a target
reference), but the raise kills the message-handler task and stalls the battle.

Scope is deliberately narrow -- KeyError, plus ONLY the team-overflow ValueError
signature, only around parse_message. Swallowing a message leaves that one update
unapplied, which is a small, local inaccuracy; letting it raise loses the entire
battle. Every swallow is counted so this stays visible rather than becoming silent
damage: call `report()` at the end of a run.
"""

from __future__ import annotations

from collections import Counter

PARSE_ERRORS: Counter[str] = Counter()
PARSE_REPAIRS: Counter[str] = Counter()
_installed = False


def install() -> bool:
    """Wrap AbstractBattle.parse_message. Idempotent; True if newly installed."""
    global _installed
    if _installed:
        return False
    from poke_env.battle.abstract_battle import AbstractBattle

    original = AbstractBattle.parse_message

    def parse_message(self, split_message):  # type: ignore[no-untyped-def]
        try:
            result = original(self, split_message)
        except KeyError as exc:
            # split_message[1] is the message type, e.g. "move" -- enough to tell a
            # recurring parser gap from a one-off without logging battle contents.
            kind = split_message[1] if len(split_message) > 1 else "?"
            PARSE_ERRORS[f"{kind}:{exc}"] += 1
            return None
        except ValueError as exc:
            # Only the Illusion signature: an unregistered revealed name makes
            # get_pokemon refuse a "fifth" team member. Anything else propagates.
            if "team already has" not in str(exc):
                raise
            kind = split_message[1] if len(split_message) > 1 else "?"
            PARSE_ERRORS[f"{kind}:illusion_team_overflow"] += 1
            return None
        # poke-env treats every |cant| message as only "the actor could not move".
        # Showdown also uses it to reveal the defender's blocking ability:
        #   |cant|p2b: Farigiraf|ability: Armor Tail|Sucker Punch|[of] p1b: Kingambit
        # Losing that fact made the bot repeat Sucker Punch for nine turns. Preserve
        # the revealed ability as durable battle state after the upstream parser has
        # applied its normal cant_move bookkeeping.
        if (
            len(split_message) >= 4
            and split_message[1] == "cant"
            and split_message[3].lower().startswith("ability:")
        ):
            try:
                ability = split_message[3].split(":", 1)[1].strip()
                self.get_pokemon(split_message[2]).ability = ability
                PARSE_REPAIRS[f"cant_ability:{ability}"] += 1
            except Exception as exc:
                PARSE_ERRORS[f"cant_ability_repair:{type(exc).__name__}"] += 1
        return result

    AbstractBattle.parse_message = parse_message  # type: ignore[method-assign]
    _installed = True
    return True


def report() -> str:
    repairs = sum(PARSE_REPAIRS.values())
    repair_text = f"; revealed abilities repaired: {repairs}" if repairs else ""
    if not PARSE_ERRORS:
        return f"poke-env parse errors: none{repair_text}"
    total = sum(PARSE_ERRORS.values())
    detail = ", ".join(f"{k} x{v}" for k, v in PARSE_ERRORS.most_common(5))
    return f"poke-env parse errors swallowed: {total} ({detail}){repair_text}"
