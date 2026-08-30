"""The poke-env parse shim must swallow exactly the two observed unsurvivable
failures and nothing else.

Scope is the whole point of this shim. Too narrow and the battle still stalls; too
broad and it hides real breakage while quietly corrupting battle state. So: KeyError
and the Illusion team-overflow ValueError swallowed and counted; every other
exception still propagates.
"""

from poke_env.battle.abstract_battle import AbstractBattle

from vgc_bench.src import pokeenv_patches as P

fails = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(label)


# Stand in for the real parser so the test drives the wrapper, not poke-env internals.
raised: list[str] = []


def fake_parse(self, split_message):
    kind = split_message[1]
    if kind == "boom-key":
        raise KeyError("round")  # the observed ladder failure
    if kind == "boom-illusion":
        raise ValueError(  # the observed league-training failure (Zoroark)
            "p2's team already has 4 pokemons: cannot add p2: Zoroark to "
            "p2: Audino, p2: Overqwil, p2: Malamar, p2: Zoroark-Hisui"
        )
    if kind == "boom-other":
        raise ValueError("something genuinely broken")
    raised.append(kind)
    return "parsed"


AbstractBattle.parse_message = fake_parse
P._installed = False  # allow install over the stub
P.PARSE_ERRORS.clear()
P.PARSE_REPAIRS.clear()
check("install reports success", P.install(), True)
check("install is idempotent", P.install(), False)


class Stub:
    pass


print("1. normal messages pass straight through")
check(
    "returns the parser's value",
    AbstractBattle.parse_message(Stub(), ["", "move", "p1a: X"]),
    "parsed",
)
check("parser actually ran", raised, ["move"])

print("2. the observed KeyError is swallowed, not raised")
try:
    out = AbstractBattle.parse_message(Stub(), ["", "boom-key", "p1a: X"])
    check("returns None instead of raising", out, None)
except KeyError:
    check("returns None instead of raising", "raised KeyError", None)
check("counted once", sum(P.PARSE_ERRORS.values()), 1)
check(
    "counted under its message type", [k for k in P.PARSE_ERRORS], ["boom-key:'round'"]
)
check("report names it", "boom-key:'round'" in P.report(), True)

print("3. the Illusion team-overflow ValueError is swallowed, not raised")
try:
    out = AbstractBattle.parse_message(Stub(), ["", "boom-illusion", "p1b: Y"])
    check("returns None instead of raising", out, None)
except ValueError:
    check("returns None instead of raising", "raised ValueError", None)
check("counted once", sum(P.PARSE_ERRORS.values()), 2)
check(
    "counted under the illusion label",
    P.PARSE_ERRORS["boom-illusion:illusion_team_overflow"],
    1,
)

print("4. other exceptions must STILL propagate")
try:
    AbstractBattle.parse_message(Stub(), ["", "boom-other", "p1a: X"])
    check("ValueError propagates", "swallowed", "raised")
except ValueError:
    check("ValueError propagates", "raised", "raised")
check("and is not counted as a parse error", sum(P.PARSE_ERRORS.values()), 2)

print("5. abilities revealed by |cant| survive the upstream parser gap")


class Mon:
    ability = None


class BattleStub:
    pokemon = Mon()

    def get_pokemon(self, _identifier):
        return self.pokemon


blocked = BattleStub()
AbstractBattle.parse_message(
    blocked,
    [
        "",
        "cant",
        "p2b: Farigiraf",
        "ability: Armor Tail",
        "Sucker Punch",
        "[of] p1b: Kingambit",
    ],
)
check("Armor Tail recorded on Farigiraf", blocked.pokemon.ability, "Armor Tail")
check("repair counted", sum(P.PARSE_REPAIRS.values()), 1)
check("repair appears in report", "revealed abilities repaired: 1" in P.report(), True)

print()
if fails:
    raise SystemExit(f"FAILED {len(fails)}: {fails}")
print("PASS - shim swallows exactly the unsurvivable case and nothing more")
