"""Type knowledge must survive an opponent denying Open Team Sheets.

Reg M-B's Open Team Sheets are OPT-IN. When an opponent denies them their Pokemon
arrive with NO stats, calculate_damage raises for every pairing, and before this fix
every calc-driven check silently answered "not immune" -- so the bot fired Dragon Claw
into a Fairy on ladder while the type chart sat unconsulted the whole time.

This reproduces the sheet-less state directly: a Pokemon with no stats set.
"""

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from poke_env.battle import Move, Pokemon

from vgc_bench.src import vgc_knowledge as K

fails = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(label)


GEN = 9


def sheetless(species):
    """A Pokemon exactly as it arrives when the opponent denied team sheets."""
    mon = Pokemon(gen=GEN, species=species)
    check(
        f"{species} really has no stats",
        not mon.stats or any(v is None for v in (mon.stats or {}).values()),
        True,
    )
    return mon


print("1. the exact ladder blunder: Dragon into Fairy, no stats available")
flutter = sheetless("flutter mane")  # Ghost/Fairy
dragon_claw = Move("dragonclaw", gen=GEN)
check("Flutter Mane is Fairy", any(t.name == "FAIRY" for t in flutter.types), True)
check(
    "type_multiplier works with no stats", K.type_multiplier(dragon_claw, flutter), 0.0
)
check(
    "deals_no_damage now catches it",
    K.deals_no_damage(None, None, flutter, dragon_claw),
    True,
)

print("2. the other classic immunities, all without stats")
cases = [
    ("earthquake", "flutter mane", None),  # levitate-ish: Ghost/Fairy is hit
    ("earthquake", "talonflame", 0.0),  # Flying is immune to Ground
    ("shadowball", "kingambit", None),  # Dark/Steel takes Ghost
    ("sucker punch", "gholdengo", None),
    ("close combat", "flutter mane", 0.0),  # Ghost is immune to Fighting
    ("thunderbolt", "garchomp", 0.0),  # Ground is immune to Electric
]
for move_id, species, want_zero in cases:
    mon = Pokemon(gen=GEN, species=species)
    mult = K.type_multiplier(
        Move("".join(c for c in move_id if c.isalnum()), gen=GEN), mon
    )
    immune = K.deals_no_damage(
        None, None, mon, Move("".join(c for c in move_id if c.isalnum()), gen=GEN)
    )
    label = f"{move_id} vs {species}"
    if want_zero == 0.0:
        check(f"{label} -> immune", (mult, immune), (0.0, True))
    else:
        check(f"{label} -> NOT immune", immune, False)

print("3. status moves and protected targets are still excluded")
check(
    "status move is never 'no damage'",
    K.deals_no_damage(None, None, sheetless("garchomp"), Move("willowisp", gen=GEN)),
    False,
)

print("4. ensure_stats makes a sheet-less Pokemon usable by the calc")
mon = Pokemon(gen=GEN, species="flutter mane")
check(
    "starts unusable",
    bool(mon.stats and all(v is not None for v in mon.stats.values())),
    False,
)
check("ensure_stats reports a change", K.ensure_stats(mon), True)
check(
    "now has every stat",
    all(
        isinstance(mon.stats.get(k), int)
        for k in ("hp", "atk", "def", "spa", "spd", "spe")
    ),
    True,
)
check("idempotent", K.ensure_stats(mon), False)
hp = mon.stats["hp"]
check(
    "HP uses the Champions formula base+ev+75",
    hp,
    mon.base_stats["hp"] + K.CHAMPIONS_EV_CAP + 75,
)

print()
if fails:
    raise SystemExit(f"FAILED {len(fails)}: {fails}")
print("PASS - type knowledge survives a denied team sheet")
