"""Unit tests for the Phase 2 guards' rule tables and pure helpers.

These guards encode facts about Pokemon, not computations over calculate_damage, so
the failure mode is a WRONG FACT rather than a crash -- and a wrong fact is invisible
in a win rate. Live battles cannot cover this: replayed states have no `|request|`, so
`available_moves` is empty and `pass` is the only legal action, which is why two
earlier replay-based tests silently reported nothing.

Helpers are duck-typed, so lightweight stubs suffice; the move facts use real
poke-env Move objects so a poke-env data change would break the test rather than the
bot.
"""

from types import SimpleNamespace as NS

from poke_env.battle import Field, Move, PokemonType, SideCondition, Status

from vgc_bench.src import guards as G

fails = []


def check(label, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {label}: got {got!r}, want {want!r}")
    if not ok:
        fails.append(label)


print("1. status immunity facts")
# Ground is immune to Thunder Wave by type; Electric cannot be paralysed at all.
check(
    "thunderwave vs Ground",
    PokemonType.GROUND in G.STATUS_TYPE_IMMUNITY["thunderwave"],
    True,
)
check(
    "thunderwave vs Electric",
    PokemonType.ELECTRIC in G.STATUS_TYPE_IMMUNITY["thunderwave"],
    True,
)
check(
    "thunderwave vs Steel (NOT immune)",
    PokemonType.STEEL in G.STATUS_TYPE_IMMUNITY["thunderwave"],
    False,
)
check(
    "willowisp vs Fire", PokemonType.FIRE in G.STATUS_TYPE_IMMUNITY["willowisp"], True
)
check("toxic vs Steel", PokemonType.STEEL in G.STATUS_TYPE_IMMUNITY["toxic"], True)
check("toxic vs Poison", PokemonType.POISON in G.STATUS_TYPE_IMMUNITY["toxic"], True)
check(
    "leechseed vs Grass", PokemonType.GRASS in G.STATUS_TYPE_IMMUNITY["leechseed"], True
)
check("spore is a powder move", "spore" in G.POWDER_MOVES, True)
check("sleeppowder is a powder move", "sleeppowder" in G.POWDER_MOVES, True)
check("willowisp is NOT a powder move", "willowisp" in G.POWDER_MOVES, False)

print("2. move ids resolve in poke-env (a typo here silently disables a rule)")
for mid in list(G.STATUS_TYPE_IMMUNITY) + list(G.POWDER_MOVES) + list(G.PROTECT_MOVES):
    try:
        check(f"{mid} exists", Move(mid, gen=9).id, mid)
    except Exception as e:
        check(f"{mid} exists", f"raised {type(e).__name__}", mid)

print("3. _norm")
check("spaces/case/punctuation", G._norm("Lightning Rod"), "lightningrod")
check("None", G._norm(None), "")
check("Safety Goggles", G._norm("Safety Goggles"), "safetygoggles")
check(
    "redirect table keyed by normalised id",
    G.REDIRECT_ABILITIES.get(G._norm("Storm Drain")),
    PokemonType.WATER,
)

print("4. _effective_speed")
battle = NS(side_conditions={}, opponent_side_conditions={}, fields={})
mon = NS(stats={"spe": 100}, boosts={"spe": 0}, status=None)
check("base", G._effective_speed(battle, mon, ours=True), 100.0)
check(
    "+1 stage",
    G._effective_speed(
        battle, NS(stats={"spe": 100}, boosts={"spe": 1}, status=None), ours=True
    ),
    150.0,
)
check(
    "paralysed halves",
    G._effective_speed(
        battle, NS(stats={"spe": 100}, boosts={"spe": 0}, status=Status.PAR), ours=True
    ),
    50.0,
)
tw = NS(
    side_conditions={SideCondition.TAILWIND: 1}, opponent_side_conditions={}, fields={}
)
check("tailwind doubles", G._effective_speed(tw, mon, ours=True), 200.0)
check("tailwind is side-specific", G._effective_speed(tw, mon, ours=False), 100.0)
check(
    "unknown speed -> None",
    G._effective_speed(battle, NS(stats={}, boosts={}, status=None), ours=True),
    None,
)

print("5. _foe_moves_first  (certainty only)")
slow = NS(stats={"spe": 80}, boosts={"spe": 0}, status=None)
fast = NS(stats={"spe": 120}, boosts={"spe": 0}, status=None)
check("faster foe", G._foe_moves_first(battle, slow, fast), True)
check("slower foe", G._foe_moves_first(battle, fast, slow), False)
check("speed tie is NOT certainty", G._foe_moves_first(battle, slow, slow), False)
check(
    "unknown foe speed is NOT certainty",
    G._foe_moves_first(battle, slow, NS(stats={}, boosts={}, status=None)),
    False,
)
tr = NS(side_conditions={}, opponent_side_conditions={}, fields={Field.TRICK_ROOM: 1})
check(
    "trick room inverts (fast foe now second)",
    G._foe_moves_first(tr, slow, fast),
    False,
)
check(
    "trick room inverts (slow foe now first)", G._foe_moves_first(tr, fast, slow), True
)

print("6. _ally_hit  (which orders splash our own side)")
ally = NS(fainted=False)
b2 = NS(active_pokemon=[NS(fainted=False), ally])


def hit(move_id, move_target=0, pos=0):
    order = NS(order=Move(move_id, gen=9), move_target=move_target)
    return G._ally_hit(b2, order, pos) is ally


check("earthquake (ALL_ADJACENT) hits ally", hit("earthquake"), True)
check("rockslide (ALL_ADJACENT_FOES) does not", hit("rockslide"), False)
check("single-target at a foe does not", hit("flamethrower", move_target=1), False)
check("single-target at our own slot does", hit("flamethrower", move_target=-2), True)
check("status move never counts as ally damage", hit("tailwind"), False)
check(
    "fainted ally is not hit",
    G._ally_hit(
        NS(active_pokemon=[NS(fainted=False), NS(fainted=True)]),
        NS(order=Move("earthquake", gen=9), move_target=0),
        0,
    ),
    None,
)

print("7. a vetoed candidate can never be resurrected by the tiebreak")
# The stack runs hard vetoes before the soft rerank precisely so their verdict stands.
# Before the fix, ko_tiebreak took a POSITIONAL prefix of a list that _demote had
# already rearranged into keep+push, so it could sort a vetoed pair back to the front.
cands = [G.Candidate((0, 0), 0.9), G.Candidate((1, 1), 0.85), G.Candidate((2, 2), 0.8)]
rep = G.GuardReport()
out = G._demote(cands, {0}, "zero_damage", rep)
check(
    "demoted pair moved to the back", [c.actions for c in out], [(1, 1), (2, 2), (0, 0)]
)
check("demotion counted", dict(rep.demotions), {"zero_damage": 1})
check("top-pick change also noted as a stage", rep.stages, ["zero_damage"])
# active_pokemon of None makes scoring a no-op without raising, isolating the ordering.
stub = NS(active_pokemon=[None, None], opponent_active_pokemon=[None, None])
out2 = G.guard_ko_tiebreak(stub, out, rep)
check("vetoed pair still last after tiebreak", out2[-1].actions, (0, 0))
check("tiebreak did not drop candidates", len(out2), 3)

# Sequential hard guards compose over the still-live prefix. If the second guard
# rejects every survivor it must stand down, not resurrect the already-vetoed pair.
rep_seq = G.GuardReport()
seq = [G.Candidate((0, 0), 0.9), G.Candidate((1, 1), 0.8), G.Candidate((2, 2), 0.7)]
seq = G._demote(seq, {0}, "zero_damage", rep_seq)
stood_down = G._demote(seq, {0, 1}, "status_immunity", rep_seq)
check(
    "later all-live veto stands down",
    [c.actions for c in stood_down],
    [(1, 1), (2, 2), (0, 0)],
)
check("earlier veto remains marked", stood_down[-1].demoted_by, "zero_damage")

print("8. team-relevant guards fire on the right states")
tw_move = Move("tailwind", gen=9)
check(
    "tailwind carries its side condition",
    tw_move.side_condition,
    SideCondition.TAILWIND,
)
b_tw = NS(
    active_pokemon=[NS(protect_counter=0), NS(protect_counter=0)],
    side_conditions={SideCondition.TAILWIND: 1},
)
b_no = NS(
    active_pokemon=[NS(protect_counter=0), NS(protect_counter=0)], side_conditions={}
)


def redundant(battle):
    c = [G.Candidate((0, 0), 0.9), G.Candidate((1, 1), 0.8)]
    # patch the decoder so this exercises the RULE, not poke-env's action encoding
    real = G._decode
    G._decode = lambda b, a, p: NS(order=tw_move, move_target=0) if a == 0 else None
    try:
        out = G.guard_redundant_side_condition(battle, c, G.GuardReport())
    finally:
        G._decode = real
    return out[0].actions


check("tailwind demoted when tailwind is already up", redundant(b_tw), (1, 1))
check("tailwind kept when it is not up", redundant(b_no), (0, 0))

prot = Move("protect", gen=9)


def protect_spam(counter):
    battle = NS(
        active_pokemon=[NS(protect_counter=counter), NS(protect_counter=0)],
        side_conditions={},
    )
    c = [G.Candidate((0, 0), 0.9), G.Candidate((1, 1), 0.8)]
    real = G._decode
    G._decode = lambda b, a, p: NS(order=prot, move_target=0) if a == 0 else None
    try:
        out = G.guard_protect_spam(battle, c, G.GuardReport())
    finally:
        G._decode = real
    return out[0].actions


check("first protect is kept", protect_spam(0), (0, 0))
check("consecutive protect is demoted", protect_spam(1), (1, 1))
check(
    "trick room is NOT treated as a side condition (it is a toggle)",
    getattr(Move("trickroom", gen=9), "side_condition", None),
    None,
)

print("9. registry")
check("order covers registry", set(G.GUARD_ORDER), set(G.GUARDS))
check("hard profile is a registry subset", G.HARD_GUARDS <= set(G.GUARDS), True)
check("ko_tiebreak runs last", G.GUARD_ORDER[-1], "ko_tiebreak")
check("zero_damage runs first", G.GUARD_ORDER[0], "zero_damage")

print()
if fails:
    raise SystemExit(f"FAILED {len(fails)}: {fails}")
print(f"PASS - all checks green across {len(G.GUARDS)} registered guards")
