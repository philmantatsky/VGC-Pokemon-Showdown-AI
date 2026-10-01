# T6 vs the Reg M-C meta: team and moveset review (2026-09-27 evening)

The user: "look up our team and then meta teams and see how it fares based on data we
could be playing the wrong team ... or maybe even need to tweak our moveset to match
the meta". Analysis only: nothing retrained, nothing run on the local server (the
doomed-Pokemon chain owns the machine), DEPLOYED.json untouched.

**Data:** our 163 T6-era ladder games (09-23 → 09-27, every brain, incl. today's two
15-game reads) and 50 T4 games; the held-out battery of the deployed T6tac (6,204
local games vs 47 human rosters); 9,947 human Reg M-C games (the 09-09..09-20 corpus +
2,247 replays through 09-27 fetched by this morning's web agent, now in the git-ignored
`battle_logs_web_mc_20260927/`), ratings ~1050-1680; their 7,400+ open team sheets;
Pikalytics tournament sheets; the project's earlier team tournaments. "vsElo" = score
above what the two ratings predict. Four high-volume accounts (SC-Control, SC-SME,
Scorecard-Pokemon, kevdan42 — 637 sides, one team) are excluded from human stats.
Scripts and tables: `results_analysis/openings_20260927/team_*.py|txt`.

## 1. Short answer

1. **T6 is not a bad team for this meta — in human hands.** Human rosters with 4+ of
   our six score exactly their rating expectation (51-47, -0.1pp); with 5+ they went
   28-11 (+18pp, strong pilots). Their building blocks are good: Farigiraf + Torkoal
   rosters +4.4pp (238-199, 126 players), Mega Blastoise rosters +4.1pp (325-279, 184
   players). No archetype dominates at our rating: every major one is within
   -1.0..+1.2pp,
   and human teams built like each of our candidates (T0-T5, T6, the Baltimore top
   two) all land within -1.9..+3.5pp. Choosing a different meta team is worth a few
   points at most; piloting is worth far more.
2. **But our bot plays T6 worse than it played T4.** T6 era: 78-85 (48%), **-2.1pp**
   vs rating over 163 games; T4: 27-23 (54%), **+4.8pp** over 50 (small sample,
   different brain). Locally T6 was -4.3pp vs T4 as laddered (09-23, before the
   T6hp / T6ctx / T6tac improvements; never re-measured). T6's failure is
   structural to a slow Trick Room team: one of our Pokemon chose a move and fainted
   before acting 217 times in 165 ladder games (the 18:55 entry), and only Water Room
   holds up locally (Support Room -14.6, Sun Room -5.2 at 02:53) — the bot uses four
   of the six.
3. **Its losses are concentrated and rising in the meta:** sand, rain + Archaludon,
   slow Trick Room (Hatterene / Camerupt), and Volcarona / Sinistcha (section 2).
   Rain is ~11% of rosters in the 09-09..09-20 corpus but 22% of top-ladder team
   sides in the 09-10..09-27 replays (web agent; overlapping labels).
4. **Set tweaks — revised by our own ladder logs (section 4b, 2026-09-28):** the meta
   and coverage favoured four tweaks, but replaying our 165 ladder games turn by turn
   keeps only two: **Charizard Weather Ball over Solar Beam** (5 KO-turns gained, 1
   lost) and **Torkoal Earth Power over Heat Wave** (10 gained, 4 lost). **Keep
   Incineroar's Passho Berry** (Chople would have saved 4 KOs + 2 maybes, Passho did
   save 4, all vs rain) and **Venusaur's Leaf Storm** (Earth Power 5 KO-turns gained,
   7 lost — Leaf Storm is our anti-rain move). Farigiraf's Helping Hand decided 10 KOs
   (9 in wins): keep. Blastoise and Farigiraf are the expert pilots' exact sets.
5. **What would settle "wrong team":** T6 has never been in a team tournament. The
   test in section 5 (same pilot, every team, same meta pool; ~2 h of local compute)
   ranks T6, a T6 with the set tweaks, T4 and T0-T5 in one table.

## 2. How T6 fares

**Ladder, 163 T6-era games, record when the opponent has it on the roster (n >= 8):**

| Worst | Record | | Best | Record |
|---|---|---|---|---|
| Volcarona | 2-11 | | Torkoal | 7-3 |
| Tyranitar | 6-18 | | Armarouge | 7-4 |
| Sinistcha | 4-11 | | Salamence | 20-13 |
| Grimmsnarl | 2-6 | | Gardevoir | 10-7 |
| Whimsicott | 2-6 | | Farigiraf | 8-6 |
| Corviknight | 3-8 | | Indeedee-F | 19-15 |
| Garchomp | 8-19 | | Kingambit | 15-12 |
| Pelipper | 7-15 | | Incineroar | 20-16 |
| Milotic | 8-16 | | Arcanine-H | 11-9 |
| Archaludon | 7-13 | | Rillaboom | 31-32 |

Why: Volcarona resists Fire and Grass and redirects with Rage Powder; Sinistcha
(Grass/Ghost) resists Water, is immune to Fake Out, redirects with Rage Powder and
sets its own Trick Room; Tyranitar takes the weather, Knocks Off Farigiraf and Rock
Slides; Archaludon resists Water and Electro Shots Blastoise; Garchomp's Ground moves
hit Torkoal and Incineroar super-effectively.

**Local battery (T6tac, 6,204 games):** rain 89.1%, Trick Room 90.5%, Tailwind 92.0%,
grassy Fake Out 93.9%, balance 98.5%. Worst rosters: Politoed rain (Gengar,
Incineroar, Rillaboom, Sneasler, Politoed, Archaludon) 62.9%; Indeedee / Hatterene /
Camerupt Trick Room 72.7%; sand (Indeedee, Sneasler, Tyranitar, Excadrill,
Corviknight, Hydreigon) 73.5%. Species on the roster (human-clone populations):
Camerupt 61%, Politoed 65%, Corviknight 67%, Excadrill / Tyranitar 68%, Hatterene
74%, Indeedee 77% (overall 90%). Ladder and local agree on the list.

## 3. The meta teams

- **Human teams built like each candidate** (4+ of its six): T6 -0.1pp (n=98), T4 sun
  -1.9 (209), T0 MB430 +2.5 (408), T1 consensus +1.3 (2,838), T2 Trick Room / Psychic
  Terrain +2.2 (357), T3 Psychic Terrain offense -1.0 (1,455), T5 rain +1.0 (661; 5+:
  +5.2), Baltimore #1 sand +1.1 (649), Baltimore #2 +3.5 (220), Raichu / Gholdengo
  balance +2.8 (1,117). All inside a few points: the meta is balanced.
- **The high-volume accounts** (the Baltimore runner-up team: Charizard-Y, Golisopod,
  Politoed, Archaludon, Grimmsnarl, Farigiraf) went 290-347, **-4.8pp**, median
  rating 1301 — frequent opponents, not strong ones.
- **Strongest pairs on human rosters** (n >= 250): Camerupt + Hatterene +12.6pp,
  Charizard + Floette +11.3, Camerupt + Indeedee-F +11.0, Garchomp + Golisopod +10.4,
  **Blastoise + Indeedee-F +10.3**, Blastoise + Sneasler +9.4, Dragonite + Gholdengo
  +9.2. Slow Trick Room (Mega Camerupt) is both a strong human archetype and our
  worst local matchup.
- **Tournaments:** sand won Baltimore (Reg M-C, 1,081 players); Day-2 conversion led
  by Staraptor, Sylveon, Kingambit, Floette, Gengar, Garchomp; Farigiraf 17.0% and
  Charizard 16.8% vs a 14.3% baseline.
- **Earlier team tournaments (none with T6):** clone pilot (09-10): T1 64.3%, T0 62.7,
  T2 52.3, T3 46.3, T4 42.3, T5 28.3; brain-v1 pilot (09-15): T4 86.7 / 82.0, T0 85.0 /
  83.0, T2 80.0 / 84.0, T1 69.7 / 72.0, T3 66.7 / 60.0, T5 66.7 / 71.7. A human-like
  pilot and our brain rank teams very differently (T4 fifth of six vs first):
  team quality and team-pilot fit are separate questions.

## 4. Movesets: ours vs the meta

Shares are of human open sheets (Pikalytics tournament sheets in parentheses).
Outcome splits by move are confounded (team, pilot) and several cells are driven by 1-4
players; the case for each change rests mainly on prevalence and coverage.

| Pokemon | Ours | Meta standard | Proposal | Confidence |
|---|---|---|---|---|
| Torkoal | Eruption, Heat Wave, Weather Ball, Protect | Eruption 98, Protect 94, Weather Ball 65, **Earth Power 51**, Heat Wave 38 | **Earth Power over Heat Wave**: super-effective on 39% of top-40 usage incl. Archaludon, Tyranitar, Gholdengo, Kingambit, Incineroar, Raichu, Sneasler (Heat Wave duplicates Eruption's Fire spread) | medium |
| Incineroar | Passho Berry | Sitrus 69%, Chople 8, Passho 7, Rocky Helmet 6 | **Chople Berry** (Sitrus is Farigiraf's; Item Clause): Passho sides 82-127, -7.7pp over 70 players; Sneasler's Close Combat is on 42% of rosters, Basculegion's Water on 22% | medium |
| Venusaur | Leaf Storm, Focus Sash | Sludge Bomb 93, Sleep Powder 69, **Earth Power 54**, Leaf Storm 50; Sash 35 / Life Orb 34 | **Earth Power over Leaf Storm**: 39% vs 11% super-effective; Leaf Storm is resisted by 61% of usage. Keep Sash (Life Orb's edge is two players) | medium |
| Charizard-Y | Heat Wave, Solar Beam, Ancient Power, Protect | Heat Wave 98, Protect 100, **Weather Ball 93** (92), Solar Beam 42 (44), Ancient Power 31 (43) | **Weather Ball over Solar Beam** (keep Ancient Power: 4x on Volcarona, 2-11 for us) — or over Ancient Power; the tournament can compare | high prevalence, weak outcome data |
| Blastoise-Mega | Fake Out, Water Spout, Water Pulse, Ice Beam | Water Spout 97, Protect 84, Shell Smash 79, Dark Pulse 51 (a Shell Smash sweeper) | **keep**: our set is dksnnfud's / thruxy's Trick Room attacker (section 1 of OPENINGS_RESEARCH_T6.md); the meta's Fake Out data is one player | — |
| Farigiraf | Trick Room, Psychic, Helping Hand, Rain Dance | Trick Room 99, Psychic 62, Protect 58, Helping Hand 56; Rain Dance 2 | keep; optional **Protect over Helping Hand** (Farigiraf takes 67% of attacks and fainted before acting 51 times; Protect +2.2 / -1.1pp) — dropped after 4b | low |

The Incineroar and Venusaur proposals above are overturned by our own games (4b).

A set change is an input the brain never practised (09-23 lesson: unseen-but-better
inputs hurt a brain) — each needs a practice cycle before it can be judged.

## 4b. What our own ladder logs say (set-swap audit, 2026-09-28)

The user: "check our ladder logs of losses / wins where our incineroar wouldve done
better with chople berry and do the rest with the other pokemon too".
`results_analysis/openings_20260927/setswap_audit.py` replays all 165 T6-era ladder
games (09-23 → 09-27) through `unit_tests/ladder_position.py` and the real damage
calculator: every item trigger on Incineroar, and at every attacking turn of Torkoal,
Venusaur and Charizard the best move of the current set vs the best move of the
proposed set (guaranteed KOs first; a likely Focus Sash at full HP blocks a one-hit
KO — 93% of Hisuian Arcanine hold one). Every moment is listed in
`setswap_audit.txt`.

| Change | Would have helped | Would have hurt | Verdict |
|---|---|---|---|
| Incineroar Chople Berry for Passho | the first super-effective Fighting hit, 8 games (all losses): 4 KOs avoided (Annihilape Drain Punch, Sneasler Close Combat, Corviknight Body Press, and a Drain Punch that left it at 8 HP to die to its own Flare Blitz recoil), 2 maybe (Staraptor Close Combat; Sneasler the turn after an Aura Sphere) | Passho fired 13 times: 4 KOs avoided (Greninja / Araquanid Liquidation, rain Weather Ball from Politoed and Pelipper) and 4 more times it left Incineroar at 16-26 HP instead of 52+ | **wash — keep Passho**: it is insurance for our worst matchup (rain); every one of these games was a loss either way |
| Torkoal Earth Power for Heat Wave | 298 attacking turns: better in 22 (18 games); a KO only Earth Power had in 10 turns / 10 games (Hisuian Arcanine after its Sash, Archaludon, Tyranitar, Incineroar, Chandelure, Basculegion) | better in 15 turns (14 games); a KO only Heat Wave had in 4 turns — spread KOs at low HP, partly hindsight (Torkoal was hit earlier in the turn) | **modest gain — test it** |
| Venusaur Earth Power for Leaf Storm | 90 turns: KO only Earth Power had, 5 turns / 3 games (Archaludon 4, one game with Leaf Storm into Archaludon three turns running) | KO only Leaf Storm had, 7 turns / 5 games (Pelipper 4, Politoed, Milotic, a Blastoise) | **loss — keep Leaf Storm** (our answer to rain's Water types) |
| Charizard Weather Ball for Solar Beam | 55 turns: better in 9 (8 games); KO only Weather Ball had, 5 turns / 4 games (Salamence, Aerodactyl, Torkoal, Grapploct) | 1 turn (Solar Beam on Tyranitar, no weather) | **clear gain — do it** |
| Farigiraf Protect for Helping Hand | it fainted before acting 52 times (41 in losses) — a Protect could have saved some, if chosen | Helping Hand decided 10 KOs the partner could not get alone (9 of them in wins), 9 more maybe | **keep Helping Hand** |

Limits: one turn at a time (the rest of the game is not replayed, and the bot would
still have to choose the new move); hidden opponent sets use the standard spread, so
KOs near the threshold are uncertain; many "new set better" turns are turns where the
bot did not play the old set's best move either.

## 5. Recommendation and the test that settles it (proposal, nothing run)

1. **Keep T6 for now** and let the doomed-Pokemon lesson (running) attack T6's main
   ladder failure; a team switch now would throw away T6tac's practised openings,
   tactical fine-tune and guards tuned on T6 games.
2. **Team tournament** — the missing measurement. `evaluation/run_team_tournament.sh`
   with the newest human clone (`results_bc/mc_A_20260920/saves_bc/seed1/2.zip`, its
   BEST) as the pilot, the same
   weighted Reg M-C pool (`data/team_weights_regmc.json`), hidden sheets, seed 83:
   T6 and **T6m** (T6 + the two tweaks that survived section 4b: Charizard Weather
   Ball over Solar Beam, Torkoal Earth Power over Heat Wave) at 600 games each, T4
   and T0-T3, T5 at 300
   (~2-2.5 h, after the doomed chain; the pool is from 09-10 — optionally refresh it
   with this week's rosters first). Readings: T6m vs T6 (does the meta set help the
   team in human-like hands?), T6 vs the field.
3. **If T6m beats T6** there: a practice cycle for our brain on T6m (as the playbook
   practice cycle did for openings), then the mirror vs T6tac and the battery, then
   ladder at the user's word.
4. **A different team is worth it only if** it beats T6 in the clone tournament AND a
   brain trained on it beats T6tac locally; T4 is the only one with our own ladder
   evidence (27-23), and its brain lacks every improvement since 09-20.
