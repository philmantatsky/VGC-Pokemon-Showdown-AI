# What the bots that topped Reg M-C do, and what to take from them (2026-10-04)

The user, 2026-10-04: "take a look into the vgc other bot thats been going on in a seperate
context and see if u can learn anything from it".

**Sources:**
- **The other context:** a Claude Code cloud session building **vgczero** on branch
  `origin/claude/nice-bohr-mvs9zh`. It made 25 commits between 18:48 on 10-03 and 02:28 on 10-04
  (CDT), about 20k lines under `vgczero/`; see `vgczero/DESIGN.md` there.
- **The two bots it studies:**
  - [mikumiku37's Smogon writeup](https://www.smogon.com/forums/threads/mikumiku37-a-self-play-vgc-bot-reached-1-on-the-vgc-reg-m-c-ladder.3789199/)
  - [Nessie123's Smogon writeup](https://www.smogon.com/forums/threads/nessie123-an-ots-vgc-bot-that-topped-the-reg-m-c-bo3-ladder.3789213/)

## The bots side by side

| | mikumiku37 | Nessie123 | antonius1 (ours) |
|---|---|---|---|
| Result | #1 Reg M-C ladder on Oct 1: 1857 Elo, 81.2% GXE; 248-110 (69.3%) over 358 games with search | #1 Reg M-C Bo3 (open sheets) | Reg M-C peak 1353 |
| Policy alone | 1747 Elo peak (100 games); search added ~110 | -- | about 1300 (argmax + guards) |
| Network | 8.7M-parameter transformer, from scratch; own view + static dex only (no damage calc, no usage stats) | 1.5M-parameter evaluator + auxiliary heads (damage, next moves, survival time, game length) | PPO transformer with damage-calc knowledge features and a set prior; guards on top |
| Training | PPO self-play vs a league of past versions, terminal reward only; **330M games in 48 h** on one RTX 5090 in a simulator ~650x faster than Showdown; 1,260 tournament teams | **375k self-play games**, about 20 h on an M4 Mac Mini + $120 of cloud; half the games with mutated Pokemon (stats, typings, moves, items) | PPO on the real Showdown server (~136 steps/s), 35% human-clone opponents, HP/faint shaping; our side always the T6 family |
| Play-time search | One turn as a simultaneous-move game: in each sampled world (16; **4 caused outlier blunders**, 64-128 suggested), a payoff table of our top 8 joint actions x their top 8 replies scored by the value net, **solved for a mixed strategy**, averaged over worlds, then **sampled** | Double-oracle matrix-game search, branching on high-impact chance (KOs, crits, secondaries), selective deepening by reach probability x entropy; samples equilibrium strategies only | Argmax policy + guards; the exact search (Aug) is built but off: it aggregated 2 deep worlds (expectation / lower tail / worst case) and played one action, and tied the champion |
| Teams | Rotates the 10 teams that did best in its training | Baltimore Regional 6-2+ sheets | One fixed team (T6e) |
| Open sheets | Declines | Open-sheet format | Accepts |

vgczero rebuilds mikumiku37's recipe:
- **Engine:** a Rust doubles engine at about 17,600 random games/s per core, roughly 690x upstream
  Showdown.
- **Parity with Showdown:** damage exact on 160k rolls, turn outcomes statistically consistent,
  request legality identical.
- **Training:** PPO + PFSP league with terminal rewards.
- **Search:** both the one-turn matrix search and a Jaxcalibur-style pUCT.
- **The rest:** team-preview search, team evolution, and a live Showdown client.

Its first laptop run (tiny model, 0.7 h) only reached parity with a greedy baseline: "Real runs
belong on a GPU". The value today is the infrastructure and the recipe, not a stronger bot.

## What it means for us

1. **Simultaneous-move search is the shared ingredient, and the biggest lever we can reach
   without a GPU.** Both #1 bots solve each turn as a matrix game and play the mixed equilibrium.
   mikumiku37's search was worth about 110 Elo on top of an already strong policy. Our August
   exact search did three things differently, each of which their results argue against:
   - **Few worlds:** it deep-searched 2 worlds, while the reference author saw blunders at 4.
   - **Aggregation instead of a game:** it scored root actions by an aggregate over replies
     rather than solving the table.
   - **One action:** it played a single action instead of sampling the equilibrium.

   The pieces exist here: the exact Showdown bridge steps a whole turn,
   `exact_sim` samples hidden-set worlds, the brain proposes our top-K and the opponent move
   model their top-K, and the critic scores leaves. Proposed next project: a one-turn
   matrix-game search on the exact bridge (K about 6-8 per side, 16+ worlds, regret-matching
   solve, sample), with latency held under 8 s by batching children.

2. **Training scale is the gap we cannot close on the Showdown server.** They used 330M games;
   we have a few million steps. A fast simulator is the prerequisite, and that is what vgczero
   builds. The long-term options:
   - train our brain inside vgczero's engine, which needs an observation bridge from its state
     to our encoder;
   - make vgczero the next bot, once it has GPU time.

3. **Brains trained across many teams do not care about one set change.** Ours lost 43.4% to
   itself when Torkoal's Weather Ball became Earth Power. Theirs train on 1,260-3,600 teams, and
   Nessie mutates half its Pokemon. For our next practice cycle: vary our own side's sets
   (mutations of T6e) instead of always playing the exact team.

4. **Terminal reward only.** Both use it; we shape on HP and faints (0.05 / 0.10). This is worth
   an A/B once practice works again; shaping can favour damage races over winning lines.

5. **Mixed strategies, no exploitation.** Both sample equilibria. Our argmax play is predictable
   to human opponents. The mirror cannot measure what unpredictability buys; only the ladder can.

6. **What not to copy at our scale:**
   - Dropping the damage calc, usage priors and guards: they replace those with 100x more games.
   - Declining open sheets: the user wants them kept.

## What this changed tonight

- **Practice failure:** the first lesson was diagnosed before reading vgczero. Practice from a
  fine-tuned brain drifts as far as practice from scratch (0.52 vs 0.48 nats), which overwrites
  the 0.24-nat fine-tune. The **KL-anchored practice** (`vgc_bench/src/anchored_ppo.py`) keeps
  it near its start: the "keep the lessons inside training" fix.
- **Recommendation for the user:** the one-turn matrix search (point 1) is the strongest idea
  in these writeups that fits our hardware.
