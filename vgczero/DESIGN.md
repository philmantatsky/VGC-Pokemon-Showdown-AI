# vgczero design

A self-play VGC bot in the style of **mikumiku37** (the bot that reached #1 on
the [Gen 9 Champions] VGC 2026 Reg M-C ladder in October 2026) and
**Jaxcalibur** (the Gen 9 Random Battle #1 bot whose writeup mikumiku37 adapted).
This directory is self-contained so it can be split into its own repository
later (`git subtree split -P vgczero`); it reuses the parent repo's team corpus
(`teams/reg_mc`) and nothing else.

## What the two reference bots do, and where each piece lives here

| Reference bot | Technique | vgczero |
|---|---|---|
| mikumiku37 | Custom simulator ~650x faster than Showdown | `engine/` (Rust). Measured here: **~17,600 random games/s on one core vs 25.6 for upstream Showdown on the same workload ≈ 690x** (`engine/src/bin/bench.rs`, `showdown/bench/showdown_speed.js`). |
| both | Small transformer, ~8.5-8.7M params, trained from scratch | `python/vgczero/model.py`, preset `base` = 8.7M params (`tiny`/`small` presets for laptops and tests). |
| mikumiku37 | Sees only its own side's view plus static dex data (types, base stats, move data); no damage calc, usage stats or speed resolver | `engine/src/obs.rs` hides everything the player could not know; static dex tables are separate inputs (`static_tables`). |
| both | PPO self-play against a league of past versions, terminal rewards only | `python/vgczero/ppo.py`, `league.py` (PFSP sampling of snapshots, current-policy self-play share). Reward is +1/-1 at the end, nothing else. |
| mikumiku37 | Trained on ~1,260 public tournament teams, some spreads guessed | 3,605 validated Reg M-C teams compiled from `teams/reg_mc` (`showdown/compile_teams.js`); 3,362 (93%) currently simulated faithfully. |
| mikumiku37 | Plays the ten teams that did best in training | `teams.py` (per-team ratings with Wilson bounds, optional focus sampling, top-k export). |
| mikumiku37 | AlphaZero-style one-turn search: in each of 16 sampled worlds, top-8 joint actions vs opponent's top-8 replies, payoff table from the value net, solved for a mixed strategy | `python/vgczero/search.py` + `matrix_game.py` + engine `worlds.rs` (hidden-set sampling) and `BattleBatch.expand` (all W×K×K children stepped in parallel in Rust). Also searches team preview. |
| Jaxcalibur | pUCT tree search 4 plies deep, 20,480 rollouts over 32 sampled worlds | `python/vgczero/search_tree.py`: open-loop decoupled simultaneous-move pUCT over sampled worlds (fresh world + dice per simulation, policy priors for both players, batched value-net leaves, default depth 4 decision points). `play.py --search tree`. |
| mikumiku37 | Accepts challenges on Showdown; declines open team sheets | `python/vgczero/live/` + `scripts/play.py`: accept / ladder / challenge modes, open team sheets rejected by default, search or raw policy, every battle logged. Tested bot-vs-bot on a local Showdown server. |

The user's requests on top of that: the team is allowed to change as the bot
learns which strategies are better (team ratings now, team evolution next), and
features from the existing antonius1 bot can be reused where they help
(its team corpus and ladder tooling; its guard/reranker layer is deliberately
*not* used, since the point of this project is that the network learns those
lessons itself).

## Architecture

```
showdown/          Node: Showdown is the source of truth
  export_dex.js      dex -> data/dex_regmc.json.gz (with each entry's scripted callbacks)
  compile_teams.js   teams/*.txt -> data/teams_regmc.json.gz (validated, stats computed by Showdown)
  parity/            differential tests vs Showdown (damage rolls, turn outcome distributions)
  bench/             Showdown speed reference
engine/            Rust crate vgczero-engine (+ Python module via maturin)
  battle/            the simulator (requests, queue, moves, damage, effects, residuals)
  obs.rs             what one player sees
  vecenv.rs          batched self-play env (rayon)
  worlds.rs          hidden-information sampling for search
  baseline.rs        random / greedy reference players
  py.rs              Python bindings
python/vgczero/    PyTorch: model, PPO + league, team stats, search, evaluation
scripts/train.py   training CLI
```

### Simulator

* **Data from Showdown, mechanics in Rust.** `export_dex.js` dumps the Champions
  mod's dex; `compile_teams.js` validates every team with Showdown's
  TeamValidator and stores Showdown-computed stats (and Mega forme stats), so the
  Champions stat formula never has to be re-derived.
* **Honest support tracking.** Every move/ability/item exports the names of its
  Showdown callbacks. The engine marks anything with scripted behaviour it does
  not implement as unsupported, and a team is trainable only if all of its
  sets are supported (`cargo run --release --bin bench` prints coverage and the
  top blockers). Currently 93.3% of teams; the largest blockers are Clanging
  Scales/Scale Shot (`selfBoost`), Revival Blessing, Stance Change, Dragon Darts,
  Ally Switch, Zero to Hero.
* **Showdown's control flow.** Requests (team preview / move / switch / wait),
  the action queue sorted by order/priority/speed with random tie-breaks and
  Gen 8+ re-sorting after every action, mid-turn switch requests (U-turn,
  Parting Shot, Emergency Exit, Eject Button), end-of-turn replacements, the
  residual order (`onResidualOrder`/subOrder/speed), Showdown's hit steps
  (invulnerability → TryHit → type immunity → TryImmunity → accuracy → hit loop),
  and the damage formula with 4096-based modifier chains and the same
  truncation points.
* **Champions specifics.** Stat points instead of EVs (from Showdown), PP
  `(pp/5+1)*4`, paralysis 1/8, sleep 2-3 turns, freeze at most 3 turns, Megas
  that do not revert, Fake Out / First Impression disabled after the first
  action, Encore re-targeting the queued move, Unseen Fist / Piercing Drill
  hitting through Protect for 1/4, opponent HP shown as floor(100·hp/maxhp).
* **Copyable state.** A `Battle` is fixed-size arrays (≈2 KB), so search clones
  it freely; the RNG lives inside the battle.
* **Correctness.** `showdown/parity/` compares exact damage for all 16 rolls
  (with/without crits, across weather, terrain, screens, items, abilities) and
  turn-outcome distributions from real mid-battle Showdown states; see its
  README for current numbers.

### Observation and network

13 entity tokens + CLS: the viewer's six Pokémon, the opponent's six (species
from team preview; moves/items/abilities only once revealed, or all of them
under open team sheets; exact stats never), and a field token (weather,
terrain, Trick Room, both sides' conditions, turn, request type). Each
Pokémon token mixes learned embeddings (species, ability, item, four moves,
last move, status, position, owner) with static dex features and ~80 numeric
features (HP, boosts, types, visible volatiles, Protect odds, "first action
available", ...).

Heads:
* **value** (tanh) for PPO and search leaves;
* **team preview**: 90 options = lead pair × back pair, scored from the chosen
  Pokémon's tokens;
* **turn actions**: two autoregressive heads over 31 per-slot actions
  (pass / switch to member k / move m × target × Mega). The second slot gets an
  embedding of the first slot's choice, so double-targeting, Protect + attack,
  Fake Out + setup and coordinated switches are representable as joint plans
  (the antonius1 brain's slot-independence was one of its documented limits).
  Joint legality (one Mega, no shared switch target) is masked exactly.

### Training

PPO with GAE computed over each player's own decision sequence (a player may
wait while the other picks a replacement), γ=1, λ=0.95, clipped value loss,
target-KL early stopping, cosine LR. Side 0 is the learner; side 1 is the
current policy (both sides then train) with probability `self_play_frac`,
otherwise a league snapshot drawn by PFSP weight (1 − learner win rate)^p.
Snapshots every `league_every` updates; the league evicts the members the
learner beats most reliably and keeps anchors. Open team sheets are on for
each side with probability `open_sheet_prob` so the network learns both modes.

Throughput math for planning: mikumiku37 played ~330M games in 48.3 h on one
RTX 5090 (≈1,900 games/s). With the engine at ~17.6k random games/s per core,
simulation is no longer the bottleneck; network inference is. On a GPU the
`base` model (14 tokens, 8.7M params) costs ~0.2 GFLOP per forward pass, so
1,900 games/s × ~10 decisions × 2 players ≈ 8 TFLOP/s of inference, roughly
10-25 TFLOP/s once the PPO backward passes are included -- within an RTX 5090's
reach in bf16; CPU cores step the engine in parallel (rayon).
Expect a MacBook (MPS) to run the `small` preset at a fraction of that.

### Search

As in mikumiku37 (one turn, simultaneous moves, solved per world), with two
additions: (1) the same machinery searches team preview, which is the decision
human players find hardest (bring four and lead two against the opponent's
likely leads), and (2) every world shares one RNG stream across its K×K cells
(common random numbers), so differences between cells come from the actions,
not the dice. Worlds sample the opponent's hidden sets from tournament sets
consistent with everything revealed (moves used, items/abilities that
activated, Mega Evolution, open sheets), and which unseen Pokémon were
brought. 16 worlds by default; the author found 4 too noisy and suggests 64-128.

### Teams

`TeamStats` records every finished game for the team the learner piloted
(both teams in pure self-play) and keeps posterior means, Wilson lower bounds
and a recent EMA. `team_focus > 0` biases the learner's team sampling toward
teams that are winning (with a uniform floor so every team keeps being
explored); `TeamStats.top(10)` is the ladder rotation (`scripts/top_teams.py`).
Team evolution (`evolve.py`, `scripts/evolve.py`) mutates the best teams (swap
a member for another pool set, swap items within the item clause, swap moves or
EV spreads among pool sets of the same species — legal without re-validation,
and `showdown/validate_team.js` double-checks), registers mutants with the
engine at runtime, scores them by self-play with the current policy against a
field of meta teams, and keeps the best by Wilson lower bound.

## Milestones

1. ✅ Data export, engine core, support report, 690x speed.
2. ✅ Observation encoder, batched env, Python bindings, baselines.
3. ✅ Model, PPO + league, team stats, evaluation.
4. ✅ One-turn matrix-game search (+ preview search).
5. 🔄 Parity vs Showdown (damage exact; turn-outcome statistics) and fixes.
6. ✅ Live Showdown client: websocket login/challenges, protocol → engine
   snapshot (own side from the request JSON, opponent from protocol messages +
   preview + placeholder sets), choices masked by the server's request
   (disagreements logged as a live parity check), open-team-sheet handling,
   ladder/challenge modes, battle logs.
7. ✅ Team evolution (`evolve.py`): legal recombination mutants of the best
   teams scored by self-play against a meta field; survivors validated by
   Showdown and playable via `play.py --team-file`. `scripts/top_teams.py`
   exports the ten best training teams for ladder rotation.
8. ✅ Deeper search (`search_tree.py`, Jaxcalibur-style). Still to measure
   against the one-turn matrix search at equal compute once a strong network
   exists; selective deepening of the matrix search's most influential leaves
   (the Nessie123 idea) is the other candidate.
9. ⏭ Coverage to ~100% of the pool (selfBoost moves, Revival Blessing, Ally
   Switch, Stance Change, Dragon Darts...).

## Practical notes

* Pin torch threads (`--threads`, `torch.set_num_threads`) when search or
  training runs next to the engine's rayon pool on the same cores:
  oversubscription slowed search ~90x in testing.
* First from-scratch CPU run (tiny model, 4 shared cores): 23.5% vs the greedy
  baseline after 8.7k games (random ≈ 8%). Real runs belong on a GPU.

* Gates before laddering, inherited from the antonius1 project's experience:
  head-to-head against the previous checkpoint and against `greedy`, with
  Wilson intervals, before any ladder games.
* The live bot will play on Showdown; the reference author chose not to
  release weights or engine because of the effect on ladders and tournaments.
  The same consideration applies to this repository's visibility.
