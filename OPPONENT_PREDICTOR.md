# Opponent predictor — side experiment

Started 2026-10-04 at the user's word. A separate track from the matrix search; it touches
nothing in the deployed decision path.

**Goal.** A model that, given the public state of a doubles battle at the start of a turn,
gives the probability of what the opposing player clicks for each of their two active
Pokemon: Protect, attack into which slot, spread move, support/setup, voluntary switch (to
which Pokemon), Mega. It learns from thousands of human replays across the whole rating
range, takes the player's Elo as an input, and works with open team sheets and without
them. The deliverable is an artifact plus a runtime class the deployed bot can call each
turn.

**Status (2026-10-05 03:50).** Built, reviewed, scored, and running in **shadow mode**
on the deployed bot since 03:43 (the user's word): the forecast is logged with every move
decision and no decision reads it. The neural predictor beats the count-table bar by 0.27
nats on our own ladder opponents (R1 passes; exact action top-1 43.6%), Elo adds 0.001-0.005
nats (R2: it does not stay, the model is Elo-blind), and the existing reranker could not use
even a perfect prediction (R4), so a first real use is a guard, after the shadow log has
been read. Details and numbers under "Results" at the bottom.

## What reconnaissance established (2026-10-04, before any code)

Eight read-only scouts mapped the ground. The findings that shape the design:

1. **The existing opponent models are weak.** `data/opponent_move_top500_regmc.pt` sees only
   species ids, four HP numbers and the turn. On our own ladder opponents it scores NLL 3.34
   against 1.89 for a plain species -> move frequency table (top-1 25.2% vs 28.7%). The saved
   weights are the last epoch, not the best (the trainers snapshot `best_state` without
   `.clone()`). Rating is used only as a filter, never as an input.
2. **Elo may carry little.** Count tables on 4,905 bo1 Reg M-C battles, players held out by
   name: adding Elo bands gains nothing (-0.0002 +/- 0.0003 nats; never positive in 12
   settings), while three state flags (turn 1, first turn on field, protected last turn) gain
   +0.033 to +0.054. The marginal action mix does shift with rating (voluntary switches
   6-7% below 1400, 9-10% above), but a count table cannot use it once species is known. So
   Elo is an input with a kill criterion, not the organising idea.
3. **The shift to our own opponents is larger than any Elo effect.** The same table scores
   1.32 nats on the bot's ladder opponents against 1.27 on held-out human players. The bot's
   own saved games are the only on-distribution test.
4. **Data.** 10,004 unique Reg M-C battles on disk (4,905 bo1, 5,099 bo3; 256k slot-turns;
   09-09..09-27). The public replay feed holds about 111,000 more (about 69k bo1, 43k bo3).
   Feed rows carry a rating, so the fetch can be stratified by Elo. Public replays are about
   4% of ladder games and 7 accounts appear in 25% of feed rows -> per-account cap.
5. **Labels are partly hidden, and not at random.** 83% of slot-turns show their action;
   8% were knocked out before acting, 4% are forfeit turns, 3% flinch/sleep/paralysis. A
   voluntary switch and a Protect are always visible, attacks are not, so training on visible
   labels alone over-predicts both. The logged target is where the move landed (after
   redirection and retargeting), not what was clicked.
6. **Train/serve parity is achievable.** A poke-env `DoubleBattle` can be driven straight
   from a replay log (0.6 ms per log), and the live bot holds the same raw event stream in
   `battle._replay_data`. One event-fed tracker serves both; the bot's own exact HP is
   rewritten to the public percent.
7. **The opponent's Elo is available live** on ladder (`battle._players`, 406 of 406 saved
   games) and absent in challenges and local games -> "unknown Elo" must be a normal input.
8. **Where it can act.** Guards touch 30% of ladder move decisions; the existing reranker's
   opponent term changes 1.7%; search is off in the deployed bot and losing its head-to-heads.
   First use is shadow mode (log the forecast, change nothing), then an opt-in guard.

## Design

### Layout

    vgc_bench/src/oppmodel/      library (no script logic)
      events.py                  pure-string log reading: turn segments, per-slot actions
      public_state.py            PublicBattle (event-fed public state), drive_log, LiveShadow
      features.py                Vocab, Featurizer, array layout, Predictor protocol
      tables.py                  count-table predictors (the baselines, and the day-one model)
      model.py                   the neural predictor
      runtime.py                 OpponentPredictor: predict(battle) -> Forecast
    datagen/oppmodel_scrape.py           feed index walk + Elo-stratified fetch (resumable)
    datagen/oppmodel_build_dataset.py    corpus -> examples, splits, manifest
    training/train_oppmodel.py
    evaluation/oppmodel_scorecard.py     the pre-registered reading
    evaluation/oppmodel_oracle.py        perfect-foresight ceiling through the existing reranker
    unit_tests/test_oppmodel_*.py
    battle_logs_feed_mc/  results_oppmodel/      data and results (git-ignored by pattern)

### Rules every module follows

- **Team-agnostic.** No species or move name in code. Behaviour classes come from dex
  properties (category, target type, priority, flags) and `move_semantics`.
- **One state path.** Features are computed only from `PublicBattle`, which consumes raw
  split events. Offline it is fed log lines, live it is fed `battle._replay_data[cursor:]`.
  Never featurise from the live battle object (private request data and exact HP would leak).
- **Absolute sides.** Drive each log once; index by `p1` / `p2`; emit examples for both
  players with the actor's roster first. Targets are the foe's slot letter (`foe_a`, `foe_b`),
  never a seat-mirrored number.
- **Nothing raises into a decision.** No bare `assert` on any path the runtime calls. Catch,
  count under a named counter, return "no forecast".
- **CPU only at runtime**, thread-safe, cache keyed by battle tag + role.
- Scripts run from the repo root with `.venv/bin/python`; ruff, pyright
  (`--pythonpath .venv/bin/python`) and pytest pass on every new file.

### Labels (events.py)

Per turn and per slot that was alive at the start of the turn, one `SlotAction`:

| field | meaning |
|---|---|
| `kind` | `move`, `switch`, `hidden` (a `cant` line or a confusion self-hit), `none` (no action seen) |
| `move`, `target` | move id; `foe_a` / `foe_b` / `ally` / `self` / `auto` (spread, field, not choosable) |
| `switch_to` | species brought in by a voluntary switch |
| `mega` | a `-mega` line for this slot this turn (visible even when the move is not) |
| `reason` | for `hidden` / `none`: `flinch`, `slp`, `par`, `frz`, `confusion`, `fainted_first`, `game_ended`, `forced`, ... |
| `switch_known` | whether we can tell if the slot switched (false only when the turn never resolved) |
| `protect_known` | whether we can tell if the slot protected |
| `target_trusted` | false when a redirector was on the field or a foe had already fainted that turn |
| `locked` | a forced continuation (`[from] lockedmove`, recharge) — not a choice, excluded from loss |

Reading rules (from the simulator source and the scouts' corpus census):

- Voluntary switch = a `switch` line for a live turn-start slot before that side's first
  `move` / `cant` and before `upkeep`. `drag`, pivots (after the slot moved), Eject Button and
  end-of-turn replacements are not choices.
- Move = the first `move` line of the Pokemon that held the slot at turn start, without a
  `[from]` tag (except `[from] move: Round`). Later untagged lines are Instruct repeats.
- `cant` with `[of]` (Armor Tail, Queenly Majesty, ...) names the ability holder as subject;
  the blocked attacker is in `[of]`.
- `[still]` with an empty target: take the target from the following `-anim` line or the
  next turn's locked-move line; else the target is unknown.
- Actions belong to the Pokemon in the slot at turn start (Ally Switch does not move them).
- Slots are tracked by slot letter, never by nickname (Illusion duplicates nicknames).

Censoring is handled, not dropped:

- `none` with `fainted_first`, or `hidden` by flinch: the slot chose a move that was not a
  switch and not a Protect-family move (those resolve first). Loss = `-log` of the summed
  probability of the remaining actions.
- `hidden` by sleep / paralysis / freeze / confusion: chose some move (not a switch).
- A `-mega` line on a censored slot also proves "a move, not a switch".
- Turn never resolved (forfeit before any event): no information, masked.
- Event calibration (switch, Protect, "my slot is attacked") is computed over every slot-turn
  where the event is known, censored ones included.

### State and features

`PublicSnapshot` is plain data (no poke-env objects), taken right after the `turn|N` line:
context (turn, weather / terrain / Trick Room with turns since start and whether set after
upkeep, bo1 or bo3, sheets open per side, each player's rating or unknown), two sides
(Tailwind, screens, Mega used, fainted, revealed, unknown-brought = 4 - revealed) and twelve
Pokemon in preview order (species and current forme, public HP fraction with 1.0 when never
seen, status, boosts, slot, revealed, fainted, is-mega, mega-possible, first turn on field,
protected last turn, protect streak, turns on field, item and its state, ability and whether
known, up to four moves each flagged revealed-in-battle / from-sheet, a small set of
volatiles, last turn's action).

Action space per actor slot: up to 12 candidate moves + `OTHER` + a pointer over the actor's
six roster Pokemon (switch). Candidate moves are the sheet's four when the sheet is open;
otherwise the moves revealed so far plus the species' most used moves from the training
split's repertoire table (stored in the artifact). Each candidate carries its id, the 84
`move_semantics` floats, and flags (revealed, from sheet, prior frequency). Target is a
5-way head per candidate move; Mega is one logit.

Derived intent classes (7): switch, Protect-family, single-target attack on `foe_a`, on
`foe_b`, spread attack, status/disruption aimed at a foe, self / ally / field support and
setup.

### Models

- `SpeciesTable` — species -> action frequencies. The floor.
- `FlagsTable` — species x (turn 1, first turn on field, protected last turn), backed off to
  the species table. **The bar the neural model must beat**, and the day-one model behind the
  runtime interface.
- `EloTable` — FlagsTable plus an Elo-band offset. Exists to test the Elo premise cheaply.
- `OppNet` — a small transformer over 15 tokens (context, two sides, twelve Pokemon) with
  candidate-move scoring; the species prior enters as a logit offset so it starts at the
  table and learns the residual. Elo of both players enters as a bucket embedding plus an
  unknown flag. Plain cross-entropy (no class reweighting: the consumer needs true
  frequencies), then one temperature per head fitted on validation.
- Legacy baseline: the shipped `opponent_move_top500_regmc.pt` / `opponent_switch_top500_regmc.pt`
  scored by the same harness.

### Splits and hygiene

- Dedup by replay id. Drop battles with zero turns, Zoroark/Zorua logs (counted), and any
  battle with the bot's own account.
- **Ladder holdout:** the opponent side of every saved own Reg M-C game (about 406 rated
  games, 400 opponents, about 4,100 visible labels). Never trained on. Battles in the human
  corpus that involve a ladder-holdout opponent are excluded from training.
- **Player holdout:** the actor's account decides the split: `crc32(name) % 10` -> 0 test,
  1 validation, 2-9 train. A player is never on both sides of the split.
- **Time slice:** the latest 10% of battles by upload time, reported inside the test players.
- **Clone hygiene:** open-sheet games in crc32 buckets 5-9 of
  `battle_logs_top_mc_merged_20260920` are excluded from training (the eval-only human clones
  were fitted on them).
- **Account cap:** an account's examples are down-weighted so no account counts for more than
  60 battles.
- Closed-sheet mode is trained and judged on real closed-sheet bo1 games; bo3 feeds the
  open-sheet mode.

### Runtime

    predictor = OpponentPredictor.load("results_oppmodel/<run>/artifact.pt")
    forecast = predictor.predict(battle)        # None when it stands down; never raises

`Forecast` holds, per opposing slot: the ranked action list with probabilities, and the
scalars a guard can read directly — `p_switch`, `p_protect`, `p_fake_out`, `p_attacks[my slot]`,
the seven intent probabilities, `p_mega`, and `support` (how much training evidence backs it).
It also converts to the existing `MovePrediction` / `SwitchPrediction` so the current
reranker and planner prior can consume it unchanged.

### How it would reach the bot (step 1 is live since 2026-10-05, see Results)

1. **Shadow mode**: compute the forecast before the search branch in `PolicyPlayer`, write it
   into the decision audit, change nothing. Every ladder session then scores the predictor for
   free. Needs one default-off flag that is dropped from run-config material when off.
2. **An opt-in guard** reading the scalars, A/B tested like any other guard. First candidate:
   gate `threat_first2` on the predicted attack actually being likely.
3. Reranker adapter; then, only if search becomes competitive, the reply prior with a
   restricted Nash response (`p * best response + (1-p) * equilibrium`).

## PRE-REGISTERED readings (2026-10-04 17:05, before any model was fitted)

Baseline numbers already seen: the scouts' count-table readings quoted above. No neural
result exists yet.

- **R1 — quality.** Metric: mean NLL per labelled slot-turn on the fine action (move x target,
  or switch destination), censored slots scored by their set loss. Sets: (a) ladder holdout,
  (b) test players, (c) test players in the time slice. `OppNet` passes if its paired
  difference to `FlagsTable` has a game-clustered bootstrap 95% interval below zero on (a) and
  (b). Top-1 / top-3 and 7-class intent accuracy are reported, not gated.
- **R2 — Elo.** Compare `OppNet` with its Elo input shuffled at evaluation, and with an
  Elo-blind retrain. Elo stays only if the gain is at least 0.02 nats on (a) and (b). "Elo
  adds nothing" is an allowed outcome; then the shipped model is Elo-blind and this file says so.
- **R3 — decision relevance (descriptive).** Share of slot-turns where P(Protect) >= 0.5,
  P(switch) >= 0.35, P(my slot is attacked) >= 0.7, and the precision there, with reliability
  tables for those three events. Any consumer's flip threshold is set from these before its A/B.
- **R4 — oracle ceiling.** Feed the opponent's actual action, as a certain prediction, to the
  existing reranker on the bot's logged ladder move decisions. If it flips fewer than 5% of
  them, or the flipped picks are not better against what the opponent really did (game-clustered
  interval excluding zero), that consumer cannot use any predictor and the first decision-changing
  use must be a guard.
- **Not claimed:** win rate. The local mirror opponent is our own bot, which a human-move
  predictor is not expected to predict; ladder is the only arbiter and a ladder trial is the
  user's call.

## Results

### Corrections to the notes above (2026-10-04 20:00, from the build)

- **Ladder is not always closed-sheet.** A saved replay page is rebuilt from
  `battle._replay_data`, which never contains `|showteam|`, so "0 of 406 saved games had open
  sheets" measured nothing. The bot's decision audit (`preview_shadow.open_sheet`) says 10
  open and 116 closed among the 126 games with a record (7.9%); 2 of the bot's 11 public
  replays show both sheets. `PublicSnapshot.sheet_open` is therefore `True / False / None`,
  and a player-view stream is `None` until the caller says. No training example carries
  "unknown"; evaluation and serving map it to closed (`features.sheet_unknown_as_closed`).
- **Label fields added by the audit:** `target_forced` (one foe stood at turn start: the
  target is not a choice, 8,882 rows), reason `overridden` (a same-turn Encore replaced the
  click: 317 rows, hidden), `forced` (Struggle). A dex-aimed move logged with `[spread]`
  (Expanding Force in Psychic Terrain, 2,264 rows) is `auto` / spread attack.
- **Split names in the dataset:** `train`, `val`, `test`, `ladder_holdout`, `own_unrated`,
  `excluded_holdout_player`, `excluded_clone`. Accounts are hashed by the server's user id,
  not the display name (77 accounts appear under more than one display name). The repertoire
  is keyed by the forme that owns the moveset, so regional formes are not merged.

### Foundation: audited, train/serve parity holds (2026-10-04)

`events.py` + `public_state.py`, three independent audits, 25 findings (0 blockers, 6 major),
23 fixed each with a test that fails without the fix. 292 experiment tests pass; ruff and
pyright clean; no bare assert in library code. `evaluation/oppmodel_parity.py` exits 0:

- Live path vs offline path on 408 saved own pages (bot p1 in 219, p2 in 189), 2,620 turns,
  three feeding modes including a mid-game reconnect: 0 snapshot mismatches, 0 exact-HP
  strings left after the rewrite.
- Spectator vs player view of the same battle (the bot's 11 public replays, 84 turns, 140
  bot-side HP strings): identical once sheets are treated the same on both sides.
- Seat symmetry on 1,500 human logs (9,924 turns, 38,120 actions): 0 mismatches.
  `drive_log` 1.4 ms per log.
- Label census on that sample: move 74.3%, voluntary switch 10.0%, hidden 3.1%, no action
  seen 12.6%; targets untrusted 7.1%.

### Dataset `results_oppmodel/v1_ondisk` (2026-10-04, feed still downloading)

- 15,558 battles kept (15,155 human, of which 5,453 already from the feed; 403 own), 208,487
  examples, 397,440 slot-turns, 15.6% censored (47,838 of those carry a set loss).
- Examples: train 121,016, val 20,020, test 20,628, ladder holdout 2,598 (401 games, 395
  opponents), excluded holdout-player 18,396, excluded clone 25,812.
- 8,911 accounts; 38 over the 60-battle cap. Three of the six heaviest accounts are opponents
  of the bot, so the holdout rule removes all their battles from training.
- True move outside the 12 candidates (closed sheets): train 1.1%, test 1.9%, ladder holdout
  2.2% (recorded closed) / 3.0% (sheet state not recorded).
- Floors through the scoring code, no model fitted: uniform over legal actions, fine NLL
  2.58 / 2.64 / 2.77 (val / test / ladder holdout); species prior with a constant switch rate
  and uniform targets 2.17 / 2.20 / 2.27, action top-1 30.0 / 30.0 / 28.3%.
- A rebuild under a new tag follows when the download is done.

### R4 — oracle ceiling: the reranker cannot use a predictor (2026-10-04)

`evaluation/oppmodel_oracle.py`, `results_oppmodel/oracle/` (13 s, CPU only). The count
reproduces: 3,433 audited records = 250 preview + 552 forced replacement + **2,631 move
decisions** in 403 battles; 2,624 rebuilt; the replay reproduces the played pick in 99.0%.

- **Perfect foresight flips 106 decisions = 4.0% [3.3, 4.8]**, below the pre-registered 5%.
  By the reading fixed above: this consumer cannot use any predictor; the first
  decision-changing use must be a guard.
- Only 53 flips (2.0%) need the true action: mostly "Protect the slot about to be hit" (24)
  and "retarget away from a slot that switches" (15). 31 more also happen when the gates are
  opened with no information at all, and 22 are replay mismatches.
- With the gates at their normal values the true action flips 28 decisions (1.1%).
- The reranker has no term for a true Protect (593 known) and cannot use a switch into a
  Pokemon it has not seen (260 of 378).
- Hindsight on a one-turn damage exchange: the 106 flips are not clearly better (+0.052
  [-0.028, +0.129]); the 53 that need the truth are (+0.141 [+0.045, +0.234]).
- Limit: where both opposing actions are visible (1,677 decisions) the share is 4.5%
  [3.6, 5.6], which reaches the bar. Flips are counted, not played out.

### R1-R3 — models (2026-10-04 23:25)

`results_oppmodel/scorecard_v1/` (`evaluation/oppmodel_scorecard.py`, 10,000 game-clustered
resamples). Sets: (a) ladder holdout, 401 own games, 4,810 labelled slot-turns; (b) test
players, 2,866 games, 38,056 slot-turns; (c) test players in the latest 10% by time (training
covers the same period, so (c) is not a forward-in-time test).

**Standing of these numbers.** `oppnet_v1` and `oppnet_v1_blind` (904,953 parameters, 14
minutes each on CPU, best epoch 16 of 22) were trained before the review fixes: their
validation split still held 1,431 rows of battles with a ladder-holdout opponent. The
reviewer measured the effect on (a) at 0.004 nats. A retrain on the fixed trainer and a
larger dataset follows; until then read the neural rows as the first reading, not the final.

**The bar was strengthened after the first reading.** The review found the pre-registered
bar easy on targets: a count layer on type effectiveness against each foe and which foe has
less HP gains 0.03-0.055 nats. `flags_table` now includes it and is the reference;
`flags_table_plain_targets` is the bar as first fitted. Both margins are given.

| predictor | fine NLL (a) | minus bar (a) [95%] | minus bar (b) [95%] | top-1 (a) | top-3 (a) |
|---|---|---|---|---|---|
| shipped MoveNet + SwitchNet | 3.605 | +1.737 [+1.603, +1.874] | +1.216 [+1.169, +1.264] | 18.3% | 41.5% |
| species table | 1.918 | +0.050 [+0.039, +0.060] | +0.075 [+0.071, +0.079] | | |
| flags table, plain targets | 1.898 | +0.030 [+0.022, +0.037] | +0.055 [+0.052, +0.058] | 30.0% | 59.9% |
| **flags table (the bar)** | 1.868 | reference | reference | 31.4% | 61.8% |
| Elo table | 1.866 | -0.002 [-0.003, -0.001] | -0.001 [-0.001, -0.000] | | |
| OppNet with Elo | 1.663 | -0.205 [-0.234, -0.177] | -0.196 [-0.205, -0.186] | 41.8% | 72.5% |
| **OppNet, Elo-blind** | 1.665 | -0.203 [-0.232, -0.175] | -0.191 [-0.201, -0.182] | 40.9% | 71.3% |

- **R1 passes.** Against the bar as pre-registered (plain targets) the margin is -0.235
  [-0.264, -0.206] on (a) and -0.251 [-0.261, -0.241] on (b); against the strengthened bar
  -0.205 and -0.196. With accounts instead of games as the resampling unit the (b) interval
  widens to [-0.212, -0.179]. Against the plain bar, 87% of 926 test accounts and 83% of 391
  ladder opponents are individually better predicted.
- **R2: Elo does not stay.** OppNet's gain from Elo: 0.0023 / 0.0023 nats over shuffled
  ratings on (a) / (b), 0.0020 / 0.0048 over the Elo-blind retrain; the Elo table gains
  0.0007 / 0.0011. The bar was 0.02. On (a) the two nets differ by -0.002 [-0.015, +0.011].
  **The model to ship is the Elo-blind one.** Accuracy is flat across rating bands (fine NLL
  on (a): 1.65 below 1100, 1.69 at 1100-1199, 1.64 at 1200-1299, 1.66 at 1300-1399).
- The shipped opponent models are far behind: 3.61 against 1.66 on our ladder opponents. Even
  with two temperatures picked on validation (not the shipped behaviour) they stay at 2.42.
- Independent check: a reviewer rewrote the scoring by hand from the array definitions and
  reproduced every table number to 7e-15 and the neural ones to four decimals; found no fitted
  quantity that reads test or ladder rows; and a label-leak probe (cut or replace turn t's
  events, re-encode) changed no feature in 7,846 examples.

**R3 — decision relevance (OppNet with Elo, ladder holdout; descriptive).**

| event | P at least | share of cases | mean predicted | happened |
|---|---|---|---|---|
| a foe attacks a given one of our two slots (both on the field) | 0.70 | 6.9% | 77.6% | 80.3% [75.7, 84.8] |
| same | 0.80 | 2.4% | 84.2% | 86.6% [79.9, 92.6] |
| one of our two slots is attacked by either foe | 0.70 | 29.9% | 79.8% | 80.4% [77.6, 83.3] |
| same | 0.80 | 13.2% | 86.4% | 85.5% [81.7, 89.2] |
| same | 0.90 | 2.9% | 92.6% | 95.3% [89.6, 100] |
| the slot uses a Protect-family move | 0.50 | 5.9% | 63.3% | 56.8% [50.6, 62.8] |
| same | 0.60 | 3.3% | 70.6% | 62.2% [54.8, 69.2] |
| the slot switches out by choice | 0.35 | 5.1% | 50.9% | 38.9% [33.2, 44.7] |
| same | 0.60 | 1.3% | 70.3% | 54.8% [42.9, 66.7] |

- "Who is attacked" is calibrated and has usable confident mass; the count table reaches 0.7
  on only 1.4% of the two-slot cases (precision 65%).
- Protect (base rate 12.4%) and switch (7.9%) are predicted well above their base rates but
  over-confidently at the top: says 51%, happens 39%. `Forecast.p_switch` / `p_protect` need a
  calibration map before any guard reads them as probabilities.
- An earlier version of these tables counted "attacked by either" only when the answer was
  yes on rows with a hidden attacker, and mixed in turns with one of our Pokemon on the field
  (where any attack hits it). Both are corrected here; the first reading (94% at 0.9 on 8.3%
  of slot-turns) was inflated.

### Second pass: fixed trainer, and more data helps a lot (2026-10-05 00:30)

One chain, single-thread and niced (44 minutes): the Elo-blind model retrained on the v1
dataset with the fixed trainer; the dataset rebuilt on everything downloaded by 23:43
(`results_oppmodel/v2_feed`: 26,847 human battles against 15,155, 228,512 training examples
against 121,016, same 401 ladder-holdout games); tables refitted; two fits on it; scorecards
`scorecard_v1r/` and `scorecard_v2/`. The ladder holdout is the same games in both builds,
each model fed by its own build's featurizer.

| model (Elo-blind unless said) | training battles | fine NLL (a) | minus its bar (a) [95%] | top-1 | top-3 |
|---|---|---|---|---|---|
| first fit, v1 | 15,155 | 1.665 | -0.203 [-0.232, -0.175] | 40.9% | 71.3% |
| fixed trainer, v1 | 15,155 | 1.672 | -0.196 [-0.223, -0.169] | 40.3% | 71.2% |
| **fixed trainer, v2** | 26,847 | **1.583** | -0.271 [-0.303, -0.239] | 43.6% | 74.6% |
| fixed trainer, v2, with Elo | 26,847 | 1.580 | -0.275 [-0.305, -0.245] | 44.1% | 75.2% |
| flags table (the bar), v2 | 26,847 | 1.855 | reference | 31.2% | 62.1% |

- **The fixed-trainer reading confirms R1**: -0.196 on (a) and -0.191 [-0.200, -0.181] on (b)
  on the v1 data. The first fit and the retrain differ by 0.007 on (a), which is the only
  measure of run-to-run noise so far.
- **More data is worth 0.09 nats on our ladder opponents** (1.672 -> 1.583) and 3 points of
  top-1, from 1.8 times the battles. The count table gains 0.014 from the same data. The
  added games are also the recent, rating-balanced part of the feed, closer to our opponents
  than the top-player scrapes, so not all of the gain is volume. Either way the model is
  data-limited and the rest of the download (about 96,000 games) is worth having.
- On held-out players (v2's own, larger test set): -0.249 [-0.257, -0.241].
- **R2 again: Elo does not stay.** With the larger data its gain is 0.0046 / 0.0027 over
  shuffled ratings and 0.0038 / 0.0013 over the Elo-blind retrain on (a) / (b); bar 0.02.
- R3 with the v2 model on (a): Protect at P >= 0.5 covers 5.9%, happens 62.4% [56.0, 68.6]
  (mean predicted 63.5%); at P >= 0.8, 0.4%, 85%; switch at P >= 0.35 covers 5.8%, happens
  44.4% (predicted 50.3%); "one of our two slots is attacked by either foe" at P >= 0.9 covers
  5.1%, happens 91.4% [85.7, 96.3]. Protect is now close to calibrated; switch still reads high.
- **The model to use is `results_oppmodel/oppnet_v2_blind/artifact.pt`.**

### Runtime class (2026-10-04)

`vgc_bench/src/oppmodel/runtime.py`: `OpponentPredictor.load(path).predict(battle)` ->
`Forecast` or `None`. `evaluation/oppmodel_shadow_replay.py` (`results_oppmodel/shadow_replay/`):

- The runtime equals the offline path on 2,615 of 2,615 turns of 408 saved own pages, fed per
  turn, per event, and through real poke-env `DoubleBattle` objects: largest difference 0.0.
- Latency per call on one thread while the machine was loaded (load 13-25, not ladder-valid):
  count table p50 0.9 ms, p99 1.5 ms; neural p99 3.6-5.2 ms, largest call 12.3 ms.
- It stands down by name instead of guessing: not at a turn start, another format, a turn
  number that jumps, a damaged stream, a malformed prediction, a missing dex. Nothing raises.
- `to_move_prediction` / `to_switch_prediction` convert a forecast to the existing dataclasses;
  reliability defaults to known moves / 4, so a closed-sheet turn does not open the
  reranker's known-set gate.

### Third build: calibration, joint reply coverage, learning-curve tooling (2026-10-05 03:30)

Three more pieces, reviewed (20 findings, 18 fixed with tests; 754 predictor tests).

**Event calibration** (`vgc_bench/src/oppmodel/calibration.py`,
`training/calibrate_oppmodel.py`; artifacts `results_oppmodel/c_oppnet_v2_blind_cal/`,
`c_oppnet_v2_cal/`). One monotone map for P(switch) and one for P(Protect | no switch),
fitted on validation only and applied inside the action distribution, so everything stays
normalised.

- On held-out players the over-confidence goes: P(switch) >= 0.35 said 49.5% / happened
  40.2% before, says 46.2% / happens 46.3% after; P(Protect) >= 0.5 said 62.3% / 55.3%,
  now 60.7% / 60.8%. Fine NLL improves by 0.003 on (a) and (b), intervals below zero.
- **What the over-confidence was.** It sits in slot-turns whose action is hidden (knocked
  out before moving, flinch, game ended), which count as "did not switch / did not
  Protect". Among visible actions alone the uncalibrated model was 3 points high for switch
  and 4 points LOW for Protect. The maps fit the turn-start event, so after the fit visible
  actions happen more often than predicted, and fine NLL on visible actions gets slightly
  worse (+0.002 test, +0.005 ladder) while hidden ones gain 0.04-0.05. Which of the two a
  guard should read is a decision for whoever writes the guard.
- **Not demonstrated on our ladder opponents.** The mean bias shrinks (switch +1.4 -> +0.5
  points, Protect +2.0 -> +1.1) but at the thresholds the gap only changes sign inside the
  noise (200-300 slot-turns). Our opponents switch less and protect more than the human
  corpus; a map fitted on human validation cannot fix that. A first-turn term in the switch
  map helps on test and not on the ladder holdout, and its acceptance rule was changed after
  test and ladder numbers had been seen. **So the model in shadow mode is the uncalibrated
  one**, and a calibration for our own opponents should be fitted from the shadow log.

**Joint reply coverage** (`vgc_bench/src/oppmodel/joint.py`, scorecard section; card
`results_oppmodel/c_scorecard_v2_final/`). How often the opponent's real pair of actions is
among the predictor's 8 likeliest joint replies, over turns where every acting slot's action
is fully visible (66% of turns). The joint is the product of the two slots' marginals.

| top-8 coverage | count table | OppNet (Elo-blind) | shipped models |
|---|---|---|---|
| ladder holdout, all (1,714 turns) | 46.4% | 62.3% [59.5, 64.9] | 25.8% |
| ladder, a move not shown before (1,271) | 36.2% | 55.4% [52.3, 58.4] | 20.5% |
| ladder, every move shown or on the sheet (336) | 93.2% | 90.2% [85.9, 93.8] | 51.2% |
| test players, all (24,749) | 45.4% | 61.6% [60.9, 62.3] | 29.3% |
| test, closed sheets (19,386) | 44.0% | 59.0% | 26.0% |
| test, open sheets (5,363) | 50.7% | 71.0% | 41.4% |

- Top-1 / 4 / 16 / 32 on the ladder holdout: 24.8 / 50.0 / 72.8 / 81.7%. Turn 1 is the
  hardest (45.6% top-8); with one acting slot it is 98.6%.
- The binding constraint is the same one the search session measured in its own worlds: a
  reply that uses a move not shown before is covered 55% of the time, against 83% when no
  unshown move is involved (sheets not open).
- What independence loses (test, two acting slots): both Protect happens 1.9 times as often
  as the product of the marginals says, both attacking the same slot 0.87 times, switch +
  Protect 1.6 times. A joint head would recover some of this.

**Learning-curve tooling** (builder `--human-fraction`, `training/oppmodel_learning_curve.py`):
nested random subsamples of one corpus, each built, fitted and scored alike, paired on the
same ladder games.

### Learning curve: every doubling still helps, the last one half as much (2026-10-05 05:37)

`results_oppmodel/lc20261005_curve/` (108 minutes on one niced thread). One frozen corpus
(the download paused while the four datasets were built: identical inputs, 0% drift),
39,025 human battles at 100%; Elo-blind model; the same 401 ladder-holdout games in every
build. One fit per point, so the intervals hold game sampling, not training noise (about
0.007 between two fits of the same data).

| corpus | human battles | fine NLL (a) | top-1 | top-3 | minus the count table of that build |
|---|---|---|---|---|---|
| 12.5% | 4,938 | 1.888 | 34.1% | 63.8% | -0.000 [-0.027, +0.029] |
| 25% | 9,821 | 1.744 | 38.3% | 68.9% | -0.127 [-0.154, -0.099] |
| 50% | 19,712 | 1.606 | 43.0% | 73.7% | -0.254 [-0.281, -0.227] |
| 100% | 39,025 | 1.539 | 44.2% | 75.7% | -0.312 [-0.342, -0.282] |

- Paired steps on the ladder holdout (larger minus smaller, same slot-turns): -0.144
  [-0.165, -0.123], -0.138 [-0.161, -0.115], then **-0.067 [-0.087, -0.049]** for the last
  doubling. On held-out players: -0.161, -0.108, -0.091 [-0.098, -0.085].
- The count table hardly moves with data (1.888 -> 1.851 on (a)): below about 5,000 battles
  the neural model is no better than the table; all of its advantage comes from data.
- The remaining download (about 78,000 games queued at 05:37) is roughly 1.6 more doublings.
  If the steps keep shrinking as they have, that is worth a few hundredths of a nat on our
  ladder opponents and more on held-out players: worth finishing, no longer dramatic.
- The 100% point here (`lc20261005_f100_oppnet/artifact.pt`, 1.539 on (a)) is better than
  the model in shadow mode (`oppnet_v2_blind`, 1.583, trained on 26,847 battles). Swapping
  the shadow model is a manifest change and waits for the user's word; the natural moment is
  the retrain after the download.

### Shadow mode is ON (2026-10-05 03:43, the user: "turn on shadow mode")

- **What runs.** `results_deployed/opponent_forecast_oppnet_v2_blind.pt` (sha 95ffbde2; the
  Elo-blind model trained on `v2_feed`, uncalibrated) is in `DEPLOYED.json`
  (`opponent_forecast`, `opponent_forecast_sha256`, an amendment quoting the user). The
  launchers pass `--opponent-forecast`; the bot computes the forecast at every move decision
  and writes it into `decisions.jsonl` under `opponent_forecast`. **No decision reads it.**
- **Where.** `PolicyPlayer._opponent_forecast_record` (called once in `_guarded_action`
  before the search branch), `_audit_decision`'s payload, `_forget_opponent_forecast` at
  battle end. The flag is not part of a replay directory's play configuration (it is in
  `VOLATILE_ARGS`), so `replay_tag` and the directories are unchanged; each run records the
  artifact's path and sha256 in `run_config.json`.
- **What it cannot do.** Every failure is counted (`opponent_forecast_failed:*`,
  `opponent_forecast_unloaded:*`) and the decision goes on; a launch line says
  `forecast: shadow mode on ...` or `SHADOW MODE NOT SERVING (...)`. A test pins that the
  record is made, passed to the audit and written, and that no other module mentions it.
- **Checked.** 18 unit tests (`unit_tests/test_opponent_forecast_shadow.py`), one of them on
  real `DoubleBattle` objects fed by saved games; the project's default test run: 919 passed,
  5 skipped; `checks/forecast_shadow_live.py`: two local games with the deployed
  configuration (one open-sheet, one closed) logged a forecast for its own turn on every move
  decision, stood down only on the forced replacements, and counted no failure. Runtime
  equals the offline path on 2,615 of 2,615 saved turns; p99 3.0 ms per call.
- **Live.** The challenge listener was restarted at 03:43 and reports
  `forecast: shadow mode on`. Ladder sessions pick it up at their next
  `tools/ladder_deployed.sh`. No real game has been logged yet.
- **A record.** `{"turn", "model", "kind", "elo", "own_elo", "sheets", "sheets_reported",
  "latency_ms", "slots": [per opposing slot: species, ranked actions with probabilities,
  intents, p_switch, p_protect, p_fake_out, p_attacks (our slot a, b), p_mega, known_moves,
  support]}`, about 2 kB; `{"stand_down": reason}` at a forced replacement.
- **To turn it off:** remove `opponent_forecast` and `opponent_forecast_sha256` from
  `DEPLOYED.json` and restart the listener.
- **To read it later:** every new ladder game is an out-of-sample test. Score the logged
  forecasts against what the opponent then did (the same label reader), by sheet state, and
  fit the calibration for our own opponents from it.
  Join a forecast to its replay by battle tag AND replay directory: a local Showdown
  server reuses battle tags after a restart (the search session met 1,127 reused tags), so a
  tag alone identifies a battle only on the real server.

### Fourth build — PRE-REGISTERED 2026-10-10 00:40, before any of its results

The user (10-10, about 00:25): "check out how the predictor did [on the ladder games] ...
and see how u can make it better ... work on your own until morning."

**State at this moment.** The download is finished (113,201 replays: 76,870 single games,
46,444 best-of-three games indexed; 24 deferred). The model in the bot is still
`oppnet_v2_blind` (26,847 battles). Since 10-09 the search reads its forecast as the reply
prior in ladder trials (the search session's work, weight 0.75): 70 ladder games were played
on 10-09 (`ladder_replays_mc_search_nash1/2/3`, `ladder_replays_mc_search_forecast1`) and
ten more are being played tonight into `ladder_replays_mc_search_forecast1`. Numbers already
seen from those games, all from the search session's reading
(`results_analysis/search_ladder_20261009/`): on the 30 trial games the predictor's own eight
likeliest reply pairs held the real reply in 96 of 170 hidden-sheet decisions (56%), its
first pair in 21%; replies holding a switch are covered worst (9 of 29 by the forecast
alone). No reading of mine on those games exists yet.
*Correction, 01:45:* the 56% / 21% above undercounted. The search session's reading tool
read a forecast switch's destination from the wrong key, so a correctly forecast switch
could never count (found by tonight's code map, confirmed and fixed by the search session,
main 859a20ce). Corrected by them: 105 of 170 (62%) at 30 games; 134 of 218 (61%) over the
40 trial games, first pair 23% — level with the search's table as played (62%).

**Sets.**
- *Development:* validation players and the OLD ladder holdout (the 401 own games played
  before 10-05, in every dataset build so far). Every design choice below is made on these.
- *Confirmation:* the FRESH ladder set = the opponent side of every own ladder game played on
  10-09 and 10-10 (about 80 games), and the test players. The fresh games are read now once,
  in aggregate, for the models that already exist (that is the user's first question), and
  once more at the end for the final candidate. No design choice is made from them and
  nobody looks through them for failure patterns.

**What is tried** (in a separate worktree, branch `oppmodel-night-1010`; the code the bot
runs does not change while ladder games play):
1. The same architecture retrained on the full download (`v3full_20261010`).
2. Explicit matchup inputs (type effectiveness and an estimated damage of each candidate on
   each target, the threat on the slot, speed order) — new arrays that an old artifact never
   reads, so the model in the bot stays bit-identical.
3. A coupling of the two slots when ranking reply pairs (a table over intent-class pairs).
4. More capacity, if the larger corpus asks for it.
5. A correction for our own opponents (they switch less and protect more than the corpus),
   fitted on the OLD ladder holdout only.

**Readings, fixed now.** Paired with the model in the bot on the same slot-turns,
game-clustered bootstrap, 95% intervals:
- **R5a.** Fine NLL on the fresh ladder set: the final candidate counts as better on our
  ladder opponents if the interval of (candidate − deployed) lies below zero.
- **R5b.** Joint top-8 coverage on the fresh ladder set, turns whose acting slots are all
  visible: reported with its interval; "not worse" is required (interval not wholly below
  zero), "better" is claimed only if it lies above zero. With about 350 such turns a gain
  under about four points cannot be shown; that is said when it happens.
- **R5c.** Coverage of replies that hold a switch and of replies that hold a Protect, and
  the switch / Protect calibration: reported, not gated (tens of cases).
- Each single change (2-5) is kept only if it improves the development sets beyond the
  fit-to-fit noise (0.007 nats between two fits of the same data); ablations are run on one
  dataset build so that only the change differs.
- **Not claimed:** win rate, or the coverage of the search's own table (that needs its live
  worlds; a quarter of real replies are legal in none of them, which no predictor changes).
- **What follows from it.** A candidate that passes R5a and is not worse on R5b is
  RECOMMENDED as the new shadow / reply-prior model. The swap is a `DEPLOYED.json` change and
  stays the user's decision.

**Amendments, 02:23, each written before its own result exists.**
- *Dropped (5), a correction for our own opponents.* Measured on the OLD ladder holdout,
  cross-fitted: two logit offsets (switch −0.27, Protect −0.32) gain 0.0044 nats
  [−0.0076, −0.0009], under the 0.007 bar, and the gain comes from slots that were stopped
  before acting; on visible actions it is worse, and a real switch falls out of the slot's
  top three more often (51.9% → 46.8%). It would cost the search exactly the replies it
  covers worst.
- *Not built: in-game style.* How a player has behaved earlier in the same game does predict
  the next action beyond the model, but by 0.0014–0.0022 nats.
- *Added (6), a fine-tune on our own old ladder games.* On the ladder the opponent always
  faces this bot and these six Pokemon; the corpus holds few such games. Grid fixed now:
  learning rate {1e-5, 3e-5, 1e-4} × passes over the ladder rows {1, 2, 4}, one part ladder
  rows to four parts corpus rows, 5 folds by game over the 401 OLD ladder games only. Taken
  only if the best setting's pooled out-of-fold fine NLL change has its interval below zero,
  joint top-8 is not lower by more than its interval's half-width, and replies holding a
  switch do not lose more than 2 points; otherwise not taken. The fresh games stay sealed.
- *The slot coupling's rule:* fitted on validation by the turn-start set likelihood, 10 folds
  by battle; taken only if the gain is above 0.002 nats per two-slot row in at least 8 folds.
  (A table fitted as a plain ratio on fully seen turns looks better on counted top-8 and is
  worse at turn start: it is selection, and it is not what is fitted.)
- *Expectation written down before the fits:* residual probes on the frozen model say the
  network already holds most of what the new inputs carry — matchup facts 0.006 nats, the
  set-conditional move prior 0.003–0.004 — so the new inputs are expected to add about a
  hundredth of a nat, and the larger corpus and a wider network most of the gain.

### Not done, and what is next

- **Read the shadow log** once real games exist; then a first guard, with thresholds fixed
  from the decision-relevance tables beforehand and the usual guard A/B.
- **Retrain when the download is done** (29,976 of about 113,000 games fetched at 03:48;
  the second pass shows the model is data-limited). The learning curve now running says
  whether the last doubling still helps.
- A joint head for the two slots (the independence table above), and a set prior for
  unshown moves: both aim at the turns the predictor covers worst.
- Not measured: the deployed brain played from the opponent's seat as a baseline (needs the
  brain on a quiet machine).
- Deferred by the reviews: the three prior trainers still save last-epoch weights
  (`train_move_model.py`, `train_switch_model.py`, `train_preview_model.py`) and
  `opponent_tactics._hp_fraction("50/100g")` returns 1.0. Both touch deployed models' lineage.
