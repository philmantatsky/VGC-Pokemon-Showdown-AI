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

**Status.** See "Results" at the bottom. Nothing here is wired into the bot yet.

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

### How it would reach the bot (not built in this pass)

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

(none yet)
