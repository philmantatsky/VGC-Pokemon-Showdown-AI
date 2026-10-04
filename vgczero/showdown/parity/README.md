# Showdown parity harness

Checks the Rust engine (`vgczero/engine`) against Pokemon Showdown for
**[Gen 9 Champions] VGC 2026 Reg M-C** (`gen9championsvgc2026regmc`), on the
teams of `data/teams_regmc.json.gz` that the engine fully supports.

| check | what is compared | pass criterion |
|---|---|---|
| `damage.js` | `getDamage` for random (attacker, defender, move) with all 16 rolls, with and without a crit | exact equality |
| `turns.js` (outcomes) | the state at the next decision point after a recorded decision, replayed N times in both simulators | no feature distribution differs (chi-square / KS, p < 1e-6) |
| `turns.js` (legality) | the legal actions of every request (moves, Mega, switch targets, passes) | exact equality |
| `cargo test` (`engine/tests/parity.rs`) | the same three checks against recorded fixtures, without Showdown | as above (turn check allows up to 1% flagged scenarios) |

## Setup

```sh
vgczero/showdown/setup.sh                 # pinned Showdown checkout + build in showdown/ps
# or: export PS_DIR=/path/to/built/pokemon-showdown
cd vgczero/engine && cargo build --release  # builds target/release/parity (and bench)
```

The scripts never modify the Showdown checkout. `PARITY_BIN` overrides the
engine binary, `FORMAT` the format id.

## Running

All commands from `vgczero/showdown/parity/`. Each exits with code 1 when it
finds a mismatch.

```sh
node damage.js --n 5000 --seed 1          # random scenarios
node damage.js --focus-all --per 20       # every supported ability and item forced in turn
node damage.js --focus item:ironball      # one ability / item; --move <id> forces a move

node turns.js --battles 300 --runs 200 --seed 1 --dump /tmp/flags.json
node turns.js --require pawmot,aegislash  # side 1 always brings one of these species
```

Damage scenarios vary boosts, weather, terrain, screens, spread, Helping
Hand, burn, Mega Evolution, items, abilities, HP, Friend Guard / auras etc.
Turn scenarios come from random Showdown battles with random legal choices;
about a third of the decision points (team preview, move requests, end-of-turn
and mid-turn switch requests) are sampled.

Debugging a flagged scenario from a `--dump` file:

```sh
node debug.js /tmp/flags.json b20.118.7         # one Showdown and one engine run, both logs
node debug_hits.js /tmp/flags.json b20.118.7    # HP change of every hit / residual, both sides
```

`--dump` also writes `<file>.legal.json` (request mismatches) and
`<file>.err.json` (engine errors on a recorded choice).

### Fixtures and `cargo test`

```sh
mkdir -p fixtures
node damage.js --n 3000 --seed 7 --fixture fixtures/damage.jsonl.gz
node turns.js --battles 100 --runs 200 --seed 7 --fixture fixtures/turns.jsonl.gz
cd ../../engine && cargo test --release --test parity -- --nocapture
```

The fixtures hold the engine-loadable state, the choices and Showdown's
outcome counts, so the Rust tests run without Node or Showdown (they are
skipped with a message when the fixtures are missing). Regenerate them after
changing the state format or the outcome features.

## How it works

* **State transfer.** `lib.js exportState` writes a Showdown battle (any
  decision point, including mid-turn requests with the remaining queue) as the
  JSON that `Battle::from_state_json` (`engine/src/snapshot.rs`) loads: field,
  side and slot conditions, party order, every Pokemon (HP, status counters,
  boosts, volatiles with their effect state, PP, stats, formes, item state,
  per-turn flags), the action queue and the pending switch slots. Anything the
  engine cannot represent makes the load fail, and the scenario is reported as
  skipped instead of compared.
* **Outcome features** (`lib.js outcome` / `Battle::outcome_features`): the
  next request, turn number, per Pokemon HP, status (+ sleep / toxic
  counters), boosts, position, item, ability, species, types, tracked
  volatiles, PP; side and slot conditions; weather, terrain, rooms; the order
  of `|move|` lines.
* **Statistics.** Each recorded decision is replayed `--runs` times in both
  simulators with fresh seeds (Showdown: deserialize + reseeded PRNG). Per
  feature a chi-square homogeneity test (values with fewer than 8 combined
  observations pooled) or a two-sample KS test for numbers; flagged when
  p < `--alpha` (1e-6, so with ~100k distributions per run a false alarm is
  rare). Flagged scenarios print both distributions and the battle context.
* **Damage.** Showdown's `getDamage` runs on a snapshot with a patched
  randomizer for each roll; the engine computes the same 32 values with
  `damage_probe` (`parity damage`).

## Results

| | before this work | now |
|---|---|---|
| damage, random | 294 / 300 exact | 5000 / 5000 exact (seeds 21, 31, 33), 0 / 160000 values differ |
| damage, every ability and item forced | not run | 4020 / 4020 exact |
| turn outcomes | 543 / 604 scenarios consistent (61 flagged) | 1784 / 1784 (seed 20), 1311 / 1311 (seed 10), 1280 / 1280 (seed 11), 440 / 440 (seed 7) |
| request legality | (no check) | 4651 / 4651 decision points (seed 20) |
| teams fully supported | 93.3 % (3362 / 3605) | 98.3 % (3545 / 3605) |
| `cargo test --test parity` | | 1000 damage cases, 250 turn scenarios, 283 requests: all pass |

Engine speed (`bench 50000`, random legal play): about 17k games/s now vs
18k games/s before this work (more mechanics simulated, more teams).

Focused turn runs (`--require`): Aegislash / Palafin forme changes, Pawmot
(Revival Blessing), Weavile / Salazzle / Delphox (Pickpocket, Shell Bell,
Magician), Lucario (Steel Beam): all consistent.

Remaining unsupported (the team is reported unsupported, never simulated
approximately): Dragon Darts, Ally Switch, Illusion, Instruct, Transform,
Sleep Talk and a few rarer moves / abilities / items; `bench` prints the
current list of blockers.

## Known approximations

These are known to differ from Showdown in rare situations; none showed up
in the runs above.

* Berries: Showdown runs the Update event only at specific points (after a
  hit, an action, weather); the engine also checks after each residual
  handler, so at the end of a turn a berry can be eaten a few handlers
  earlier (e.g. poison damage, then Sitrus, then Leech Seed).
* Destiny Bond takes effect when the damage is dealt rather than at the
  end-of-hit faint check.
* Quick Claw is not re-rolled when Encore changes the queued move.
* Emergency Exit does not trigger from entry-hazard damage on switch-in.
* Showdown's quirk of asking a second time for a slot that passed in a
  two-slots-one-replacement mid-turn switch request is not reproduced (only
  for Revival Blessing).
* Handler ordering ties (equal speed) are resolved with the engine's RNG,
  which only matters for log order.
