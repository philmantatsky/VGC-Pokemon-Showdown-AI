# VGC Bot Project Status

## Ready for the morning, nothing run: the search against the battery's rosters, and the ladder launcher's search flags (2026-October 4, 23:30)

- **`evaluation/search_roster_ab.py` + `tools/search_roster_arms.sh`.** The head-to-head is
  one matchup: our team on both sides and our own brain as the search's model of the
  opponent. The new harness plays the deployed bot against each of the held-out battery's
  six opponents on its 47 held-out rosters twice per cell -- as deployed and with the
  search -- and compares the arms roster by roster (bootstrap over rosters, as
  `run_guard_ab.py` does). Defaults are the battery's own (seed 20923, 10 a category, its
  exclusions, 11 games a cell: 1,034 games an arm and opponent). One process per opponent
  (the searching side plays one battle at a time), cells appended whole (resumable), a
  `STOP` file ends a process at its next cell. Smoked end to end against the heuristic and
  the human clone: 125 searched decisions on five foreign rosters, one budget fallback, no
  error. **Not run** -- the machine is V6's; a full pass is about 7-8 hours with six
  processes (4 hours at 6 games a cell). This is the gate I would put between a mirror win
  and a ladder trial.
- **A ladder trial can now play what the mirror measures** (the user's word starts it,
  nothing here does). `ladder_ourteam.py`: `--search-anchor`, `--search-argmax`,
  `--search-replies`, `--search-leaf-calibration`, `--search-streams` (`--device cpu`
  existed). `tools/ladder_trial.sh`: `TRIAL_SEARCH="..."` (refused unless it carries
  `--search` and `--device cpu`; the measured flags are written in its header and parsed
  in a test); the read loop passes `SEARCH`, `ladder_deployed.sh` sets it empty.
  - The new flags stay out of a replay directory's recorded configuration at their
    defaults: parsing the running challenge listener's own arguments gives exactly the 52
    keys its directory holds (`build_parser()` and `material_config()` are functions now,
    so this is a test -- the same class of change refused a listener restart on 10-04).
  - Found on the way: `ladder_ourteam.py --help` has raised since 09-26 (a bare `%` in a
    help string). Fixed; the help is formatted in a test.
- **What the search changes** (the pilot's 800 games, 512 overrides, 6.0% of decisions,
  2.2 a game; `evaluation/search_override_report.py`, whose per-game table joined shards
  by battle tag alone and so counted 233 games of 800 -- fixed): Water Spout -> Water Pulse
  146, Throat Chop -> Flare Blitz 55, Throat Chop -> Fake Out 46, Ice Beam -> Water Pulse
  34, Rain Dance -> Helping Hand 30. Three in five overrides are "the other attack" on
  Blastoise or Incineroar -- the move-choice faults the ladder reviews kept finding and
  the guards patch one at a time. Median edge 0.22, median prior ratio 1.8, median turn 5;
  guards sent back another 446.

## While V6 runs: a leaf sees what the bot sees; replacement decisions are noise to the search; half the overrides are thin (2026-October 4, 22:55)

Three read-only probes (scratch wrappers around `evaluation/mirror_guard_ab.py`, one
low-priority process at a time; nothing in the repo changed, V6 runs frozen code).
- **Leaf fidelity: 37 of 41.** For every table cell whose turn ended at a move request the
  probe kept the leaf's observation, and at the next real decision compared it with the
  bot's own observation whenever the predicted position WAS the real one (same turn,
  actives, HP, status, boosts, weather, fields, side conditions). 8 games, 80 searched
  decisions: 41 such cells, **37 identical in all 12,276 entries**; the other 4 differ only
  in the gender of an opposing Pokemon that came in during the simulated turn (a world
  draws its own gender for a Pokemon the live battle has not seen on the field). The
  live-view machinery is faithful; what a leaf gets wrong is which branch happens, not what
  the branch looks like. Gender: small, open.
- **Forced-switch decisions: the search's values there are noise.** They are searched too
  (not by the one-turn table: the older two-ply search). When the two candidates are the
  SAME last Pokemon into slot A or slot B -- a difference the critic prices at 0.007 on
  165 positions, live-anchored or rebuilt views alike -- the search's two expected values
  differ with **sd 0.21 (mean |gap| 0.175; 70% above 0.05, 31% above 0.2; n = 218** from V4
  and the pilot). The override threshold at anchor 0.07 is 0.05-0.15, so replacements are
  overridden on noise: 83 of 3,339 (2.5%), 65 of them the meaningless slot flip, 18 a
  different Pokemon. Small in games (one real change in ~60), but the right rule is plain:
  **keep the bot's own replacement** until that search is rebuilt. Not in V6 (frozen).
- **Move decisions: stream noise is small, and half the overrides are thin.** 24 games, 211
  move decisions with 4-6 random streams each: the paired difference between a candidate
  and the bot's pick varies from stream to stream with a median sd of 0.035 (p75 0.09, p90
  0.18). Of 13 overrides, 7 have edge / standard error >= 2 (6 of them > 3) and 6 are
  below 2 (5 below 1). **Next idea (V7): an override must clear its own standard error**
  (a lower confidence bound on the paired edge instead of the mean). What the standard
  error cannot see is error common to every stream: the critic's own, and a wrong reply
  model.
- V6 health at 22:48 (six shards + one niced probe): one-stream arms median 1.0 s, 7% of
  decisions at the budget; streams arms median 1.75 s, 17-19% at the budget (KO-heavy
  tables with four streams still run out of time: the cheaper leaf did not cure that).
  First rounds: raw 56 / 100, calibrated 57 / 100.

## The search's copies of the battle read a zeroed threat block (30% of lookups): fixed; encoder a third faster with the same bytes; V5 ended as a pilot, V6 pre-registered (2026-October 4, 22:08, before the run)

- **Found while making the encoder faster** (its identity test failed on a *second* call on
  the same position). `PolicyPlayer._knowledge_for` and `_threat_for` cache one result per
  state, and the cached vectors are keyed by `id(pokemon)` -- but the cache key had no
  object in it. A second battle object in the same state hit the first one's entry, found
  none of its own Pokemon there and read zeros.
  - **The exact search:** every leaf is a copy of the live battle. In a 4-game searched
    probe **3,231 of 10,617 threat lookups (30.4%) returned another object's entry**; 0 of
    11,178 on the fixed code. All search runs so far (V1-V4, the V5 pilot) valued about a
    third of their leaves, and ranked many replies, without the threat block.
  - **The mirror harness** (both players in one process): with the same leads at full HP on
    turn 1 one side read a zeroed block. The harness already swapped the knowledge cache
    for `_NoCache` for this reason; the threat cache was missed.
  - **The ladder bot** has one battle object per process: unaffected, and the fix is
    identical for it (the key is more specific by a per-object constant).
  - **Size:** zeroing the block on 310 played-out positions moves the critic by 0.064 on
    average (sd 0.10, mean +0.03, max 0.43) and the first slot's favourite in 3.2% of
    positions (none on turns 1-3, 3% on turns 4-6, 8% after) -- so the mirror's turn-1 case
    was small; the search's 30% was noise of the same order as its override threshold.
  - **Fix** (`vgc_bench/src/policy_player.py`): the key carries `id(battle)`, and a hit must
    name every active it is asked for (an address is reused once a copy is gone).
    `unit_tests/test_observation_identity.py`: a copy reads its own blocks, the same object
    still reuses its entry, a stale entry at a reused address is not served (all three fail
    on the old code).
- **Encoder, same bytes, less time.** `embed_move` read the move's category, target and type
  once per enum member (38 property reads a move, each walking the move's data entry; the
  target runs a regex); `embed_pokemon` did the same for types, tera, gender and status and
  walked all 226 `Effect`s for every Pokemon. Each is read once now, and a Pokemon without
  volatile effects takes a constant. `ExactPolicyAdapter.rank` updates the joint mask once
  per first action instead of once per choice. Same load, cpu: **a leaf 3.5 -> 2.2 ms, a
  ranking 17 -> 7 ms.** The encoding as it stood at 0cb9ed91 is kept verbatim in the test as
  the reference: every move in the vocabulary (unused and used) and 2,000+ Pokemon tokens
  of played-out positions (both seats, sheets open and hidden, with and without volatile
  effects, with a status) encode to the same bytes.
- **V5 ended as a pilot.** Its four arms were asked to stop after their first round (their
  own `_STOP` files, 21:49; nothing was killed -- an earlier attempt to kill and relaunch
  with fewer shards was refused by the permission system). 200 games an arm on the old code
  (`../vgc-bench-v5`, f40463d2): reported apart, never pooled with V6. No result of it had
  been read when this was decided.
  - Its load reading stands: 8 searching shards saturate the machine (load 14-17; a shard
    is 0.8 core of python + 0.26 of the bridge), and under it the streams arms sat at the
    time budget in 21% of decisions (6-7% for the one-stream arms).
- **Null check on the frozen V6 code** (worktree `../vgc-bench-v6` at 19445d0e; cpu, streams
  4, anchor 1e12): **0 of 93** decisions differ from the bot's own (main tree: 0 of 96).
  Calibrated + streams smoke: 8 overrides in 82 decisions, expected edge +0.26, realized
  +0.49 on the four whose reply was in the table, none negative (a smoke).
- **V6 PRE-REGISTERED (replaces V5; the same question and the same 2 x 2).** Side A = the
  deployed bot + nash search (critic leaf, 4 worlds, 8 s, argmax, anchor 0.07, champion
  boost 2, 8 likeliest replies, live views, the player's guards), side B = the deployed
  bot, both sides' networks on cpu, one battle at a time per process, code 19445d0e.

  | arm (`results_mirror_<name>`) | leaf | streams | shards | port | SEED |
  |---|---|---|---|---|---|
  | `search_nash6` | raw | one per world | 1 | 7622 | 60924 |
  | `search_nash6cal` | calibrated | one per world | 1 | 7623 | 70924 |
  | `search_nash6s4` | raw | >= 4 | 2 | 7624 | 80924 |
  | `search_nash6s4cal` | calibrated | >= 4 | 2 | 7625 | 90924 |

  - **Six shards, not eight** (the load reading above); the streams arms play about 1.7x
    slower, so two shards there and one on the others gives the arms similar game counts.
    Rounds of 100 games a shard, pooled after every round. The one-stream arms start when
    the pilot's one-stream arms have ended (about 22:11), the streams arms when the pilot's
    have (about 22:35), so no more than six search processes ever share the machine.
  - **Stop by the clock, not by the numbers:** a detached timer touches the four `_STOP`
    files at 07:30 on 10-05; rounds in play finish; every complete shard counts. Expected:
    about 1,800-2,400 games an arm.
  - **Reading (unchanged):** per arm, pooled Wilson lower bound > 50% = the search wins;
    upper bound < 50% = it loses; otherwise no detectable difference. Four arms are four
    chances: a lone win whose lower bound is within a point of 50% needs a replication.
    Factors, each over both levels of the other (difference of pooled rates, 95% Newcombe
    interval): streams >= 4 vs one; calibrated vs raw. `evaluation/search_arms_report.py`.
  - Secondary, descriptive: open vs hidden halves; decisions at the budget; overrides and
    guard send-backs; expected vs realized edges; reply coverage; the pilot's 200 games an
    arm next to the same arm here.
  - A winning arm is a candidate for a ladder trial with `--device cpu`, the user's
    decision. Search stays OFF in the deployed bot; DEPLOYED.json is not touched.
  - **Launched:** the one-stream arms 22:15, the streams arms 22:32 (when the pilot's had
    ended); six shards, every manifest checked; all at normal priority (zsh starts `&`
    jobs at nice 5 -- every detached run before this one, the pilot included, ran niced).
    The 07:30 timer is armed (`results_mirror_search_nash6_clockstop.log`).
- **The V5 pilot's one round** (old code, 8 shards on a saturated machine, 200 games an
  arm; a pilot, not a result): raw 107 = 53.5% [46.6, 60.3]; calibrated 123 = 61.5% [54.6,
  68.0]; streams 107 = 53.5% [46.6, 60.3]; streams + calibrated 104 = 52.0% [45.1, 58.8];
  together 441 / 800 = 55.1% [51.7, 58.5].
- **The machine:** `~/Desktop` is in iCloud Drive. Creating the V6 worktree at 22:02 sent
  the load average to 97 for eight minutes (`bird`, `fileproviderd`, Spotlight), and they
  take one to two cores whenever results are written. A worktree or any bulk write under
  Desktop belongs well before a timed run, never during one.

## Search spent 85% of its time on GPU overhead (cpu is 15x faster for it); 41% of searched decisions saw one run of the dice; V5 amended to four cpu arms (2026-October 4, 21:30, before the run)

- **Found while checking the machine load for V5** (a wall-clock stack sampler around four
  searched games, the code V5 was frozen on):
  - V4's searched decisions took 2.6 s at the median and 7.3 s at the 90th percentile of a
    7.35 s planning budget; 17% were flagged truncated (the flag under-counts: a cell cut at
    the deadline with its replacement pending is not flagged).
  - **85% of that time was network overhead on mps, not simulation.**
    `torch.embedding_renorm_` is unsupported on mps and falls back to cpu on every forward
    pass (19.5% of all search time); `ExactPolicyAdapter.rank` updates the joint mask once
    per legal choice with a tensor on mps (16% + 12%); the simulator bridge is 11%.
  - One position (T6ep, turn 1, 174 legal choices), cpu vs mps: **ranking 14.4 ms vs
    211.6 ms, a critic leaf 2.8 ms vs 11.2 ms**; probabilities equal to 5e-7, the value to
    1.3e-6, the top ten in the same order.
  - The same four games with `--device cpu` (new flag of `evaluation/mirror_guard_ab.py`,
    both sides; default mps = every earlier run): 41.8 s in search instead of 114.8 s, the
    median decision 0.74 s, tables of 530 nodes at the 90th percentile instead of 240-276
    (the KO-heavy tables finish). On cpu the cost is `PolicyPlayer.embed_battle` (47%, about
    2 ms a position) and the bridge (24%); neither was touched.
- **Second finding: one run of the dice.** Every world resolves its cells on its own random
  stream. **41% of V4's searched decisions were planned over ONE world** -- an open sheet
  once all four opposing Pokemon are known (29%), and any turn the worlds were rebuilt (11%:
  the "concentrate on the likeliest world" rule kept from the multi-turn search) -- so one
  stream of damage rolls, misses and crits decided the comparison. Overrides were more
  frequent there: hidden sheets 7.9% vs 2.9% of decisions (anchor 0.07) and 2.0% vs 1.0%
  (0.2); open sheets 3.3% vs 1.5% (0.2), 6.5% vs 8.1% (0.07).
  - New: `--a-search-streams N` (`LiveExactSession(min_streams=)`,
    `PolicyPlayer(exact_min_streams=)`). A decision averages at least N streams = worlds x
    streams per world (N=4: one world -> 4, three -> 2, four or more -> 1) and a rebuild turn
    keeps its worlds. Default 0 = as before. Explicit streams are seeded from each world's
    own PRNG state (reproducible; different between worlds).
  - Also new: `--a-search-chance-samples` (a fixed count per world; not used tonight);
    `tools/search_mirror_rounds.sh` runs in the checkout it is called from and takes `SEED`.
- **Checked and NOT changed: a deadline-cut cell.** Out of time, a cell whose turn ended in
  a KO is valued with the replacement still pending. On 3,983 such positions from 127
  simulated T6e mirrors: pending minus resolved value = -0.006 (they replace, n=1,826, sd
  0.07), +0.019 (we replace, n=1,781, sd 0.15), -0.037 (both, n=376). Noisier, not biased.
- **Null check on the frozen code** (worktree `../vgc-bench-v5` at f40463d2; cpu, streams
  4, anchor 1e12): **0 of 85** decisions differ from the bot's own (main tree: 0 of 82 with
  streams, 0 of 83 without). Calibrated + streams smoke from the worktree: 8 overrides in 89
  decisions, expected edge +0.24, realized +0.29, none negative (a smoke).
- Decision time on cpu with two probes and another session's 8-battle mirror on the
  machine: one stream p50 0.7-1.1 s, p90 4.8-5.2 s; streams 4 p50 1.5-1.7 s, p90 7.3 s
  (about one decision in eight still reaches the budget).
- **What this does to the old numbers:** every search result so far (V1-V4, and August's
  "exact search only ties") was measured with the networks on mps, i.e. with a search that
  had a quarter to a third of the thinking it could have had. They stand as measurements of
  that configuration only.
- **V5 AMENDED before launch (replaces the 20:45 plan; no V5 game had been played).** Four
  arms, a 2 x 2 of leaf (raw / calibrated) x streams (one per world / at least 4). Common to
  all: side A = the deployed bot + nash search (critic leaf, 4 worlds, 8 s, argmax, anchor
  0.07, champion boost 2, 8 likeliest replies, live views, the player's guards), side B =
  the deployed bot, **both sides' networks on cpu**, one battle at a time per process.

  | arm (`results_mirror_<name>`) | leaf | streams | port | SEED |
  |---|---|---|---|---|
  | `search_nash5` (the 20:45 plan's raw arm) | raw | one per world | 7612 | 20924 |
  | `search_nash5cal` (its calibrated arm) | calibrated | one per world | 7613 | 30924 |
  | `search_nash5s4` | raw | >= 4 | 7614 | 40924 |
  | `search_nash5s4cal` | calibrated | >= 4 | 7615 | 50924 |

  - Two shards per arm (8 processes), rounds of 100 games a shard, pooled after every round;
    frozen code = worktree `../vgc-bench-v5` at f40463d2 (the main tree stays free for the
    other sessions). **Stop by the clock, not by the numbers:** `_STOP` files at 07:30 on
    10-05 whatever the pooled lines say; rounds in play finish; every complete shard
    counts. Expected: about 2,000-3,500 games an arm.
  - Changes from the 20:45 plan, all made before any game: cpu instead of mps; 2 shards and
    a clock stop instead of 3 shards x 2,400 games; the two streams arms.
  - **Reading, per arm (unchanged):** pooled Wilson lower bound > 50% = the search wins;
    upper bound < 50% = it loses; otherwise no detectable difference. Four arms are four
    chances: a lone win whose lower bound is within a point of 50% is a candidate that needs
    a replication. **Factors** (each over both levels of the other, difference of pooled
    rates with its 95% interval): streams >= 4 vs one; calibrated vs raw.
  - Secondary, descriptive: open vs hidden halves; decisions at the budget; overrides and
    guard send-backs; expected vs realized edges; reply coverage; the one-world share.
  - A winning arm is a candidate for a ladder trial with `--device cpu`, which is the
    user's decision. Search stays OFF in the deployed bot; DEPLOYED.json is not touched.
  - **Launched 21:31** (after the other session's 21:00 mirror had ended): 8 shards, every
    manifest checked (cpu, anchor 0.07, 8 likeliest replies, streams 0 / 4, calibration sha
    a29504a3 on the two calibrated arms, T6ep on T6e, 14 guards). A detached timer touches
    the four `_STOP` files at 07:30 (`results_mirror_search_nash5_clockstop.log`).

## Opposing Megas keep their pre-Mega stats: confirmed, measured on 404 ladder games; opt-in `forme_stats` (guards only) built; mirror pre-registered (2026-October 4, 21:00, before the run)

- **The lead** (from the opponent-predictor session, by reading): `vgc_knowledge.ensure_stats`
  gives an opposing Pokemon a stat estimate (32 HP, 32 in the better attack, 2 Speed, neutral
  nature) the first time a calculation needs one and returns early afterwards. poke-env
  updates `base_stats` on `-mega` / `detailschange` / `-formechange` but not `stats`, and the
  live bot runs the calculator against every active foe at every decision, so a foe is always
  estimated in the forme it came in with. **Confirmed, and wider than reported:**
  - open sheets too (`_update_from_teambuilder` + `impute_stats` freeze the same line), and
    every forme change, not only Megas (Palafin-Hero: 11 ladder turns; Aegislash);
  - Mega Blastoise Special Attack 137 for 187, Mega Charizard Y 161 for 211, Mega Raichu Y 110
    for 212, Mega Charizard X Attack 104 for 182, Mega Gengar Speed 132 for 152;
  - `stats_were_synthesized` reads False for such a line; our own side is fine (the server's
    request carries the Mega's stats);
  - the replay audits of 10-03 could not see it: a position rebuilt in one go estimates the
    foe AFTER its Mega (`unit_tests/test_forme_stats.py`, the "two answers" test).
- **Measured before changing anything** (`evaluation/forme_stats_audit.py` ->
  `results_analysis/forme_stats_20261004/`: every saved Reg M-C game rebuilt turn by turn as
  the live bot held it, on the opponent-predictor session's `Position`; one core, 40 s):
  - *Exposure.* A stale opposing forme is on the field in **286 of 404 games (70.8%) and 904
    of 2,620 turns (34.5%)**; in 917 of 2,641 logged move decisions (34.7%). Most turns: Mega
    Salamence 110, Garchomp 91, Gardevoir 90, Raichu Y 88, Tyranitar 78, Charizard Y 46.
  - *Accuracy against what really happened* (clean hits only -- no crit, multi-hit, knock-out,
    resist berry, Helping Hand, or change of the board or of either Pokemon earlier in the
    turn; real damage / middle of the calculator's range):
    - hits BY the stale foe on ours, n=155: stored line **1.27** (the real damage is above
      the whole range in 82% of hits), right line **1.01**, control (opposing Pokemon whose
      forme is right, n=456) 1.01; the right line is the closer one in 86.5% of hits;
    - our hits INTO it, n=132: stored 0.91, right 1.06, control (n=484) 0.97; mean |log
      error| 0.217 -> 0.167 (control 0.168); the right line is closer in 50%.
    The stored line understates what an opposing Mega does to us by a quarter; with the right
    line a Mega is estimated as well as any other Pokemon.
  - *Decisions* (the logged candidates through today's 14-guard profile, stored numbers
    against the right ones; the replay reproduces the logged guard stage in 99.2% of
    decisions, 97.5% where a guard fired): **36 picks change -- 1.4% of move decisions, 3.9%
    of the exposed ones, 0.09 per game** (the T6-family directories alone: 31 of 2,235, the
    same rates). By guard: guaranteed_ko 10, dominated_spread 9, dominated_attack 6,
    threat_first2 4, dominated_throat_chop 4, drop_free_finish 2, resisted_target 1,
    focus_boosted 1. The reranker and sticky corrections after the guards undo none of them
    and would change no further pick if they used the right numbers too. A lower bound: the
    audit keeps 8 of up to 64 candidates. Every change is listed in `report.txt` with the
    damage ranges on both lines, what was played and who fainted.
  - *Were the flipped verdicts right?* They are borderline cases and the games sit between
    the two answers. The stale foe our played pair attacked went down that turn in 31.5% of
    the 333 cases where both lines called the pair's minimum rolls a knock-out, **21.7% of the
    46 where only the stored line did**, 6.2% of the 385 where neither did. One of ours went
    down in 11.6% of 835 cases where neither line said a foe could knock it out, **20.0% of
    the 85 where only the right line said so**, 43.7% of 721 where both did (the knowledge
    block's "expected knock-out": 15.4%, **31.6% of 76**, 50.8%).
  - *The observation, if it were recomputed too* (it is NOT): the knowledge and threat
    blocks would move in 98% of exposed decisions (8.3 of ~74 floats on average; the largest
    move averages 0.49 of a health bar) and a 0/1 fact would flip in 43% of them. That is a
    third of all decisions with inputs no brain has seen. The token also carries the foe's
    base stats, which DO follow the Mega, so the brain may have learned to read the stale
    damage numbers beside them.
- **Built, default off: the guard-profile entry `forme_stats`** (`vgc_bench/src/forme_stats.py`,
  `guards.apply_guards`). With it on, the guard stack runs on the estimate for the forme each
  opposing Pokemon is in; the stored line (the same dict object) is back when `apply_guards`
  returns, so the observation and everything else read what they always read.
  - Only an opponent's line is replaced, and only when it is exactly `ensure_stats`' estimate
    for another forme of the same Pokemon (computed by `ensure_stats` itself on a copy, so
    the two cannot drift; with the sheet's nature when the stale line shows one -- see the
    amendment below). A Transform is left alone.
  - It is a name in `GUARDS` (a no-op placeholder), so `--guards-extra`, `--guard`,
    `run_guard_ab.py --guards`, `tools/ladder_trial.sh` and `DEPLOYED.json` take it without a
    launcher change. Strictly opt-in: "every guard" (`enabled=None`) does not turn it on; the
    search session made searched picks follow the same rule (7fe46162).
  - Never silent: every decision in which a line was replaced counts
    `forme_stats:corrected`, the stack also runs once on the stored numbers (on copies), and a
    different pick is noted as the stage `forme_stats` (the mirror's and the battery's
    "changed actions"); the decision audit records the species and the pick the stored
    numbers made. Cost: 3.5 ms -> 7.0 ms per exposed decision.
  - `vgc_knowledge.py` is byte-identical (it is a pin of the battery's reference arm), the
    default path is unchanged, **`DEPLOYED.json` is not touched and the deployed bot plays as
    before.** 25 tests (`unit_tests/test_forme_stats.py`): the defect, the recognition
    (Megas, Floette-Eternal, Aegislash both ways, a returning Mega, lines that are not ours),
    the swap and its restore after an error, the opt-in, two end-to-end positions (threat_first2
    answers an evolved Raichu; a knock-out Mega Gengar survives is no longer promoted), the
    observation blocks identical before and after, the audit record. The suite passes (1,267).
- **Pre-registered** (the opt-in guard rule of 09-28 / 10-03, `evaluation/guard_ladder_gate.py
  go <battery> <mirror> forme_stats`):
  1. **Mirror, tonight** (the window before the search session's V5 starts at 21:45):
     `mirror_guard_ab.py --guard forme_stats --games 2000 --port 7620` -- the deployed T6ep
     setup on both sides, side A with the entry. Not lost if the Wilson upper bound >= 50%;
     "wins close games" only if the lower bound > 50%. Also read: corrections and changed
     picks per game (the whole candidate list this time), by block. The tool stops on any
     guard error. Killed at 21:40 if unfinished (then it is not a result).
  2. **Held-out battery, not tonight** (V5 needs a quiet machine until about 07:45):
     `run_guard_ab.py --guards forme_stats --without-arm results_brain_ab_t6e_ep1` (6,204
     games; `--prepare-only` passes). Deploy-eligible if the pooled upper bound >= 0 and no
     population is below -3pp; guard errors within 1% of the games.
  3. GO for a ladder trial = both hold. Adding it to `guards_extra` is the user's decision.
  - **Power, stated before the run:** at about one changed pick in eleven ladder games neither
    run can see the effect of the change itself (the mirror's Megas are our own two, so its
    rate will differ); they can show it is not broken and does not lose. The case for the
    entry is the accuracy reading above, not a win rate.
- **Amended 21:20, before the rerun and the battery below exist** (two faults of mine, found
  while the first mirror ran; its hidden-sheet blocks had not finished):
  - *The entry did nothing with open sheets.* The first open-sheet block ended with **0
    changed picks in 500 games**. A 12-game probe per sheet mode: hidden 124 corrections and 2
    changed picks, open 0 and 0. With an open sheet the stored line carries the sheet's NATURE
    (poke-env writes the sheet's numbers, `ensure_stats` tops them up and keeps each stat's
    multiplier: the opposing Blastoise reads Special Attack 150, Speed 90), so it never was
    "exactly the neutral estimate" and was not recognised. Fixed: the recognition recovers the
    nature from the stale line and the replacement keeps it (Mega Blastoise 205 / 90; the real
    ones are 205 / 88); three more tests, one on the live line. The probe after the fix: open 103
    corrections and 3 changed picks, hidden 128 and 6. Hidden-sheet lines are untouched by this
    -- the ladder audit's summary is identical, and so are tonight's two hidden-sheet blocks --
    but tonight's two open-sheet blocks compared the bot with itself. **The gate's mirror is
    therefore a full rerun** (`results_mirror_forme_stats2`), same reading; tonight's run is
    reported as it is.
  - *The battery command above was wrong.* `run_guard_ab.py` plays a plain guard arm on the
    reference study's team (T6), and `results_brain_ab_t6e_ep1` was played on T6e: the
    comparison would have been the team's, and the tool would have stopped after the first
    population ("arms differ beyond output"). The with side now goes through the candidate
    path -- the deployed brain's own file, `--candidate-plans data/opening_plans_t6e.json`,
    port 7610 like the reference arm -- so the two arms differ in `forme_stats` alone
    (`--prepare-only`: same brain sha, plans, team, guards, seed, rosters). Both sides play
    the 12 guards that arm was played with (`wasted_fake_out` and `throat_chop_main_threat`
    came later and are off on both). Same rule.
  - Both run from **`tools/forme_stats_chain.sh`** (started tonight; it waits): once no
    search, mirror, battery or training job has run for ten minutes in a row it plays the
    mirror (about 30 min) and the battery (about 45 min) and writes the verdict to
    `results_analysis/forme_stats_20261004/chain.log`. No ladder, no promotion.
    `touch results_analysis/forme_stats_20261004/STOP` ends it while it waits.
- **Tonight's mirror, as it is** (`results_mirror_forme_stats`, 20:57-21:31, the code of
  21:00; a compact copy in `results_analysis/forme_stats_20261004/mirror_result_first.json`):
  **A 991 / 2,000 = 49.6% [47.4, 51.7] -> not lost** by the pre-registered reading (upper
  bound >= 50%); no detectable difference; no guard error.
  - Open sheets, where that code did nothing: 502 / 1,000 = 50.2% [47.1, 53.3], 0 changed
    picks -- the bot against itself, and a measure of the noise.
  - Hidden sheets, where it worked as the final code does: **489 / 1,000 = 48.9% [45.8,
    52.0]** (251 and 238 by block), **290 changed picks, 0.29 per game** -- three times the
    rate of the ladder replays, because every mirror game has an opposing Mega and the whole
    candidate list is in play.
  - Nothing here says the entry helps and nothing says it hurts: at 0.29 changed picks per
    game it would take several points of win probability per changed pick to show in 1,000
    games. The rerun (all four blocks on the final code) and the battery are the gate. The
    search session's V5 now ends about 08:30 on 10-05; the chain starts ten quiet minutes
    later and needs about 85 minutes.
- **Note, 22:20, before the rerun and the battery exist:** they will play the tree as it is
  then, 19445d0e or later. That commit (the search session's) fixes the observation caches --
  a second battle object in the same state read a zeroed threat / knowledge block: in a mirror
  on turn 1 with the same leads at full HP, and in the battery only when the opponent has our
  own two species out in our slot order at the same HP -- and makes the encoder read each
  property once (same bytes, `unit_tests/test_observation_identity.py`). Tonight's mirror and
  the battery's reference arm predate it. The search session's measurement of a zeroed turn-1
  block: the first slot's favourite unchanged in 40 of 40 positions, the value moves 0.03 --
  small beside what these runs can resolve, but it is a second difference between the
  battery's arms and belongs in the verdict's write-up. The chain logs the commit it plays
  (`TREE` in `chain.log`). Checked on that commit: the `forme_stats` tests pass and the audit
  gives the same numbers.
- **What this leaves open (the user's call):** the larger half is the brain's own view of an
  opposing Mega's damage, which this entry does not touch. Options: (a) guards only (this
  entry, after its gates); (b) the observation as well, which needs a fine-tune on the
  corrected inputs and its own ablation on the deployed brain first; (c) nothing. A cheap
  first reading for (b), not built: the deployed brain with the two blocks computed on the
  right lines and nothing retrained, mirror and battery against itself (an opt-in switch around
  the observation, about an hour of code, two hours of machine). If that does not lose, (b)
  needs no fine-tune; if it loses, the fine-tune is the price.
- **Second lead, checked: where a reranker runs without sticky corrections.** The mechanism
  is as reported (`opponent_reranker.rerank_candidates`: with any evidence it re-sorts the
  eligible pairs by log policy ratio plus utilities, so a pair a guard promoted starts behind
  the one it replaced). The deployed launchers pass sticky (`ladder_deployed.sh`,
  `challenges_deployed.sh`, `ladder_trial.sh`). Reranker on, sticky off:
  1. `evaluation/eval_counterfactual.py` `_player` -- both rerankers hard-coded on, no way to
     pass sticky: the gate battery (`run_gate_battery.py`), `run_rollout_gate.py`, the team
     tournament / grid / confirmation scripts, the league verdict scripts, the guard and
     mixing probes;
  2. `ladder_ourteam.py` run bare (rerankers default on, `--sticky-corrections` default off):
     `tools/ladder_read_loop.sh` called directly without `STICKY`, and `exhibition_mode.sh`
     with a `CHECKPOINT` / `TEAM` override (it takes sticky only for the deployed pair);
  3. `mirror_guard_ab.py --rerankers`: side B always, side A unless `--a-sticky` (no chain
     passes `--rerankers`).
  - The larger gap is the other way round: the held-out battery (`run_guard_ab.py` ->
    `opening_study`) and every chained mirror run with **no reranker stage at all**, so
    neither current gate plays the ladder bot's reranker + sticky stage. Guard promotions
    always stand there, as they do on ladder with sticky on; what the gates do not see is the
    reranker's own picks. Nothing changed; reported for the user.

## V4: the fixed search no longer loses (50.7% and 54.5%); V5 pre-registered for the night (2026-October 4, 20:45, before the run)

- **V4 results** (pre-registered 18:35; side A = deployed bot + nash search on the stack as
  of 18:31, side B = deployed bot):
  - **Anchor 0.2** (`results_mirror_search_nash4_pooled.json`): **304 / 600 = 50.7% [46.7,
    54.7]** -> no detectable difference. Open sheets 54.0 / 50.0%, hidden 47.3 / 51.3%. It
    left the bot's pick in 120 of 6,535 decisions (1.8%; guards sent back another 94).
  - **Anchor 0.07** (`..._nash4b_pooled.json`, two shards -- the third died at launch):
    **218 / 400 = 54.5% [49.6, 59.3]** -> no detectable difference by the pre-registered
    reading (the lower bound is 49.6). Open sheets 60.0 / 50.0%, hidden 55.0 / 53.0%. 244
    overrides in 4,348 decisions (5.6%); guards sent back 213 more (hidden sheets: 128 sent
    back against 86 accepted).
  - Against 38.8%, 39.1% and 34.9% for V1-V3: the twelve defects were the loss. Whether the
    search now *adds* anything is not shown: +4.5 points on 400 games is inside the noise.
  - What the 0.07 arm changed (`evaluation/search_override_report.py`): median payoff edge
    +0.22, median prior ratio 2.9, median turn 5; Rain Dance -> Helping Hand 39, Water
    Spout -> Water Pulse 37, Throat Chop -> Flare Blitz 22; 53 switches, 43 Protects.
  - Side A won 61.8% of its 217 games with no override and 43.1% of the 130 with one. That
    is selection, not effect: the search finds big edges when it is in trouble. Only the
    pooled rate is the experiment.
  - Caveats: hidden blocks ran while short probes and a calibration collection shared the
    machine; V4 lacks every fix listed in the 19:05 entry (pivot replacements failed in it:
    17 and 21 error fallbacks).
- **Leaf calibration fitted** (`results_leaf_calibration_T6ep/calibration.json`, sha256
  a29504a303800327...; 1,600 plain mirror games, 14,303 move decisions, side A won 50.2%):
  - Turns 1-3: AUC 0.55; Brier 0.320 raw -> 0.246 calibrated (held out; a coin is 0.250).
    Raw 0.3 to 0.8 means 46-60%.
  - Turns 4-6: AUC 0.81; Brier 0.199 -> 0.177. Turns 7+: AUC 0.96; 0.088 -> 0.078.
  - The raw value is about +0.45 too high (raw 0.48 is an even game) and steep at the top
    (late turns: raw 0.69 wins 74%, 0.79 wins 97%).
  - 16-game probe with it at anchor 0.07: 10 overrides in 161 decisions (6.2%), 11 sent back;
    expected edge +0.21, realized edge against the reply that came +0.22 (9 of the 10 had
    that reply in the table). Side A won 12 of 16 -- a probe, not a result.
- **Null check on the code V5 runs** (anchor 1e12, 8 likeliest replies): 0 of 147 decisions
  differ from the bot's own, no fallbacks; with oracle worlds 0 of 82. Real reply searched:
  96% open, 63% hidden; with oracle sets 97% / 91%.
- **PRE-REGISTERED V5** (`tools/search_mirror_rounds.sh`; 6 processes; start about 21:45
  when the other session's CPU training has stopped; about 10 hours). Side A = the deployed
  bot + nash search on the current stack (everything through commit a89f2e1a: champion
  anchor, live views, opponent-view snapshot, hidden worlds conditioned on what was shown,
  pivot fix, **8 likeliest replies**, critic leaf, 4 worlds, 8 s, argmax, anchor 0.07);
  side B = the deployed bot. Two arms, 3 shards x 200 games x 4 rounds = **2,400 games
  each**:
  - `NAME=search_nash5 ... EXTRA="--a-search-argmax --a-search-anchor 0.07
    --a-search-replies 8"`: the raw clipped leaf.
  - `NAME=search_nash5cal PORT=7613 ... EXTRA="<the same> --a-search-leaf-calibration
    results_leaf_calibration_T6ep/calibration.json"`: leaf = 2 x calibrated win
    probability - 1.
  - **Reading per arm:** pooled Wilson lower bound > 50% = the search wins; upper bound <
    50% = it loses; otherwise no detectable difference. Two arms are two chances at a false
    positive: an arm whose lower bound clears 50% by less than a point needs a replication.
  - Secondary, descriptive: open against hidden halves (hidden is ~92% of ladder, and our
    own team is far from the meta the set model describes); calibrated against raw;
    overrides, guard vetoes, expected against realized edges; reply coverage.
  - Stopping: all four rounds, unless the machine is needed in the morning
    (`touch results_mirror_<NAME>_STOP` ends an arm after its current round; whatever
    finished is pooled and reported as that).
  - A win is a candidate for a ladder trial, which is the user's decision. Search stays OFF
    in the deployed bot.
- **20:55, housekeeping with the two other sessions on this tree:**
  - V5 runs from a frozen worktree, `../vgc-bench-v5` at fe5957ca (venv, simulator and the
    six champion zips linked in), so edits to the shared tree cannot reach it mid-run. Its
    results are written there (`results_mirror_search_nash5*`).
  - `evaluation/run_guard_ab.py` `ALLOWED_CHANGED_PINS` gains the five search files changed
    today (exact_observation, exact_planner, live_exact, live_snapshot, set_particles): no
    battery arm searches; importing the harness and its arm runner loads none of the first
    four, and set_particles is imported for `team_roster` alone. The battery had been
    refusing to start on today's tree.
  - `live_exact._apply_live_hard_guards`: a profile switch (the stale-Mega session's
    `forme_stats`) is on for a searched pick only when the player's flags name it.

## Search, while V4 runs: the opponent model saw a stale board, hidden worlds ignored what was shown; reply-coverage and calibration instruments (2026-October 4, 19:05)

- **V4 is running** (launched 18:31, 5 shards alive). One of the six shards
  (`search_nash4b_s3`) died within a minute of launch with no traceback, so the 0.07 arm has
  400 games, not 600; its chain will fail to pool and the two finished shards are pooled by
  hand. It was not restarted: the code on disk has moved on and a late shard would not be
  the same experiment.
  - Early audit: anchor 0.2 leaves the bot's pick in 2.4% of decisions (guards veto another
    1.7%); anchor 0.07 in 7.1% (4.5% vetoed, mostly resisted_target, guaranteed_ko,
    dominated_attack, protect_spam). No fallbacks; search p50 1.6 s, p90 7.3 s.
- **New instrument -- was the opponent's real reply in the table?** (`reply_coverage` in the
  audit, `opponent_reply_was_searched` in `search_audit.py`). The search scores our moves
  against six predicted replies per world; if the reply that happens is not among them the
  payoffs describe a turn that was not played. The older `planned_*_coverage` fields divide
  by every retained world and read zero after any world refresh, so they could not say.
  - First reading (8 games, after the first fix below): the real reply was in some searched
    world in 60% of decisions (hidden sheets 46%).
- **Two more defects behind that, both fixed** (not in V4):
  - **The opponent model was shown a stale board.** Only our own view received the public
    snapshot. In a world recreated mid-battle the opponent's view had our Pokemon at full
    health, no Trick Room or Tailwind, turn zero, everyone on a first turn.
    `apply_public_snapshot(..., perspective="p2")` gives it the public facts (never our
    hidden item or ability; its own exact HP stays its request's).
  - **Hidden sheets: recreated worlds ignored everything already shown.** `_belief`
    conditioned only with open sheets, so a world rebuilt on turn 4 was drawn from the
    untouched prior (a Blastoise with Shell Smash and no Water Pulse, after it had used Water
    Pulse) and every such world was then marked impossible together. Now the belief is
    conditioned on shown moves, items and abilities, and when fewer worlds agree with the
    evidence than the search uses, the worlds are drawn again (once per new piece of
    evidence). 12-game probe: 7.2 of 8.0 hidden worlds agree with the evidence; the real
    reply was searched in 62% of hidden-sheet decisions (46% before), world mass 0.53 (0.29).
  - A Mega's ability is no longer read as evidence about its set.
- **Is the leaf any good?** Value of the played pair against who won, 377 decisions of
  mirror games on the fixed stack: AUC 0.50 on turns 1-2, 0.62 on 3-4, 0.85-0.89 from turn
  5; 0.95 in the last two turns of a game. Calibration is poor: raw 0.6-0.8 (as a win
  probability) won 40-52%, anything below 0.6 won 25-43%. The critic learned against
  opponents the brain usually beats.
  - Tool ready, not yet run: `evaluation/leaf_calibration.py collect|fit` (the bot plays
    itself without search, raw leaf value at every move decision against the result; one
    isotonic map per phase of the game) and `LeafCalibration` in `critic_leaf.py`
    (`--a-search-leaf-calibration`): leaf = 2 x calibrated win probability - 1.
- `evaluation/search_override_report.py`: what the search changed (kept / overridden /
  vetoed, payoff edges, kinds of change) and side A's results by overrides per game.
- **Power, stated plainly:** at anchor 0.2 the search changes about one move in four games.
  No 600-game mirror can see that; only an effect of 5 points or more is visible in an
  evening. A real answer for a small effect needs thousands of games on one configuration.
- **19:30, three more (none in V4):**
  - **A pivot's replacement always failed.** The snapshot did not say which of our slots
    must switch, so reconcile guessed "the fainted ones"; after Parting Shot or U-turn nobody
    has fainted and every world died with "Choices are done immediately after a request" (3
    of 12 probe games; since August). `public_snapshot` now passes our request's
    `forceSwitch`.
  - **The table spent its replies on coverage.** Nash inherited the risk search's rule of
    covering every move family first, so most of the six replies were pairings picked for
    coverage. Rows and replies are now the most probable as they come
    (`PlannerConfig.nash_likeliest`; `--a-search-table diverse` for the old rule;
    `--a-search-replies N`). Real reply in some searched world, 12-game probes: diverse 6
    replies 0.60-0.64; likeliest 6: 0.62 (open sheets 0.79); **likeliest 8: 0.82** (open
    0.86, hidden 0.75), search p50 3.0 s.
  - Hidden sheets, by whether the reply held a move never shown before: 30% searched if so,
    71-80% if not. Our own team is far from the meta the set particles describe (stored
    sets that hold all four of our Blastoise's moves: 0%; Farigiraf 0%; Charizard 0%;
    Venusaur 0%; Torkoal 9%; Incineroar 72%), so the mirror's hidden-sheet half is a harsh
    test of the set model, harsher than ladder opponents on common sets.
  - With live views a choice the live battle masks (a trapped switch, a disabled move) is
    skipped instead of costing the whole world.
  - Each override's expected payoff edge is now logged next to the edge in the cells of the
    reply that really came (`override_edges` in `search_audit.py`).
  - `tools/search_mirror_rounds.sh`: a long head-to-head as several short rounds, so a dead
    shard costs one round's share; pools what finished after every round.

## Opponent predictor side experiment started (separate session); the challenge listener's restart hazard fixed (2026-October 4, 18:15)

- The user: a predictor of what the opponent clicks each turn (Protect / attack which slot /
  switch / Mega), learned from thousands of human replays across the rating range, Elo as an
  input, open or closed sheets, to hand to the deployed bot. **Source of truth:
  `OPPONENT_PREDICTOR.md`** (design, label rules, splits, and four PRE-REGISTERED readings,
  committed in 147001a4 before any model was fitted). Nothing is wired into a decision path.
- **What reconnaissance found** (details and numbers in that file):
  - The shipped opponent move model is worse than a species -> move frequency table (NLL 3.34
    vs 1.89 on our own ladder opponents), and its saved weights are the last epoch, not the
    best (`best_state` is not cloned in the three prior trainers).
  - Elo bands add nothing at count-table level (-0.0002 +/- 0.0003 nats on held-out players);
    three state flags add +0.03 to +0.05. Elo is an input with a kill criterion (+0.02 nats).
  - The public replay feed holds 123,314 Reg M-C rows (76,870 bo1, 46,444 bo3) against 10,004
    battles on disk.
- **Running:** `datagen/oppmodel_scrape.py` (detached since 17:09; index done, fetch at ~0.86
  requests/s, single thread, about 36 h for everything, rating bands taken in turn). Log
  `battle_logs_feed_mc/scrape.log`; `touch battle_logs_feed_mc/STOP` ends it cleanly; it resumes.
- **In progress, uncommitted until its audit fixes land:** `vgc_bench/src/oppmodel/` (event
  reader + event-fed public state shared by training and runtime), then features, dataset,
  count-table and neural models, scorecard, runtime class.
- **Listener fix (ec1ca71c):** `--search-solution` / `--search-leaf` entered the run-config
  material with their defaults at 11:38, after the challenge listener started (11:22). Its
  directory records neither key, so its next reconnect restart would have stopped with
  "already records a different configuration" and ended the reconnect loop. They are now
  recorded only when not the default (`ladder_ourteam.record_run_config`; test in
  `unit_tests/test_run_config_hardening.py`).

## The search was playing a different game: twelve defects fixed, search rebuilt around the bot's own pick; V3 loses 34.9%; V4 pre-registered (2026-October 4, 18:35, before the run)

- **V3** (`results_mirror_search_nash3_pooled.json`; V2's settings + default spreads):
  **279 / 800 = 34.9% [31.7, 38.2]** -> loses. Open sheets 39.0 / 41.5%, hidden 30.0 / 29.0%.
  Not the pre-registered recovery. With what follows, V1-V3 do not measure the matrix search:
  side A was not playing the deployed bot's game plus search.
- **How it was found -- the null search.** With the anchor temperature at 1e12 the search can
  only return the policy's own favourite, so side A should play exactly what side B plays.
  - It did not: 11 of 52 decisions differed (21%), and side A won 2 of 12 smoke games.
  - New audit fields made that visible per decision: `champion_actions` (the pair the bot
    submits with search off, from the real unsearched path with its side effects held back;
    its own audit goes to `<log>_champion.jsonl`), `actions`, `champion` (kept / overridden /
    override_vetoed), `policy_choice`, `guard_scope`, `views`.
  - **After the fixes: 0 of 140 decisions differ** (12 games, `--a-search-anchor 1e12`), no
    error fallbacks (was ~3% of decisions). The machinery is now neutral.
- **What the search changed in V1/V2** (16,278 searched decisions): it left the policy's
  favourite in 43% / 31%; 86% / 92% of the changed slots swap one move for another. V2's top
  swaps: Ice Beam -> Water Spout 467, Ice Beam -> Water Pulse 347, Throat Chop -> Flare Blitz
  299. Part of that was the guards' work side A had lost (below), part was noise: battle
  4605249 turn 3, Ice Beam + Eruption prior 0.512 payoff +0.373 against Water Pulse +
  Eruption prior 0.117 payoff +0.377 -- the equilibrium played Water Pulse for an edge of 0.004.
- **The defects, by layer** (tests: `unit_tests/test_live_exact_seat.py` 24,
  `test_live_views.py` 13, `test_exact_planner_nash.py`, `test_matrix_game.py`):
  - *Decision.*
    1. **An equilibrium has no margin**: any payoff edge moves all its weight. Fix: the
       anchored game (`matrix_game.solve_anchored`, `PlannerConfig.nash_anchor`): both sides
       pay `temperature x KL(strategy || prior)` (piKL, solved by magnetic mirror descent).
       An action overtakes a preferred one only with an edge of `temperature x ln(prior
       ratio)`; the opponent is its prior leaning against us, not a perfect adversary.
    2. **Searched picks skipped the 14 opt-in guards** (`_apply_live_hard_guards` enabled
       HARD_GUARDS only): side A played ~95% of its turns without them.
    3. **The search ranked the raw policy's favourites, not the bot's pick.** The guards work
       on the whole candidate list; the search ranks a handful, often without the pair the
       guards would promote. Fix: the champion anchor -- the unsearched pair is always ranked
       (`include` in the planners) and under the anchor counts as `nash_champion_boost` (2) x
       the top prior, so leaving it takes an edge of 0.14 at temperature 0.2.
    4. **Re-running the guard stack on those few rows moved the bot's own pick.** Now: if the
       search agrees with that pair it stands; if it overrides and the guards object, the
       bot's pair comes back (not whatever the stack promotes among the few).
    5. A pair a guard adds itself raised KeyError and cost the whole search step.
  - *View -- what the policy and the critic were shown inside the search.*
    6. **The rebuilt view never matched the live one: 0 of 77 decisions, a median 642 of
       12,276 observation cells off.** It held four of our six Pokemon in another order and
       forgot which opponents had appeared; in a world recreated mid-battle (after almost
       every newly revealed reserve) our Mega was shown un-evolved, every Pokemon was on its
       "first turn", and with hidden sheets a just-switched-in opponent was missing; with
       open sheets a spent item was shown as held. Fix: **live-anchored views**
       (`exact_observation.LiveAnchor` / `live_view`): our side's root view is the live
       battle itself; a child is a copy of it that has watched the world's new protocol
       (seats and opponent nicknames translated). The policy prior at the root is now the
       live policy's, exactly.
  - *World -- the simulated physics.*
    7. **Hidden sheets: every active opponent lost its item.** poke-env's placeholder
       "unknown_item" went into the snapshot as an item, and reconcile wrote it over the
       sampled one: no Focus Sash, berry, Choice item or Life Orb in any hidden-sheet world.
    8. **A Mega the world never played was a transform**: right stats on the field, but
       "Blastoise, L50" in every request and reverted on its first switch. Reconcile now does
       a real forme change (`applyForme` in `tools/exact_showdown_bridge.js`).
    9. **One Pokemon in both active slots** when it had stood in the world's other slot
       (our last Pokemon in slot b): every choice gave it two moves and none could be encoded.
       This was 290 of V2's 296 error fallbacks -- the endgames.
    10. **The seat was hard-coded.** In seat 2 (every "B challenges" block, about half of
        ladder games) each side's last move, item changes and spent Mega were read from the
        OTHER side's events, the opponent's observed action was our own, and set evidence
        came from our own Pokemon. `live_snapshot.live_roles`.
    11. **Seat 2's targets were mirrored** (`_target_location`); the simulator numbers the
        opposing side the same for both seats (now a test against the simulator itself).
    12. Zero stat points in every world (the previous entry).
  - 7-11 also applied to the August-September "risk" search runs, including on ladder.
  - Parity report regenerated on the patched bridge: 1,000 / 1,000 states, 0 mismatches.
- `mirror_guard_ab.py`: `--a-search-anchor`, `--a-search-guards player|hard`,
  `--a-search-champion on|off`, `--a-search-views live|rebuilt` (the old behaviours stay
  reachable); each battle's tag, result and length in `result.json`.
- **PRE-REGISTERED V4** (6 shards share the machine; V1-V3 had 4). Side A = the deployed bot
  + nash search on the fixed stack (critic leaf, 4 worlds, 8 s, argmax, champion anchor,
  player guards, live views); side B = the deployed bot. Two anchor temperatures:
  - `NAME=search_nash4 SHARDS=3 EXTRA="--a-search-argmax --a-search-anchor 0.2"`: leaving the
    bot's own pick takes a payoff edge of 0.14 (for the policy's favourite) to 0.6.
  - `NAME=search_nash4b SHARDS=3 PORT=7613 EXTRA="--a-search-argmax --a-search-anchor 0.07"`:
    edges of 0.05 to 0.2 -- more overrides.
  - 600 games each (3 x 200), same seeds. Reading per arm: pooled lower bound > 50% = the
    search wins; upper bound < 50% = it loses; in between = no detectable difference. Two
    arms are two chances at a false positive: a lower bound barely above 50% in one arm needs
    a replication before any claim, and no ladder trial without the user's word.
  - Reported with it: how often the search left the bot's pick (`champion_outcomes`), the
    payoff edges of those overrides, overrides against per-battle results, searched share,
    latency.
  - Not in this round: the oracle arm (real opponent sets in every world) -- next, once
    there is an effect to explain.

## Found: the search's hidden worlds built every opponent with ZERO stat points; fixed; V3 pre-registered (2026-October 4, 16:08, before the run; launched 16:10)

- **The conservative variant also loses**: `results_mirror_search_nash2` (argmax of 0.5 x
  equilibrium + 0.5 x policy prior) **313 / 800 = 39.1% [35.8, 42.6]**.
  - Open sheets 49.0 / 48.5% (the first version: 44.5 / 44.0).
  - Hidden sheets **30.0 / 29.0%** (35.5 / 31.0).
  - 94.9% of 8,615 decisions searched; search overrode the brain in 2,504 of 8,179 (31%).
  - Blending fixes open-sheet play (parity) and does nothing for hidden sheets.
- **Cause** (`vgc_bench/src/set_particles.py`): the Reg M-C set data has moves, items and
  abilities but no spreads.
  - Of the 9-12 particles of each of our six species, 0 carry a spread.
  - `determination_team_text` then wrote "Serious Nature" and no EVs: every opponent in every
    world had zero stat points. Our real Torkoal: 32 HP / 32 SpA Quiet.
  - The search planned against frailer, weaker, differently ordered opponents. With hidden
    sheets it also guessed their moves and items.
  - Open sheets do not reveal spreads either, so open-sheet worlds had the same zero investment
    with the right moves.
  - The live damage features were never affected: `vgc_knowledge.ensure_stats` already assumes
    max HP + max attacking stat.
- **Fix:** `set_particles.default_spread` -- max HP, max in the attacking stat (by the set's
  damaging moves, else the higher base stat), 2 Speed, neutral nature -- whenever a particle has
  no spread. It also reaches the exact preview planner and the counterfactual generator, which
  build worlds the same way. Test in `unit_tests/test_exact_planner_nash.py`.
- **PRE-REGISTERED V3:** `NAME=search_nash3 EXTRA="--a-search-argmax --a-search-prior-mix 0.5"
  tools/search_mirror_chain.sh`: V2's settings, so the spread fix is the only change. 800 games,
  same seeds, same reading (lower bound > 50% wins; upper < 50% loses). Expected if the cause is
  right: hidden-sheet blocks recover toward the open-sheet level; open-sheet blocks may rise
  above parity.
- Prepared but not run: `datagen/generate_outcome_dataset.py --format/--opponent-glob/
  --train-split-only/--moveset-prior` and `mirror_guard_ab.py --a-search-outcome`, for
  retraining the outcome-net leaf on T6ep games if the leaf is still the limit after this.

## The matrix search LOSES as built: 38.8% [35.4, 42.2] over 800 games vs the deployed bot (hidden sheets 31-35%); a conservative variant is next (2026-October 4, 14:00)

- **Result:** `results_mirror_search_nash1_s1..4` (pooled `results_mirror_search_nash1_pooled.json`):
  side A (T6ep + nash search, critic leaf, 4 worlds, 8 s, sampled) **310 / 800 = 38.8% [35.4,
  42.2]** -> loses, by the pre-registered reading.
  - Open sheets 44.5 / 44.0%; **hidden sheets 35.5 / 31.0%**.
- **Health was fine:** 8,544 side-A move decisions, 94.8% searched; 225 coverage fallbacks, 220
  error fallbacks (2.6%, mostly hidden-world budget exhaustion); search time p50 3.1 s, p90 7.2 s,
  max 8.1 s. The search overrode the brain's favourite in **3,521 of 8,099 (43%)** decisions.
- **Reading:** the mechanics work, but a one-turn table scored by our critic is not a better judge
  than the brain plus guards, and the hidden-set worlds make it worse.
  - mikumiku37's search added ~110 Elo on top of a value net trained on 330M games. Ours is the
    PPO critic of far fewer games -- the open question flagged in `RESEARCH_TOP_BOTS.md`.
  - Other suspects: sampling a mixed strategy against a deterministic opponent, and reply
    candidates from the human move model rather than our brain.
- The chain process exited after its shards without logging; its script had been edited and
  restored mid-run, a mistake now noted. The shards were complete and were pooled by hand.

## PRE-REGISTERED: the conservative matrix-search variant (2026-October 4, 14:00, before the run)

- `NAME=search_nash2 EXTRA="--a-search-argmax --a-search-prior-mix 0.5" tools/search_mirror_chain.sh`:
  identical to the first test except that side A plays the **argmax of 0.5 x equilibrium +
  0.5 x policy prior**, with no sampling. It overrides the brain only where the search
  disagrees strongly. 800 games (4 x 200), same seeds, same reading.
- What it separates: if it reaches ~50% or better, the loss came from sampling and from
  overriding too often; if it still loses clearly, the table itself (leaf value / worlds /
  reply model) is wrong, and the next step is a better leaf value, not a different solution
  concept.

## PRE-REGISTERED: the matrix-search head-to-head (2026-October 4, 11:40, before the run; launched 11:43)

- **Arms** (`tools/search_mirror_chain.sh`):
  - Side A: the deployed T6ep + 14 guards + **nash exact search**: critic leaf, 4 hidden worlds,
    8 s budget, every move turn, the equilibrium sampled.
  - Side B: the deployed T6ep + 14 guards, no search.
  - Both on T6e with the deployed preview, over the mirror's four blocks (open / hidden sheets x
    who challenges).
- **Scale:** 4 processes x 200 games (seeds 20924 + 1000k, one game at a time per process, one
  local server), pooled by `evaluation/pool_mirrors.py` -> 800 games.
- **Reading:**
  - Pooled Wilson 95% lower bound > 50% -> the search wins close games; a ladder trial is the
    user's call.
  - Upper bound < 50% -> it loses; next variants: argmax instead of sampling, or mixing the
    policy prior in.
  - Otherwise inconclusive.
  - **Health**, reported alongside: share of side A's move decisions really searched (expect
    >= 90%), error fallbacks, and how often search overrode the brain. Latency under 4 parallel
    processes is not ladder-valid.
- **Smoke games, not part of the test:** outcome-net leaf 0/4; critic leaf 4/8 (84 of 85
  decisions searched, p50 2.3 s, overrode the brain 35 of 84).

## Matrix search built: nash solution mode in the exact planner, critic leaf, side-A search in the head-to-head tool; first smoke games (2026-October 4, 11:30)

- The user: "start the matrix search". It runs mikumiku37's turn as a matrix game on our exact
  Showdown bridge (`RESEARCH_TOP_BOTS.md`).
- **`PlannerConfig(solution="nash")`** (`vgc_bench/src/exact_planner.py`): per hidden-information
  world, one payoff table of our candidates x the opponent's `opponent_width` likely replies.
  - Our candidates are those with at least 10% of the top candidate's prior, the live low-prior
    rule's own ratio, so a sampled pick is never vetoed for that.
  - The table is built through `_score_actions`: exact one-turn children, common random numbers,
    the KO penalty, and forced replacements resolved.
  - It is solved by regret matching+ (`vgc_bench/src/matrix_game.py`). The strategies are averaged
    over worlds by probability (`_aggregate_nash`) and sampled; every tabled action counts as
    searched (depth one). Risk mode is unchanged.
  - Tests: `unit_tests/test_exact_planner_nash.py` (matching pennies, dominance, prior cut,
    two-world averaging, sampling frequencies).
- **Leaf value `critic`** (`vgc_bench/src/critic_leaf.py`): the deployed brain's PPO critic plus
  the shaping potential it was trained with (0.10 faints, 0.05 HP; critic = E[result] - Phi), in
  place of the August outcome net calibrated on the Reg M-B champion. Selected by
  `LiveExactSession(leaf=...)` / `PolicyPlayer(exact_leaf=...)`; tests
  `unit_tests/test_critic_leaf.py`.
- **Plumbing:**
  - `ladder_ourteam.py --search-solution risk|nash --search-leaf outcome|critic`; the defaults
    keep the old search.
  - `evaluation/mirror_guard_ab.py --a-search nash --a-search-leaf critic --a-search-worlds N
    --a-search-budget S --concurrency C`: side A searches every move turn and is audited to
    `a_decisions.jsonl`. Search failures are counted, not fatal; they fall back to the champion
    plus guards.
  - `evaluation/search_audit.py` summarises those audits.
- **Smoke, outcome-net leaf** (4 games, 4 worlds, 8 s, one game at a time): the mechanics work.
  - 40 of 42 move decisions searched; 2 fell back (`DeterminizationBudgetExhausted`).
  - Latency: median 2.6 s, p90 7.2 s, max 7.5 s.
  - Play: the search overrode T6ep's favourite in **18 of 40** decisions, and side A lost all
    four games -> the outcome-net leaf is suspect. The critic-leaf smoke follows.

## PROMOTED at the user's word: T6ep (the Earth Power brain) is the deployed brain; matrix search started (2026-October 4, 11:25)

- The user: "make the earth power brain official and start the matrix search".
- `results_deployed/champion_mc_T6ep.zip` (sha f92248c6, a new file copied from
  `results_tactical_t6e_ep1/sft/tactical_e4.zip`; sidecar role production). `DEPLOYED.json`:
  - **Unchanged:** team T6e, the 14 guards, sticky corrections, learned preview, sheet preview.
  - **New:** replay tag **T6ep_guards14**, evidence (52.8% head-to-head, battery -0.66pp, gate GO,
    ladder 6-9).
  - **Archived:** T6tac moves to `previous_deployed`, with its three amendments and ladder
    record, and to `history`.
- T6tac (`champion_mc_T6tac.zip`) joins the immutable prior champions. The challenge listener was
  restarted on T6ep.
- Matrix search: the plan is next in this log.

## priority_block's swallowed errors: poke-env's "recharge" move has no priority -> fixed (2026-October 4, 08:10)

- `results_analysis/guard_errors_20261004/priority_block_repro.py`: 1,000 practice-style games
  (5,568 decisions) with the guard wrapped to log its exceptions. One error, logged in
  `priority_block_errors.txt`: an opponent bot's Sylveon recharging after Hyper Beam. Its only
  action decodes to poke-env's "recharge" pseudo-move, whose entry has no `priority`, so
  `_effective_priority` raised `KeyError: 'priority'`.
- **Our team has no recharge move**, so the deployed bot never hit this. The 7 errors in last
  night's practice data were opponent-side, and harmless there too: a recharging slot has one
  legal action.
- Fix: `guards._move_priority` (0 when the entry carries none), used by `_effective_priority`
  and `_plain_attack`; test in `unit_tests/test_guard_repairs.py`.

## KL-anchored practice keeps the lessons (drift 0.10 vs 0.52; 51.7% vs 35.2% for plain practice) but adds nothing on top of its start -> HOLD; the Earth Power candidate stays the night's best brain (2026-October 4, 08:00)

- `training/t6e_anchor_chain.sh` (pre-registered 03:40), from the Earth Power candidate on T6e:
  the T6ctx recipe +983,040 steps with beta * KL(start || policy), target 0.10 (04:48-06:46,
  116 min).
  - **The anchor held:** KL 0.12-0.15 per update with beta only 0.05-0.11. Final drift on the
    T6e positions **0.098 / 0.068** (slot 1 / slot 2), top slot-1 action unchanged 80.0%.
    Plain practice from T6tac had drifted **0.52 / 0.37** (61.7% unchanged).
- **Head-to-head** vs the deployed bot (`results_mirror_t6e_anchor1`): **51.7% [49.6, 53.9]**.
  - Open sheets 55.6 / 55.8%, hidden sheets **47.4 / 48.2%**.
  - Its start, the Earth Power candidate, scored 52.8% (open 51.1, hidden 54.4).
- **Battery** (`results_brain_ab_t6e_anchor1`): pooled **-0.34pp [-1.24, +0.58]**; populations
  human_new +0.1, frozen -1.16, rotation1 -0.19, rotation2 +0.1, human_previous -1.35,
  heuristic +0.48.
- **Gate (`training/t6tac_practice_gate.py go`): HOLD** -- head-to-head not won.
- **Reading:**
  - The anchor removes the washout. Plain practice from a fine-tuned brain fell to 35.2%;
    anchored practice stays at its start's level.
  - The practice itself adds no measurable strength from here. It moved strength from
    hidden-sheet to open-sheet games, which is the wrong direction for a ladder with no open
    sheets. Past head-to-heads show the same lean in some re-fits (tactical2, tactical4), while
    the practice that won (T6hp -> T6ctx: 61.2 / 57.4) improved both, so the open-sheet share
    of practice games (50%) is not clearly the cause.
- **Recommendation:** the Earth Power candidate (`results_tactical_t6e_ep1/sft/tactical_e4.zip`,
  GO) is the brain to deploy if the user wants one from tonight. Further practice needs a
  different lever than more of the same games, e.g. the matrix search (`RESEARCH_TOP_BOTS.md`)
  or a lead-selection fix.

## Ladder read of the Earth Power candidate: 6-9 over 15 (the user's word); Earth Power right 3 of 4 times it was the best attack; the losses are the old ones -- losing a Pokemon first (8 of 9 losses) and the Venusaur + Torkoal lead (0-3) (2026-October 4, 05:00)

- `ladder_replays_mc_t6e_ep1` (04:06-04:47; table: `results_analysis/ep_ladder_20261004/ladder_read.txt`):
  **6-9**. Five of the six wins were opponent forfeits on turns 4-7, not turn-1 quits, so all count.
  Rating 1269 -> about 1220. Opponents 1129-1326; three losses to players rated 1134-1143. No parse
  errors, no timeouts. The challenge listener was paused for it and restarted at 04:50.
- **Earth Power** (`ep_audit.py`): 45 Torkoal turns had Earth Power as a valued attack.
  - It was the teacher's best by more than 0.05 three times; the bot clicked it twice. The miss:
    game ...586104 turn 6, Eruption when Earth Power was ahead by 0.09.
  - It was best by 0.05 or less once; clicked.
  - It was worse 41 times; 0 of those 23 attacks were Earth Power.
  - Compare the pre-lesson brain's 31% attack share when Earth Power was best. The lesson shows
    on ladder, but the moment is rare: about one turn in four games.
  - It played out in two losses: Earth Power into Chandelure (game 8) and into Incineroar (game 11).
- **Where the games went:**
  - Losing a Pokemon first: 9 games, 1 win (an opponent forfeit), against 5-1 otherwise.
  - Leads: **Venusaur + Torkoal 0-3** (games 3, 9 and 11: Torkoal switches out on turn 1,
    Venusaur is double-targeted, or Farigiraf arrives and falls before setting Trick Room);
    Farigiraf + Incineroar 3-3; Blastoise + Farigiraf 2-2; Charizard + Venusaur 1-1.
  - Specific losses:
    - **Game 6:** a Trick Room war (their Indeedee's Trick Room cancelled ours), plus
      Expanding Force Mega Alakazam.
    - **Mega Raichu-Y Zap Cannons:** games 3 and 4.
    - **Mega Garchomp's Earth Power spam:** games 9 and 11.
    - **Hisuian Arcanine's Head Smash:** game 15.
- **The lead pattern holds over the whole T6 era** (`tools/ladder_loss_profile.py` on all 16
  T6-family ladder dirs: 356 games, 49.4%):

  | Our lead | Record |
  |---|---|
  | Charizard + Venusaur | 7/21 (33%) |
  | Torkoal + Venusaur | 9/24 (38%) |
  | Farigiraf + Torkoal | 21/50 (42%) |
  | Blastoise + Farigiraf | 105/202 (52%) |
  | Farigiraf + Incineroar | 17/30 (57%) |

  The two pure-sun leads trail by about 2 SE: a candidate for a preview-level test. The
  brought-four splits are biased by forfeits, since early wins reveal fewer of our Pokemon.
- **Reading:** 6/15 is consistent with anything from about 16% to 68% (95%), so this read cannot
  overrule the 2,000-game head-to-head (52.8% vs the deployed bot). It confirms that the
  structural problems are still the early-faint and lead-selection ones from the 10-03 studies.
  The Earth Power lesson works where it applies. Deployment of the candidate: the user's call.

## The Earth Power lesson PASSES its gate (GO): head-to-head 52.8% [50.6, 54.9] vs the deployed bot, battery -0.66pp [-1.48, +0.15]; the lesson moved but short of its bar (2026-October 4, 04:05)

- `training/t6e_ep_chain.sh` (pre-registered 02:25): 4,000 practice games on T6e (02:25-02:53;
  22,116 decisions, **6,671 Earth Power positions**, 0 teacher errors) -> the focus fine-tune of
  T6tac (`results_tactical_t6e_ep1/sft/tactical_e4.zip`, sha f92248c6, epoch 4 = lowest validation
  cross-entropy; drift on untaught positions 0.008).
- **Lesson** (728 validation Earth Power positions): agreement with the teacher's best attack
  82.4% -> 85.9% (bar 91.2%: **not met**). When Earth Power WAS the best attack the brain gave it
  **30.9% -> 49.3%** of its attack mass (the user's point: it under-used Earth Power); when it was
  worse, 6.0% -> 10.8%.
- **Head-to-head** vs the deployed bot (both T6e + 14 guards, `results_mirror_t6e_ep1`): **52.8%
  [50.6, 54.9]** over 2,000 games -- open sheets 52.6 / 49.6%, hidden sheets **56.4 / 52.4%**
  (where the unpractised T6e had lost 36-39% to T6).
- **Battery** vs the deployed brain on T6e (`results_brain_ab_t6e_ep1`): pooled **-0.66pp [-1.48,
  +0.15]**, worst population -1.5pp -> deploy-eligible.
- **Gate (`training/t6e_ep_gate.py`): GO** (better = head-to-head won, safe = battery). Deployment:
  the user's call. Ladder read of 15 games started 04:06 at the user's word
  (`ladder_replays_mc_t6e_ep1`; the challenge listener is paused for it).
- Side note: the always-on priority_block guard raised 7 times in the 22,116 practice decisions
  (counted, not logged); the guard sits out those turns. To investigate when the machine is free.

## PRE-REGISTERED: the 15-game ladder read of the Earth Power candidate, then anchored practice on T6e (2026-October 4, 03:40, before either runs)

- The user (02:56): "no matter better or not run 15 games on ladder and analyze them, do more
  training afterwards too and also take a look into the vgc other bot thats been going on in a
  seperate context and see if u can learn anything from it, use your best judgement through the
  night".
- **Ladder read** (after `training/t6e_ep_chain.sh` ends; the challenge listener pauses -- same
  account, ladder is serial): `TRIAL_CHECKPOINT=results_tactical_t6e_ep1/sft/tactical_e4.zip
  tools/ladder_trial.sh 15 ladder_replays_mc_t6e_ep1` -- the deployed configuration (T6e, 14
  guards, sticky corrections, learned preview, sheet preview) with the candidate brain. No gate
  (the user's word: whatever its local result). Analysis: the record, with free wins (turn-1
  quits) apart; every Earth Power decision (how often Earth Power was the teacher's best attack,
  and how often the bot clicked it then and when it was worse); guard firings; each loss's
  turning point (who fainted first, to what).
- **Anchored practice** (`training/t6e_anchor_chain.sh`, launched once the ladder read ends): the
  T6ctx practice recipe (+983,040 steps) on T6e with ONE change -- beta * KL(start || policy) in
  the PPO loss, beta adapted toward 0.10 nats (slot 1 + slot 2; `vgc_bench/src/anchored_ppo.py`,
  `training/run_t6e_anchor_trial.py`). Why: plain practice from T6tac moved the brain 0.52 nats
  from its start (slot 1, measured on T6 positions) -- as far as the practice that improved T6hp
  (0.48) -- over the tactical fine-tune's 0.24-nat lessons, and lost 35.2%. START = the Earth
  Power candidate if its head-to-head was not lost (upper bound >= 50%) and its battery is
  deploy-eligible, else T6tac. Gate (`training/t6tac_practice_gate.py go`): head-to-head vs the
  deployed bot won (Wilson lower bound > 50%) AND the battery vs
  `results_brain_ab_t6e_unpractised` deploy-eligible -> GO, else HOLD. Also reported: the
  candidate's drift KL(start || candidate) on the T6e positions (the anchor should hold it near
  0.1). Deployment: the user's call.

## PRE-REGISTERED: the Earth Power lesson on T6e (2026-October 4, 02:25, before any run)

- The user: "start the earth power practice training but like its pretty obvious just use ep
  when its super effective on a pokemon and it does better damage than the rest of the moves and
  theres no better switch in".
- **Recipe** (`training/t6e_ep_chain.sh`): 4,000 practice games by the deployed bot on T6e
  against the training-role opponents (`training/gen_tactical_data.py --focus-moves earthpower`:
  train-split rosters, 8 cells x 500; it records which legal actions are Earth Power) -> fine-tune
  of the deployed brain T6tac (`training/tactical_sft.py --lessons focus`, lr 1e-4, 4 epochs, the
  epoch with the lowest validation cross-entropy). The lesson acts ONLY where our Pokemon can
  click Earth Power as a valued attack: the calculator teacher re-spreads that Pokemon's attack
  mass by damage value (tau 0.1), so the hardest-hitting attack gets most of it -- Earth Power
  when super effective and stronger than the rest (e.g. rain vs Incineroar: Earth Power 0.51,
  Eruption 0.34; sun at full HP: Eruption 1.04 stays first). Attack vs Protect vs switch keeps
  the brain's own proportions ("no better switch in" stays its call), and every other position
  targets the brain's own distribution, so nothing else is retrained. Plain RL practice is not
  run: from T6tac it lost 35.2% / 37.2% (2026-10-03). Smoke run (24 games): 36 of 130 decisions
  were Earth Power positions; the chain stops if the 4,000 games give fewer than 300.
- **Readings** (`training/t6e_ep_gate.py`): *lesson* -- on the validation Earth Power positions
  (>= 100) the brain's disagreements with the teacher's best attack at least halve; *better* --
  head-to-head vs the deployed bot (both T6e + the 14 guards, `results_mirror_t6e_ep1`, 2,000
  games) won, Wilson lower bound > 50%; *safe* -- held-out battery vs the deployed brain on T6e
  (`results_brain_ab_t6e_ep1` vs `results_brain_ab_t6e_unpractised`, both with the 12 guards
  that arm used) deploy-eligible (pooled upper >= 0, no population < -3pp). **GO** = better and
  safe; **NEUTRAL** = lesson, safe and the head-to-head not lost (upper >= 50%); **HOLD**
  otherwise. Deployment and any ladder trial: the user's call either way.
- Also: the ladder / challenge / exhibition launchers now count `gen_tactical_data.py` and
  `tactical_sft.py` as heavy jobs (a ladder session cannot start during the practice games).

## Deployed at the user's word: T6e (Torkoal Earth Power) + 14 guards + the open-sheet preview; the challenge listener now reconnects after a lost connection (2026-October 3, 23:45)

- The user: "keep the guards, keep the team sheet stuff its gonna be useful later, and keep
  earth power". `results_deployed/DEPLOYED.json` amended (third amendment, with the previous
  values and the evidence): team **`teams/candidates_mc/T6e.txt`** (sha 5d85cac3; T6 with
  Torkoal's Earth Power for Weather Ball -- same six, same preview model), `guards_extra` +=
  **wasted_fake_out, throat_chop_main_threat** (14 guards), **`sheet_preview: true`**, replay tag
  T6tac_guards12 -> **T6tac_T6e_guards14**. The brain stays T6tac (sha 05e82218); both HP-move
  guards stay off. `tools/deployed_config.py` verifies it; launchers need no change.
- What each piece rests on -- each measured alone against the 12-guard bot, the combination is
  unmeasured: the two guards, rare trio battery -0.55pp [-1.14, +0.06] (with the HP-move guard,
  which drew nearly all the changes), 7 of 7 ladder firings correct; the sheet preview, battery
  -0.52pp [-1.32, +0.27], non-regression only (acts in challenges, ladder opponents never share
  sheets); T6e, battery +0.50pp [-0.52, +1.52] but head-to-head vs the same brain on T6 43.4%
  (hidden-sheet blocks 36-39%) -- HOLD by its pre-registered rule, deployed over it by the user's
  decision. **What to watch on ladder:** the hidden-sheet games against teams like ours.
- **Challenge listener fix.** The listener started at 15:20 went deaf at 17:52: Showdown dropped
  the websocket ("no close frame received or sent"); poke-env logs that and stops reading, but
  `accept_challenges` keeps waiting, so the process looked alive for six hours.
  `ladder_ourteam.play_until_disconnect` now watches poke-env's listener and on a lost
  connection ends the run with **exit 75** and a line in `<replay_dir>/disconnects.jsonl`
  naming any unfinished battle; **`tools/reconnect_loop.sh`** (which `tools/challenges_deployed.sh`
  now execs) reruns the listener after 30 s -- after 5 s with `--rejoin-battle` when a battle was
  open; any other exit ends it, ten short runs in a row give up. Ladder mode exits the same way
  (`tools/ladder_read_loop.sh` already restarted dead sockets by log pattern and handles it as
  a session end). Tests `unit_tests/test_reconnect_loop.py` (11: the helper with fakes; the loop
  with a fake listener -- rejoin, first room once, other exits, give-up, Reg M-C rooms only).
  Live `checks/disconnect_live.py` (real poke-env players, a throwaway server on 7612 killed
  with SIGKILL): **returned in 0.0 s while waiting and 0.3 s mid-battle (with the room)**, while
  the old bare `accept_challenges` was still waiting 3 s after the kill -> PASS. Full suite 710
  passed, 5 skipped; ruff and pyright clean on the changed files.
- Listener restarted at 23:30 on the new configuration and at 23:41 on the reconnecting launcher
  (`challenge_replays_mc_deployed_T6tac_T6e_guards14`, log
  `results_analysis/challenges_20261003/listener_T6tac_T6e_guards14.log`). Stop it by stopping its
  `ladder_ourteam.py` process (the loop ends on any exit but 75).

## The rare-guard trio PASSES its rule (battery -0.55pp [-1.14, +0.06], mirror 48.6% [46.5, 50.8]); but the HP-move guard still leans negative where it acts -> recommend the two rare guards only (2026-October 3, 22:25)

- `results_guard_ab_rare_trio` / `results_mirror_rare_trio` (wasted_fake_out +
  throat_chop_main_threat + hp_move_after_spread vs the deployed bot): battery **pooled -0.55pp
  [-1.14, +0.06]**, worst population -1.1pp; mirror **48.6% [46.5, 50.8]** -> **RARE_TRIO_PASS**
  (`review1003_gate.py guards`).
- **Where each fired** (battery cells): hp_move_after_spread 178 firings in 51 cells **-3.21pp
  [-7.49, +0.71]**; throat_chop_main_threat 74 in 16 cells +0.57pp [-9.09, +9.09];
  wasted_fake_out 3 in 2 cells (one or two games); cells where none fired -0.16pp. In the
  mirror all 1,572 changes were hp_move_after_spread (our own six show many spread attacks
  from faster Pokemon), and side A won 48.6%.
- **Reading:** both HP-move guards lean negative wherever they act (broad: -1.57pp cells,
  mirror 46.2%; narrow: -3.2pp cells, mirror 48.6%), though their swaps look right by the
  calculator 70-75% of the time -- the shown spread attack often does not come again, and Heat
  Wave misses 10%. The user's turn-7 read was right for that game; as a standing rule it does
  not pay. **Recommendation:** add wasted_fake_out and throat_chop_main_threat (rare, 7 of 7
  correct on the ladder replay, no harm measured); keep both HP-move guards off. The user's call.

## The open-sheet preview passes its non-regression rule (pooled -0.52pp [-1.32, +0.27]); it changes the plan in ~4% of open-sheet games and its effect there is unmeasured; the rare-guard trio test started (2026-October 3, 21:11)

- `results_guard_ab_sheet_preview` (the deployed bot + `sheet_preview` vs the deployed bot):
  **pooled -0.52pp [-1.32, +0.27]**, worst population -1.4pp, no preview errors ->
  **SHEET_PASS** (`review1003_gate.py sheet`, rule fixed before the run).
- It ran in 506 of 517 open-sheet games per population and changed the plan in 22 (~4%):
  the same two held-out rosters (MC2566, MC53), 11 games each, in every population. Those 12
  cells: -3.03pp [-19.70, +12.88] (e.g. MC2566 91% -> 18% vs human_previous but 55% -> 82% vs
  frozen) -- unmeasured; open-sheet cells it never changed -0.40pp, hidden-sheet cells (it
  cannot act there: an A/A check) -0.52pp. The late-sheet re-plan was checked live
  (`checks/late_sheet_live.py`). Open sheets only happen in challenges (ladder opponents never
  accept), so turning it on is low-stakes either way; the user's call.
- **Started now:** the rare-guard trio (wasted_fake_out, throat_chop_main_threat,
  hp_move_after_spread, registered now) by the rules pre-registered at 19:16:
  `tools/rare_trio_chain.sh` (mirror + battery, gate `review1003_gate.py guards`).

## T6e (Torkoal Earth Power for Weather Ball) on the unpractised deployed brain: battery fine (+0.50pp), head-to-head vs T6 lost 43.4% -> HOLD by the pre-registered rule (2026-October 3, 20:25)

- `results_mirror_t6e_unpractised` (the deployed brain on T6e vs the same brain on T6, both with
  the 12 guards): **43.4% [41.2, 45.5]** -- open-sheet blocks 49.8% / 48.4%, hidden-sheet blocks
  **36.4% / 38.8%**.
- `results_brain_ab_t6e_unpractised` (held-out battery vs the deployed bot): **pooled +0.50pp
  [-0.52, +1.52]**, worst population -1.2pp (human_new -0.48, frozen +2.13, rotation1 +1.84,
  rotation2 +0.87, human_previous -0.19, heuristic -1.16); its hidden-sheet halves are fine too
  (-0.58 .. +1.93), so the hidden-sheet loss is specific to our own team.
- **Gate (`review1003_gate.py team`): T6E_HOLD** -- by the rule fixed before the run, T6e needs a
  practice round first. The same shape as T6m (battery +0.34, head-to-head 43.8%): the changed
  set costs the brain in close games against our own six, not against the other teams. Whether
  to switch anyway is the user's call (one DEPLOYED.json field); the practice recipe itself just
  failed (it washes out the tactical fine-tune), so "practise first" needs a better recipe.

## The three review guards HOLD together (mirror 46.2%, battery -0.90pp), all from hp_move_after_hits; its narrow version and the two rare guards pre-registered for their own test (2026-October 3, 19:16)

- **Readings** (`results_mirror_review1003`, `results_guard_ab_review1003`, gate
  `evaluation/review1003_gate.py guards` -> **GUARDS_HOLD**): mirror **46.2% [44.1, 48.4]**
  (changes: hp_move_after_hits 2,529 in 2,000 games, the other two 0); battery **pooled
  -0.90pp [-1.98, +0.08]**, human_new -2.71 (firings in 6,204 games: hp_move 2,170,
  throat_chop_main_threat 70, wasted_fake_out 3). Cells where hp_move fired -1.57pp [-3.16,
  -0.05], fired 3+ times -2.35pp [-4.69, -0.13], never fired +0.12pp: the broad
  Eruption / Water Spout guard hurts, dose by dose.
- **Why** (`results_analysis/review1003/hp_move_calibration.py|txt`, 943 logged ladder
  decisions with Eruption / Water Spout on top): it expects 23% HP lost before the move,
  16% is; where it swaps, 49% expected vs 32% real. 75% of its ladder swaps were right in
  hindsight, but it swaps about every other game and those swaps cost.
- **hp_move_after_spread** (narrow, `vgc_bench/src/game_review_1003.py`): only a spread attack
  a foe has shown (or its open sheet lists), from a foe that certainly moves first, and our
  partner's damage not counted in advance (the shown spread attack hits it too; turn 7's
  Incineroar fainted before its Flare Blitz). Ladder replay: 19 of 943 HP-move decisions,
  expected 51% vs 40% real where it swaps; it still turns the user's turn-7 Eruption into Heat
  Wave (test).
- **Rare guards on 2,115 replayed ladder decisions** (`rare_guard_audit.py|txt`):
  throat_chop_main_threat changes 6 (Throat Chop -> Flare Blitz on the same target: Sneasler
  twice -- Dark is resisted --, Heliolisk, Baxcalibur, Annihilape, the user's Volcarona),
  wasted_fake_out 1 (the user's turn 2): correct in all 7.
- **Pre-registered now** (after the review chain; rules fixed here): the rare trio
  wasted_fake_out + throat_chop_main_threat + hp_move_after_spread -- mirror (2,000) + battery vs
  `results_guard_ab_threat_first2`, the same gate (battery deploy-eligible, mirror upper >=
  50%, errors <= 1%). For rules this rare the gate is a non-regression check and correctness
  carries the case. Deployment: the user's call.

## The T6tac practice cycle FAILS its gate: head-to-head 37.2% / 35.2% vs the deployed bot; battery about neutral (2026-October 3, 18:05)

`training/t6tac_practice_chain.sh` (the user: "do that next training"):
- **RL save 23,101,440** (T6tac + 983,040 steps of the T6ctx recipe on T6): head-to-head vs the
  deployed bot (12 guards both sides) **35.2% [33.1, 37.3]**.
- **Tactical re-fit** (`results_tactical_t6tacp1/sft/tactical_e4.zip`): **37.2% [35.2, 39.4]** ->
  picked for the battery: **pooled +0.03pp [-0.97, +1.06]** (human_new -0.58, frozen -0.68,
  rotation1 +1.16, rotation2 +1.45, human_previous -1.06, heuristic -0.10).
- **Gate: HOLD** (head-to-head not won) -> no ladder, nothing deployed. The chain's idle() before
  its gate waited on the challenge listener (a ladder_ourteam process the user asked for);
  stopped by hand at 18:01 and the gate read by hand (logged in its chain.log).
- **Reading:** practice from the tactical fine-tune washes out what the fine-tune taught: T6tac
  beat T6ctx 58.0% after one re-fit; 983,040 more RL steps give a brain that wins only 35% against
  T6tac, and a second re-fit on its own games wins back 2 points. T6m's cycle had the same shape
  (39.7% -> 43.8%). Against the battery's bots the brains are even, so the loss is in the close,
  same-team games the head-to-head measures. A next practice cycle should keep the tactical lessons
  inside training (e.g. an auxiliary loss toward the tactical teacher), not re-fit afterwards --
  the user's call.

## Pre-registered: the user's game-review requests -- three guards, open-sheet preview with late re-plan, T6e (Torkoal Earth Power); measured after the practice chain (2026-October 3, 16:03)

The user, after a challenge game with open sheets (challenge_replays_mc_deployed_T6tac_guards12,
battle 2692336128, lost to Mega Raichu Y / Sylveon / Volcarona / Rillaboom): "i need it to
also factor in things it knows from open team sheets when it has them, and for the plan to
change if the opponent accepts open team sheets last second", notes on turns 2, 5, 6 and 7,
and "remove weather ball for earth power on torkoal". The decision log confirms each note:
turn 2 Fake Out into a Raichu asleep since last turn beside a recharging Sylveon (ranked 19%
vs Flare Blitz 16%); turn 5 Venusaur had picked Sludge Bomb on Volcarona (Sleep Powder not
among its 8 ranked pairs); turn 6 Throat Chop (23%) kept by dominated_throat_chop because
Volcarona's sheet shows Bug Buzz, the switch to Torkoal ranked 4th (9%); turn 7 Eruption
(36%) over Heat Wave (12%), which would have knocked out both foes.
- **Built** (`vgc_bench/src/game_review_1003.py`, tests rebuilt from the game's turns):
  `wasted_fake_out` (a foe that must recharge, or fell asleep and has not failed a move
  attempt yet, cannot act -- Champions sleep lasts 2-3 attempts -- so Fake Out gives way to
  the strongest attack), `throat_chop_main_threat` (the sound block counts only when a sound
  move is the target's strongest attack on our side, or Parting Shot / Perish Song),
  `hp_move_after_hits` (Eruption / Water Spout re-scored at the HP left after the foes that
  move first: a certain order fully, an uncertain one half; spread hits fully, single-target
  half). Not built yet: the turn-6 sun reset (switch Torkoal in so Chlorophyll outspeeds),
  a three-step plan.
- **Built** (`vgc_bench/src/sheet_preview.py`, `PolicyPlayer(sheet_preview=True)`): with their
  open sheet, the preview model's top 6 plans are re-scored by a turn-1 damage race (with
  Fake Out and Wide Guard) computed with their sheet's sets MINUS the same race with their
  species' usual sets (2.0 log-probability per unit of HP), so only what the sheet adds can
  move a plan; their sheet is written to the decision log. A sheet that arrives after our
  preview went out on the 20 s wait re-plans and resends a changed `/team` (Showdown
  replaces a choice until both players have chosen; a late sheet means they have not).
  Switch: `--sheet-preview`, DEPLOYED.json `sheet_preview` -> the launchers. Default off.
- **Built:** `teams/candidates_mc/T6e.txt` = T6 with Torkoal Earth Power for Weather Ball
  (validator passes; manifest entry; `data/opening_plans_t6e.json`).
- **Measured after the T6tac practice chain** (`tools/review1003_chain.sh`; readings fixed now
  in `evaluation/review1003_gate.py`):
  1. the three guards together: 2,000-game mirror + battery vs `results_guard_ab_threat_first2`
     (the deployed bot, 12 guards). PASS = battery deploy-eligible (pooled upper >= 0, no
     population below -3pp), mirror upper >= 50%, each guard's errors <= 1%.
  2. T6e on the deployed brain, unpractised: head-to-head vs the same brain on T6 + battery.
     PASS (switch now) = battery deploy-eligible AND head-to-head upper >= 50%; else T6e needs
     a practice round first (the T6m lesson: changed moves cost an unpractised brain).
  3. the open-sheet preview: battery. PASS = deploy-eligible and preview errors <= 1%;
     descriptive: open- vs hidden-sheet halves (the hidden half cannot change), how often
     the plan changed.
- Deployment is the user's call for each; DEPLOYED.json untouched.

## Pre-registered: the T6tac practice cycle -- one more practice round on our own team, then the tactical re-fit (2026-October 3, 13:57)

The user: "add threat_first2 to official bot and do that next training". The
remaining gap from the early-loss work is play after the first exchange (human
pilots of our six win 44% of games where they lose a Pokemon first, we 25%), which no
guard reaches; practice of whole games is the tool. The two latest promotions each
came from one such round (T6ctx 59.3% vs T6hp, T6tac 58.0% vs T6ctx).
- **Training** (`training/run_t6tac_practice_trial.py`, `results_brainv1_t6tac_practice1`):
  the T6ctx recipe unchanged (35% human-clone games, human-model previews for both
  sides, 50% hidden sheets, reward and shaping unchanged, held-out rosters zeroed) from
  the deployed brain T6tac (22,118,400) on the deployed team T6: +983,040 steps
  (-> 23,101,440), saves every 491,520.
- **Tactical re-fit** (the T6tac recipe): `gen_tactical_data.py` on the final save's own T6
  games (250 per cell) -> `tactical_sft.py --lessons base --lr 1e-4 --epochs 4`, the
  lowest-cross-entropy epoch.
- **Head-to-heads** (2,000 games each, `mirror_guard_ab.py --a-checkpoint`, both sides the
  deployed setup with the 12 guards): the RL save and the re-fit; the battery plays
  whichever won more.
- **Battery** (`run_guard_ab.py --candidate --without-arm results_guard_ab_threat_first2`, the
  deployed brain with all 12 guards).
- **Gate** (`training/t6tac_practice_gate.py`, fixed now): "better" = head-to-head Wilson
  lower bound > 50% (the same team on both sides, so a tie is not enough) AND the
  battery deploy-eligible (pooled upper >= 0, no population below -3pp).
- **Ladder only on the user's word:** if the gate passes and
  `results_brainv1_t6tac_practice1/LADDER_OK` exists, a 15-game trial (continued to 40
  unless it starts 4-11 or worse) -- the user had not authorized ladder for this cycle
  when it started. **No promotion by me**; DEPLOYED.json untouched.
- Chain: `training/t6tac_practice_chain.sh` (detached, caffeinate, ~4 hours by the T6m
  cycle's timings). Prepared and dry-checked: league built, battery and head-to-head
  set up against the 12-guard bot; the gate holds on the T6m cycle's results as it
  should.

## threat_first2 DEPLOYED at the user's word (12 guards, replay tag T6tac_guards12) (2026-October 3)

The user: "add threat_first2 to official bot and do that next training".
`results_deployed/DEPLOYED.json`: `guards_extra` += `threat_first2`, `replay_tag`
T6tac_guards11 -> **T6tac_guards12** (fresh single-config replay dirs), amendment
recorded with the evidence (battery +0.44pp [-0.39, +1.26], +1.49pp where it acted;
mirror 49.9%; ladder trial 22-18 over 40, 3 changes in 345 decisions). Brain, team,
preview model and sticky corrections unchanged. `tools/deployed_config.py` verifies;
the deployed-config / launcher tests pass. Later guard A/Bs reuse
`results_guard_ab_threat_first2` as their without side (its guards are now all
deployed).

## Rating check: the T6tac era holds the account's Reg M-C highs -- peak 1353 (09-28), 1339 this morning (2026-October 3, 12:10)

`results_analysis/threat_first3_20261003/rating_trace.py|txt` (our rating at the start
of each of 391 saved Reg M-C ladder games): **peak 1353** at game 288 (T6tac + the
review guards, 09-28) and **1339** at game 373 (T6tac + 11 guards + threat_first2, this
morning); latest 1289 before the last game (a loss). Earlier highs: T4 1321 (09-20),
T6ctx 1282, T6hp 1239. Within any 20 games the rating swings 50-150 points, so this
is a level, not a trend; but the deployed configuration plays at the account's best
Reg M-C level so far.

## Research: how far a threat guard can reach -- the knockout threat is seen in 63% of early first faints, but an answer exists in ~2% of games (2026-October 3, 12:00)

`results_analysis/threat_first3_20261003/` (likely_set_audit, fake_out_reach; .py +
.txt; live settings: usage prior on, Reg M-C format):
- **Detection is decent:** in 123 first faints of ours on turns 1-3 (T6-era ladder), the
  deployed threat check (threat_first2.threats) flags the actual killer as a knockout
  threat to the victim in **77 (63%)**. Assuming the foe's most-likely ability and item
  from usage (>= 50% of its sets) adds only **2** (both Armarouge).
- **The miss that prompted it** (threat_first2 trial, battle 2692164095): Farigiraf +
  Incineroar led into Mega Staraptor + Sylveon; Incineroar Parting Shot, and Sylveon's
  Hyper Beam knocked out the full-HP Farigiraf before Trick Room. The calculator gave
  44-52% (unrevealed Pixilate); with Pixilate 78-93%; the rest is Fairy Feather, which
  poke-env's calculator does not apply to a Pixilate-changed move (it checks the item
  against the move's original Normal type, damage_calc_gen9.py:949). And Incineroar's
  Fake Out into Sylveon was not among the ranked pairs at all.
- **The answer is the bottleneck:** of 35 turn-1 first faints (games with decision
  logs), the killer was flagged in 12; one of our leads could Fake Out the killer in 7;
  that Fake Out was played 0 times, ranked but not played 5 (threat_first2's territory),
  not ranked 2. Our usual leads (Blastoise, Farigiraf) carry no Protect, so beyond
  Fake Out the answers are switches (doomed_switch failed) or nothing.
- **Reading:** guard-level fixes for early faints top out around 2% of games -- in line
  with threat_first2's +0.44pp. No new guard built from this. Bigger levers are
  structural (a Protect on a lead -- a team question) or the brain's mid-game play
  (experts win 44% of games after losing a Pokemon first, we 25%; small human sample).

## The turn-1 script alone is NEUTRAL locally and fails the pre-registered floor (human_new -3.6pp) -> no ladder A/B; not recommended (2026-October 3, 11:30)

`results_guard_ab_turn1_script` (the deployed bot + the Water Room turn-1 script on our
own preview, `playbook_script_only`; vs `results_guard_ab_review_guards_0928`):
- **Pooled -0.26pp [-1.43, +0.97]**; human_new **-3.58**, frozen -2.13, rotation1 -0.58,
  rotation2 +1.45, human_previous -0.97, heuristic **+4.26**. It changed turn 1 in
  **3,293 of 6,204 games (53%)**, 0 playbook / preview errors. Gate (10:40 rules):
  **HOLD** -- a population below -3pp -> **no ladder A/B** (`TURN1_HOLD`, 11:26).
- **Split** (`results_analysis/turn1_script_20261003/battery_split.py|txt`): the cells
  where it changed turn 1 -0.11pp [-2.10, +1.85], where it never did -0.46; the other-
  lead cells (identical bots on both sides, an A/A check) -0.66 [-2.84, +1.42]. By
  population in the Blastoise + Farigiraf cells: heuristic +4.55 [+2.21, +6.99], the
  human-clone populations human_new -4.08 / human_previous -0.47. Early in the run the
  changed cells were worst against Tailwind (-14pp, 14 cells) and Trick Room archetypes
  (-9pp): Fake Out on a Tailwind setter spends Blastoise's turn where their Tailwind
  would not have mattered under our room (ladder history agrees, small: vs Tailwind
  leads Blastoise's Fake Out 8-8, Ice Beam / other 12-4).
- **Verdict:** forcing the experts' turn 1 on this brain does not help locally; with the
  history analysis (10:50: the games it would change won 52% vs 55%, so a small effect
  at most) I do not recommend a ladder A/B of it. The 09-27 idea stands: a turn-1 plan
  has to be practised by the brain (plan features + a training cycle), not forced by a
  guard. The pre-registered floor decided this read (the pooled number alone is
  neutral); overriding it for a ladder A/B is the user's call.
- Nothing deployed, DEPLOYED.json untouched; `playbook_script_only` stays as an option
  (default off).

## Research while the battery runs: what the turn-1 script can and cannot fix in our own ladder history (2026-October 3, 10:50)

`results_analysis/turn1_script_20261003/` (turn1_history, room_failures, script_audit,
script_split; .py + .txt), over our 196 T6-era ladder games that led Blastoise +
Farigiraf (free wins excluded):
- **Blastoise's turn 1 there:** Water Spout 85 (won 45%), Fake Out 75-76 (56%), Ice Beam
  21 (67%). The current brain Fakes Out much less: 3/24, 7/29, 7/28 in the last three
  reads (earlier reads 7/12-9/14).
- **Our Trick Room went up on turn 1** in 95% of Fake Out games, 67% of Water Spout
  games; room-up games won 55%, the rest 37%. Water Spout failures: Imprison 10, their
  own Trick Room 8, Farigiraf knocked out before moving 5, flinch 2.
- **But the Fake Out vs Water Spout gap is mostly confounding.** The Imprison users were
  Indeedee 10 (Psychic Terrain), Farigiraf (Armor Tail), Gengar (Ghost), Oranguru
  (Inner Focus): Fake Out cannot touch any of them, so those games are Water Spout
  games by necessity. Replaying the script on the real turn-1 positions (with the
  usage prior on, as on ladder; it correctly skips an opposing Farigiraf): it changes
  turn 1 in **100 of 198** Water-lead games; the Water Spout games it would change won
  **52%** (room up 74%) vs 55% (98%) where the bot already Faked Out the scripted
  target; the Water Spout games it would NOT change (no reachable foe) won **37%**
  (room up 60%). Of the 43 failed rooms the script changes turn 1 in 19 (Imprison 2 of
  13).
- **Expectation for the A/B:** a higher room-up rate in the games it changes, a small
  win-rate effect (a few points at most) -- below what 80 ladder games can see. The
  untouched hole is the Fake-Out-proof Trick Room leads (Indeedee / Psychic Terrain,
  Farigiraf, Ghosts): 43 games at 37%.

## Pre-registered: the experts' turn 1 alone -- the Water Room turn-1 script on our usual preview; battery, then an alternating ladder A/B if it holds up (2026-October 3, 10:40)

Under the user's delegation ("try new things"; "run ladder if tests are good"), the
early-loss lever from the overnight summary, separated from the playbook's plan
choice (which is what lost locally on 09-27; the script alone was about neutral,
-1.1pp over 924 matching battery games):
- **Built:** `PolicyPlayer(playbook_script_only=True)` (`ladder_ourteam.py
  --playbook-script-only`, `TRIAL_SCRIPT_ONLY=1` in `tools/ladder_trial.sh`,
  `run_guard_ab.py --playbook-script-only`): our four and leads stay the deployed
  preview model's; the playbook card (`data/playbook_t6_trial.json`: Water Room for
  every opponent) is attached only for its turn-1 script, which the opt-in guard
  `playbook_opening` plays only when the preview led the card's own pair, Blastoise +
  Farigiraf: Mega Blastoise Fake Out (their Trick Room / Tailwind setter, else the
  biggest threat to Farigiraf; never into Armor Tail / Psychic Terrain / a Ghost) +
  Farigiraf Trick Room. Other leads are untouched. 6 new tests; 669 pass.
- **Battery** (`results_guard_ab_turn1_script` vs `results_guard_ab_review_guards_0928`,
  the deployed bot with its 11 guards): holds up = pooled upper bound >= 0, no
  population below -3pp, playbook / learned-preview errors <= 1% of games, and the
  script changed turn 1 in >= 5% of games (else there is nothing to test). Reported:
  the split by cells where it changed turn 1 vs not. **No mirror:** its opponent is
  our own team, whose Farigiraf's Armor Tail blocks every Fake Out, so there the
  script is only "Trick Room into their Trick Room".
- **If it holds up -- ladder A/B** (`tools/turn1_script_chain.sh`): serial, in
  alternating 10-game blocks so both arms meet the same rating range: script (S,
  `ladder_replays_mc_T6tac_turn1_script`) and the unchanged deployed configuration
  (C, `ladder_replays_mc_T6tac_turn1_control`, `tools/ladder_trial.sh` with no
  additions), order **S C C S C S S C** (40 + 40, about 3.5 hours). It stops at a
  block boundary if the user says so (or a `STOP` file in
  `results_analysis/turn1_script_20261003/`). Reading: each arm's record without
  free wins and the Newcombe 95% CI of the difference; mechanism: in games we led
  Blastoise + Farigiraf, the share where we lost a Pokemon on turns 1-3, per arm;
  how often the script changed turn 1. 80 games cannot show a few points: a read,
  not proof. **No promotion by me**; DEPLOYED.json untouched.

## Overnight summary (the user's 8-hour delegation): threat_first2 is the keeper (battery +0.44pp, ladder 22-18); doomed_switch dropped; the guards fire rarely on ladder -- the early-loss lever is turn-1 play (2026-October 3, 10:15)

- **threat_first2 ladder trial** (`ladder_replays_mc_T6tac_threat_first2`, 08:25-10:12, T6tac
  + 11 guards + threat_first2; 40 games by the pre-registered continuation): **22-18**,
  no free wins (first 20: 13-7). We lost a Pokemon first in 17 games and won 3. It
  changed **3 of 345 decisions** -- the turn-1 Fake Out onto a Rillaboom about to knock
  out Blastoise (it flinched, the room went up; that game was lost later) and two others.
- **threat_first ladder trial** (earlier tonight): 18-22 (16-22 without 2 free wins),
  1 change in 332 decisions (Ice Beam on a 49% Rillaboom under our room; a win).
- Together the two reads are essentially the deployed bot with its 11 guards: 40-40
  over 80 games (38-40 without free wins). Neither guard fires often enough on ladder for
  40 games to see it; the evidence for threat_first2 is local -- battery +0.44pp
  [-0.39, +1.26], +1.49pp in the cells where it acted -- and every ladder change it
  made was the intended one.
- **Verdicts:** threat_first2 passes every gate and supersedes threat_first ->
  recommended for `guards_extra` (the user's call). doomed_switch failed its battery
  (-2.97pp [-4.37, -1.74]) -> dropped.
- **Where early losses really come from:** guards act on a few percent of decisions,
  while we lose a Pokemon first in about half our games and win a fifth of those. The
  expert pilots of our six differ in turn-1 play (Blastoise Fake Out : Water Spout = 4 :
  1, ours 1 : 1; Farigiraf never lost early). The one configuration that forced a turn-1
  Fake Out + Trick Room script on ladder (T6tac + the Water Room playbook, 2026-09-27)
  went 10-4 without free wins. Next lever (the user's call): a 40-game ladder A/B of
  that turn-1 script against the plain deployed bot -- our local instruments cannot
  judge it (the mirror's Armor Tail blocks Fake Out; the battery field is too weak).

## doomed_switch FAILS its battery (-2.97pp [-4.37, -1.74]) -> no ladder; threat_first2's 20-game ladder trial started (2026-October 3, 08:25)

- **doomed_switch:** mirror 49.4% [47.2, 51.5]; battery **-2.97pp [-4.37, -1.74]**
  (worse than the deployed bot), populations {'human_new': -1.2, 'frozen': -5.0, 'rotation1': -4.7, 'rotation2': -2.8, 'human_previous': -3.5, 'heuristic': -0.6}; it changed the pick
  1613 times in 6,204 games (far more than the ladder replay's 1.8% of
  decisions suggested). Gate: HOLD, no ladder. Lesson: switching out a Pokemon the
  calculator calls doomed costs more than it saves -- the predicted KO often does not
  come (the foe targets or moves differently), and the switch-in takes damage and the
  tempo is lost. The 66-of-67 "a benched Pokemon survives the hit" counterfactual was
  hindsight on the turns where the KO did happen.
- **threat_first2:** its 20-game ladder trial started 08:25
  (`ladder_replays_mc_T6tac_threat_first2`), continued to 40 by the 08:00 rule.

## threat_first2 passes (battery +0.44pp); its ladder continuation pre-registered; doomed_switch's battery running badly (2026-October 3, 08:00)

- **threat_first2:** mirror 49.9% [47.7, 52.1]; battery **+0.44pp [-0.39, +1.26]**, worst
  population -1.5pp, 0 errors in 6,204 games, 583 changes in 128 of 564 cells -- cells
  where it fired **+1.49pp**, where it did not +0.13pp, cells with 3+ changes +0.67pp
  (threat_first by the same split: -0.30 / -0.04 / +0.00). Gate: **PASS** -> 20 ladder
  games next (`tools/guard_round2_ladder.sh`).
- **Fixed now, before its first ladder game:** continue threat_first2's trial to 40
  games unless it wins 6 or fewer of the first 20 (`tools/threat_first2_continue.sh`),
  as threat_first's rule.
- **doomed_switch:** mirror 49.4% [47.2, 51.5]; battery so far human_new -1.2pp,
  **frozen -5.0pp** (160 / 321 changes per 1,034 games) -- likely to fail the -3pp
  floor; read when complete.

## Round 2 ladder step pre-registered (2026-October 3, 06:30)

Before any round-2 battery result exists: each round-2 guard that passes the gate
(`ROUND2_PASS` in `results_analysis/threat_first_20261003/round2.log`) gets a 20-game
serial ladder trial alone on top of the deployed bot (`TRIAL_GUARDS=<guard>
tools/ladder_trial.sh 20 ladder_replays_mc_T6tac_<guard>`), threat_first2 first, then
doomed_switch (`tools/guard_round2_ladder.sh`, started now, waiting for round 2 to
finish). No continuation rule (time: the user returns around 10:45), no promotion.
threat_first2's mirror: **49.9% [47.7, 52.1]** (10 changes in 2,000 games); its
battery is running.

## threat_first ladder trial: 18-22 over 40 (16-22 without 2 free wins); it fired once (correctly); research: expert pilots of our six protect Farigiraf and Fake Out on turn 1 (2026-October 3, 06:00)

- **Ladder** (`ladder_replays_mc_T6tac_threat_first`, 04:05-05:56, T6tac + 11 guards +
  threat_first): **18-22**, 2 free wins -> 16-22 (42%); the first 20 were 8-12, so it
  continued to 40 by the pre-registered rule. We lost a Pokemon first in 21 games and won
  3 of them. **threat_first changed 1 of 332 decisions**: turn 2 under our room,
  Blastoise's Water Spout -> Ice Beam on a 49% Rillaboom (Farigiraf's Psychic took it to
  8%, Ice Beam knocked it out before its Wood Hammer); that game was a win. So the read
  is essentially the deployed bot with 11 guards (its last 40-game read: 20-17 without
  free wins); the guard is correct but rare on ladder, and its battery was -0.10pp.
- **Research, our ladder vs human pilots of our six** (`results_analysis/early_faints_20261003/
  experts_vs_us.py|txt`; the 37 corpus games where a side brought >= 5 of our six, vs
  our 253 T6-era ladder games): they win 68% (we 48%); they lose a Pokemon first 43% of
  games (we 52%) and still win 44% of those (we 25%); they lose one on turns 1-3 in 35%
  (we 44%) and **never Farigiraf there (0 of 13; ours 43 of 111)**; they switch on turn
  1-2 in 54% (we 42%); their Blastoise's turn 1 is Fake Out 8 : Water Spout 2 (ours 67 :
  61). Small human sample, but it points at the same place: protecting the setter on turn
  1 (Fake Out the biggest threat) and switching more. A turn-1 Fake Out rule cannot be
  judged by our mirror (the opposing Farigiraf's Armor Tail blocks Fake Out) and the
  battery measured the playbook's turn-1 script as about neutral -- it would need ladder.
- Round 2 (threat_first2, doomed_switch) started 05:56 (`tools/guard_round2_chain.sh`).

## threat_first passes its tests (mirror 49.7%, battery -0.10pp) -> 20-game ladder trial running; round 2 built and pre-registered: threat_first2 and doomed_switch (2026-October 3, 04:10)

- **threat_first results:** mirror **49.7% [47.5, 51.9]** over 2,000 (inconclusive -- it
  changed 3 actions in 2,000 games: our own team rarely creates its situations);
  battery **-0.10pp [-0.68, +0.48]**, worst population -1.5pp, 0 errors in 6,204 games,
  355 changes in 120 of 564 cells (cells where it fired -0.30pp, where it did not
  -0.04pp, cells with 3+ changes +0.00pp: no signal). Gate: **GO** -> the ladder trial
  (T6tac + 11 guards + threat_first, `ladder_replays_mc_T6tac_threat_first`) started
  04:05: 20 games, continued to 40 unless it wins 6 or fewer.
- **Round 2, built during the tests** (opt-in, registered after the battery ended; the
  running ladder trial does not enable them):
  - `threat_first2` (`vgc_bench/src/threat_first2.py`, 5 tests): threat_first that
    counts the foe's likely Mega forme -- a pre-Mega Raichu holding Raichunite Y is
    judged as Mega Raichu Y, which threat_first missed (Zap Cannon KO'd our Blastoise
    on turn 1 three times) -- and answers a faster threat with a first-turn Fake Out
    (or moves a Fake Out from a harmless foe onto the threat); the knockout claim must
    hold against both formes. Replayed: 11 changes in 1,131 ladder decisions (incl.
    three turn-1 Fake Outs onto Rillaboom / Mega Raichu).
  - `doomed_switch` (`vgc_bench/src/doomed_switch.py`, 3 tests): when a foe knocks one
    of ours out (>= 90%) before it can move, cannot also KO our partner, and nothing in
    the top pair answers it, switch it to a benched Pokemon taking <= 60% of its HP from
    that move -- never a Trick Room (or other status) click. Basis: in 66 of 67 turn 1-3
    first faints a benched Pokemon would have survived the killing hit
    (`results_analysis/early_faints_20261003/switch_counterfactual.txt`). Replayed: 20
    changes in 1,131 decisions; in 11 the Pokemon did faint that turn in the real game.
  - (`fake_out_threat` alone fired twice in 1,131 decisions -- folded into threat_first2.)
- **Round 2, fixed now** (`tools/guard_round2_chain.sh`, launched after the ladder
  trial ends; ladder and local runs never share the machine): for each of
  threat_first2 and doomed_switch alone on the deployed bot, the 2,000-game mirror and
  the battery vs the deployed bot with its 11 guards, read by the same gate (battery
  deploy-eligible, mirror upper bound >= 50%, errors <= 1%). A passing guard gets a
  ladder trial after that; no promotion by me.
- Heading time of the threat_first pre-registration corrected to its commit time
  (02:53, commit 4e8a770).

## threat_first: a guard against losing a Pokemon early, built and pre-registered; test-and-ladder chain started (2026-October 3, 02:53)

The user: "yes build the guard and test it, then run ladder if tests are good.
operate on your own until i give next instructions in about 8 hours".
- **The guard** (`vgc_bench/src/threat_first.py`, registered opt-in in guards.py after
  `drop_free_finish`; 6 tests in `unit_tests/test_threat_first.py`): when a foe can
  knock out one of our active Pokemon this turn (its best likely move, revealed plus
  the most-used set, KOs from our current HP with >= 50% of the roll, accuracy
  included), and the top pair does not already answer it (an attack that knocks it out
  first, a Fake Out on it, or the threatened Pokemon Protecting or switching), it
  promotes the best-ranked pair that keeps the partner's action and replaces one of
  our plain attacks with an attack of the same Pokemon (same Mega/Tera) that knocks the
  threat out (>= 85%, a likely Focus Sash at full HP excluded) before the threat's
  move: higher priority, or certainly faster (certainly slower under Trick Room) over
  every Champions build of the foe, its possible Mega and a possible Choice Scarf.
- **Replayed over 1,131 logged ladder decisions** (`results_analysis/threat_first_20261003/
  firing_audit.py`): the first version moved attacks onto the wrong foe when two foes
  threatened the same Pokemon, so it now acts only when removing the threat saves the
  Pokemon, and never moves a single-target attack off a foe that is itself a threat. It
  then fires on 11 decisions (1.0%), all sensible: Water Spout -> Ice Beam into
  Rillaboom / Salamence / Serperior, Water Pulse into Gengar / Raichu / Aerodactyl /
  Glimmora, Farigiraf's Psychic onto a Hisuian Arcanine about to KO it.
- **Fixed before running** (`tools/threat_first_chain.sh`, gate
  `evaluation/guard_ladder_gate.py`): mirror `mirror_guard_ab.py --guard threat_first
  --games 2000` (side A adds the guard to the deployed bot); battery `run_guard_ab.py
  --guards threat_first --without-arm results_guard_ab_review_guards_0928` (the deployed
  bot with its 11 guards, 6,204 held-out games). **"Tests are good" = the battery
  deploy-eligible (pooled upper bound >= 0, no population below -3pp) AND the mirror not
  lost (upper bound >= 50%) AND guard errors <= 1% of games** -- the review guards' rule.
  Then a serial ladder trial (`TRIAL_GUARDS=threat_first`, `ladder_replays_mc_T6tac_threat_first`):
  20 games, continued to 40 unless it wins 6 or fewer. Records reported with free wins
  excluded. No promotion by me.

## Losing a Pokemon first: when and how (analysis for the user's question, 2026-October 3)

The user: "what do u think for the losing a pokemon problem". Analysis only, the 236
T6-era ladder games (`results_analysis/early_faints_20261003/early_faints.py|txt`).
- We lose a Pokemon first in 107 games and win 21% of them (23-84); when they lose
  one first we win 71% (84-35). 92 of our 107 first faints come on turns 1-3.
- **73 of those 92 are our two leads -- Blastoise 37, Farigiraf 36 -- and neither set
  has Protect.** Half fainted before acting that turn (Farigiraf 26 of 36), half were
  already below 75% HP, and 52 of 92 happened UNDER OUR OWN Trick Room. Only 19 were
  hit twice that turn: mostly chip on turn 1-2, then a finishing hit.
- Killing hits: Rillaboom's Wood Hammer 13 (7 on Blastoise, 6 on Farigiraf); on
  Blastoise Psychic 12 (Expanding Force under Psychic Terrain, Psychic), Normal 7
  (Hyper Voice), Electric 6 (Zap Cannon); on Farigiraf Dark 9 (Knock Off, Kowtow
  Cleave, Darkest Lariat), Bug 4. What the bot had chosen for the victim: Trick Room
  18, Water Spout 16, Water Pulse 14, Psychic 9, Rain Dance 7, Ice Beam 7.
- Why the doomed lesson (tactical 2-4) could not help much: it only moved mass to a
  legal Protect, and the two Pokemon that die early have none.

## Ladder trial of the T6m candidate: 6-9 over the 15 games asked for (7-9 over 16: the restart played one extra); no case for T6m over T6 (2026-October 2, 23:10)

`ladder_replays_mc_t6m1`, 22:17-23:06, serial; the configuration as fixed at 22:20.
- **Record: 6-9 over the user's 15** (7-9 over all 16); no free wins -- 5 of the 6
  wins were opponents quitting from behind at turns 2-5. Opponents median 1177
  (1098-1313). T6tac's reads: 7-8 alone, 20-17 with the review guards (free wins
  excluded). 15 games cannot resolve a few points either way.
- **What happened to the run:** the app stopped the first session after 8 games (its
  background time limit); relaunched detached, the read loop continued from the
  saved games. Game 9 straddled the restart (disconnected at turn 4, reconnected with
  40 s to spare) and was already lost on turn 3 -- Indeedee-F's Trick put a Choice
  item on Farigiraf, whose locked second Trick Room cancelled our own room. The
  resumed session finished that game and still played its 7, so a 16th game (a win)
  was played beyond the 15.
- **T6m's changes in play:** Charizard was never brought, so Weather Ball on it went
  untested; Torkoal used Eruption 18 times, Weather Ball 7, Earth Power 5 (into
  Incineroar and Armarouge, both 2x), Protect 4. Sand teams met once (a win), rain
  twice (1-1). When the opponent lost a Pokemon first we won 6 of 8; when we did,
  0 of 7.
- **Reading:** together with the local evidence (battery level with T6tac, the close
  games 6 points behind), nothing argues for T6m over T6. DEPLOYED.json unchanged
  (T6tac on T6 with 11 guards); the T6m candidate stays a candidate.

## Ladder trial of the T6m candidate at the user's word: 15 games (2026-October 2, 22:20)

The user, after the held gate: "run the ladder trial anyway with 15 games". Fixed now:
`results_tactical_t6m1/sft/tactical_e4.zip` (sha 5c8e52b9; sidecar stamped:
knowledge features required, team T6m) on `teams/candidates_mc/T6m.txt`, with
everything else the deployed configuration (11 guards, the learned preview model,
sticky corrections, Reg M-C set data) via `TRIAL_CHECKPOINT` / `TRIAL_TEAM
tools/ladder_trial.sh 15 ladder_replays_mc_t6m1`, serial, under caffeinate. It is a
read, not a test that can settle a few points; reported with free wins (opponent
gone by turn 1) excluded and compared with T6tac's reads (7-8; 20-17 with the
review guards). DEPLOYED.json is not touched; promotion stays the user's call.

## T6m diagnostics: every changed configuration loses the head-to-head to T6tac on T6 by 6-9 points; T6m is not yet an improvement in our bot's hands; no ladder played (2026-October 2, 04:50)

Both diagnostics ran 03:44-04:43 (`results_brainv1_t6m_practice1/diagnostics.log`;
results copied to `results_analysis/t6m_practice_20261002/`), read as fixed at 03:43
(commit 716133c).
- **D1, the deployed T6tac on T6m (unpractised) vs T6tac on T6: 41.0% [38.9, 43.2].**
  Practice effect in this matchup: 43.8 - 41.0 = +2.8pp (about +/-3.1pp: not
  significant).
- **D2, the candidate on T6 (its old team) vs T6tac on T6: 41.0% [38.9, 43.2]** (open
  sheets 35.9%, hidden 46.2%). Its upper bound is below 50%, so by the pre-registered
  reading the brain itself is weaker on T6 too -- drift or forgetting of its old
  sets, not separable.
- **Together** (T6tac on T6 = 50%): the same brain on T6m -9.0; the candidate on T6
  -9.0; the candidate on T6m -6.2. Each brain plays the team it practised best (the
  candidate is +2.8 on T6m over its own T6 play and over T6tac's T6m play), and the
  long-practised T6 configuration still beats the T6m one (983,040 steps of
  practice) in close games. A structural matchup edge (T6 keeps Solar Beam for the
  opposing Blastoise) and practice depth cannot be separated with these runs.
- **Reading:** against the held-out field the candidate on T6m is level with the
  deployed bot (+0.34pp [-0.98, +1.69]; rain rosters +4.2pp, descriptive), where the
  bot wins ~92% whatever it plays; in the one close-game instrument we have it is 6
  points behind; and the clone tournament's sand gain did not show on the 3 held-out
  sand rosters. T6m is not shown to be an improvement in our bot's hands yet.
- **State:** DEPLOYED = T6tac on T6 with 11 guards (unchanged since the amendment).
  Candidate kept, not stamped for ladder: `results_tactical_t6m1/sft/tactical_e4.zip`
  (sha 5c8e52b9). No ladder games were played tonight; nothing is running.
- **The user's options:** (1) another practice round on T6m (+983,040 steps from the
  RL save, re-fit, the same tests; the unpractised 41.0% is now the baseline to
  beat) -- my recommendation if T6m is worth pursuing; (2) a ladder trial of the
  candidate anyway (`TRIAL_CHECKPOINT=... TRIAL_TEAM=teams/candidates_mc/T6m.txt
  tools/ladder_trial.sh 15 ladder_replays_mc_t6m1` after stamping its sidecar): we
  never meet our own T6 on ladder, but 15-40 games cannot resolve a few points; (3)
  keep T6 and drop T6m.
- Lesson for the next team variant: a head-to-head against the old team is one
  matchup and is confounded by the sets changed -- pre-register the unpractised
  baseline (D1) as the comparison, not 50%.

## T6m practice cycle: level with the deployed bot against the field (+0.34pp), better vs rain, but it LOSES the head-to-head (43.8%) -> gate HOLD, no ladder; two diagnostics pre-registered (2026-October 2, 03:43)

The chain ran clean 23:47 -> 03:42 (`results_brainv1_t6m_practice1/chain.log`); the
pre-registered rules were applied by `training/t6m_practice_gate.py`.
- **Training** (T6tac + 983,040 steps with T6m as our team, 23:47-01:47, ~136 steps/s):
  ep_rew_mean 0.63 -> 0.79; saves 22,609,920 and 23,101,440. **Tactical re-fit** on
  its own T6m games (11,190 positions from 2,000 games, which it won 92.3%): teacher
  agreement 70.2% -> 78.8%, epoch 4 (`results_tactical_t6m1/sft/tactical_e4.zip`,
  sha 5c8e52b9).
- **Head-to-head vs the deployed bot** (candidate on T6m, T6tac on T6, 11 guards and
  the preview model on both sides, 2,000 games each): RL save **39.7% [37.6, 41.9]**,
  re-fit **43.8% [41.6, 46.0]** (every block 41.8-46.2%) -- both LOSE; the re-fit is
  the pick.
- **Battery of the pick** (on T6m vs the deployed brain on T6 with all 11 guards, 6,204
  held-out games, `results_brain_ab_t6m1_23101440_tactical`): pooled **+0.34pp
  [-0.98, +1.69]**, worst population -1.4pp -> deploy-eligible, not better.
- **Gate (pre-registered): HOLD** -- the head-to-head is lost (upper bound 46.0% <
  50%). No ladder games were played; DEPLOYED.json is unchanged since the guard
  amendment.
- Descriptive (`results_analysis/t6m_practice_20261002/battery_readout.txt`): rain
  rosters **+4.2pp [+0.7, +7.9]** (89.3 -> 93.5%), trick_room -1.9, balance -1.1,
  grassy Fake Out -0.6, tailwind +0.7; the Charizard + Venusaur lead 88.6 -> 94.4%
  (6% of games). **The 3 held-out sand rosters: 82.1 -> 81.1% (-1.0pp [-11.4, +5.3])**
  -- the clone tournament's +11.2pp against sand did not show up in our brain's hands
  (3 rosters, 396 games: weak evidence either way).
- **Two diagnostics, fixed before running** (head-to-heads, 2,000 games each, the
  same harness; they explain the lost head-to-head and trigger nothing):
  D1 = the deployed T6tac itself on T6m (unpractised) vs T6tac on T6 -> practice
  effect in this matchup = 43.8% minus D1. D2 = the candidate on T6 (its old team) vs
  T6tac on T6 -> if its interval reaches 50%, the brain is not weaker in close games
  and the loss is the T6m-vs-T6 matchup (the team); if its upper bound is below 50%,
  the brain itself is weaker on T6 too (drift or forgetting, not separable).
- The first notification waiter (a 10-minute background command) was stopped at its
  time limit; a log monitor replaced it. The chain itself ran detached throughout.

## The three review guards DEPLOYED (the user's word); the T6m practice cycle pre-registered; pushed to GitHub (2026-October 1, 23:50)

The user: "add the guards to the bot and start the practice run, im not going to be
available to give u instructions afterwards but if it does well start 15 ladder
games and go from there on your own" and "can u also push everything to gh".
- **Pushed:** `main` to origin (b2683c3..998a2c0, 159 commits; `pre-squash-backup`
  untouched). Tonight's commits are pushed as they land.
- **Guards deployed** (an amendment in `results_deployed/DEPLOYED.json`):
  `guards_extra` += `drop_free_finish`, `fake_out_partner_acts`, `switch_the_crippled`
  (11 now); `replay_tag` T6tac -> T6tac_guards11 (fresh single-config replay dirs).
  Evidence: battery +0.26pp [-0.53, +1.03] deploy-eligible, ladder trial 23-17
  (20-17 without free wins). Brain, team, preview model, sticky corrections unchanged.
- **The T6m practice cycle, fixed before any training** (`training/t6m_practice_chain.sh`,
  gate `training/t6m_practice_gate.py`):
  - **Training** (`training/run_t6_teamvariant_trial.py`, new): from T6tac
    (`results_tactical1/sft/tactical_e4.zip`, the deployed sha) +983,040 steps on the
    recipe that made T6ctx (35% human-clone games, human-model previews for both
    sides, 50% hidden sheets, held-out rosters zeroed; the training command is
    identical) with ONE change: our team file is `teams/candidates_mc/T6m.txt`. Then
    the tactical re-fit of the final save on ITS OWN T6m games
    (`gen_tactical_data.py --checkpoint --team`, new: 8 cells x 250 games; base
    lessons, lr 1e-4, 4 epochs, the lowest validation CE).
  - **Pick:** whichever of the two (RL save, re-fit) wins more of its head-to-head --
    the candidate on T6m against the deployed T6tac on T6, both with the 11 deployed
    guards and the preview model, 2,000 games each (`mirror_guard_ab.py --a-team`, new).
  - **Battery:** the pick on T6m vs the deployed brain on T6 with all 11 guards (the
    arm of `results_guard_ab_review_guards_0928`), 6 populations x 47 held-out
    rosters x 22 games (`run_guard_ab.py --candidate-plans`, new: the arms may differ
    in the brain and our team's sets, nothing else).
  - **"Does well" = the ladder gate:** the battery is deploy-eligible (pooled upper
    bound >= 0 and no population below -3pp) AND the head-to-head is not lost (Wilson
    upper bound >= 50%).
  - **Ladder, only if it does well:** a 15-game serial trial of the pick on T6m with
    the deployed guards, preview model and sticky corrections (`tools/ladder_trial.sh`
    with `TRIAL_TEAM`, new), replays in `ladder_replays_mc_t6m1`; continued to 40
    games unless it wins 4 or fewer of the first 15.
  - **No promotion by me:** DEPLOYED.json stays T6tac on T6 with the 11 guards;
    making the T6m brain the bot is the user's call. If the gate holds: no ladder, a
    write-up, stop.
  - Descriptive, not part of the gate: the battery's sand rosters, candidate vs
    reference (the matchup T6m is for).
- Dry-runs before the start: the training preparation verified; tactical data (2
  games per cell), a head-to-head (8 games) and one battery cell with the T6m plans
  ran clean; 648 tests pass, Ruff and Pyright clean on the changed files.

## T6 set-variant tournament: T6m beats T6 against sand (+11.2pp [+7.2, +15.2]) and is even on the full pool -> T6m qualifies for a practice cycle (2026-September 28, 23:59)

Resumed at the user's word ("continue this", 22:36, on AC power); 6 arms x 1,000
games, clean (no retries); read by the pre-registered script
(`results_analysis/t6variants_20260928/reading.txt`, summaries copied there).
- **Full pool** (clone pilot, 3,610 teams): T6 46.7%, T6m 48.3% (d +1.6pp [-2.8,
  +6.0]), T6mAS 47.7% (+1.0 [-3.4, +5.4]) -- no detectable difference.
- **Sand pool** (371 Tyranitar teams): T6 24.8%, **T6m 36.0% (+11.2pp [+7.2, +15.2],
  better)**, T6mAS 30.2% (+5.4 [+1.5, +9.3], better).
- **Decision (pre-registered rule):** both qualify; T6m has the higher full-pool
  estimate -> **T6m is the practice-cycle candidate** (the user's call).
- Descriptive, not part of the decision: the gain sits in games where the opponent
  brought Tyranitar (full pool: T6 21.3% -> T6m 40.2%, T6mAS 33.6%; games without
  it 49-50% for all three). Aura Sphere for Water Pulse on top of T6m did worse vs
  sand (30.2 vs 36.0%): the 10:40 audit's hindsight KO counts did not carry over to
  a pilot choosing its moves -- a caution for such audits.
- The clone is a human-like pilot, not our brain. In our bot's hands T6m needs the
  practice cycle first (the 09-27 playbook practice recipe with T6m as our team:
  T6tac + 983,040 steps, then the tactical re-fit), then the head-to-head vs T6tac on
  T6 and the battery, before any ladder.

## T6 set-variant clone tournament pre-registered: T6 / T6m / T6m + Aura Sphere, full pool and sand pool; paused (2026-September 28, 11:02)

The user: "yes start the tournament with all three teams". Local only: no ladder,
nothing promoted, DEPLOYED.json untouched.
- **Teams:** `teams/candidates_mc/T6.txt` (as deployed), `T6m.txt` (Torkoal Earth
  Power for Heat Wave, Charizard Weather Ball for Solar Beam: TEAM_REVIEW_T6.md 4b),
  `T6mAS.txt` (T6m + Blastoise Aura Sphere for Water Pulse: the 10:40 Blastoise
  audit). All three pass the simulator's validator (an illegal control fails).
- **Pilot:** the human clone `results_bc/mc_A_20260920/saves_bc/seed1/2.zip` (its
  BEST) plays our team and every opponent (stochastic); hidden sheets (ladder is
  closed-sheet); seed 83, so every arm meets the same opponent-team sequence; 8
  workers; `evaluation/run_team_tournament.sh` (new: `TEAM_WEIGHTS` swaps the pool).
- **Run A, full pool:** `data/team_weights_regmc.json` (3,610 teams, ladder-usage
  weights), **1,000 games per team** -> `results_team_tournament/clone_t6variants_20260928/`.
- **Run B, sand pool:** `data/team_weights_regmc_sand.json`, the same weights with
  every team lacking a sand setter at 0 (371 teams, all Tyranitar; 10.1% of the full
  pool's weight), 1,000 games per team ->
  `results_team_tournament/clone_t6variants_sand_20260928/`. Smoke (8 games, T6mAS,
  sand pool): clean, Tyranitar brought in 6 of 8.
- **Reading, fixed now:** per variant and pool, d = win rate - T6's, with a 95%
  interval for a difference of two proportions (unpaired, conservative): better =
  lower bound > 0; worse = upper bound < 0; else no detectable difference
  (resolution about +/-4.4pp).
- **Decision rule:** a variant qualifies for a practice cycle (the user's call) if it
  is better on run A, or better on run B with run A's point estimate >= -2pp. Both
  qualify -> the higher run-A point estimate (tie -> T6m, the smaller change).
  Neither -> T6 stays and no practice cycle.
- The clone is a human-like pilot, not our brain (09-15: the two rank teams
  differently); a qualifying variant still needs a practice cycle, then the mirror
  and battery vs T6tac, before any ladder.
- **Paused at the user's word (11:10, "pause it for now ill do it later")** during
  the first arm (T6, full pool; committed a4d098c at 11:02:39, run started 11:02:49).
  No arm finished, so nothing was read; the partial logs are in
  `results_team_tournament/clone_t6variants_20260928/paused_1110/`. Resume, unchanged:
  `./evaluation/run_t6variants_20260928.sh` (runs both pools under caffeinate, skips
  finished arms, then prints the pre-registered reading), about 75 minutes.

## Blastoise's moves on ladder: Ice Beam earns its slot; Protect for Fake Out and Dark Pulse for Ice Beam not supported; Aura Sphere for Water Pulse is the swap our games favour (2026-September 28, 10:40)

The user: "how often do we use ice beam on blastoise ... and how often does it
actually do good damage", then "protect instead of fake out and then dark pulse
instead of ice beam ... is dark better for blastoise in this meta?". Analysis only
(`results_analysis/blastoise_set_20260928/blastoise_audit.py` / `.txt`, the 220 T6-era
ladder games); nothing trained, nothing on the server.
- **Ice Beam:** 49 of Blastoise's 475 moves (Water Spout 270, Water Pulse 92, Fake
  Out 64), in 35 of the 169 games Blastoise played; chosen 76 times, and in 22 of
  those Blastoise fainted before moving. Of the 49: 13 KOs + 13 hits of 50%+ (53%;
  Water Pulse 41%), 8 at 25-49%, 8 under 25%, 7 into Protect. 10 of the 13 KOs on 4x
  targets (Salamence 7, Dragonite, Garchomp, Altaria); of the 21 aimed at a 4x-weak
  target 10 KO'd, 5 hit Protect, 6 hit a switch-in. On its intended target Ice Beam
  out-powered Water Pulse in 43 of 49 (type x weather x Mega Launcher).
- **Fake Out:** 54 of 64 flinched their target, 7 hit without a flinch, 2 were
  blocked by Psychic Terrain, 1 Protected. Blastoise fainted before moving 6 times on
  turn 1 in 169 games (63 on later turns) -> Protect for Fake Out would give up ~54
  turn-1 flinches (the expert pilots' turn-1 line) to save few turns: not supported.
- **Move variants** (the best move of each set at each of Blastoise's 411 attacking
  turns, real damage calc with Mega Launcher, KOs first then +20% damage; hindsight,
  not the bot's choices): Dark Pulse for Ice Beam better 49 / worse 41, KOs +25 / -28
  (gains Sinistcha, Indeedee, Milotic; loses Garchomp 10, Salamence 9) -- a wash that
  gives up our Garchomp answer (Garchomp rosters 11-29, our worst common one); Dark
  Pulse for Water Pulse +25 / -18; **Aura Sphere for Water Pulse 59 / 31, KOs +40 /
  -20** (Kingambit 12, Archaludon 10, Tyranitar 7 -- the sand and rain problem
  Pokemon: Tyranitar rosters 8-21, Archaludon 11-20); Dark Pulse + Aura Sphere 92 / 54,
  +59 / -39 (loses Garchomp and Salamence).
- The set-swap audit's verdict rule let damage outrank a KO; re-run with KOs first (a
  scratch copy), its Torkoal / Venusaur / Charizard counts are identical, so T6m's
  numbers stand.
- **So:** keep Fake Out and Ice Beam; if Blastoise changes, Aura Sphere over Water
  Pulse, a candidate third team for T6m's clone team tournament (the user's call).
  Any set change needs a practice cycle before our brain can use it.

## Ladder has never had open team sheets (0 of 766 games); how our Reg M-C wins ended (2026-September 28, 10:05)

> **CORRECTED 2026-10-04** (found by the opponent-predictor session, re-counted here): the
> "0 of 766" count measured nothing. Saved pages are rebuilt from `battle._replay_data`,
> and poke-env consumes `|showteam|` before that stream, so a saved page can never hold the
> line. The bot's own decision audit (`preview_shadow.open_sheet` in
> `ladder_replays_mc*/decisions.jsonl`) says **10 open, 115 closed among the 125 Reg M-C
> battles with a record: about 8% of ladder games are open-sheet.** Hidden-sheet blocks
> still describe ~92% of ladder. The win accounting below is unaffected.

The user: "check in our ladder games out of all the games with open team sheets how
many did we win that werent forfeits". Analysis only.
- **Open team sheets: 0 of 766 saved ladder games** (all 30 `ladder_replays*` folders,
  Reg M-B and M-C). Every game offered them (`otsrequest`) and the bot accepts
  (`accept_open_team_sheet=True`; poke-env sends `/acceptopenteamsheets`), but no
  opponent ever did -- no `|showteam|` line anywhere (the simulator writes one per side
  when sheets open). **Ladder is 100% closed-sheet: only the hidden-sheet blocks of
  local mirrors / batteries describe ladder conditions.**
- **Reg M-C, 295 games, 150-145:** wins played out **64**, by opponent forfeit 82, by
  opponent inactivity 4; losses 145, all played out (the bot never forfeits or times
  out). Played-out record 64-145 -- biased low, because only opponents forfeit: at 69 of
  the 86 forfeit / timeout wins we had more Pokemon left, 17 even (12 by turn 1), 0
  behind. Counting the 69 as wins and dropping the 17: 133-145 (48%).
- **T6 era (11 folders, 220 games, 110-110):** played out 46-110; 64 forfeit / timeout
  wins (51 ahead, 13 even, 0 behind; 7 by turn 1).

## The openings research re-checked on the 83 ladder games played after it; correction: opponent forfeits are wins (2026-September 28, 10:05)

The user: "take a look at openings research that i had another context make".
Analysis only (`results_analysis/openings_oos_20260928/oos_check.py` / `.txt`, reusing
the research's parser); nothing trained, nothing on the server, DEPLOYED.json untouched.
- **Correction to my "played out" records (09-27/28):** excluding every opponent
  forfeit threw away real wins. Of the 64 opponent forfeits / inactivity losses in the
  220 T6-era ladder games, 49 came with us ahead on Pokemon, 8 even, and only 7 by
  turn 1 (free wins, 3 of them at team preview). Counting all but free wins: T6tac +
  Water Room **10-4** (reported "played out 4-4"), the practised brain + Water Room
  5-9 (3-9), tactical fine-tune 2 6-8 (2-8), T6tac + the review guards **20-17**
  (11-17), T6tac alone 7-8 (4-8). No decision rested on these (the night's holds were
  mirror verdicts). From now on ladder records exclude only free wins.
- **Out of sample (the research's 134 games -> the 83 after it):** sand replicates --
  sand rosters 6-16 -> 2-5 (both 8-21), Tyranitar leads 1-9 -> 0-3 (**1-12**; 0-7 for
  the human-preview brains); Wide Guard rosters 12-25 -> 5-9 (17-34); rain 8-14 ->
  7-10 (15-24). Did not replicate: Indeedee / Psychic Terrain leads 5-11 -> 9-8, rain +
  Archaludon 3-9 -> 7-7, Volcarona 1-10 -> 6-3, Water Room vs Trick Room 12-4 -> 8-10.
  Rosters with none of Psychic Terrain / sand / rain: 53-46.
- **Wide Guard:** 1-13 in the 14 games where it blocked us (52 blocked Water Spouts /
  Eruptions / Heat Waves), but the repeats were before the `wide_guard` guard
  (09-26); since then only the first, unrevealed Wide Guard lands (one turn per game,
  5 of 7 Pelipper). The research's "Water Pulse / Ice Beam once shown" is deployed.
- **The experts' turn 2 is already the brain's:** after its own turn-1 room it plays
  Rain Dance + Water Spout on turn 2 in 52 of 103 Water Room games (33-19,
  correlational); Imprison blanked our turn-1 Trick Room 7 times. Vs sand it Rain
  Dances by turn 2 in 10 of 20 games (3-7; 4-6 without), and in 3 of the 4
  human-preview Tyranitar-lead losses the research lists, our own Torkoal re-set sun
  right after our Rain Dance (the research: back Incineroar + Venusaur vs sand).
- **Mechanics:** Champions disables Fake Out after the first action
  (`pokemon-showdown/data/mods/champions/moves.ts`), so Encore -> Struggle holds. "Mold
  Breaker ignores Armor Tail" is not modelled: `guards._priority_is_blocked` and the
  teacher's priority-block fact ignore the attacker's Mold Breaker (2 Mold Breaker
  games in our 220; minor). The Baltimore runner-up team: met once in the 83 (a win).
- **So:** draft 2 as card rules is not worth building -- its main new rule (Support
  Room vs Psychic Terrain) rests on a hole that did not replicate, and that card lost
  -14.6 locally. What carries forward: sand, the one hole that holds out of sample and
  on the local battery (the sand roster 73.5%, among the three worst); T6m from
  `TEAM_REVIEW_T6.md` (Torkoal Earth Power: 2x on Tyranitar, Excadrill, Archaludon, and
  single-target, so Wide Guard cannot block it) aims at it, through that review's
  clone team tournament first. The user's call.

## Night summary: T6tac + the review guards 23-17 over 40 ladder games; no brain beat T6tac; nothing promoted (2026-September 28, 06:05)

- **Ladder** `ladder_replays_mc_T6tac_review_guards` (40 serial games, 01:22-05:58):
  **23-17 (57.5%)**; 12 wins by forfeit / inactivity, played out 11-17. The three
  review guards changed 3 ladder decisions in 40 games (all `drop_free_finish`), so
  this read is essentially T6tac's level tonight (its earlier 15-game read: 7-8).
- **Candidates tonight:** the review guards (battery +0.26 [-0.53, +1.03],
  deploy-eligible; they fix the mistakes the user found; ladder above) and three
  tactical fine-tunes from T6tac, none better head-to-head (mirrors 48.25 / 41.2 /
  47.4%; batteries +0.35 / +0.34 / +0.61).
- **For the user (not done tonight):** add `drop_free_finish`,
  `fake_out_partner_acts`, `switch_the_crippled` to DEPLOYED.json `guards_extra`
  (the evidence above); T6tac stays the brain.

## Tactical fine-tune 4 (base lessons only) also fails its mirror -> no ladder; the L1 read continues to 40 games (2026-September 28, 05:10)

- **Tactical 4** (`results_tactical4/tactical_e4.zip`, `--lessons base` on
  `results_tactical3/data`): useless mass 0.095 -> 0.055, agreement 0.76 -> 0.80.
  **Mirror 47.4% [45.2, 49.6] -> loses close games** (open 52.2, hidden 42.6);
  **battery +0.61 [-0.55, +1.81]** (deploy-eligible, worst -1.0). L3 held.
- So the doomed lesson is not the (only) reason: a second tactical round on T6tac's
  own games (tactical 2 / 3 / 4: mirrors 48.25 / 41.2 / 47.4%, batteries +0.35 /
  +0.34 / +0.61) does not beat T6tac head-to-head, while the first round (T6tac vs
  T6ctx: 58.0%) did. The mirrors' hidden-sheet halves are the weak side (40.7 / 45.3
  / 42.6%). Lesson: the first tactical round took the easy gain; more of the same
  lessons from the brain's own play adds little and costs the mirror.
- **Pre-registered now (L1b):** the only configuration that passed its local test
  tonight -- T6tac + the three review guards -- continues its ladder read from 20 to
  40 games in the same directory (`tools/ladder_trial.sh 40
  ladder_replays_mc_T6tac_review_guards`), to give the morning decision (adding the
  three guards to the deployed configuration) more than 20 games. The stop rule
  applies to the whole read (<= 25% wins ends it).

## Tactical fine-tune 3 loses its mirror (41.2%) -> no ladder; tactical fine-tune 4 (base lessons only) pre-registered (2026-September 28, 04:00)

- **Tactical 3** (`results_tactical3/sft/tactical_e4.zip`; 3,200 games from T6tac,
  no teacher errors): validation wasted-move mass in doomed spots 0.805 -> 0.594,
  useless mass 0.095 -> 0.056, agreement 0.76 -> 0.80; the Fake Out pairing lesson
  had 9 validation rows and did not move (0.12 -> 0.12). **Mirror 41.2% [39.1,
  43.4] -> loses** (open 37.3, hidden 45.3); **battery +0.34 [-0.60, +1.31]**
  (deploy-eligible, worst -1.7). L2 held by its pre-registered rule.
- Both brains taught the doomed lesson lost or failed to win their mirror (tactical
  2: 48.25%, tactical 3: 41.2%) while the fine-tune without it (T6tac itself: 58.0%
  vs T6ctx) won; the doomed lesson is the prime suspect.
- **Pre-registered now: tactical fine-tune 4** = the same data
  (`results_tactical3/data`) and recipe from T6tac, but only the base lessons
  (`training/tactical_sft.py --lessons base`: useless actions, attack values with the
  self-drop cost) -> the pick by validation CE -> mirror vs T6tac (2,000) -> battery
  vs `results_brain_ab_tactical1_e4`. **L3** (the L2 rule): a 20-game ladder read
  with the review guards (their local test passed) only if the mirror WINS close
  games (lower > 50%) and the battery is deploy-eligible.

## L1 ladder read (T6tac + the review guards): 11-9, played out 7-9 (2026-September 28, 02:20)

`ladder_replays_mc_T6tac_review_guards` (20 serial games, 01:22-02:13): **11-9
(55%)**, 4 wins by forfeit / inactivity, **played out 7-9** -- the best played-out
record of the recent reads (T6tac 4-8, T6ctx 3-9). But the three guards changed only
3 ladder decisions (all `drop_free_finish`), so this read mostly re-measures T6tac:
a read, not proof. Tactical fine-tune 3 started at 02:14 (the chain's next step).

## The review guards: battery +0.26 (deploy-eligible), mirror inconclusive (never fired) -> L1 ladder read running (2026-September 28, 01:30)

- **Battery** `results_guard_ab_review_guards_0928` (T6tac + the three vs T6tac):
  **pooled +0.26pp [-0.53, +1.03]** -> no clear change, deploy-eligible (human_new
  +0.6, frozen +2.2 [+0.4, +4.3], rotation1 +0.2, rotation2 +0.2, human_previous +0.7,
  heuristic -2.3 [-4.5, -0.2]). Fired in 6,204 games: `drop_free_finish` 226,
  `switch_the_crippled` 59 (56 promoted, 3 built), `fake_out_partner_acts` 21; no
  errors.
- **Mirror** `results_mirror_review_guards_0928`: **48.15% [45.97, 50.34] ->
  inconclusive**, and none of the three fired in its 2,000 games: against our own
  team T6tac's preview almost never brings Venusaur (the only self-dropping attack,
  Leaf Storm) and the other two situations barely arise -- the mirror cannot see
  these guards; the battery is their test.
- By the pre-registered L1 rule (battery deploy-eligible, mirror upper >= 50%) the
  20-game ladder read of T6tac + the three guards started at 01:22
  (`ladder_replays_mc_T6tac_review_guards`).

## Set-swap audit over our own ladder games: two of the four meta tweaks survive (2026-September 28, 00:33)

The user: "can u also check our ladder logs of losses / wins where our inceneroar
wouldve done better with chople berry and do the rest with the other pokemon too".
`results_analysis/openings_20260927/setswap_audit.py` (+ `.json` / `.txt`, every moment
listed) replays the 165 T6-era ladder games through `unit_tests/ladder_position.py`
and the real damage calculator; analysis only.
- **Incineroar Chople vs Passho: a wash, keep Passho.** Chople would have avoided 4
  KOs (+2 maybe) on the first super-effective Fighting hit; Passho did avoid 4 KOs
  (rain Weather Ball, Liquidation) and kept it out of 16-26 HP four more times; all
  of these games were losses either way.
- **Torkoal Earth Power for Heat Wave: modest gain.** Of 298 attacking turns the
  proposed set was better in 22 and worse in 15; a KO only it had in 10 turns vs 4
  (Heat Wave's low-HP spread KOs). Earth Power's KOs: Hisuian Arcanine after its Sash
  (93% hold one), Archaludon, Tyranitar, Incineroar.
- **Venusaur Earth Power for Leaf Storm: a loss, keep Leaf Storm** (KO-only turns 5
  vs 7; Leaf Storm KOs Pelipper / Politoed / Milotic in rain).
- **Charizard Weather Ball for Solar Beam: clear gain** (better in 9 turns, worse in
  1; KO-only 5 vs 1).
- **Farigiraf: keep Helping Hand** -- it decided 10 KOs the partner could not get
  alone (9 in wins).
- `TEAM_REVIEW_T6.md` updated (section 4b); the proposed tournament's **T6m is now
  T6 + Charizard Weather Ball + Torkoal Earth Power** only.

## Pre-registered: tonight's plan under the user's delegation (2026-September 28, 00:25)

The user, going to sleep: "continue training as needed and run ladder whenever u
feel the need to, im sleeping so use your best judgment from here on out". Fixed
now, before the review-guard A/B (running) or anything else reports:
- **DEPLOYED.json stays T6tac**; no promotion tonight (candidates and evidence go to
  the morning report). Ladder only through `tools/ladder_trial.sh`, serial, never
  sharing the machine with a local run.
- **L1 -- the three review guards on T6tac:** a 20-game ladder read
  (`ladder_replays_mc_T6tac_review_guards`) if their battery is deploy-eligible
  (pooled upper >= 0, no population below -3pp) and their mirror does not lose
  (upper >= 50%). If the bundle fails, no ladder for it; a subset is only laddered
  after its own local test.
- **T1 -- tactical fine-tune 3 from T6tac** (the lessons the reviews keep finding,
  taught to the brain, not only guarded): the existing move / damage lessons; the
  doomed lesson with its two ladder-exposed flaws fixed (slot 2 does not count a
  foe our slot 1 Fakes Out; no Protect shift when both our Pokemon are doomed);
  finishing drop-free (a self-dropping attack's value carries a cost per stage
  dropped, so a drop-free knockout wins the tie); and a Fake Out's partner acts
  (slot 2 conditioned on slot 1: a Fake Out beside a Protect moves the Protect /
  Fake Out mass to that Pokemon's other moves). Same recipe as T6tac (positions
  from T6tac, lr 1e-4, 4 epochs, lowest validation CE).
- **L2 -- tactical 3:** a 20-game ladder read (with the review guards if L1's local
  test passed) only if its mirror WINS close games (Wilson lower > 50%) and its
  battery is deploy-eligible.
- **Stop rule:** a read at 5-15 or worse ends ladder play for the night.

## Ladder read of the doomed-lesson brain: 7-8 (played out 2-8); the user's review -> three opt-in guards, their A/B pre-registered (2026-September 28, 00:40)

`ladder_replays_mc_tactical2_e4` (15 games, 23:21-00:04): **7-8, 5 wins by forfeit
or inactivity -> played out 2-8**, the weakest read yet; a Pokemon of ours fainted
before acting in 7 of the 8 losses (2.1 per loss; the read before it 2.8). Not
promoted; T6tac stays deployed.
- **The user's review of one loss (vs s9mmow):** "the bot leaf stormed a low hp
  raichu for no reason killing its spA ... it switched torkoal out but it wouldve
  been much better to switch out venu cuz of sun + it would get its spA back ...
  we used fakeout and then protected with the other mon which makes no sense cuz
  the fake out was for us to attack with the other mon". The decision log
  confirms all three: turn 7 Leaf Storm -> Raichu 9% (p 0.31) with Sludge Bomb ->
  Raichu ranked (0.06), both knockouts, dominated_attack never counts the -2;
  turn 9 Torkoal (7%) switched out while Venusaur (1%, -2 Sp. Atk) stayed to Leaf
  Storm again (no Venusaur switch ranked); turn 10 Torkoal Protect + Incineroar
  Fake Out -> Volcarona (0.41), Heat Wave + Fake Out ranked (0.12), the free Garchomp
  knocked Incineroar out. Turn 8 was a double Protect that bought nothing -- the
  doomed lesson's flaw: it moves attacks to Protect without asking whether the
  Protect buys anything (the partner's Fake Out, a double Protect).
- **Three opt-in guards** (`vgc_bench/src/guards.py`, tests from that game's turns
  in `unit_tests/test_review_guards_0928.py`): `drop_free_finish` (a self-dropping
  attack likely to knock out its targets gives way to a ranked drop-free attack of
  the same Pokemon that knocks them out at least as surely), `fake_out_partner_acts`
  (our Fake Out beside the partner's Protect gives way to the best-ranked pair with
  the same Fake Out user and the partner acting), `switch_the_crippled` (when one of
  ours switches out while the other attacks at -2 or worse in the stat its move uses,
  the crippled one takes the switch and the other acts with its best-ranked
  non-switch action; built if not ranked).
- **Pre-registered A/B** (T6tac, the three together, per-guard firing reported):
  mirror `--guard drop_free_finish,fake_out_partner_acts,switch_the_crippled`
  2,000 games (wins close games if the Wilson lower bound > 50%, loses if the upper
  < 50%); battery `--guards` the same `--without-arm results_brain_ab_tactical1_e4`
  (better if the pooled lower bound > 0; deploy-eligible if the pooled upper >= 0 and
  no population below -3pp; the fixes made since that arm ride along). Deployment:
  the user's call.

## Ladder read of the doomed-lesson brain (the user's word, 2026-September 27, 20:05)

The user: "yes run 15 ladder games with it". 15 serial games with exactly the
tested configuration: brain `results_tactical2/sft/tactical_e4.zip` (sha in its
sidecar, now with `requires_knowledge_obs: true`), the deployed team / preview model /
8 guards / sticky corrections, no playbook -- via `tools/ladder_trial.sh`
(TRIAL_CHECKPOINT) into `ladder_replays_mc_tactical2_e4`; DEPLOYED.json untouched. A
read, not proof (15-game reads of near-identical configurations have ranged 6-9 to
11-4); reported with the forfeit-adjusted record and how often a Pokemon about to be
knocked out Protected.

## Doomed-Pokemon lesson: learned, no clear change locally (deploy-eligible); the mirror splits by sheet mode (2026-September 27, 19:55)

(The 18:55 pre-registration was written and committed before the run started at
18:12 wall-clock; its header time is a typo.)
- **Data** `results_tactical2/data`: 3,200 games by the deployed T6tac on train-split
  rosters, 17,807 decisions, no teacher errors (4 `priority_block_error`, the known
  rare one); doomed positions ~4% of slots (mean chance 0.56).
- **Fine-tune** `results_tactical2/sft/tactical_e4.zip` (from T6tac, lr 1e-4, 4
  epochs; the pick by validation CE 2.019 -> 1.888): validation **mass on wasted
  moves in doomed positions 0.772 -> 0.531** (the lesson is learned, partly);
  useless mass 0.072 -> 0.036, teacher agreement 0.78 -> 0.82, drift on lesson-free
  rows 0.019.
- **Mirror** `results_mirror_tactical2`: **48.25% [46.07, 50.44] -> inconclusive**;
  open sheets 55.8% (wins), hidden sheets **40.7%** (loses).
- **Battery** `results_brain_ab_tactical2_e4`: **pooled +0.35pp [-0.64, +1.37]** ->
  no clear change, deploy-eligible (every population within about 1pp: human_new
  +0.3, frozen -0.3, rotation1 +0.7, rotation2 +0.9, human_previous +0.9,
  heuristic -0.3). By sheet mode the battery is the mirror's opposite: hidden
  +1.06 (all six positive), open -0.35.
- Reading: the brain learned to Protect a Pokemon that is about to be knocked out,
  but locally that neither wins nor loses games measurably. The mirror's
  hidden-sheet loss is the one matchup where the usage-based set guesses (which the
  lesson leans on) fit worst: our own unusual sets (Water Spout Blastoise vs the
  meta's Shell Smash, etc.). Ladder and promotion: the user's call.

## Team and moveset review: T6 vs the Reg M-C meta -- `TEAM_REVIEW_T6.md` (2026-September 27, 19:40)

The user: "look up our team and then meta teams and see how it fares based on data we
could be playing the wrong team ... or maybe even need to tweak our moveset to match
the meta". Analysis only (the doomed chain owns the machine); nothing trained, run or
deployed.
- **Data:** 163 T6-era ladder games (incl. today's two reads), the T6tac battery
  (6,204), 9,947 human Reg M-C games (this morning's 2,247 web replays through 09-27
  added, git-ignored in `battle_logs_web_mc_20260927/`), 7,400+ open sheets,
  Pikalytics sheets, the earlier team tournaments. Scripts / tables:
  `results_analysis/openings_20260927/team_*`.
- **Not the wrong team for the meta in human hands:** human rosters with 4+ of our six
  -0.1pp vs rating (5+: 28-11, +18pp); Farigiraf + Torkoal rosters +4.4pp (126
  players), Mega Blastoise rosters +4.1pp (184); every major archetype within
  -1.0..+1.2pp and teams built like each candidate within -1.9..+3.5pp.
- **But our bot pilots it worse than it piloted T4:** T6 era 78-85, -2.1pp vs rating
  (163 games) vs T4 27-23, +4.8pp (50); losses concentrate on Volcarona 2-11,
  Tyranitar 6-18, Sinistcha 4-11, Garchomp 8-19, Pelipper 7-15, Archaludon 7-13, and
  locally on Politoed rain 62.9%, Hatterene / Camerupt Trick Room 72.7%, sand 73.5%.
- **Sets:** Torkoal Earth Power over Heat Wave, Incineroar Chople Berry over Passho
  (Passho sides -7.7pp over 70 players), Venusaur Earth Power over Leaf Storm,
  Charizard Weather Ball (93% of sheets); Blastoise / Farigiraf are the expert
  pilots' sets -- keep. A set change needs a practice cycle before it can be judged.
- **Proposed (not pre-registered, needs the user's word):** a clone team tournament
  (`evaluation/run_team_tournament.sh`, pilot `results_bc/mc_A_20260920` BEST, pool
  `data/team_weights_regmc.json`, hidden sheets, seed 83): T6 and T6m (the four
  tweaks) 600 games each, T4 and T0-T3, T5 300 each, after the doomed chain; T6 has
  never been in a team tournament.

## Pre-registered: the doomed-Pokemon lesson (tactical fine-tune 2) (2026-September 27, 18:55)

The user chose it ("2"): teach the brain not to waste a turn with a Pokemon that
will be knocked out before it moves. In 165 T6-era ladder games one of our Pokemon
chose an attack and fainted before acting 217 times (191 in losses; 187 with no
Trick Room up; Torkoal 53, Farigiraf 51, Blastoise 51, Incineroar 28, Charizard 22,
Venusaur 12); the brain had ranked Protect among its top pairs in only 50.
- **Teacher** (`training/tactical_teacher.py`: `doomed_probability`,
  `doomed_facts`, `doomed_target`): the chance our Pokemon is knocked out before it
  moves -- a foe that certainly acts first (speed bounds over every Champions build
  of its species, Trick Room / Tailwind / paralysis / weather-speed abilities, or a
  priority move -- Grassy Glide in Grassy Terrain included -- that our Armor Tail
  does not block) has a likely damaging move that knocks it out from its current HP
  (damage-roll share x accuracy x the foe attacking, 0.8 x the move aimed at us, 1.0
  spread / 0.6 single-target). When the Pokemon has a legal Protect it did not use
  last turn and the chance is >= 0.25, that share of its wasted moves' mass moves
  onto Protect (on top of the existing move and damage lessons). Switching out is
  not taught (whether the incoming Pokemon survives is far less certain), so the
  lesson reaches Torkoal, Charizard and Venusaur (87 of the 217 events).
- **Run:** positions from the deployed T6tac on train-split rosters
  (`training/gen_tactical_data.py --games-per-cell 400 --output
  results_tactical2/data`, 3,200 games), fine-tune from
  `results_deployed/champion_mc_T6tac.zip` (never modified) exactly as T6tac was
  made (lr 1e-4, 4 epochs, tau 0.1); the pick = the epoch with the lowest validation
  cross-entropy.
- **Readings:** (1) mirror, the pick vs the deployed T6tac setup, 2,000 games: wins
  close games if the Wilson lower bound > 50%, loses if the upper bound < 50%. (2)
  battery vs `results_brain_ab_tactical1_e4`: better if the pooled lower bound > 0;
  deploy-eligible if the pooled upper bound >= 0 and no population below -3pp. (3)
  Diagnostic: the validation `doomed_wasted_mass` before vs after (the lesson was
  learned). Promotion and ladder: the user's word only.

## No-script test: the practised brain's own opening is WORSE -> no ladder; the turn-1 script helps it (2026-September 27, 18:05)

The practised brain + Water Room WITHOUT `playbook_opening` (it plays its practised
Mega Water Spout + Trick Room):
- **Mirror** `results_mirror_practice1_water_noscript`: **44.5% [42.3, 46.6] ->
  loses** (open sheets 35.0%, hidden 53.9%).
- **Battery** `results_brain_ab_practice1_23101440_tactical_water_noscript`: **pooled
  -1.81pp [-3.93, +0.66]**; human_new -1.8, frozen +0.5, rotation1 -2.6, rotation2
  -1.0, human_previous +0.1, **heuristic -6.0 [-9.8, -2.4]** -> does not hold up (a
  population below -3pp): **the user's condition failed, no ladder** (automatic hold
  17:58).
- Against the same brain with the script (-0.06 [-2.68, +2.63]) the script is worth
  about **+1.75pp** locally: my 16:55 hypothesis (the script hurt the practised brain
  on ladder) was wrong -- the practised turn 1 is the weaker one, and the 6-9 ladder
  read looks like noise around an even configuration.
- Where the playbook stands: practice made its openings playable (+3.64pp over
  unpractised), and the practised brain + Water Room + the turn-1 script is even with
  T6tac locally; nothing in the playbook beats T6tac + the human-trained preview yet.
  T6tac stays deployed. The steadiest ladder loss pattern today: in all 9 losses of
  the 15-game read one of our Pokemon fainted before it acted (slow Pokemon outside
  or at the end of Trick Room).

## Pre-registered before the no-script battery result: the user's ladder condition (2026-September 27, 17:00)

The user: "if it holds up run 15 ladder games". Holds up = the same bar on
`results_brain_ab_practice1_23101440_tactical_water_noscript`: pooled upper bound >=
0, no population below -3pp, no playbook errors in more than 1% of games (and the
arm really ran without `playbook_opening`). Then 15 serial ladder games with exactly
that configuration: brain `results_brainv1_t6_playbook1/tactical/tactical_e4.zip`,
the deployed team / preview model / 8 guards / sticky corrections, playbook
`data/playbook_t6_trial.json` (Water Room only), NO turn-1 script -- via
`tools/ladder_trial.sh` into `ladder_replays_mc_practice1_water_noscript_t6`;
DEPLOYED.json untouched. If it does not hold up: no ladder, report.

## Ladder: the practised brain + Water Room 6-9 (played out 3-9); the turn-1 script overrode its practised opening -> test without the script (2026-September 27, 16:55)

`ladder_replays_mc_practice1_water_t6` (15 serial games, 15:56-16:44, the user's
pre-registered condition): **6-9**; 3 wins by forfeit / inactivity, played out
**3-9** (T6tac 7-8 / 4-8; last night's T6tac + Water Room 11-4 / 4-4). 15 games are a
read, not proof, but two findings stand:
- **The turn-1 script fought the practised brain:** it changed turn 1 in 13 of 15
  ladder games (T6tac last night: 7 of 15) and in 5,273 of the 6,204 battery games
  (85%; 3,866 injected = the brain did not even rank the scripted pair). The
  practised brain learned **Mega Water Spout + Trick Room** as its Water Room turn 1
  (its top pick in 9 of the 13, p 0.41-0.83); training has no guards, so the
  script's Fake Out was never practised. The practised brain's own opening has not
  been tested yet.
- Every one of the 9 losses had one of our Pokemon faint before it acted (2.8 per
  loss; wins 0): the slow team outside or at the end of Trick Room, as before.
- **Pre-registered now (local only):** the practised brain + Water Room WITHOUT
  `playbook_opening`: mirror (2,000 vs the deployed T6tac setup; wins close games if
  the Wilson lower bound > 50%) and battery vs `results_brain_ab_tactical1_e4` (holds
  up = pooled upper >= 0 and no population below -3pp; better if the lower bound > 0);
  descriptive: the script's effect on the practised brain = this arm minus the
  with-script arm on the same cells. Ladder only on the user's word; T6tac stays
  deployed.

## The practised brain + Water Room only HOLDS UP (even with T6tac) -> 15 ladder games running (2026-September 27, 16:00)

`results_brain_ab_practice1_23101440_tactical_water` (the practised re-fit +
`data/playbook_t6_trial.json` + `playbook_opening` vs `results_brain_ab_tactical1_e4`):
**pooled -0.06pp [-2.68, +2.63]**; human_new -1.0, frozen -0.4, rotation1 +1.1,
rotation2 +0.8, human_previous +2.0, heuristic -2.9 (the worst, inside the -3pp
bar); Water Room chosen in all 6,204 games, no errors -> deploy-eligible, not better.
With the mirror (the practised brain + playbook beat T6tac 54.4%; its Water Room
blocks 54.5% / 54.4% for the two saves) it is the first playbook configuration that
is at least even locally. By the user's pre-registered condition the 15 ladder
games started automatically at 15:56 (`ladder_replays_mc_practice1_water_t6`).

## Pre-registered before the Water-Room-only battery result: the user's ladder condition (2026-September 27, 15:30)

The user: "yes if it holds up run 15 ladder games". Holds up = the same bar --
pooled upper bound >= 0 and no population below -3pp, and no playbook errors in
more than 1% of games -- on `results_brain_ab_practice1_23101440_tactical_water`.
Then 15 serial ladder games with exactly the tested configuration: brain
`results_brainv1_t6_playbook1/tactical/tactical_e4.zip` (sha b0ad945b), the
deployed team / preview model / 8 guards / sticky corrections, playbook
`data/playbook_t6_trial.json` (Water Room only) + `playbook_opening`, via
`tools/ladder_trial.sh` into `ladder_replays_mc_practice1_water_t6`; DEPLOYED.json
untouched. If it does not hold up: no ladder, report.

## Practice cycle result: practice helps (+3.6pp) and wins the mirror, but the full playbook is still worse than T6tac's own preview -> no ladder; Water Room only re-tested (2026-September 27, 15:20)

`results_brain_ab_practice1_23101440_tactical` (the practised brain's re-fit +
playbook draft 1.1 + `playbook_opening` vs `results_brain_ab_tactical1_e4`):
- **Pooled -4.38pp [-7.38, -1.42]** -> not better, NOT deploy-eligible (every
  population negative: human_new -1.9, frozen -4.1, rotation1 -4.1, rotation2 -7.0,
  human_previous -5.3, heuristic -4.0) -> **the user's condition ("if the battery
  holds up run 15 ladder games") failed: no ladder** (LADDER_HOLD 15:10, automatic).
- **Practice effect** (vs last night's unpractised playbook arm, same cells):
  **+3.64pp [+1.93, +5.48]**; mirror 54.4% [52.2, 56.6] (wins; unpractised 30.6%).
- **Per card** (T6tac / unpractised / practised): Water Room 90.9 / 88.1 / 90.5%
  (**-0.3**); Sun Room 96.0 / 90.8 / 91.8 (-4.1); Support Room 92.7 / 78.1 / 84.0
  (**-8.6**). Practice closed most of the gap for Water Room, part of it for Support
  Room: against these held-out teams, our rule-picked Support Room and Sun Room
  openings stay worse than the human-trained preview's choice (there mostly Blastoise
  + Farigiraf), even practised. The mirror (our own team) hides this.
- Lesson: the human-trained preview picks better fours than our card RULES; the
  playbook's value so far is Water Room (even), the turn-1 facts (Fake Out
  reachability: Indeedee leads 4-0 on ladder) and the explanation layer. Draft 2
  (the openings research) should re-key Support Room before it is practised again.
- **Pre-registered now (the machine is idle; local only):** the practised brain +
  Water Room only (`data/playbook_t6_trial.json`, last night's ladder-trial
  playbook) + `playbook_opening`, battery vs `results_brain_ab_tactical1_e4`:
  holds up = pooled upper >= 0 and no population below -3pp. Caveat: Water Room was
  chosen after seeing these rosters' per-card results (its own cells -0.3); the
  new information is Water Room on the other 26 rosters. Ladder only on the user's
  word.

## Practice cycle: both mirrors WIN; the user's ladder condition pre-registered before the battery result (2026-September 27, 14:35)

- Training (11:21-13:19) clean: ep_rew_mean 0.48 -> 0.80; practice previews Support
  Room 47%, Water Room 38%, Sun Room 16% (with the 15% exploration), no errors.
- **Mirrors** (candidate + playbook draft 1.1 + `playbook_opening` vs the deployed
  T6tac setup, 2,000 each): RL save 23,101,440 **53.75% [51.6, 55.9]**, its
  tactical re-fit **54.4% [52.2, 56.6]** -- both win close games (last night's
  unpractised brain with the same playbook: 30.6%; its Support Room blocks 13.6%,
  now 54.0 / 53.4%). The re-fit is the battery pick (pre-registered rule).
- **The user, while the battery runs: "if the battery holds up run 15 ladder games
  with it".** Fixed now: "holds up" = the pre-registered deploy-eligible bar --
  pooled upper bound >= 0 and no population below -3pp -- plus no playbook errors in
  more than 1% of games. Then 15 serial ladder games with the tested
  configuration: brain `results_brainv1_t6_playbook1/tactical/tactical_e4.zip`
  (sha b0ad945b; its sidecar now carries `requires_knowledge_obs: true`), the
  deployed team / preview model / 8 guards / sticky corrections, the playbook
  (draft 1.1) and `playbook_opening`, except that a card whose battery cells lose
  >= 5pp over >= 500 games is switched off (last night's rule). Via
  `tools/ladder_trial.sh` (new `TRIAL_CHECKPOINT`), replay dir
  `ladder_replays_mc_practice1_playbook_t6`; DEPLOYED.json untouched. If it does not
  hold up: no ladder, report.

## Pre-registered: the playbook practice cycle (2026-September 27, 11:40)

The user, after last night's report: "start the practice cycle". Fixed now, before
any training:
- **Training** (`training/run_t6_playbook_trial.py` + `training/playbook_preview.py`,
  new): from T6tac (`results_tactical1/sft/tactical_e4.zip`, the deployed sha)
  +983,040 steps (two saves) on the recipe that made T6ctx (35% human-clone games,
  human-model previews for the opponents, 50% hidden sheets, held-out rosters
  zeroed) with ONE change: our previews come from the playbook (draft 1.1: the card
  its rules pick for the opponent, else with p = 0.15 another non-experimental card,
  so every opening is practised). No roster weighting. Then the tactical re-fit of
  the final save (same data and settings as before; the lowest validation CE).
- **Pick:** whichever of the two wins more of its mirror (candidate + playbook
  draft 1.1 + `playbook_opening` vs the deployed T6tac setup, 2,000 games each).
- **Readings:** (1) the pick's mirror: wins close games if the Wilson lower bound
  > 50%, loses if the upper bound < 50%. (2) Battery, the pick + playbook +
  `playbook_opening` vs `results_brain_ab_tactical1_e4` (T6tac with its own
  preview): better if the pooled lower bound > 0; deploy-eligible if the pooled
  upper bound >= 0 and no population is below -3pp; per-card breakdown (a card
  counts as practised if its cells are no longer below -3pp). (3) Descriptive: the
  practice effect = this arm minus last night's untrained playbook arm (-8.03
  [-11.04, -4.96]) on the same cells; the two guard fixes made after that battery
  (`severe_attack_drop_switch` vs Fake Out, the script keeps the card's Mega) ride
  along in the new arm.
- No ladder and no promotion without the user's word. About 4.5 h (training ~2 h,
  re-fit, two mirrors ~1 h, battery ~50 min).

## Playbook ladder trial: 11-4 (played out 4-4); review of every game; two fixes (2026-September 27, 03:45)

`ladder_replays_mc_deployed_T6tac_playbook_t6` (15 serial games, 02:50-03:22; the
verified T6tac configuration + `data/playbook_t6_trial.json` = Water Room for every
opponent after the per-card rule + `playbook_opening`): **11-4 (73%)**. A read, not
proof (vs T6tac 7-8, T6ctx 6-9 in the same size of read; 11/15 vs 7/15 is not a
significant difference). 7 of the 11 wins are opponent forfeits (turn 0 once,
turns 3-5 six times); played-out games **4-4** -- the best played-out record of the
recent reads (T6tac 4-8, T6ctx 3-9, wide guard 4-7, Throat Chop 3-7; opponents
forfeited 3-5 times in each). No errors; every game logged its card and reasons.
- **Turn-1 script** (`plan_audit.py`): changed turn 1 in 7 games (5-2): Fake Out
  on their Tailwind setter instead of an attack (G1 W, G2 W, G8 L -- G8 gave up a
  4x Ice Beam on Mega Salamence for a 7% Fake Out), a different Fake Out target
  (G4 L: Golisopod, the threat to Farigiraf, while Rotom's Volt Switch took 65%
  from Blastoise; G13 W; G15 W), and once a Fake Out without the card's Mega (G12
  W). Against Indeedee / Psychic Terrain leads (4 games) no Fake Out could land, so
  the policy played Water Spout + Trick Room: **4-0** (our history there 5-11).
- **Losses:** G6 -- a HARD-guard bug: after a double Intimidate,
  `severe_attack_drop_switch` read Fake Out as a crippled physical attack and
  switched Mega Blastoise out for Torkoal on turn 1 (p=0.00); Torkoal and Farigiraf
  fell on turn 2. G4 -- Water Pulse into Rotom (resisted) and Rain Dance with
  Blastoise at 35%. G8 -- three turns of attacks into Protect while Trick Room ran
  out. G9 (sand) -- the expert turn-2 Rain Dance, then two critical hits and a lost
  Sableye endgame. Mistake flags as in the T6tac read (fewer attacks into Protect in
  wins: 1.18 vs 2.14 per win).
- **Fixed (tests from the real positions):** `severe_attack_drop_switch` never
  counts Fake Out (a FIRST_TURN_ONLY move) as a crippled attack -- on ladder it
  fired twice in all T6-era logs, once here; this changes the DEPLOYED guard stack
  in exactly that case (the user may want it measured first). `playbook_opening`
  promotes a ranked pair only if it carries the card's Mega; otherwise it builds the
  Mega pair (legality-checked).
- **For draft 2 (not made):** the Fake Out target order from the openings research
  (Tailwind setter -> screens -> a reachable Trick Room setter -> Tyranitar -> their
  Mega / fastest attacker), and never trade a likely KO (4x Ice Beam) for Fake Out.

## Playbook evaluation: the planner's own openings LOSE locally, its turn-1 script is about neutral; the ladder trial plays Water Room only (2026-September 27, 03:00)

Pre-registered readings (00:55 / 01:20 / 01:50 / 01:55):
- **Mirror** (`results_mirror_playbook_t6`): **30.6% [28.6, 32.6] -> loses** (the
  pre-registered verdict). Open sheets (Support Room vs our own team) 13.6%;
  hidden sheets (Water Room) 47.5% [44.4, 50.6] -> no hold (01:50 amendment).
- **Battery** (`results_guard_ab_playbook_playbook_t6_playbook_opening` vs
  `results_brain_ab_tactical1_e4`): **pooled -8.03pp [-11.04, -4.96]** -> not
  better, not deploy-eligible; every population negative (human_new -5.5, frozen
  -11.0, rotation1 -9.7, rotation2 -13.3, human_previous -3.4, heuristic -5.2). No
  errors: a card in all 6,204 games, the turn-1 script changed turn 1 in 3,983.
- **Per card** (same roster x sheet cells, T6tac -> playbook): Support Room 92.7%
  -> 78.1% (**-14.6pp**, 2,574 games), Sun Room 96.0 -> 90.8 (-5.2, 990), Water
  Room -2.7 (2,640). The pre-registered per-card rule switched Support Room and Sun
  Room off for the ladder trial (`data/playbook_t6_trial.json`: the same cards
  with `experimental: true` on those two).
- **Where the loss is** (descriptive, `playbook_decompose.py`): in the Water Room
  cells where T6tac's own preview already played the card's exact lead + four,
  the arms differ only by the turn-1 script (and the Fake Out fix): **-1.1pp over
  924 games, no clear effect**, although the script changed turn 1 in 61% of Water
  Room games. Where the card's plan differs from T6tac's own: -3.6 (Water Room's
  fixed back line), -14.6 (Support Room; T6tac itself led Incineroar + Farigiraf in
  36 of those 234 cells), -5.2 (Sun Room). **The loss is the plan choice, not the
  script:** rule-picked fours and leads lose to the human-trained preview this brain
  practised with -- the unpractised-opening collapse again (2026-09-23: Farigiraf +
  Incineroar 62% where the practised lead scored 90%+; openings research section 8).
- **Ladder trial** (the user's word; hold checks passed: no errors, hidden-sheet
  mirror upper 50.6% >= 40%, battery upper -4.96 >= -5): started 02:50 with Water
  Room for every opponent + the turn-1 script, into
  `ladder_replays_mc_deployed_T6tac_playbook_t6`.
- **Next (the user's call):** a plan has to be PRACTISED before it is played -- a
  training cycle whose preview for our side comes from the playbook (the way
  `training/human_preview.py` sampled the human model), then these same tests;
  draft 2 from `OPENINGS_RESEARCH_T6.md` (Support Room keyed to physical threats or
  a Psychic attacker, not to two Fake Out users).

## Amended mid-mirror, before its hidden-sheet blocks: the mirror's ladder hold now reads the Water Room blocks only (2026-September 27, 01:50)

The mirror's two open-sheet blocks: **A 13.8% and 13.4%** (136 / 1,000). An
8-game diagnostic of side A with a decision log (port 7611) shows **no bug**:
every game Support Room is chosen by its rule ("fake out users 2 (blastoise,
incineroar)") against our own T6 -- a special-attacking team with 1 physical
threat whose Farigiraf's Armor Tail blocks every Fake Out -- exactly the case
`PLAYBOOK_T6.md` lists under Support Room's "Beaten by: special attackers
(Intimidate does nothing)". The turn-1 script correctly drops the unreachable Fake
Out (0 changed actions); Incineroar + Farigiraf simply lose to Blastoise +
Farigiraf here. On our 135 T6-era ladder opponents draft 1.1 picks Water Room 75,
Support Room 44, Sun Room 16, and only 2 of the 44 Support Room picks face fewer
than 3 physical threats.
- **Amendment (made before the hidden-sheet blocks, the battery and the per-card
  breakdown exist):** hold rule (b) was a bug detector; the bug question is
  answered, so the mirror can hold the ladder only through its hidden-sheet blocks
  (Water Room, what 56% of ladder games get): hold if their upper bound < 40%.
  The open-sheet blocks are reported, not a hold. The battery clauses (pooled
  upper < -5pp; errors) and the per-card rule are unchanged -- Support Room plays
  on ladder only if its battery cells (21 of the 47 held-out rosters) do not lose
  >= 5pp. The pre-registered mirror verdict is read on all 2,000 games as fixed.
- For the morning (a tweak, not made tonight): Support Room's rule should need
  physical threats (or avoid special teams / an Armor Tail side), not just two
  Fake Out users.

## Playbook wired in; draft 1.1; its evaluation and the ladder trial running (2026-September 27, 01:55)

- `PolicyPlayer(playbook_path=...)`: at team preview the planner picks the card for
  this opponent (our four, our leads, each Pokemon's job, the turn-1 script, the
  reasons -- now also the cards it passed over and why, e.g. "not sun_room:
  tailwind (whimsicott)"); the preview model keeps predicting THEIR plan. The card
  is attached to the battle for the opt-in guard `playbook_opening` (registered
  FIRST in the guard order) and written to the decision log as a turn-0 row. A
  playbook failure counts `playbook_error:<type>` and falls back to the normal
  preview.
- `resisted_target` no longer retargets Fake Out (T6tac ladder game 8 turn 1; two
  earlier reviews); both sides wherever the guard is on.
- **Draft 1.1, amended before any playbook result** (the openings-research
  session's flags, 01:40): Sun Room also avoids Tailwind teams, as the approved
  text's Plan A already said (the JSON had missed it: 18 of our 25 ladder Tailwind
  teams got Sun Room, a lead the current brain has not played in 84 ladder games);
  turn-1 Fake Out only at a foe it can reach (not a Ghost, not grounded under
  Psychic Terrain, no Armor Tail / Dazzling / Queenly Majesty on their side -- the
  priority_block guard's facts), else no Fake Out step; a scripted action must be
  legal in the live request (a Mega the request forbids is dropped). Held-out card
  mix (hidden sheets): Water Room 21, Support Room 21, Sun Room 5 (draft 1:
  17 / 10 / 20). Mirror: Water Room with hidden sheets, Support Room with open ones
  (our own T6 sheet shows two Fake Out users).
- **Added to the ladder-trial pre-registration (before any result):** the battery's
  per-card breakdown (playbook arm vs T6tac arm on the same roster x sheet cells,
  six populations pooled); a card whose cells lose >= 5pp over >= 500 games is
  marked experimental for the ladder trial only (its opponents fall to the next
  card; the default card never is) -- the known failure mode is an unpractised lead
  collapsing (2026-09-23: sun -23, Incineroar -33). The local verdicts read draft
  1.1 unchanged.
- Harness: `learned_preview_study.py --playbook`, `run_guard_ab.py --playbook` (arm
  `playbook_<guards>`; the playbook and its sha in the manifest). Fixed a latent
  `run_guard_ab.py` flaw: a guard arm against a `--without-arm` played the
  REFERENCE study's brain (T6hp), not the deployed one -- harmless while T6hp was
  deployed (every earlier guard run), wrong since T6ctx. It now plays the without
  side's own brain file (sha checked against the deployed brain); `same_study`
  compares the checkpoint path too.
- Ladder: `ladder_ourteam.py --playbook` (its sha in run_config.json; a playbook
  written for another team is refused); `PLAYBOOK` passes through every launcher
  (DEPLOYED.json fields `playbook` + `playbook_sha256`, sha-verified by
  `tools/deployed_config.py`); `tools/ladder_trial.sh` = the verified deployed
  configuration + `TRIAL_PLAYBOOK` / `TRIAL_GUARDS` in its own replay dir,
  DEPLOYED.json untouched.
- `playbook.set_share` no longer crashes on recorded sets with `item: null` (found
  by the openings research; no caller used that path yet).
- Suite 621 passed + the new playbook / opening / preview / harness / launcher
  tests; Ruff and Pyright clean on every changed file (the 8 pre-existing
  `policy_player.py` Pyright errors unchanged).
- Running (chain under caffeinate): mirror (2,000) -> battery -> the pre-registered
  hold check and per-card rule -> 15 ladder games
  (`ladder_replays_mc_deployed_T6tac_playbook_t6`).

## Opening research for T6: `OPENINGS_RESEARCH_T6.md` (2026-September 27, 01:40)

The user: "can you research different openings for our bot to play + first move
strategies that go alongside it or like against which opponents". Research only: no
code, no runs, nothing wired; the playbook trial pre-registered at 01:20 is untouched.
- **Evidence:** 7,432 human Reg M-C games at our rating band (median ~1200), our 134
  T6-era ladder games, two read-only web-research agents (Baltimore Regional Reg M-C,
  Pikalytics sheets, 1,248 top-ladder replays, the replays of thruxy and dksnnfud --
  the two strongest pilots of our exact six -- and Champions write-ups), and the
  Champions engine mod. Scripts and tables: `results_analysis/openings_20260927/`.
- **Default confirmed:** Water Room with the experts' script (turn 1 Mega + Fake Out +
  Trick Room, turn 2 Rain Dance + Water Spout, Torkoal on the first faint under the
  room): ours 35-30 (12-4 vs Trick Room, 8-3 vs Tailwind); humans 10-3 with that turn
  1; thruxy / dksnnfud 10-1 in their uploaded replays.
- **Losses concentrate** in Psychic Terrain (4-8; Indeedee lead 5-11), sand (Tyranitar
  lead 1-9) and rain (8-14); rain + Archaludon has no known winning line for this team
  (the experts went 1-3).
- **Draft-2 proposals (to practise and test, section 8 of the doc):** Support Room vs
  Psychic Terrain teams with a Psychic attacker; vs sand, Incineroar + Venusaur back and
  turn-2 Rain Dance; two new openings (Charizard + Farigiraf 24-15, Charizard +
  Incineroar 23-18 in human games); turn-1 facts for the tactical teacher (Fake Out
  reachability incl. Mold Breaker, Trick Room order and cancellation, Imprison,
  weather order).
- **Flags:** draft 1 picked Sun Room for 39 of our 134 ladder opponents (18 of 25
  Tailwind teams), a lead the brain has not played since the human-opening retraining
  (0 of 84 ladder games) and the strong pilots never lead; `playbook_opening` could aim
  Fake Out at unreachable foes; `playbook.set_share(..., item=...)` crashed on
  `item: null`. All three addressed in draft 1.1 (`c7b76b2`, the playbook session):
  Water 74 / Support 44 / Sun 16 of our 134 opponents now. Still open: Psychic
  Terrain teams get Water Room (13 of 15; research says Support Room). Four
  high-volume mid-ladder accounts (31% of a 15-hour sample) run the Baltimore
  runner-up team (Charizard-Y, Golisopod, Politoed, Archaludon, Grimmsnarl, Farigiraf).

## Rain/sand cycle fails its pre-registered test: no weather router, T6tac stays alone (2026-September 27, 01:30)

The rain/sand-trained save (`results_brainv1_t6_weather1/saves_fp_xt_hs_wt/reg_mc/
seed1/23101440.zip`; its mirror 35.45%, the tactical re-fit's 35.3%, both losing to
T6tac) on the held-out battery against T6tac (`results_brain_ab_weather1_23101440`
vs `results_brain_ab_tactical1_e4`):
- **Pre-registered router reading, the 14 rosters with a rain / sand setter:
  -0.92pp, 95% [-3.03, +1.08]** -- the lower bound is not above 0, so no router;
  the cycle is recorded as failed (the user's idea stays right in principle: the
  specialist simply is not better at its specialty).
- Unflagged 33 rosters -0.44 [-2.23, +1.38]; all 47 -0.58 [-1.97, +0.81]; per
  population human_new -0.5, frozen +0.7, rotation1 -1.4, rotation2 +1.1,
  human_previous -2.5, heuristic -0.9.
- Lesson: two rounds of RL practice with rain / sand rosters doubled did not make
  the brain better against rain / sand even locally, and cost 15 points in the
  mirror. Rain needs a different answer than more of the same practice (the
  openings research: rain + Archaludon has no known winning line for this team).

## Pre-registered: tonight's ladder trial of the playbook (2026-September 27, 01:20)

The user, going to sleep: "test everything out and map it all togehter and try it
on some ladder to see it" -- the explicit word for a ladder trial. Fixed now,
before the rain/sand result and before any playbook result:
- **Order (serial, never sharing the machine):** rain/sand battery -> wire the
  playbook (tests, commit) -> the pre-registered mirror and battery (00:55 entry)
  -> 15 serial ladder games -> review of every game, wins and losses.
- **Configuration:** `tools/ladder_trial.sh` (new): the verified DEPLOYED
  configuration (T6tac, T6, the preview model now only predicting THEIR plan, the
  8 guards, sticky corrections) + `TRIAL_PLAYBOOK=data/playbook_t6.json`
  (draft 1, unchanged) + `TRIAL_GUARDS=playbook_opening`, replay dir
  `ladder_replays_mc_deployed_T6tac_playbook_t6`. DEPLOYED.json is not changed:
  a trial, not a promotion (that stays the user's call).
- **The ladder runs whatever the local verdict**, because the user asked to see
  it and the battery barely moves (T6tac ~92% there), EXCEPT: (a) the playbook
  errors or never fires locally (`playbook_error` / `playbook_opening_error` in
  more than 1% of games, or no `playbook:<card>` counts) -> fix and re-check
  first; (b) a result that looks like a bug rather than a strategy -- mirror
  upper bound < 40% or pooled battery upper bound < -5pp -> hold the ladder and
  report. No card is tweaked before the trial; tweaks go to the morning report.
- **What 15 games can say:** a read, not proof (T6ctx 6-9, T6tac 7-8 in the
  same size of read). Reported: record, card mix and its reasons, whether each
  turn-1 script happened as planned (Fake Out target, Trick Room), and the
  mistake review.
- Weather routing, if its pre-registered reading passes, is built and tested
  separately, never mixed into this trial.

## Playbook planner built; its evaluation pre-registered (2026-September 27, 00:55)

The user: "we need to think of this moreso as a handoff, like a strategy playbook
... our team picker needs to come up with a plan of action as to why its choosing
its specific 4 and why it wants its first two ... we need to be the human picking
and choosing our own strategy not just basing it off of what others did" -- then
"draft them" and, of the draft, "i like them all".
- `PLAYBOOK_T6.md` / `data/playbook_t6.json`: four plan cards (Water Room =
  default, Sun Room only against clean fire-weak teams, Support Room against 4+
  heavy physical attackers or 2+ Fake Out users, Fast Sun kept but experimental),
  each with jobs, turn-1 script, beaten-by and fallback, plus seven rules from our
  mistakes. Grounded in the 135 T6-era ladder games: Farigiraf + Torkoal is 14-10
  against clean teams but 7-19 against sand / Wide Guard / Imprison / Psychic
  Terrain; Blastoise + Farigiraf 54% either way; Charizard + Venusaur 1-6.
- `vgc_bench/src/playbook.py`: opponent features from the Reg M-C set data and the
  type chart (open sheet when shown), the card by rules, its reasons logged.
  `vgc_bench/src/playbook_opening.py`: opt-in guard `playbook_opening` -- on turn 1
  the card's script goes on top (Fake Out their Trick Room / Tailwind setter, else
  the biggest threat to our setter; Farigiraf Trick Room), FIRST in the guard order
  so every factual veto still judges it. 13 tests.
- Wiring (after the running battery): PolicyPlayer `playbook_path` (the card at
  preview; the preview model keeps predicting THEIR plan), the card attached to the
  battle and written to the decision log; `resisted_target` stops retargeting Fake
  Out (its flinch is the point -- T6tac ladder game 8 and two earlier reviews).

**Pre-registered:** (1) mirror `--a-playbook data/playbook_t6.json --guard
playbook_opening`, 2,000 games vs the deployed T6tac setup (both with the Fake Out
fix; the opponent is our own T6, one matchup, so this tests that card and its
script): wins close games if the Wilson lower bound > 50%, loses if the upper
bound < 50%. (2) Held-out battery, T6tac + playbook + playbook_opening against
`results_brain_ab_tactical1_e4` (played before the Fake Out fix, so that fix rides
along): better if the pooled lower bound > 0; deploy-eligible if the pooled upper
bound >= 0 and no population is below -3pp; the card mix over the 47 rosters is
reported. Deployment and ladder are the user's call.

## Pre-registered: weather routing (the user's idea), before the rain/sand battery result (2026-September 27, 00:35)

The rain/sand-trained save LOSES its mirror against T6tac badly (35.4% [33.0,
37.9] after 1,500 games, sun-vs-sun: no rain or sand in the matchup). The user:
"the weather checks really only should apply if the opponent even has weather
it shouldnt overthink those checks without that". So the rain/sand brain is not
a replacement; it could only be a SPECIALIST: **route** each game at team
preview -- the rain/sand brain when the opponent's roster has a rain or sand
setter (a species whose Reg M-C set data, `data/joint_sets_regmc.json`, gives
P(Drizzle or Sand Stream) >= 0.5: Pelipper, Politoed, Tyranitar, Hippowdon, ...;
property-derived, no species list), T6tac otherwise. The router flags 14 of the
47 held-out battery rosters (10 rain, 2 Trick Room, 2 Tailwind categories).
**Reading (fixed now):** from the battery arm the chain plays (the better of the
two weather mirrors) against `results_brain_ab_tactical1_e4` (T6tac), the delta
on the 14 flagged rosters (whole-roster bootstrap). The router is worth building
only if that delta's 95% lower bound is above 0; otherwise the weather cycle is
recorded as failed and T6tac stays alone. The unflagged rosters would play
T6tac unchanged (and the mirror is unflagged: no change there).

## T6tac ladder read: 7-8 (T6ctx 6-9); the guards corrected its own pick 35% of the time vs 45%; rain/sand cycle restarted (2026-September 26, 21:50)

`ladder_replays_mc_deployed_T6tac` (T6tac, 8 guards, learned preview, sticky
corrections on): **7-8** (L L W W L W L L L W L W L W W), rating 1140 -> ~1080
low -> ~1105; no guard or parse errors. Every game of this read and of T6ctx's
was a closed-sheet game (the opponent did not accept open team sheets in time).
- **The fine-tune's target moved on ladder too:** the guards changed the brain's
  own pick in **34 of 98** move decisions (35%) vs 44 of 98 (45%) for T6ctx;
  dominated_attack **8 vs 20**.
- **Sticky corrections fired 0 times** (the reranker changed the pick twice in
  121 decisions): in closed-sheet games the reranker has no switch evidence, so
  the fix -- worth 57.7% in open-sheet mirror games -- rarely has anything to do.
- Mistakes seen (hand review): attacks into the Trick Room last-turn double
  Protect again (games 2, 9; blocked 2.1 per win, 1.9 per loss), where a switch
  to Chlorophyll Venusaur for the post-TR turn was the better line (game 2);
  Water Spout + Eruption both at ~20% HP (game 1 turn 5); `resisted_target`
  retargeting a Fake Out away from Tyranitar (game 8 turn 1 -- the flinch is the
  point; a guard fix, after the training run); losses again to sand/rain/
  Archaludon/Kingambit teams (games 5, 7, 8, 9).
7-8 vs 6-9 is within noise; the 2,000-game mirror (58.0%) remains the strong
evidence. The rain/sand cycle restarted at 21:45 from `tactical_e4` (= T6tac),
its comparisons against T6tac as amended.

## PROMOTED by the user: T6tac (tactical fine-tune) + sticky guard corrections ON; rain/sand cycle paused for the ladder (2026-September 26, 21:00)

The user: "turn on the sticky fix and make tactical official then more ladder".
`results_deployed/champion_mc_T6tac.zip` (new file, sha 05e82218..., copy of
`results_tactical1/sft/tactical_e4.zip`, sidecar role production). DEPLOYED.json:
checkpoint -> T6tac, **`sticky_guard_corrections: true`** (the launchers pass
`--sticky-corrections` to ladder_ourteam.py), guards / team / preview model
unchanged, fresh `replay_tag` **T6tac**; the outgoing T6ctx moves to
`previous_deployed` and `history` with its ladder record (6-9, peak 1282) and is
now an immutable prior champion (seven in all). Evidence recorded: mirror 58.0%
[55.8, 60.2] vs T6ctx, battery -0.40 [-1.61, +0.84] (no clear change), sticky
mirror 53.4% [51.2, 55.5]. Suite: 589 passed.

The rain/sand training cycle (started 20:31 from `tactical_e4`) was stopped at
about 10% so the ladder can run alone (ladder and heavy local runs never share
the machine); its directory is in `_cleanup_2026-09-26/` (nothing deleted). It
restarts from scratch after the ladder. **Pre-registration amended before any
weather result:** its comparisons are now against the DEPLOYED brain T6tac -- the
mirror's side B is the deployed configuration, and the battery's without arm is
`results_brain_ab_tactical1_e4`, which played exactly T6tac (same sha) with the 8
guards -- same readings as before. Next: 15 ladder games with T6tac.

## Sticky guard corrections WIN their mirror: 53.4% [51.2, 55.5]; 59.4 / 56.0% in open-sheet games (2026-September 26, 20:31)

`results_mirror_sticky_corrections_rerankers` (pre-registered 18:25): deployed
T6ctx configuration on both sides WITH the ladder's opponent/tempo reranker;
side A keeps guard corrections the reranker would undo. **A 1,067 / 2,000 =
53.4% [51.2, 55.5] -- wins close games.** By block: open sheets 59.4% (A
challenges) and 56.0%, hidden sheets 51.0% and 47.0%. The fix fired **1.01 and
1.08 times per game with open sheets** and 0.07 / 0.04 with hidden ones -- the
reranker's switch evidence only runs when the opponent's sets are known -- so
the effect sits exactly where the fix acts (open: 577 / 1,000 = 57.7% [54.6,
60.7]); the hidden blocks are close to an A/A comparison. Deploy-eligible under
the mirror rule (2,099 changed actions, upper bound >= 50%, no block below 47%:
the lowest is exactly 47.0%, a block where it fired 18 times). The mistake it
removes is the one found on ladder (6 of 37 corrections undone; Leaf Storm into
Archaludon three turns running). Turning it on for ladder is the user's call;
ladder_ourteam needs a flag and DEPLOYED.json a field (to add at the user's word).

## Tactical fine-tune: wins its mirror 58.0%, battery no clear change (-0.40 [-1.61, +0.84]); guards correct it 31% less (2026-September 26, 20:00)

`tactical_e4` (T6ctx + the tactical fine-tune), against the deployed T6ctx, both
with the 8 deployed guards and the learned preview:
- **Mirror: 58.0% [55.8, 60.2]** over 2,000 games -- wins close games (every
  block > 54%). `results_mirror_tactical1`.
- **Held-out battery: pooled -0.40pp [-1.61, +0.84] -- no clear change**; no
  population below -3pp (human_new -1.3, frozen -1.5, rotation1 0.0, rotation2
  -1.9, human_previous +0.4, heuristic +1.9); rain rosters 0.0, Trick Room +1.7,
  grassy Fake Out -2.0, Tailwind -1.7. `results_brain_ab_tactical1_e4` against the
  new reference arm `results_brain_ab_deployed_T6ctx` (T6ctx + all 8 guards: 93.0 /
  95.3 / 93.0 / 96.0 / 88.5 / 91.1%).
- **What it learned:** in those 6,204 games the guards that only our player runs
  changed its pick **0.98 times per game vs 1.43** for T6ctx (-31%);
  dominated_attack **0.38 vs 0.77 per game** (halved).
Same evidence profile as T6ctx at its promotion (mirror 59.3%, battery no clear
change): deploy-eligible under the mirror rule; promotion is the user's call.
By the pre-registered rule the rain/sand cycle starts from `tactical_e4`.

## Next training cycle pre-registered: rain/sand curriculum (2026-September 26, 18:12)

Why (from the mistake review): across the 120 T6-era ladder games the bot is
56-64 overall but **4-11 against opponents that set rain**, 4-10 / 4-11 / 2-7 /
4-9 when Pelipper / Milotic / Archaludon / Basculegion come in, **6-14 against
sand-setting Tyranitar** -- and the reading mistakes (predictable Protects, a
doomed mon left in, attacks into Protect on Trick Room's last turn) sit in those
games. Against Trick Room rosters it is 20-20, so no Trick Room boost.
`training/run_t6_weather_trial.py`: the recipe that made T6ctx (35% human-clone
games, human-model previews, 50% hidden sheets, reward unchanged) with ONE
change: rosters with a rain or sand setter (Drizzle, Sand Stream, read from the
roster text) weigh x2 -- 28% -> 44% of the training games; held-out evaluation
rosters zeroed. +983,040 steps from the start checkpoint's own step.

**Pre-registered, before the tactical results:** the start is the tactical
fine-tune (`results_tactical1/sft/tactical_e4.zip`, the pre-declared pick: its
validation cross-entropy was lowest) unless its mirror LOSES (upper bound < 50%)
or its battery shows a population below -3pp; otherwise T6ctx. The pick is the
final save; it is judged by a 2,000-game mirror vs T6ctx (same reading as
above) and the held-out battery against `results_brain_ab_deployed_T6ctx`, with
the rain-category delta as the secondary reading. The midpoint save is only
evaluated if the final save fails. Amended 18:40 (before any weather result):
because the tactical fine-tune won its mirror, the same fine-tune (same data and
settings) is re-applied on top of the final save, and BOTH the final save and its
re-fit get a 2,000-game mirror vs T6ctx; the battery plays whichever won more of
its mirror. Runs after the sticky mirror; no ladder, no promotion without the
user.

**Mirror (pre-registered): `tactical_e4` beats T6ctx 58.0% [55.8, 60.2] over
2,000 games -- wins close games** (open 60.0 / 54.2, hidden 60.2 / 57.6 by who
challenges; every block above 54%). `results_mirror_tactical1`. Held-out battery
next (T6ctx baseline arm, then the candidate).

Fine-tune result (offline, validation split, pre-declared metrics): teacher
agreement 64.4% -> 75.8%, probability on certainly-useless actions 16.3% ->
9.6%, drift on lesson-free rows KL 0.034; epoch 4 had the lowest validation
cross-entropy (2.49 -> 1.98) -> candidate `tactical_e4`. Positions: 2,000 local
games, 11,198 decisions, 0 teacher errors.

## Sticky guard corrections (opt-in) + pre-registered mirror with the ladder's rerankers (2026-September 26, 18:25)

From the mistake review above (class 2): on ladder the opponent/tempo reranker
put back pairs a guard had corrected away in 6 of 37 corrections. New per-player
option `sticky_guard_corrections` (policy_player.py, default off): if the
reranker's top pick is exactly the pair the guards corrected away, the guards'
pick goes back on top (`keep_guard_correction`); any other reranker choice
stands, and the predicted-KO survival pick is exempt. `mirror_guard_ab.py`
gained `--rerankers` (both sides run the ladder's opponent/tempo reranker with
the Reg M-C move and switch models, which the local arms otherwise leave out)
and `--a-sticky`. **Pre-registered:** `mirror_guard_ab.py --rerankers --a-sticky
--games 2000` (deployed T6ctx config on both sides): wins close games if the
Wilson lower bound > 50%, loses them if the upper bound < 50%; deploy-eligible
under the mirror rule (>= 20 changed actions, upper bound >= 50%, no block below
47%). Runs after the tactical chain; deployment is the user's decision.

## T6ctx ladder read (6-9) + mistake review of wins AND losses -> tactical fine-tune (2026-September 26, 18:05)

The user: "run 15 ladder games with the new brain, then make your own decision on
how to train it further BASED ON MISTAKES not only losses, evaluate wins and
losses too".

**Ladder read** (`ladder_replays_mc_deployed_T6ctx`, deployed config, mixing off):
**6-9** (W W W L L L W L L W L L L W L), rating 1219 -> **peak 1282** before game 4
(the team's highest) -> ~1150. No guard or parse errors.

**Mistake review.** Every decision of the 15 games reviewed by hand, and the 30
earlier ladder games of today (T6hp, same guards) by two independent reviews
(`agent_review_*.json` in the session scratchpad), plus automatic flags over all
120 T6-era games. What recurs, in wins as much as losses:
1. **Wrong attack or target** (by damage and matchup) -- the largest class: 22 of
   40 mistakes in one review (11 in wins, 11 in losses), 8 of 23 in the other.
   The network's own pick was changed by a guard in **44 of 98** move decisions
   of this read (45%) vs 56 of 118 for T6hp (47%): the human-opponent cycle did
   not teach it (Heat Wave over a full-HP Eruption, Fake Out into Psychic
   Terrain with ~95% of its mass, Leaf Storm into a 4x resist, Water Spout at low
   HP, Solar Beam ignored in sun; Rain Dance clicked into active rain 3 times).
2. **The guard patch leaks on ladder**: the opponent/tempo reranker that runs
   after the guards (ladder only; local evaluation runs without it) undid **6 of
   37** guard corrections this read (Leaf Storm back into Archaludon three turns
   running, game 6), 3/32 and 4/56 in the earlier reads -- a promoted pair only
   ties the original's probability and the reranker's tactical terms ignore base
   damage. And guards choose among the network's top eight: when the right move
   has ~0% (Weather Ball KO, game 5 turn 4), the fallback is wrong.
3. Reading mistakes with no exact teacher: a doomed slow mon left in (7 of 23 in
   one review), attacks into predictable Protects (alternating-Protect
   Tyranitar, the Trick Room last-turn double Protect; blocked 1.2 per win vs 2.2
   per loss), our Trick Room cancelling the opponent's on turn 1 (games 4, 11).

**Decision: teach the network the move and damage facts the guards patch
(tactical fine-tune), then practise the reading mistakes by RL if it holds.**
Class 1 is the most frequent, habitual (as common in wins), untouched by the
last RL cycle, and has an exact teacher -- the damage calculator and mechanics
the validated guards already use -- while the guard patch is leaky (class 2).
`training/tactical_teacher.py`: certainly-useless actions (a damaging move every
target is immune to, priority into Psychic Terrain / Armor Tail, Fake Out after
turn 1, status into immunity, weather already up, Helping Hand without a
partner) and the value of every plain attack of the normal band (accuracy x
damage capped at the HP left after the partner's attack + 0.5 x P(KO), Solar
Beam counted in sun, spread moves into a shown Wide Guard at 15%). The target is
the brain's OWN distribution: useless mass removed, the mass on attacks re-spread
over them by value (softmax, tau 0.1); attack vs Protect vs switch vs support and
Mega timing keep the brain's proportions. `training/gen_tactical_data.py`: the
deployed bot plays 2,000 local games against training-role opponents only (the
two Reg M-C human clones of the last cycle, T6hp, itself; train-split rosters,
open and hidden sheets) and records what the network saw at every move decision.
`training/tactical_sft.py`: actor-only fine-tune of T6ctx (critic frozen: it has
its own extractor), slot 2 conditioned on slot 1's played action, lr 1e-4, 4
epochs, 10% validation split by battle; saves are new files under
`results_tactical1/sft/` (T6ctx is never modified).

**Pre-registered (before any result):** the candidate is the epoch with the
lowest validation cross-entropy. (1) Offline, on the validation split: teacher
agreement and the probability left on useless actions, before vs after.
(2) Head-to-head mirror, candidate vs T6ctx, both with the 8 deployed guards and
the learned preview, 2,000 games: wins close games if the Wilson lower bound
> 50%, loses them if the upper bound < 50%, else inconclusive. (3) Held-out
battery against a NEW reference arm of the deployed configuration
(`run_guard_ab.py --baseline` -> `results_brain_ab_deployed_T6ctx`, T6ctx + all
8 guards; this is also the T6ctx reference arm future guard A/Bs need): better if
the pooled lower bound > 0, a regression flag if any population is below -3pp.
Promotion is the user's decision; no ladder without the user's word.
`run_guard_ab.py` gained `--baseline` and accepts a without arm that played the
deployed brain (the old reference is T6hp, no longer deployed).

## PROMOTED by the user: T6ctx (save 22,118,400 of the human-opponent cycle) is the deployed brain (2026-September 26, 16:57)

The user: "make the new brain official". `results_deployed/champion_mc_T6ctx.zip`
(new file, sha 5f20cecf..., copy of
`results_brainv1_t6_contexts1/saves_fp_xt_hs_wt/reg_mc/seed1/22118400.zip`,
metadata sidecar role production). DEPLOYED.json: checkpoint -> T6ctx, team,
preview model and the 8 guards unchanged, fresh `replay_tag` **T6ctx**; the
outgoing T6hp moves to `previous_deployed` (with its amendments and its ladder
record, 29-26 over 55 games, peak 1239) and `history`. Evidence recorded:
head-to-head 59.3% [57.1, 61.4] vs T6hp with every deployed guard; held-out
battery +1.27 [-0.16, +2.68] (no clear change) -- the user's decision, no gate
passed on its own. T6hp is now an immutable prior champion (six in all). Suite:
569 passed. Follow-up for the next guard A/B: `run_guard_ab.py` checks its
reference arm against the deployed brain, so guard A/Bs on T6ctx need a T6ctx
reference arm (e.g. `results_brain_ab_contexts1_22118400`, T6ctx + dominated_attack).

## Ladder read with the 8-guard stack: 15 games, 8-7, rating 1175 -> peak 1239 -> ~1218 (2026-September 26, 12:40)

`ladder_replays_mc_deployed_T6_humanpreview1_wideguard` (T6hp brain, learned
preview, guards resisted_target, overkill_split, dominated_weather_ball_weather,
dominated_attack, dominated_throat_chop, dominated_spread, focus_boosted,
wide_guard; mixing off): **8-7** (L W W L W L W W L W L W L L W), rating 1175 at
the start -> **peak 1239** (before game 13) -> ~1218 after game 15, the highest
this team has reached. The guards changed the pick in 72 of 138 logged decisions
(dominated_attack 27, resisted_target 20, dominated_spread 10, overkill_split 7,
guaranteed_ko 3, wide_guard 2, Throat Chop 1, ...); no guard errors, no parse
errors. Loss contexts: attacks into Protect in 5/7 losses vs 5/8 wins (the gap
narrowed); into a shown Wide Guard in 1/7 losses (wide_guard fired twice);
low-HP Water Spout / Eruption in 2/7 (dominated_spread fired 10 times); Trick
Room's last turn stalled with Protect in 4/7 losses vs 2/8 wins. Pending the
user: whether save 22,118,400 (head-to-head 59.3% vs this brain) becomes the
deployed brain.

## Head-to-head: save 22,118,400 BEATS the deployed brain 59.3% [57.1, 61.4]; ladder read of the deployed configuration next (2026-September 26, 12:01)

`results_mirror_brain_contexts1_22118400`: side A = save 22,118,400 of the
35%-human-opponent cycle, side B = the deployed T6hp brain, both with the 8
deployed guards and the learned preview, T6 vs T6: A **1,186 / 2,000 = 59.3%
[57.1, 61.4]**, blocks 62.6 / 59.8 (open sheets) and 54.4 / 60.4% (hidden) --
by the pre-registered reading it **wins close games**. Against the held-out
battery (mostly opponents the bot already beats ~90%) it was +1.27 [-0.16,
+2.68], no clear change; against an opponent as strong as itself the extra
practice against human-style play shows. It passed no promotion gate, so the
deployed brain is unchanged: promotion is the user's call. The 15 ladder games
now play the deployed configuration as announced (fresh dir
`ladder_replays_mc_deployed_T6_humanpreview1_wideguard`).

## Head-to-head mirror of save 22,118,400 vs the deployed brain pre-registered; then 15 ladder games (2026-September 26, 11:32)

The user: "do the head to head mirror then run 15 ladder games".
`evaluation/mirror_guard_ab.py --a-checkpoint` (new): side A = save 22,118,400 of
`brainv1_t6_contexts1`, side B = the deployed T6hp brain; both with every
deployed guard (8) and the learned preview model, T6 on both sides, 2,000 games
in the four blocks, output `results_mirror_brain_contexts1_22118400`. Reading
(pre-registered now): save 2 **wins close games** if its Wilson 95% lower bound
is above 50%, **loses** them if the upper bound is below 50%, else inconclusive.
It already has "no clear change" on the held-out battery, so it passes no gate
either way: promotion stays the user's decision. The 15 ladder games then play
the DEPLOYED configuration (T6hp + 8 guards, fresh dir
`ladder_replays_mc_deployed_T6_humanpreview1_wideguard`) unless the user says
otherwise.

## Training cycle contexts1: both saves NO CLEAR CHANGE (+1.02 / +1.27 pooled, lower bounds just below 0); not promoted (2026-September 26, 07:37)

Each save vs the deployed brain (`results_brain_ab_contexts1_<save>`, both sides
with dominated_attack, 6 populations x 1,034 games, whole-roster bootstrap):

| Population | 21,626,880 | 22,118,400 |
|---|---:|---:|
| New held-out human clone | -0.1 [-3.5, +3.5] | +2.8 [-0.1, +6.3] |
| Frozen PPO | -1.3 [-5.9, +2.3] | +1.0 [-1.8, +3.7] |
| Rotation 1 | +0.6 [-1.9, +3.1] | -0.2 [-3.4, +3.3] |
| Rotation 2 | +2.9 [-0.2, +6.4] | +1.9 [-1.6, +5.8] |
| Previous held-out human clone | **+3.5 [+0.1, +7.4]** | +2.6 [-0.4, +5.8] |
| Scripted heuristic | +0.5 [-1.5, +2.4] | -0.5 [-2.8, +1.6] |
| **Pooled** | **+1.02 [-0.05, +2.14]** | **+1.27 [-0.16, +2.68]** |

By the pre-registered rule both are **no clear change** (neither worse nor
better; no population below -3pp). The shape is the intended one -- the gains
sit against the held-out human clones and mostly with open sheets (save 2: human
clones +4.3 / +4.4 open, +1.4 / +0.8 hidden) -- but the bar was a pooled lower
bound above zero and it is not met. Not promoted; the deployed brain stays
T6hp. The chain completed 07:36:43.

## Training cycle contexts1 complete; both saves being evaluated against the deployed brain (2026-September 26, 06:10)

`results_brainv1_t6_contexts1`: 04:12 -> 06:10, 21,135,360 -> 22,118,400 at ~140
steps/s, no worker failures, the end-of-run source/data re-check passed. Saves
21,626,880 and 22,118,400. Training telemetry (small samples, sanity only):
ep_rew_mean 0.62 -> 0.82, episode length 9.3 -> 8.7 turns; the per-save quick
evals vs the BC clone 0.83 -> **0.90** and vs the heuristic 0.79 -> 0.83 (the
human-opening cycle read 0.81 / 0.80 and 0.83 / 0.89 at its saves). The
pre-registered comparison (`run_guard_ab.py --candidate`, 6 populations x 1,034
games, both sides with dominated_attack) started 06:10 with 21,626,880;
22,118,400 follows. Promotion only at the user's word.

## focus_boosted + wide_guard DEPLOY-ELIGIBLE (pooled +0.53 [-0.08, +1.19]) -> DEPLOYED; training cycle running (2026-September 26, 04:12)

`results_guard_ab_focus_boosted_wide_guard` (both guards on our side vs the
dominated_attack arm; 6 populations x 1,034 games): human_new -1.2 [-2.8, +0.3],
frozen **+1.9 [+0.6, +3.5]**, rotation1 0.0, rotation2 -0.8, human_previous +2.7
[-0.5, +6.3], heuristic +0.5; **pooled +0.53 [-0.08, +1.19]** -- not worse, no
population below -3pp, not "better". wide_guard changed 679 actions (191 against
the previous human clone), focus_boosted 179 (89 against the new human clone).
By the pre-registered rule both deploy: `guards_extra` += focus_boosted,
wide_guard, fresh `replay_tag` **T6_humanpreview1_wideguard** (607102c).
Deployed stack now: resisted_target, overkill_split,
dominated_weather_ball_weather, dominated_attack, dominated_throat_chop,
dominated_spread, focus_boosted, wide_guard. The training cycle
`brainv1_t6_contexts1` started 04:12:02 (35% human-clone opponents, from T6hp).

## Near-tie mixing LOSES (41.8% [39.6, 43.9]); stays off; focus + Wide Guard A/B running (2026-September 26, 03:29)

`results_mirror_mixing_neartie` (side A samples among its top 3 pairs with >= 15%
of the top pick's probability, weights = policy probabilities, never undoing a
guard or reranker correction; side B the same stack deterministic): A 835 / 2,000
= **41.8% [39.6, 43.9]**, every block 40.4-44.0%; the wheel changed 3,371 picks
(1.69 per game), so each spin cost roughly 5% of the game. Against a copy of
itself the policy's second choice is not a near-equal move even at an 85/15
split: its probabilities rank moves, they are not an equilibrium mix. By the
pre-registered rule mixing stays off (September's opening-only mixing cost 1-3pp;
every-turn near-tie mixing costs ~8pp here). Unpredictability has to come from
training that makes the policy's own mixes worth playing, not from sampling a
policy trained to be played at argmax. The focus_boosted + wide_guard held-out
A/B started 03:28:45.

## dominated_spread WINS CLOSE GAMES (56.4% [54.2, 58.6]) -> DEPLOYED (2026-September 26, 03:02)

`results_mirror_dominated_spread`: A 1,128 / 2,000 = **56.4% [54.2, 58.6]**, blocks
55.4 / 59.2 / 52.8 / 58.2%, 1,424 changed actions (0.71 per game). The user's
point measured: a spread hit whose HP-scaled power (or split damage) has dropped
below one clear single-target hit loses close games. By the pre-registered rule
it is deployed: `guards_extra` += dominated_spread, fresh `replay_tag`
**T6_humanpreview1_spread** (a6f4e3d). The near-tie mixing mirror had already
read the configuration when this changed (started 03:01:50), so it compares
mixing against the stack without dominated_spread on both sides.

## Ladder read with Throat Chop: 15 games, 8-7; loss contexts confirmed (2026-September 26, 02:35)

`ladder_replays_mc_deployed_T6_humanpreview1_throatchop`: **8-7** (W L W L W L L W L L
L W W W W), rating 1144 -> low 1079 (after game 11) -> ~1170; stopped between
games after game 15 at the user's word ("lower the ladder games to like 15"; a
watcher ended the session 0.2 s after game 15's replay, no battle left
unfinished). Throat Chop changed 3 moves, dominated_attack 11, resisted_target 9
of 123 logged decisions; no guard errors. Loss contexts in this read (7 losses
vs 8 wins): our moves blocked by Protect in 6/7 losses (2.9 per loss) vs 3/8
wins; spread moves into a shown Wide Guard in 3/7 losses vs 0/8; Trick Room ran
out in 6/7 vs 3/8, and its last turn was stalled with Protect in 3/7 losses
(double Protect in 2) vs 0/8 wins. The pre-registered chain (5c03f3a) started
02:34:57 with the dominated_spread mirror.

## Pre-registered: Wide Guard, weak spread, near-tie mixing, focus measurements and a human-opponent training cycle (2026-September 26, 02:09)

The user tonight, during the Throat Chop ladder read: "it needs to be taught wide
guard if its gonna use water spout and eruption too" (game 2: Water Spout, then
Eruption, both with Helping Hand, into Aerodactyl's shown Wide Guard); "water
spout and eruption are HP based moves ... sometimes its better to just use the
plain water move or heat wave" and "there are also scenarios where we could be
low hp but the spread attack is better bc one of the opposing mons is really low
on hp"; a mixed strategy ("like 85% chance to click this one move and 15% to
click another ... still isnt a horrible move"); "analyze the losses and train it
on picking correct moves in different contexts"; "lower the ladder games to
like 15" (the run stops itself after game 15, between games).

Built (dcd37fd, c4aa0ca, d68bc6b; 562 tests): `wide_guard` (demote spread attacks
while an able foe has shown Wide Guard, unless our Fake Out surely flinches it
first); `dominated_spread` (with both foes up, a spread attack gives way to a
single-target attack of the same Pokemon worth 1.25x its TOTAL and 0.05 more --
a low-HP spread hit that still finishes one foe and chips the other keeps its
place); near-tie mixing (`mixing_min_ratio`: only pairs with >= that share of the
top pick; `mixing_keep_corrections`: never mix a pick a guard or the reranker
corrected, because a promoted pair inherits the top pick's probability); the
mirror's `--a-mixing`; `run_guard_ab.py --candidate` (another brain vs the
reused deployed arm); `training/run_t6_contexts_trial.py` (prepared: pool built,
sources hashed -- no vgc_bench source may change until it ends).

Loss analysis (81 T6 ladder games with replays; scratch `loss_analysis.py`):
our moves blocked by Protect 11.6% in wins vs **19.5%** in losses; Trick Room ran
out in 9/37 wins vs **28/44** losses (we win inside the room; opponents stall its
last turns -- double Protect on a last room turn only in losses); Water Spout /
Eruption below half HP in 5/37 vs 25/44 (partly a symptom of losing); spread
into Wide Guard 1/37 vs 5/44.

**Measurements, in this order, after the ladder stops (scratch
`post_ladder_chain.sh`):**
1. Mirror `dominated_spread`, 2,000 games; the guard mirrors' reading
   (deploy-eligible: >= 20 changes, upper >= 50%, no block < 47%; better: lower
   > 50%; worse: upper < 50%).
2. Mirror near-tie mixing on side A (always, top 3, T=1, min ratio 0.15,
   corrections kept), 2,000 games: against a deterministic copy it measures what
   sampling COSTS (its benefit is against adaptive players, i.e. ladder).
   Deploy-eligible -> mixing on for the next ladder read (the user asked for
   it); worse -> off.
3. Held-out A/B `focus_boosted` + `wide_guard` (`--without-arm
   results_guard_ab_dominated_attack`), dominated_attack's rule on the pair;
   each guard deploys only if it changed >= 20 actions, otherwise it is
   unmeasured and the user decides.
4. Training cycle `brainv1_t6_contexts1`: from the deployed T6hp brain, +983,040
   steps, saves 21,626,880 and 22,118,400; the ONE change vs the human-opening
   recipe: 35% of games against human-style opponents (two training-role Reg
   M-C clones) instead of 20% -- practice against the Protect, Wide Guard and
   Trick Room stalling that beat us.
5. Each save vs the deployed brain (`run_guard_ab.py --candidate`, both sides
   with dominated_attack, 6 populations x 1,034 games): **better** if the pooled
   95% lower bound > 0 (any population below -3pp flagged); **worse** if the
   upper bound < 0; else no clear change. If both are better, the higher pooled
   point estimate is the pick. Promotion only at the user's word; ladder
   likewise.

## Ladder read with Throat Chop: 25 games started; weather guard stays off (user); focus_boosted A/B after the ladder (2026-September 26, 01:43)

The user: "leave weather off, plugged in now, run 25 ladder games".
`keep_our_weather` stays off by the user's decision. `tools/ladder_deployed.sh 25`
into the fresh dir `ladder_replays_mc_deployed_T6_humanpreview1_throatchop`
(guards_extra = resisted_target, overkill_split, dominated_weather_ball_weather,
dominated_attack, dominated_throat_chop; T6hp brain; learned preview; serial;
credentials sourced shell-side). The AC-gated A/B chain was stopped before its
A/B started (ladder and heavy local runs never share the machine); the
focus_boosted A/B runs after the ladder, unchanged. At launch `pmset` still
reported battery power (58%).

## dominated_throat_chop DEPLOYED (pre-registered rule); combined mirror not worse; focus_boosted A/B waits for AC power (2026-September 26, 00:33)

`results_mirror_ladder_fixes` (Throat Chop + focus_boosted, which never fired in a
T6 mirror, as expected): A 994 / 2,000 = **49.7% [47.5, 51.9]**, blocks 47.6 /
52.0 / 49.6 / 49.6%, 107 Throat Chop changes -> **not worse**. With the first
mirror, Throat Chop over 4,000 games: **50.05% [48.5, 51.6]** -- neutral, about one
changed action per 19 games.

By the rule pre-registered before any game (amended 23:40 before any affected
result), `dominated_throat_chop` is deployed: DEPLOYED.json `guards_extra` +=
dominated_throat_chop, fresh `replay_tag` **T6_humanpreview1_throatchop**
(`amendments` records it, with the guards not deployed and why). Checkpoint,
team and preview model unchanged; `tools/deployed_config.py` resolves it.
Not deployed: `trick_room_counter` (loses close games, 39.1%), `keep_our_weather`
(unmeasured -- 3 changes in 2,000 mirror games; the user decides).
`focus_boosted`'s held-out A/B starts when the MacBook is on AC power
(`fixes_chain3.sh`); DEPLOYED.json is not a pinned input of that A/B, so this
amendment does not affect it. Ladder play still waits for the user.

## Fix mirror 3: keep_our_weather UNMEASURED (3 changes in 2,000 games); combined mirror running; the A/B waits for AC power (2026-September 26, 00:07)

`results_mirror_keep_our_weather`: A 1,027 / 2,000 = 51.4% [49.2, 53.5], but the
guard changed only **3** of A's actions in 2,000 games (blocks 2 / 0 / 1 / 0):
in a T6 mirror our own Rain Dance with Blastoise staying in and a weather setter
coming in almost never happens, so the 51.4% is the mirror's noise. By the
pre-registered rule it is **unmeasured here** (< 20 changes): stays off, the
user decides. So the amended combined mirror runs `dominated_throat_chop` +
`focus_boosted` (inert in a T6 mirror) -- in effect a second 2,000-game sample
of the Throat Chop guard (`results_mirror_ladder_fixes`, started 00:06).

The MacBook has been on battery since the chain began (68% at 00:06, ~2h45m at
this load); the ~2 h focus_boosted A/B would risk dying part-way, so it now
starts only on AC power (house rule for long runs; scratch `fixes_chain3.sh`
waits for `pmset` to report AC). The A/B itself is unchanged.

## Fix mirrors 1-2: Throat Chop deploy-eligible (50.4%); Trick Room counter LOSES (39.1%); combined mirror amended (2026-September 25, 23:40)

`results_mirror_dominated_throat_chop`: A 1,008 / 2,000 = **50.4% [48.2, 52.6]**,
blocks 49.2 / 53.8 / 49.6 / 49.0%, 106 changed actions (0.05 per game) ->
**deploy-eligible** (not "better").

`results_mirror_trick_room_counter`: A 782 / 2,000 = **39.1% [37.0, 41.3]** ->
**the guard loses close games; stays off.** Open sheets 48.2 / 45.2% (it
changed exactly one action per game: their Farigiraf's Trick Room is on the
sheet, so A never pressed its own on turn 1), hidden sheets 30.8 / 32.2%. In a
Trick Room mirror, yielding the room hands the opponent its timing; with hidden
sheets both rooms cancel on turn 1 and the rule then stops A from ever
contesting it. The user's instinct -- contest their room -- is the better rule.

**Amendment (before any keep_our_weather or combined result was read):** the
pre-registered combined check ("the all-four mirror is not worse") would now
mostly measure trick_room_counter, which is already shown to lose and will not
ship. The combined mirror is replaced by one of the guards that could ship:
`dominated_throat_chop`, `focus_boosted` (inert in a T6 mirror) and
`keep_our_weather` only if its own mirror is deploy-eligible or better; same
2,000 games, same reading, output `results_mirror_ladder_fixes`. The chain is
restarted after the running keep_our_weather mirror (unchanged) finishes; the
focus_boosted A/B follows as planned.

## Four ladder-mistake fixes built as opt-in guards; measurements pre-registered (2026-September 25, 22:45)

The user: "yes do all 4 fixes" (Trick Room into a counter, focus a foe that set
up, Throat Chop in the attack check, no weather-overwriting switch-in).

Checked against all 75 T6 ladder games first, two of my claims were overstated:
the Imprison and Taunt that stopped Trick Room in games 11 and 19 came on turn
1, before either was shown (Sneasler Taunt is 0.6% in the M-C prior), and the
bot did not retry; re-setting Trick Room into a foe that had shown it happened
3 times (1 loss, 2 wins). Torkoal replaced our rain beside Blastoise in 3 games
(1 loss, 2 wins). Throat Chop: 12 uses in 6 games, all lost. 11 single-target
attacks went past a foe at +2 or more.

Guards (opt-in, off; `vgc_bench/src/guards.py`, `trick_room_guard.py`):
- `dominated_throat_chop`: dominated_attack's comparison (same 1.25x / +0.05
  margins, runs right after it) with Throat Chop scored as a plain attack unless
  a foe it hits has a sound move (revealed, or >= 0.5 in the set prior), both
  ways. Game 8's Tyranitar is NOT changed: Flare Blitz is 1.5x but only +4.8% of
  its HP into the resisting Mega.
- `focus_boosted`: when exactly one of two foes is ahead with >= +2
  Attack/Sp. Atk/Speed stages (Defense too with a shown Body Press), each foe's
  share of dominated_attack's value is weighted 1 + 0.25 x stages and a ranked
  pair of the same Pokemon attacking the boosted foe is promoted at 1.25x /
  +0.05. The partner-capped HP keeps it from overkilling a boosted foe the
  partner already KOs (game 21's 3% Serperior).
- `keep_our_weather`: demotes switching in (by choice or after a faint) a
  weather setter over OUR weather with >= 2 turns left when the partner staying
  in is clearly better off in it (strongest move >= 1.25x there, or its speed
  ability). `pokeenv_patches` now records which side set the weather and on
  which turn (poke-env forgets the setter and rewrites the start turn at every
  upkeep); nothing else reads it.
- `trick_room_counter`: with Trick Room down, demotes pairs that set it while an
  active foe has shown a counter: Imprison up (Showdown refuses the choice even
  if its user will faint first -- game 11 turn 2, where the policy's top six
  pairs all pressed it), a Taunt that would land, or its own Trick Room (can
  cancel or reverse ours; not in our fast mode -- the user's counter). For Taunt
  / reverser it stands down when our partner's half stops that foe first (Fake
  Out, or a certain KO before it acts), which also promotes Heat Wave + Trick
  Room over Protect + Trick Room.

Replay audit (scratch `guard_audit.py`: 571 logged decisions of the 75 games
rebuilt from their replays with the Champions stats the server sends, deployed
stack vs + guards): Throat Chop 3 changed decisions, focus 3, weather 3 (1 lost
game, 2 won), Trick Room 4; all four together 12 in 11 games (8 lost). Tests on
the real positions + boundaries: 33 new (`unit_tests/test_dominated_throat_chop.py`,
`test_focus_boosted.py`, `test_keep_our_weather.py`, `test_trick_room_counter.py`,
shared `ladder_position.py`); suite 547 passed, 5 skipped. Also: dominated_attack
refactored into a shared core (its 15 tests unchanged and passing);
`mirror_guard_ab.py` takes several guards; `run_guard_ab.py --without-arm` reuses
`results_guard_ab_dominated_attack` as the without side (dominated_attack on
both sides), accepting the `pokeenv_patches` pin change.

**Measurements (pre-registered now, before any game), run in this order:**
1. Mirror matches (`evaluation/mirror_guard_ab.py`, 2,000 games, four blocks
   each): `dominated_throat_chop`, `trick_room_counter`, `keep_our_weather`,
   then all four together. `focus_boosted` gets no single mirror: T6 has no
   boosting move, so a T6 mirror cannot trigger it.
2. Held-out A/B for `focus_boosted`: `run_guard_ab.py --guards focus_boosted
   --without-arm results_guard_ab_dominated_attack`.

Reading. A mirror: "better" (wins close games) if A's Wilson 95% lower bound
> 50%; "deploy-eligible" if the guard changed >= 20 of A's actions, the upper
bound >= 50% and no block's point estimate < 47%; "worse" if the upper bound
< 50%. The A/B: dominated_attack's rule (fires; pooled upper bound >= 0; no
population below -3pp; "better" if the lower bound > 0). A guard is added to
DEPLOYED.json (the user asked for the fixes) if its own measurement makes it
deploy-eligible or better AND the all-four mirror is not "worse"; one that fires
< 20 times is "unmeasured here" and stays off for the user to decide; "worse"
stays off. Fresh replay_tag on any change; ladder play waits for the user.

## Ladder read with the attack check: 11-11 (25-game read 13-12); loss causes (2026-September 25, 21:10)

`ladder_replays_mc_deployed_T6_humanpreview1_attackcheck`: 22 serial games,
**11-11**, no parse errors; rating 1144 -> peak 1227 (after 3-0) -> low 1069 ->
~1140. With the 3 games before the fix (2-1) the user's 25-game read is
**13-12** (T6 before it: 21-29; T4: 27-23). `dominated_attack` changed the pick
in 37 of 181 logged decisions, e.g. Eruption -> Heat Wave at low HP and Leaf
Storm -> Sludge Bomb in game 1 (both won).

Loss causes from the replays (`scratch/ladder_loss_causes.py`; a loss can have
several), share of WINS vs LOSSES showing each:

| Feature | Wins | Losses |
|---|---:|---:|
| Our Trick Room Taunted / Imprisoned / reversed | 0/11 | 4/11 |
| Opponent set up a boost (Shell Smash, Curse, Bulk Up, Swords Dance...) | 1/11 | 4/11 |
| Sun lead (Charizard + Venusaur) into a rain team | 0/11 | 2/11 |
| Tyranitar / sand team | 1/11 | 2/11 |
| Attacked into Protect 2+ times | 4/11 | 7/11 |

Also 5 losses came after being ahead on knockouts (2-on-1 endgames lost in
games 8 and 10). Concrete gaps seen in the replays: Throat Chop is never swapped
by the attack check (its sound-block side effect made it "utility") -- 4 resisted
Throat Chops into a lone Tyranitar in game 8; Torkoal switched in and replaced
our own Rain Dance for Blastoise (game 10); Farigiraf kept re-setting Trick Room
into a reverser (game 5) and into Taunt / Imprison (games 19 / 11); a Shell Smash
Blastoise went unpunished (game 19). Candidates, strongest evidence first: stop
spending turns on Trick Room into a revealed counter; focus a foe that just set
up; Throat Chop as a plain attack unless the target has a sound move; no
weather-overwriting switch-in; the sun lead vs rain as a training target.
Nothing changed on the bot during the read.

## Ladder read resumed with the attack check: 22 games (2026-September 25, 19:48)

The user: "continue our earlier task" (the 25-game read they paused on 09-24 to
fix the move-choice mistakes). `tools/ladder_deployed.sh 22` into the fresh dir
`ladder_replays_mc_deployed_T6_humanpreview1_attackcheck`; its run_config records
`guards_extra` = resisted_target, overkill_split, dominated_weather_ball_weather,
dominated_attack, the learned preview (`data/preview_t6_focus_20260923.pt`) and
the T6hp brain; serial, credentials sourced shell-side. The 3 games before the
fix (2-1) stay in `ladder_replays_mc_deployed_T6_humanpreview1`. Codex idle since
09-22 (heartbeat PAUSED); nothing else running.

## Mirror match: the attack check WINS CLOSE GAMES -- 71.5% [69.5, 73.4] over 2,000 games (2026-September 24, 16:15)

`results_mirror_dominated_attack/`: the deployed bot against itself, side A with
`dominated_attack`, side B without (same brain, team, preview model, other
guards; baseline exactly 50%). A won **1,430 / 2,000 = 71.5%, Wilson 95% [69.5,
73.4]**, no ties; pre-registered reading: **the guard wins close games**. Every
block agrees -- open sheets 69.4% (A challenging) / 68.6% (B challenging),
hidden sheets 75.8% / 72.2% -- so it is not the seat. It changed A's attack 2.69
times per game here (1.1 against the held-out opponents, whose games the bot
mostly wins regardless). This is the user's point measured: "at higher elo ...
each move matters so much" -- against an opponent as strong as the bot, the
weaker-attack mistakes decide most games. The guard is already in the deployed
configuration (50d6699). Ladder still OFF; 22 of the 25 games remain (resume
needs the user's word; fresh replay dir `ladder_replays_mc_deployed_T6_humanpreview1_attackcheck`).

## trick_room_direction HURTS where it acts (-9.8pp); A/B stopped at 3/6 populations; NOT deployed (2026-September 24, 15:12)

Populations: new human clone -1.6 [-4.0, +0.2], frozen +0.8 [-1.4, +3.1],
rotation 1 -2.8 [-4.8, -1.0]. On the held-out Trick Room rosters: -4.1 / +1.8 /
-6.4. Per 11-game cell against the same cell of the reference: cells where the
rule never changed the action -0.5pp (250 cells, noise); cells where it changed
3+ actions **-9.8pp** (25 cells); every sub-rule's cells negative (counter -6.1,
reverse -5.4, no-gift -5.8, keep-ours -5.5). Stopped early (the decision can only
be "do not deploy"); evidence tracked in `results_guard_ab_trick_room_direction/`.

Why (my design, not the user's lesson): T6 is itself a Trick Room team (Torkoal,
the slowest, behind Farigiraf's room). The rule judged "whom Trick Room helps"
from the two ACTIVE Pokemon only, and took every Pokemon of a >= 2-setter roster
at minimum Speed. With Blastoise + Farigiraf out it concluded the room helps
them, then reversed or refused rooms our Torkoal plan needed and cancelled rooms
that were good for us. The user's lesson holds for the case they described --
our FAST mode (e.g. sun Charizard/Venusaur, or a fast four) against a heavy
Trick Room team -- which a redesign must identify from our whole brought four and
game plan, not the two actives, and without blanket minimum-Speed assumptions.
The guard stays registered, opt-in and off.

## dominated_attack DEPLOYED at the user's word; mirror match pre-registered (2026-September 24, 15:15)

The user, on the locally neutral result: "we still need to fix that problem,
because at higher elo our opponents will win because of it esp cuz its 2v2 and
only 4 mons each move matters so much". DEPLOYED.json amended (50d6699):
`guards_extra` += `dominated_attack`, fresh `replay_tag`
T6_humanpreview1_attackcheck (the 3 games before the change stay in their own
dir); the checkpoint and preview model are unchanged; `amendments` records it.
`tools/deployed_config.py` now refuses unknown guard names (the bot would
silently ignore a typo). Ladder still OFF (resume needs the user's word).

**Mirror match (pre-registered now, runs after the Trick Room A/B):**
`evaluation/mirror_guard_ab.py --guard dominated_attack --games 2000`: the
deployed bot against itself, side A with the guard, side B without -- same
brain, team, preview model and other guards, so the baseline is exactly 50% and
every game is as close as the bot can make it (the user's "each move matters"
case the held-out opponents cannot show). Four blocks of 500: open/hidden sheets
x which side challenges. Reading: **the guard wins close games** if A's Wilson
95% lower bound is above 50%; loses them if the upper bound is below 50%; else
inconclusive. The shared knowledge-observation cache is disabled for the mirror
(its key cannot tell the two sides apart).

## dominated_attack A/B: DEPLOY-ELIGIBLE, locally neutral (pooled +0.05pp); trick_room_direction A/B running (2026-September 24, 13:45)

`results_guard_ab_dominated_attack/` (the deployed configuration + the guard on
our player vs its promotion arm; 6,204 games; delta = with minus without):

| Opponent | With | Without | Delta [95%] |
|---|---:|---:|---:|
| New held-out human clone | 91.0% | 89.3% | +1.7 [-0.4, +4.3] |
| Frozen PPO | 93.1% | 92.7% | +0.4 [-2.1, +2.8] |
| Rotation 1 | 92.5% | 91.9% | +0.6 [-1.7, +2.9] |
| Rotation 2 | 93.1% | 92.4% | +0.8 [-1.9, +3.2] |
| Previous held-out human clone | 84.1% | 85.6% | -1.5 [-3.6, +0.7] |
| Scripted heuristic | 91.9% | 93.6% | -1.7 [-3.8, +0.3] |
| **Pooled** | | | **+0.05 [-0.97, +1.05]** |

It changed the played attack 6,818 times in 6,204 games (~1.1 per game). By the
pre-registered rule it is **deploy-eligible** (fires; not worse; no population
below -3pp) but **not "better"**: locally neutral. Reading: the fixes are the
exact ladder mistakes the user flagged, yet against local opponents the bot
already wins ~90%, so a better attack rarely changes a result; the ladder is
the arbiter. Deployment is the user's call.

`trick_room_direction` registered (c18bb25, opt-in; 509 tests) and its
pre-registered A/B launched 13:43 (`results_guard_ab_trick_room_direction/`):
early firing in 99 games = 33 changed actions (18 reversals of their room, 15
counters), 81 keep-ours demotions.

## Repo cleanup at the user's request: 63 GB -> 27 GB, nothing deleted (2026-September 24, 12:35)

Moved (not deleted) to `../_cleanup_2026-09-24/` (README inside; the user drags
it to the Trash when satisfied): `trajs_foundation` (25 GB), the Reg M-B human
trajectories A/AC/C/D (~10 GB), `results_brainv1_attempt1_invalid`,
`results_league2`'s checkpoints (its tracked manifest stays), and the old
`vgc-bench-brainv1` worktree (branch brain-v1 merged and kept; worktree entry
pruned). Each was checked with git grep: nothing in code or configs loads them.
Kept: `trajs/`, `trajs_regmb_human_B`, every referenced results dir, battle logs,
ladder replays, the champions and evaluation reference arms. The 179 run logs /
outputs from the repo root now live in `logs/` (ignored); new launcher logs still
land at the root. The running dominated_attack A/B was not touched.

## The user's Trick Room lesson -> `trick_room_direction` guard; its A/B pre-registered (2026-September 24, 12:20)

The user, on ladder game 2: Armor Tail blocks Fake Out both ways (the bot
already knew: `priority_block` demoted its Fake Outs 12 times that game -- my
Fake Out suggestion was wrong), and "its sometimes a good move to predict an
opponents trick room and undo it because they have a heavy trick room team ...
sending out our fast pokemon but with farafarig for the sole purpose of
countering their TR". In game 2 the bot set Trick Room on turns 1 and 4 while
its Mega Blastoise (88) and Farigiraf (80) outsped their Trick Room team; it
worked only because their own Trick Room reversed ours both times. Nothing in
the stack decided whether Trick Room helps us or them (the tempo reranker uses
its speed comparison only to choose Protect while Trick Room is up).

`vgc_bench/src/trick_room_guard.py` (ac3c86e, tests on rebuilt positions):
Trick Room up (>= 2 turns) and favouring them -> promote the ranked pair that
reverses it; up and favouring us -> demote pairs that undo it; down and it would
favour them -> with a likely setter of theirs active (revealed, or species rate
>= 0.5) promote the ranked pair that uses ours this turn (both rooms cancel: the
user's counter), otherwise demote pairs that set it. Rosters with >= 2 likely
setters are taken at minimum Speed (the plain hidden-spread range, 72-123 for a
base-60 Pokemon, orders nothing). Registered after the dominated_attack A/B so
that run stays valid.

**Its A/B (pre-registered now, runs after the current one):** the same harness
and rule as dominated_attack: `run_guard_ab.py --guards trick_room_direction`
vs the deployed configuration's arm; deploy-eligible (the user decides) if it
fires, the pooled delta is not worse (upper bound >= 0) and no population's
point estimate is below -3pp. Its per-archetype read on the held-out Trick Room
rosters is reported as the mechanism check.

## Ladder stopped by the user after 3 games (2-1); dominated_attack guard built; its A/B PRE-REGISTERED (2026-September 24, 12:25)

The user watched games 1-2 (pasted replays): "water moves are halved in sun",
"we mightve won if our bot just clicked sludge bomb", "why are we using water
pulse on camerupt when we can just water spout and hit both", "why did we ice
beam the farafarig?", then "do that and turn off the ladder games rn". Ladder
stopped 11:36 at 2-1 (`ladder_replays_mc_deployed_T6_humanpreview1`, 3 games, no
game abandoned mid-play). The decision log confirms the policy picked the weaker
of its own attacks while ranking the better one (Leaf Storm 0.76 vs Sludge Bomb
0.07; Ice Beam into Farigiraf 0.22 vs full-HP Water Spout 0.19; Water Pulse into
Camerupt 0.21-0.26 vs Water Spout 0.09-0.20); no guard compared one move with
another (`resisted_target` only re-aims the same move). Human pilots DO bring
Blastoise with Torkoal (12/30 known fours; their most common four is exactly
game 1's) -- they manage it with Ice Beam in sun and Farigiraf's Rain Dance.

**Fix:** `dominated_attack` (a8f6b05, opt-in): per slot of the top pair using a
plain attack, the ranked pairs keeping the partner's action and using another
plain attack of the same Pokemon are scored with the calculator (expected damage
per foe, capped at the HP left after the partner's attack, x accuracy, +0.5 per
expected KO); promote the best when >= 1.25x and +0.05. Never swaps utility
attacks (priority, Fake Out, pivots, guaranteed secondaries, effect moves),
never promotes ally-hitting or other-Mega actions, only pairs the policy ranked
(always legal). Tests rebuild the ladder positions with poke-env's parser + the
real calculator (Sludge Bomb 21% vs Leaf Storm 13% expected; Water Spout 41% +
75% vs Ice Beam 22%; Water Spout KO + 62% vs Water Pulse KO).

**A/B (pre-registered now, before any game):** `evaluation/run_guard_ab.py
--guards dominated_attack`: the deployed configuration + the guard (our player
only) vs the deployed configuration's promotion arm reused as "without" (valid:
only `vgc_bench/src/guards.py` changed among the pins, the guard was off there,
every new population's manifest must equal the reference's but the output path).
6 populations x 47 held-out rosters x 22 games; pooled equal-population delta,
whole-roster bootstrap 95%. **Deploy-eligible** (the user decides) if the guard
fires, the pooled delta is not worse (upper bound >= 0) and no population's point
estimate is below -3pp; "better" if the lower bound > 0. Otherwise it stays off.

**Amendment 12:40 (before any outcome was read):** the first launch changed the
played action ~2 times per game (338 promotions in 165 games) -- far beyond the
reported failure class, because the guard could also switch the TARGET for a
bigger number, overriding which foe the policy chose to hit (threat, focus fire).
Every reported mistake was a move choice against the same foe(s). Stopped after
165 games (set aside in `attempt1_unrestricted/`, never scored); the guard now
requires the alternative to hit every foe the policy's attack hits (a spread
move may add the other foe), with a test that a bigger hit on the other foe is
not promoted. The harness also re-checks the guard module's hash between
populations. Same rule, same design, relaunched.

## PROMOTED by the user: T6 trained on human openings + the model preview; 25-game ladder read running (2026-September 24, 11:30)

The user, after the results: "make it the official bot and run 25 ladder games".
Promotion (a9de37c): save 21,135,360 copied to `results_deployed/champion_mc_T6hp.zip`
(sha c5e92237..., verified against the source; metadata sidecar stamped
`requires_knowledge_obs: true`); DEPLOYED.json `deployed` = that checkpoint, team
T6, the three opt-in guards, `set_prior_reg: mc`, `learned_preview: true`,
`preview_model: data/preview_t6_focus_20260923.pt` (+ sha), `replay_tag:
T6_humanpreview1`; T6 -> `previous_deployed` and history (21-29 on ladder), T4's
history entry keeps its 27-23. Workspace CLAUDE.md: five immutable prior
champions. Ladder read started 11:27 (`tools/ladder_deployed.sh 25`, serial,
dir `ladder_replays_mc_deployed_T6_humanpreview1`): session log confirms
`preview : learned; opponent model=data/preview_t6_focus_20260923.pt`,
knowledge_obs on, search off, opponent-aware rerank on (as for every
deployment).

## Human-opening cycle: BOTH SAVES BETTER than the deployed T6 brain; recommended save 21,135,360 + the model's preview; promotion and ladder await the user (2026-September 24, 05:20)

Chain complete 05:02 (training 01:33-03:32, 983,040 steps, ~138 steps/s, return
invariant held; probes heuristic 0.83 -> 0.89, clone 0.81 -> 0.80). Each save
played with OUR preview chosen by `data/preview_t6_focus_20260923.pt` (argmax),
vs the pinned deployed-T6 arm with its own preview; 6 populations x 47 held-out
rosters x 22 games = 6,204 games per arm; delta = arm minus deployed:

| Opponent | Control (old brain) | Save 20,643,840 | Save 21,135,360 |
|---|---:|---:|---:|
| New held-out human clone | -1.5 [-10.9, +7.1] | +3.8 [-4.6, +12.7] | +4.9 [-2.8, +13.2] |
| Frozen PPO | -0.4 [-7.8, +7.4] | +6.9 [+0.2, +14.1] | +9.7 [+3.5, +16.5] |
| Rotation 1 | -1.6 [-9.8, +6.3] | +6.6 [+1.0, +12.8] | +7.8 [+1.8, +14.2] |
| Rotation 2 | -1.7 [-10.0, +7.0] | +8.2 [+2.0, +15.1] | +7.5 [+1.3, +14.6] |
| Previous held-out human clone | -2.1 [-10.8, +6.2] | +5.6 [-3.4, +14.9] | +7.4 [-1.0, +16.1] |
| Scripted heuristic | -7.6 [-12.4, -2.9] | -1.5 [-5.9, +2.9] | +5.0 [+1.6, +8.4] |
| **Pooled (equal-population)** | **-2.51 [-6.93, +1.95]** | **+4.93 [+1.22, +8.82]** | **+7.08 [+3.69, +10.51]** |
| Verdict (pre-registered rule) | no clear change | **better** | **better** |

Win rates, save 21,135,360 vs deployed: 89.3/84.3, 92.7/83.1, 91.9/84.0,
92.4/84.8, 85.6/78.1, 93.6/88.6. Both sheet modes positive in every population
(hidden +3.7..+11.2, open +3.3..+10.8). Tie-break (declared 04:10, before save
2 reported): higher pooled mean -> **21,135,360** (+7.08 vs +4.93).

**What practice added** -- each save vs the control arm, identical openings
(the model's preview is deterministic per matchup): +7.45 [+4.71, +10.32] and
**+9.59 [+6.72, +12.52]**. By the lead the model chose (all 6 populations):

| Lead (share of games) | Deployed, own lead | Control | Save 1 | Save 2 |
|---|---:|---:|---:|---:|
| Blastoise + Farigiraf (83%) | 82.4% | 84.5% | 90.3% | **92.3%** |
| Farigiraf + Incineroar (11%) | 90.2% | 62.4% | 82.3% | 84.1% |
| Charizard + Venusaur (6%) | 92.4% | 72.0% | 79.3% | 83.6% |

Mechanism (the user's ask, 2026-09-23: bring different Pokemon by matchup):
3 lead pairs and 5 different fours by opponent (deployed: 1 and 1); Farigiraf +
Torkoal 0%; Incineroar brought 68%, Venusaur 45% (deployed: 0% / 0%).

By opponent archetype (save 2 vs deployed): Trick Room 77.7 -> 86.6 (+8.9, the
ladder's pain point), Tailwind +7.3, Grassy/Fake Out +12.0, balance +11.8,
**rain 89.5 -> 86.3 (-3.3)**: the remaining weakness, concentrated where the
model opens Farigiraf + Incineroar (83.1% vs rain) or Charizard + Venusaur
(81.1%) -- the two openings still below the old Trick Room line on their
matchups (-6.1 / -8.8). Next training target: those openings, rain first.

**Recommendation (the user decides):** promote save 21,135,360 with the model
preview as a new artifact `results_deployed/champion_mc_T6hp.zip`
(DEPLOYED.json `learned_preview: true`, `preview_model`, sha-verified by the
launchers; `replay_tag` T6_humanpreview1 = fresh replay dirs), then a serial
ladder read. Caveats: local gains have not always transferred to ladder (T6
itself went 21-29); ladder play keeps the opponent-aware layer on, which local
evaluation does not (as for every deployment so far); the focus model then also
supplies the opponent-plan belief (checked equal to today's model, 01:05 entry).
Nothing promoted; bot offline; no ladder game played.

## Tie-break declared before save 2 reports (2026-September 24, 04:10)

The pre-registered rule (0765aad) grades each save but never says which to
recommend if BOTH pass "better". Declared now, while save 1 is four
populations in (+3.8 / +6.9 / +6.6 / +8.2) and save 2 has not started: **the
higher pooled mean delta; within 0.5pp, the later save (more practice).** A save
that fails its own grade is never recommended. Recommendation only: promotion
and ladder stay the user's call.

## Control arm complete (no clear change; unpractised leads collapse); training attempt 1 crashed at worker start and was fixed; attempt 2 running (2026-September 24, 01:50)

**Control arm** (deployed T6 brain unchanged, our preview chosen by
`data/preview_t6_focus_20260923.pt`, vs the pinned deployed-T6 arm with its own
preview; 6 populations x 47 held-out rosters x 22 games):

| Opponent | Model preview | Own preview | Delta [95%] |
|---|---:|---:|---:|
| New held-out human clone | 82.8% | 84.3% | -1.5 [-10.9, +7.1] |
| Frozen PPO | 82.7% | 83.1% | -0.4 [-7.8, +7.4] |
| Rotation 1 | 82.4% | 84.0% | -1.6 [-9.8, +6.3] |
| Rotation 2 | 83.1% | 84.8% | -1.7 [-10.0, +7.0] |
| Previous held-out human clone | 76.0% | 78.1% | -2.1 [-10.8, +6.2] |
| Scripted heuristic | 80.9% | 88.6% | -7.6 [-12.4, -2.9] |
| **Pooled** | | | **-2.51 [-6.93, +1.95]** |

Verdict by the pre-registered rule: no clear change. By the lead the model
chose (same rosters for both arms): Blastoise + Farigiraf (83% of games) 84.5%
vs 82.4% own lead (+2.1); Farigiraf + Incineroar (11%) 62.4% vs 90.2%;
Charizard + Venusaur (6%) 72.0% vs 92.4%. The model leaves Trick Room exactly
where the old brain's Trick Room line already wins 90%+, and the old brain
cannot pilot the replacements -- the gap the training cycle is meant to close.
Read-out script: scratch `hp_analysis.py` (each save is also compared with this
control arm on identical openings, isolating what practice adds in battle).

**Training attempt 1 crashed at worker startup** (01:26:29-01:26:41, zero
steps): Python 3.13's forkserver preloads a path-run `__main__`, so the launcher
`training/human_preview.py` installed its patch -- importing poke-env and torch
and loading the model -- inside the forkserver, and every worker was forked from
that process. poke-env starts its event-loop thread at any import and never
restarts it after a fork (dead loop in every worker), and macOS killed the
workers ("+[NSNumber initialize] may have been in progress in another thread
when fork() was called"). My earlier fork check passed only because its toy
children never used the loop. Fix 87712a6: `prepare_workers()` empties the
forkserver preload, so each worker re-runs the launcher after the fork (tests:
real forkserver workers are patched and run a coroutine on the poke-env loop;
negative control shows the dead loop). The trial's pinned launcher hash was
amended in `experiment.json` with the reason (recipe unchanged; attempt-1 log
kept as `training_attempt1_forkcrash.log`). **Attempt 2 running since 01:33**:
9 installs (main + 8 workers, each in its own process), human openings sampled
for both sides, ~136 steps/s, ep_rew_mean 0.42 -> 0.58.

## Resumed the human-opening chain; launcher fixes; learned-preview deployment plumbing ready (2026-September 24, 01:05)

The user: `continue`. 00:50: Showdown on 7610, the control arm resumed with its
recorded command (33 complete frozen cells kept, incomplete cells never written),
the chain (`human_preview_chain.sh`, scratch) waiting behind it: then training
(`training/run_t6_human_preview_trial.py`, verified with `--prepare-only`) and both
saves' evaluations. No Codex activity since 09-22 12:09; its heartbeat is PAUSED.
Control arm so far: new human clone -1.5 [-10.9, +7.1], frozen PPO -0.4 [-7.8,
+7.4] (model preview vs the brain's own, same deployed brain).

Found and fixed while it runs (334bb8b, fdacffb):
- **The ladder launchers would not have refused during this chain**: their
  heavy-job pattern missed `run_candidate_vs_t6.py`, its `learned_preview_study.py`
  child and `training/human_preview.py` (whose command line never contains
  `vgc_bench.train`). Added, with a regression test over sample command lines
  (`unit_tests/test_launcher_scripts.py`); `ladder_read_loop.sh` now refuses.
- **The default challenge listener crashed at launch** on this Mac:
  `/usr/bin/env bash` is bash 3.2, where an empty `"${EXTRA[@]}"` under `set -u`
  is an unbound-variable error (it only worked with a rejoin room). All array
  expansions guarded; the test fails on any bare one.
- `ladder_ourteam.py --learned_preview` without `--preview_model` now refuses:
  the opponent-model default would have silently chosen our preview.

Deployment plumbing (unused until the user promotes something): every launcher
(`ladder_deployed.sh`, `challenges_deployed.sh`, `exhibition_mode.sh`) reads
DEPLOYED.json through `tools/deployed_config.py`, which sha-verifies the
checkpoint, the team (new for the ladder launcher) and, when `learned_preview:
true`, `preview_model` against `preview_model_sha256`; `replay_tag` (default: the
team stem) names fresh single-config replay dirs. The current deployment's ladder
command is byte-identical (traced). Belief check for that case: on ladder,
`--preview_model` also feeds the opponent-aware layer's opponent-plan belief.
On the 50 real T6 ladder opponents, paired:

| Model | Lead top-1 / top-3 | Lead log-p | Bring top-3 | Bring log-p | Brier |
|---|---|---:|---:|---:|---:|
| top-500 M-C (ladder today) | 0.10 / 0.30 | -4.47 | 0.38 | -3.95 | 0.268 |
| general 09-20 | 0.08 / 0.22 | -5.29 | 0.38 | -4.13 | 0.284 |
| T6 focus 09-23 | 0.16 / 0.26 | -4.56 | 0.38 | -3.92 | 0.274 |

So one focus model can serve both (no loss for the belief). Side finding: all
three are overconfident at this Elo -- log-p below uniform (-2.71 over 15 leads
or 15 fours) while ranking above it; opponents at ~1100-1250 lead unlike the
replay corpus. Worth a temperature on the belief later; not touched now.

## Paused at the user's word; control arm read on one population (2026-September 23, 10:08)

The user: `pause this for now`. Stopped 10:07: the chain (before training
started), the control-arm evaluation, caffeinate, both local servers and the
watcher; nothing is running; the bot is offline; T6 stays deployed.

Done before the pause -- control arm (deployed T6 brain, preview chosen by
`data/preview_t6_focus_20260923.pt`), new held-out human clone, 1,034 games:
82.8% vs 84.3% own preview, delta -1.5 [-10.9, +7.1]. By the lead the model
chose, against the deployed brain's own-preview results on the SAME rosters:

| Lead chosen by the model | Games | Control | Deployed, own lead |
|---|---:|---:|---:|
| Blastoise + Farigiraf | 858 | 87.2% | 83.1% |
| Farigiraf + Incineroar | 110 | 56.4% | 89.1% |
| Charizard + Venusaur | 66 | 69.7% | 92.4% |

Reading: the human Fake Out lead the old brain can already play (Blastoise +
Farigiraf) is +4pp over its own lead on those matchups; the openings it never
practised collapse (-33 / -23pp), as in Codex's forced-lead pilot. Human
openings help where the brain can pilot them; practising them (the queued
training cycle) is what should unlock the rest. Frozen-PPO population stopped at
363/1,034 (only full cells are kept; incomplete cells rerun).

**Resume** (nothing to rebuild; the trial league is prepared, no status.json):
the chain `human_preview_chain.sh` (scratch) waits for the control arm, then
trains (`training/run_t6_human_preview_trial.py`) and evaluates both saves; the
control arm resumes with the same command (completed cells are skipped):
`caffeinate -is .venv/bin/python -u evaluation/run_candidate_vs_t6.py --candidate
results_deployed/champion_mc_T6.zip --label deployed_learned_preview
--preview-model data/preview_t6_focus_20260923.pt` (Showdown on 7610). Or skip
the rest of the control arm and train straight away (the chain skips the wait
when no control process runs).

## Teaching matchup-dependent previews from human pilots of our kind of team; control arm running, training cycle PRE-REGISTERED (2026-September 23, 09:45)

The user, ~07:15: switching back to T4 "doesnt solve the problem" and T6 "needs
to understand to bring other pokemon in and not the same everytime because of
opponent matchups" -- find "a player with a lot of replays and show the bot how
many different combinations they make based off their opponents". (Fact noted
once: T6's ladder opponents averaged Elo 1179, T4's 1239; T6 did not face
stronger opposition.)

**The human evidence.** Full public Reg M-C histories of every player who ran
teams close to ours (`battle_logs_players_t6like/`, ~185 new games). Best
match: **gankyburner**, 44 games, 19 with five of our six
(Charizard/Farigiraf/Incineroar/Torkoal/Venusaur + Salamence): **15-4**, **7
different lead pairs and 7 different brought fours** by opponent -- Farigiraf +
Incineroar 11 games (won 9), Incineroar + Salamence 2, Charizard + Torkoal 2,
Torkoal + Venusaur, Charizard + Farigiraf, ... Never Farigiraf + Torkoal, the
lead our bot used in 50/50 ladder games. dksnnfud (who built our team) led
Blastoise + Farigiraf in all 4 of his games.

**Teaching it.** `training/train_preview_model.py` gained a team focus (replays
of these pilots added; examples sharing >= 4 of our six weighted x10):
`data/preview_t6_focus_20260923.pt` (validation lead top-1 0.268 / top-3 0.506,
bring top-1 0.280 / top-3 0.544 over 15 options each; reproduces gankyburner's
lead in 17/19 of his games, in-sample). For our six against our 50 ladder
opponents it leads Blastoise + Farigiraf 35, Farigiraf + Incineroar 7, Charizard
+ Venusaur 5, Torkoal + Venusaur 3 -- never Farigiraf + Torkoal; the back row
varies by matchup. Codex's pilot is the warning: forcing the Incineroar +
Farigiraf lead on a brain that never practised it won 8/48. So both halves:

- **Control arm (running since 09:34):** the deployed T6 brain unchanged, its
  preview chosen by the model (`evaluation/learned_preview_study.py`: only our
  side, opponent belief discarded, so only the opening differs) --
  `results_candidate_vs_t6_deployed_learned_preview/`.
- **Training cycle (pre-registered now, launches when the control arm ends):**
  `training/run_t6_human_preview_trial.py` (a6ded3e): the deployed T6 brain
  +983,040 steps with `--no_teampreview` and both sides' previews sampled from
  the model at temperature 1 (`training/human_preview.py`), so the battle policy
  practises the human openings; one variable vs Codex's T6 recipe; holdout
  rosters zeroed. Each save then evaluated with the model choosing its preview.

**Evaluation rule** (unchanged instrument, `evaluation/run_candidate_vs_t6.py
--preview-model`): each arm vs the pinned deployed-T6 arm (own preview), six
populations x 47 held-out rosters x 11 repeats x both sheet modes = 6,204 games;
pooled equal-population delta with whole-roster bootstrap 95%; **better** if the
lower bound > 0 and no population below -3pp; **worse** if the upper bound < 0;
else no clear change. Mechanism: lead pairs, share of Farigiraf + Torkoal,
Incineroar / Venusaur brought. Reading: the control arm measures human openings
with the old battle brain; the trained saves measure what practising them adds.
No promotion, no ladder without the user.

## Preview-entropy training cycle: NO CLEAR CHANGE on either save; the back row adapts, the lead does not (2026-September 23, 07:05)

Training 03:33-05:20 (`results_brainv1_t6_preview1/`, boost 9, return
invariant held: 0.38-0.90; ~153 steps/s). The entropy term rose from 1.7 to
~5.3 (Codex's unboosted T6 run: 1.2 -> 1.0), so the bonus acted. Training-time
probe (100 battles, bare policy): scripted bot 0.91 (deployed) -> 0.76 / 0.73;
pool clone 0.78 -> 0.75 / 0.79. Evaluation (`evaluation/run_candidate_vs_t6.py`,
6,204 games per save vs the pinned deployed-T6 arm; delta = candidate minus
deployed; pre-registered rule applied as written):

| Opponent | Save 20,643,840 | Save 21,135,360 |
|---|---:|---:|
| New held-out human clone | +0.9 [-3.6, +5.7] | -4.2 [-10.6, +2.3] |
| Frozen PPO | +0.2 [-4.0, +4.4] | -3.4 [-8.6, +1.6] |
| Rotation 1 | +3.2 [-1.0, +7.8] | +4.1 [-0.7, +9.0] |
| Rotation 2 | +0.6 [-4.2, +5.8] | +3.1 [-0.8, +7.0] |
| Previous held-out human clone | -0.6 [-6.0, +4.6] | +1.4 [-4.4, +7.6] |
| Scripted heuristic | -2.6 [-5.3, -0.0] | -2.4 [-5.2, +0.3] |
| **Pooled (equal-population)** | **+0.27 [-1.81, +2.47]** | **-0.24 [-2.76, +2.27]** |

Verdict: **no clear change** for both. Mechanism: the BACK ROW became matchup-
dependent (save 1 brings Venusaur 41%, Charizard 39%, Incineroar 19% of games,
where the deployed brain always brought Blastoise + Charizard); the LEAD barely
moved (Farigiraf + Torkoal 96% in save 1, Charizard + Farigiraf 4%) and fully
re-collapsed by the final save (100%). So the lever reaches the preview, but the
lead -- where the ladder losses concentrate (Farigiraf KO'd before Trick Room:
1-10) -- stays fixed. Likely reason: the in-battle policy only knows how to pilot
the Trick Room line, so alternative leads lose during training and PPO keeps
choosing the Trick Room lead; more preview entropy alone cannot fix that.
Nothing promoted; no ladder. Evaluation summaries tracked
(`results_candidate_vs_t6_preview_*/`).

**Where the bot needs training** (ladder + this cycle): (1) the lead -- teach
the in-battle policy to pilot the other leads (sun: Charizard + Venusaur; Fake
Out support: Blastoise or Incineroar + Farigiraf) by starting a share of
training games from those leads, with those forced preview choices kept out of
the policy gradient, then let PPO choose (Codex's "multiple openings per
matchup" idea); (2) play after Trick Room is up (20-19), against physical
attackers (56% of our knockouts; Tyranitar alone 19). **Recommendation for the
user**: the Charizard T4 bot is stronger on both instruments (ladder 27-23 vs
21-29; locally +4.3pp [0.9, 7.9] over T6); switch the deployed configuration
back to T4 (with `set_prior_reg: mb`) for ladder, and keep the T6 lead work
offline -- the user's call.

## T6 ladder read: 21-29, Elo 1253 -> ~1100; loss analysis; training cycle chosen and its evaluation PRE-REGISTERED (2026-September 23, 03:40)

**Ladder** (`ladder_replays_mc_deployed_T6`, serial, 01:08-03:33, canary 4-6 ->
audit clean -> 11-14 at 25 (above the 8-17 stop line) -> 50): **21-29 (42%,
Wilson 29-56)**, mean opponent Elo 1179, performance ~1123, pre-game Elo 1253 ->
1113 before game 50. Zero forfeits, timeouts or errors on our side (ability
reveals repaired and counted: 9). Same account, 09-20, T4: 27-23, mean opponent
1239, performance ~1267, Elo 1104 -> ~1270. Consistent with the local read
(T6 vs T4-with-its-data -4.30pp [-7.90, -0.89]); 50 games each cannot separate
~140 performance points at the usual +-140, but both instruments point the
same way.

**Loss analysis** (`tools/ladder_opening_audit.py`, 8981c96;
`results_analysis/t6_ladder_opening_audit_20260923.json`):
- One fixed script: lead Farigiraf + Torkoal **50/50**, back Blastoise +
  Charizard; Incineroar and Venusaur never reached the field (0/50); decision
  log: lead probability Farigiraf 0.998, Torkoal 0.915; turn 1 Protect + Trick
  Room at 0.95.
- Trick Room up on turn 1 in 39/50 -> **20-19**; not up -> **1-10**. Farigiraf
  knocked out before Trick Room in 11 games -> **1-10** (10 of the 29 losses).
- First faint ours -> **4-22**; theirs -> 17-7. Lost a Pokemon on turn 1 -> 1-9.
- 125 knockouts on our side through 44 games: physical 56%, special 42%;
  Tyranitar alone 19 (Rock Slide, Knock Off; its sand also removes our sun);
  then Archaludon Electro Shot, Arcanine-Hisui Head Smash, Rillaboom.
- A tested hypothesis that failed: "Trick Room hurts us against slow teams" --
  T6 went 3-0 against the slowest brought teams, 2-5 against the fastest.
Reading: the losses are not one tactical blunder; they are what a predictable
opening costs. Humans see the team at preview and focus Farigiraf; the one
member built to blunt physical attackers (Incineroar: Intimidate, Fake Out,
Parting Shot) is never brought because the preview never varies.

**Training cycle** (the one pre-registered cycle): `training/run_t6_preview_trial.py`
(ec651ea) -- the deployed T6 brain + 983,040 steps (saves at 20,643,840 and
21,135,360), Codex's T6 recipe with ONE change: a team-preview-only entropy
bonus (`training/preview_entropy.py`, boost 9 -> preview entropy coefficient
0.2 vs the 0.02 floor), applied at launch so `vgc_bench/` stays byte-identical.
Held-out roster split (786 teams, incl. every evaluation roster) zeroed.

**Evaluation, registered now, before training** (`evaluation/run_candidate_vs_t6.py`):
each save vs the deployed T6 arm of `results_t6_vs_deployed_v1` (pins verified),
six populations x 47 held-out rosters x 11 repeats x both sheet modes = 6,204
games per save. Primary: pooled equal-population delta (candidate minus
deployed), whole-roster bootstrap 95%. Verdict per save: **better** if the lower
bound > 0 and no population's point estimate < -3pp; **worse** if the upper
bound < 0; otherwise **no clear change**. Two saves = two looks: a single
"better" is reported with that caveat. Mechanism check: distinct lead pairs,
top-lead share, rosters with more than one lead, Incineroar / Venusaur brought.
No promotion and no ladder: the user decides in the morning.

## T6 ladder read + loss analysis + one training cycle, user-delegated; rules PRE-REGISTERED (2026-September 23, 01:10)

The user, ~01:05: `test it first, than after u see the results go from your own
judgement, analyze the losses and see where the bot needs training. im going to
sleep so u got this`. Committed before any game is played:

1. **Ladder read of the deployed T6** (`tools/ladder_deployed.sh`: sha-verified
   `DEPLOYED.json`, serial, three opt-in guards, Reg M-C set data, restart loop,
   credentials sourced shell-side only; fresh dir `ladder_replays_mc_deployed_T6`).
   10-game canary -> mechanical audit (no crash, no parse errors, no forfeit or
   timeout on our side, guards sane) -> continue to 25 only if clean with >= 2
   wins -> at 25 continue to 50 unless the record is <= 8-17 (<= 32%) or a
   mechanical problem appears -> bot offline. Reference: T4 27-23 over 50 (block
   1 16-9 at mean opponent Elo 1195, block 2 11-14 at 1283); the account starts
   at ~1245. 25-50 games measure mechanics and gross strength, not a rating claim.
2. **Loss analysis**: per game -- leads and brings both sides, whether and when
   Trick Room went up and who set it, whether our setter acted before fainting,
   turn-1/2 knockouts and what did them, opponent Fake Out, weather, first faint,
   archetype of the opposing team; decision logs of the losses; compared against
   Codex's local audits (Farigiraf KO'd before Trick Room).
3. **At most ONE bounded training cycle** afterwards, chosen from the dominant
   loss mechanism: from the deployed T6 brain into NEW artifacts, with a local
   evaluation registered before it starts, paired against the deployed T6 on
   held-out rosters. No promotion and no further ladder play without the user.
   Ladder and training never share the machine.

## Set-data side-by-side COMPLETE: T4 is better with its old data; T6 is measurably weaker than T4-as-laddered (2026-September 23, 01:10)

`results_set_prior_ablation_T4/REVIEW.md` (6,204 new games, 00:18-01:00; the
new-data arm is Codex's verified T4 arm). The T4 brain with Codex's Reg M-C set
data vs the Reg M-B data it trained and laddered with: pooled **-1.97pp, 95%
[-3.66, -0.32]** (new minus old); new human clone -4.0, rotation1 -5.0,
rotation2 -3.8, frozen +0.7, previous clone -0.2, heuristic +0.5. The loss sits
in HIDDEN-sheet games (-8.5 / -7.7 / -6.6 against the new clone and the two
rotations; open-sheet deltas within +-2.3) -- exactly where set guesses enter
the observation. A brain reads the guesses it trained with; "more accurate"
inputs it never saw make it worse. Codex's data is right for T6 (fine-tuned on
it), wrong for T4.

Consequence: Codex's T6-vs-T4 study ran T4 handicapped. Against T4 with its own
data on the same rosters (`t6_vs_t4_old_data.json`): T6 **-4.30pp, 95% [-7.90,
-0.89]** pooled (frozen -7.6, rotation2 -6.4, previous clone -6.3, rotation1
-3.3, new clone -1.6, heuristic -0.6); equal-population means T4 88.1% vs T6
83.8%. Locally T6 is measurably weaker than T4 as laddered. T6 stays deployed:
the user's decision; switching back is a `DEPLOYED.json` change.

Made actionable (tests pass, 443): `DEPLOYED.json` records `set_prior_reg` per
brain (T6 `mc`, T4 `mb`); `tools/ladder_deployed.sh`, the challenge listener and
`exhibition_mode.sh` export it as `VGC_SET_PRIOR_REG`; `ladder_ourteam.py`
records it in each replay dir's material config, so a dir can never mix set data
(new test). Local Showdown server stopped; nothing running.

## PROMOTED by the user: T6 (Kanto starters) is the deployed configuration; Codex's work committed; set-data side-by-side running (2026-September 23)

The user, 2026-09-23: `commit codex's work and run the side by side test, also
we want that new team that i showed the one with all three starters`.

**Committed** Codex's 09-20/21 work as one change (82db6ba): format-aware set
priors, +4 threat-evidence floats (token 1023), fixed human-clone share, the T6
repair trial, both T6 studies, the challenge listener; reviews, scorecards and
manifests tracked, per-game rows stay local (.gitignore); challenge replays
ignored like ladder replays. 442 unit tests pass.

**Promoted T6 on the user's decision**, as a NEW artifact:
`results_deployed/champion_mc_T6.zip` (verified copy of
`results_brainv1_t6_repair1/.../20152320.zip`, sha256 bed6840e..., the save
Codex tested; sidecar role production) + `teams/candidates_mc/T6.txt` (sha
a326dfd7...), Reg M-C, the same three opt-in guards, recorded in
`DEPLOYED.json` with the evidence, the previous deployed configuration and a
history. The T6 brain is native to the current code (projection input 1209);
it was trained with the Reg M-C set data. Evidence on record, stated plainly:
locally T6 is NOT better than T4 (83.8% vs 86.2% over 12,408 games; frozen
PPO -8.3pp); known weaknesses: Farigiraf KO'd turn 1 before Trick Room, and the
same four + leads every game (Torkoal + Farigiraf, Blastoise + Charizard). No
ladder games yet. `results_deployed/champion_mc_T4.zip` becomes an immutable
prior champion (27-23 on ladder). Launchers: `tools/ladder_deployed.sh` and
`exhibition_mode.sh` read `DEPLOYED.json` (sha-verified); the challenge
listener's default replay dir now follows the deployed team; all three refuse
while `opening_study.py`, `run_t6_*` or `run_set_prior_ablation` run.

**Side-by-side running** (f191823, `evaluation/run_set_prior_ablation.py`,
`results_set_prior_ablation_T4/`, launched 00:18, Showdown on 7610): does the
T4 brain do better with the Reg M-C set data (Codex's change) or the Reg M-B
data it trained and laddered with? The new-data arm IS the T4 arm of
`results_t6_vs_deployed_v1` -- all 105 files that study pinned still hash the
same (checked before and between populations), so only the old-data arm is
played (`VGC_SET_PRIOR_REG=mb`, recorded per arm): same checkpoint, team,
47 held-out rosters, 11 repeats, both sheet modes, seed 20923, six populations
= 6,204 games (~70-90 min). Whole-roster bootstrap per population and pooled
(equal-population). Stops for review; nothing is promoted or laddered by it.

## Deployed Reg M-C challenge listener added (September 21)

User requested direct challenges on the live Showdown server for Reg M-C.
Added `tools/challenges_deployed.sh`: it pins and hash-verifies both artifacts in
`results_deployed/DEPLOYED.json`, accepts only `gen9championsvgc2026regmc`, keeps
play serial, uses the deployed guard profile, and stores challenge replays and
decision logs separately from rated ladder data. Direct challenges are unranked;
normal `tools/ladder_deployed.sh` behavior is unchanged. Credentials remain
shell-side in the existing environment and are neither printed nor passed as CLI
arguments. The launcher refuses concurrent ladder/challenge or heavy local jobs.
Added an explicit Reg M-C room-recovery path after the first listener socket timed
out mid-challenge. It reconnects the same account, rejoins the audited battle room,
lets Showdown resend room history/current request, and reconstructs live state
before resuming decisions. The interrupted battle was recovered through turn 5.
Recovery is dispatched on poke-env's dedicated event loop; invalid/closed rooms
fail closed and release the serial battle slot.

## Follow-on COMPLETE; keep T4 deployed and pause automation (September 21)

`results_t6_vs_deployed_v1` finished all 12,408 local games. T4 5,346/6,204
(86.2%) versus candidate T6 5,201/6,204 (83.8%). T6 deltas by population:
human-new +2.3 pp, frozen -8.3, rotation1 +1.7, rotation2 -2.6,
human-previous -6.1, heuristic -1.1. Equal-population pooled delta -2.34 pp,
whole-roster bootstrap 95% interval [-5.75,+0.93]; improvement not established.
Frozen arm interval [-15.1,-2.4] is the clearest regression. This is a different-
team configuration comparison, not weight-only causation or a ladder forecast.

All pinned hashes and 12 complete arm files verified, zero recorded guard failures,
correct own teams and both modes, no duplicate games. Natural opponent previews
are intentionally unpaired across T4/T6. No evaluation/training processes remain;
local Showdown servers left untouched. Representative raw replays plus decision
logs reviewed: same opening vulnerability persists (Farigiraf KO before Trick Room
in both sampled human losses and 24/25 sampled frozen losses; also present in wins).
Review and uncertainties: `results_t6_vs_deployed_v1/REVIEW.md`.

The ONE follow-on cycle is complete. Pause `continue-vgc-validation`; do not
launch more work without user direction. Keep T4 deployment and all champions
unchanged. Recommended next experiment, NOT STARTED: controlled training-side
opening-survival comparisons (supported Trick Room and sun alternatives), then
a matchup-conditioned selector only if full-game benefits replicate. Do not
retrain on the reviewed holdout or continue the same broad PPO run blindly.

## Confirmation reviewed; one follow-on T4 comparison running (September 21)

`results_t6_confirmation_v2` COMPLETE: 12,408 games. Candidate T6 5,248/6,204
(84.6%) versus generalist-on-T6 3,509/6,204 (56.6%). All six populations and
both sheet modes improve; zero recorded guard failures, pairing drift or missing
cells; pinned hashes intact. Not a comparison against deployed T4 and not a ladder
claim. Full integrity checks, intervals and replay/decision review:
`results_t6_confirmation_v2/REVIEW.md`.

All 6,204 candidate games use Torkoal/Farigiraf leads and Blastoise/Charizard back.
Reviewed all three sampled human-clone losses and representative frozen losses.
Farigiraf died turn one before Trick Room in all 3 sampled human losses and 25/26
frozen losses (also 19 frozen wins). Sample is roster-biased, not causal evidence.
No demonstrated mechanical/pipeline defect: low-HP Eruption flags often reflect
damage AFTER choosing; no-weather Weather Ball versus Gallade respects revealed
Wide Guard; sun Water Spout can still KO both foes. Do not patch blanket rules.

The ONE authorized follow-on cycle is now `evaluation/run_t6_vs_deployed.py`,
under caffeinate, output `results_t6_vs_deployed_v1/`. Read its `status.json`,
logs and processes FIRST on subsequent heartbeats; do not relaunch confirmation
or start another experiment. Compares immutable deployed T4 weights/team from
`DEPLOYED.json` with unchanged candidate T6 save 20,152,320, same production guard
stack, six populations, 47 reserved rosters, 1,034 games/configuration/population,
12,408 total. Same roster blocks but independent battle RNG and **natural opponent
previews for each different own team**, not artificial cross-team preview pairing.
This measures configuration performance, not a weight-only improvement. Frozen
source manifests, 8 local battles maximum, one MPS process, per-child one-hour and
whole-invocation three-hour limits. No training or ladder. Stop for review, then
PAUSE heartbeat `continue-vgc-validation` and report recommendation to user.

Validation: 20-game two-configuration smoke completed, zero guard errors; smoke
scores are not strength evidence. Full suite 435 passed, 5 skipped; new runner
and tests Ruff-clean; diff whitespace clean. No production decision code changed,
no champion/deployment edits, no credentials, commits or pushes.

## Automatic follow-through authorized (September 21)

User requested automatic next steps after the confirmation finishes. Created
thread heartbeat `continue-vgc-validation`, active every 30 minutes (updated at
the user's request). It checks
actual process/results, stays quiet on unchanged healthy progress, reviews the
completed six-population results and candidate losses, then performs ONE bounded
evidence-driven local repair/validation cycle (or a local comparison against the
deployed T4 configuration if clean). It pauses after that cycle or if user input
is required. No ladder, promotion, champion overwrite, credentials, cloud compute,
commit/push, or unbounded training is authorized. Local Mac must remain awake and
the app running. Latest observed phase: rotation2 baseline; first three completed
populations favored the candidate, but the final verdict/review remains pending.

User's usage constraint: keep the same Codex model, with no alternate-model
fallback or delegation. If allowance runs out, leave work pending until the
normal reset. Do not redeem reset credits or purchase credits to bypass limits.

## T6 multi-population confirmation started; fixed preview found (September 20)

User authorized broader local tests and tactical review after the short trial.
The trial is COMPLETE: generalist 190/300 (63.3%), save 20,152,320 225/300
(75.0%), save 20,643,840 197/300 (65.7%) against eval_mcB_20260920, same 25
round-held-out rosters, independent battle RNG. Save 20,152,320 is selected for
confirmation, not deployment. No more training, ladder, or promotion authorized.

Important review finding: BOTH trained checkpoints chose Torkoal + Farigiraf,
back Blastoise + Charizard, in **300/300** games each. The generalist varied its
preview. Improved sampled win rate does not establish matchup-aware planning.
Earlier-save category results (60 games each): TR 33 vs baseline 36; grassy/Fake
Out 56 vs 44; Tailwind 38 vs 34; rain 46 vs 36; balance 52 vs 40. These small
subgroups identify follow-up questions, not statistically settled regressions.

Running under caffeinate: `evaluation/run_t6_confirmation.py`, artifacts
`results_t6_confirmation_v2/`, phase in `status.json`, progressive population
results in `scorecard.json`. Compare init vs save 20,152,320 using the SAME
policy+guard stack as the short trial, no new search/tempo/preview models.
Six opponent populations: fresh and 09-13 quarantined human clones (stochastic),
64opp / 8opp / tuned frozen PPOs (deterministic), simple heuristic. Each model
plays **1,034 games per population**: 47 roster-disjoint additional test teams,
11 repeats, both sheet modes. Total 12,408. Excludes the first 25 confirmation
selection rosters as well as this fine-tune's training roster partition. As
before, the GENERALIST may have seen these rosters in earlier training.
Coverage: 10 TR, 10 rain, 10 grassy/Fake Out, 10 Tailwind, 7 balance; no separate
sun or psychic-terrain bucket available under the current mutually-exclusive
categorizer/split. This is not exhaustive archetype coverage.

Opponent previews replay exactly from the control arm; battle RNG is independent.
Uncertainty resamples whole opponent rosters, not correlated repeats as independent
matchups. No automatic promotion decision. Each subprocess has a one-hour timeout;
individual cells have a five-minute timeout and fail on guard errors/pairing drift.
Sampled replay + decision audits from first roster per category/mode for the human
and 64opp populations. `tactical_review_queue.json` is a context-flag index, NOT a
verdict that a move was wrong. Human tactical review remains pending sufficient
candidate loss samples.

Harness hardening discovered during live smoke/resume:

- Persist fingerprints at preview time, not after the opposing six have shrunk to
  four revealed mons. Old live in-memory pilot pairing worked, but cross-process
  replay of old post-battle fingerprints was invalid; smoke correctly rejected it.
- Accept Showdown HP color suffixes (`50/100g`) in the post-game review parser.
- Use unique local guest accounts on restart: default repeated names rejoined
  abandoned battles before policy loading. The failed restart added ZERO result
  rows. Preserved 264 completed control games; unfinished cell rerun. Both source
  revisions are explicit sidecars next to the updated confirmation manifest.
  Policy weights/decision code/team/matchups did not change during these repairs.

Verification after repairs: 428 tests pass, 5 skip; Ruff and diff whitespace
checks clean. Live run resumed beyond the preserved 264 games, with zero pairing
mismatches in the completed cells. At 341 games, measured cell time was 0.66 s
per game; the full 12,408-game run is roughly 2-3 hours, subject to matchup length.
No bot-strength claim from the smoke results. T4 deployment is unchanged.

## T6 diagnosis-first repair trial (2026-September 20, Codex)

User paused Claude's long T6 run and authorized the diagnosis-first approach.
The old run produced no new checkpoint; it remains stopped. No ladder games,
promotion, commits or pushes were requested in this pass. Deployed T4 weights
and deployment manifest are untouched.

Completed repairs:

- Shared format-specific set loader (`src/set_priors.py`): observations, guards,
  preview helpers, search and determinization use the actual battle regulation.
  Missing current-format evidence stays unknown, never silently becomes M-B data.
  Explicit `VGC_SET_PRIOR_REG=mb` is available for controlled ablation only.
- `data/joint_sets_regmc.json`: 3,788 unique training-bucket replays (buckets 0-4
  of the merged 09-20 corpus), 186 species, provenance/hashes in its sidecar.
  Fixed packed nickname/species parsing and duplicate sheets. Rillaboom now has
  plausible current-format moves. No fabricated Smogon spreads were added.
- Impossible sets no longer reappear as confident guesses after contradictory
  reveals. Threat caching includes revealed moves/items/abilities and stat changes.
- Appended four threat-evidence floats at the token tail: revealed/inferred move
  fractions for each enemy. Token 1019 -> 1023, projection 1205 -> 1209; old input
  weights zero-extend. These inputs are available to learn, not already understood
  by the old model. Compatibility regressions cover both +4 and +12 extensions.
- Fictitious-play sampling can retain a fixed human-clone probability as new
  saves accumulate (`--fixed_opponent_stems 100,200`, fraction 0.20).

Opening pilot: `results_opening_t6/pilot_train.jsonl` / `pilot_summary.json`,
**240 completed games**, six roster categories, hidden/open sheets, five opening
arms, four repeats/cell. Seven labels exist in the categorizer but no independent
psychic-terrain bucket was selected (classification gives TR precedence). Policy
control was 24/48: rain 1/8, grassy/Fake Out 0/8, Tailwind 2/8, balance 6/8,
TR 7/8, sun 8/8. These are ONE opposing roster per category and tiny samples,
not strength estimates. T6's preview already changes across matchups; the T4
"same four every battle" observation does not carry over. Some alternative
openings helped against rain and hidden-sheet Rillaboom; every tested opening
lost all four open-sheet Rillaboom games. Do not turn cell winners into rules.
Opponent preview is reused between arms, but battle RNG is **not paired**.

Current bounded experiment: `training/run_t6_repair_trial.py`, under caffeinate,
`results_brainv1_t6_repair1/status.json` is the live source of truth. Starts from
generalist 19,660,800 on T6; ends at 20,643,840 (+983,040), saves every 491,520,
8 environments, one MPS learner, 50% hidden sheets, fixed 20% human-clone sampling,
unchanged full-game reward/shaping. No forced openings or tiny-pilot reweighting.
Roster-hash 20% of the team pool is excluded from THIS fine-tune. This is NOT
unseen-team generalization: the generalist may have trained on those rosters.
The eval-only human clone remains excluded from the training population by role
and content. Verified league plus pinned weights/data/code hashes in experiment.json.

Supervisor stops on invalid returns, 15 minutes without progress, process failure,
or the 150-minute training ceiling. On successful training it compares the initial
model and both new checkpoints against the eval-only 09-20 clone, same reserved
rosters, both sheet modes, up to 8 concurrent LOCAL games, then stops for review.
This is a diagnostic comparison, not the multi-population promotion gate. The
training/deployment guard mismatch remains: PPO trains bare policy; comparisons
use the same production guard profile for every arm. No automatic continuation,
promotion or ladder command exists in this supervisor.

Verification: 421 tests passed, 5 server-dependent skips; Ruff clean on changed
files and diff whitespace clean. Pyright still reports 29 errors; an isolated
HEAD checkout reproduced the SAME 29 pre-existing errors (not new regressions).
The training process confirmed 132 tensors copied, 3 zero-extended, none replaced.
Model strength remains unproven until the comparisons finish.

## Team change at the user's word: T6 (Kanto starters, sun + Trick Room) rebuilt from a player's replays; T6 round training since 20:28 (2026-September 20, 20:35)

The user, 20:15: change the team to the player **dksnnfud**'s "Incineroar
version of the team with all 3 starters", scrape his replays and guess what
cannot be seen. Found: 4 public replays in total (3 Reg M-C on 09-17/18 at Elo
1394-1587, 1 Reg M-B), NO open team sheet in any; roster Charizard / Venusaur /
Blastoise / Torkoal / Farigiraf + a flex slot (Incineroar in the two newest
games; Rillaboom, Raichu earlier). Four other top players run the same core
(xdstyfh, jxghxhx, thruxy, Arthuritiis; flex Pawmot / Raichu / Dragapult /
Incineroar): 13 games of the archetype in the corpus, 8-5 (dksnnfud 3-1).
How it is played: Blastoise + Farigiraf lead in 4/4 of his games (Fake Out +
Trick Room, Mega Blastoise, Rain Dance into Water Spout), Torkoal / Venusaur in
the back; others lead Charizard-Y + Venusaur (sun mode). Charizard and
Incineroar were NEVER on the field in his games; Incineroar never in any game
of this version. Reconstruction (`teams/candidates_mc/T6.txt`, validated by the
Reg M-C simulator; per-Pokemon provenance in `manifest.json`): seen in his games
= Blastoisinite + Fake Out / Water Spout / Water Pulse / Ice Beam; Farigiraf
Sitrus Berry + Trick Room / Psychic / Helping Hand / Rain Dance; Venusaur Focus
Sash + Sludge Bomb / Leaf Storm / Sleep Powder / Protect; Torkoal Drought +
Eruption / Heat Wave / Protect. Seen from the other players of this team =
Charizardite Y + Heat Wave / Solar Beam / Ancient Power / Protect. Inferred
from the 3,610 top-player open sheets = Torkoal's 4th move Weather Ball
(37/83) + Charcoal (81/83) + Quiet; Incineroar Fake Out / Flare Blitz / Parting
Shot / Throat Chop (517/1038), Passho Berry (most common once Sitrus is taken:
item clause), Careful; natures, abilities; EVs never visible: offence + Speed
for the sun attackers, HP + offence for the Trick Room attackers, HP + defence
for the two supports (the generic nature rule would have maxed their Speed).

Round spec2 (T4 + fresh clone) was STOPPED at 20:28 before its first save: its
question is moot once the team changes; nothing but ~1 h of compute lost. The
T6 round launched 20:28:24 (`brainv1_t6_chain.log`, training log
`brainv1_t6_202832.log`, artifacts `results_brainv1_t6/`): init = the brain-v1
GENERALIST save 7 (it has piloted Farigiraf + Torkoal Trick Room lines on T2
and the sun pieces on T4; the T4 specialist's preview collapsed to one fixed
four), our side T6 only, +8 intervals to 27,525,120 (~14.5 h), pool = the
specialist recipe with the fresh 09-20 clone at stems 100/200, all three
eval-only clones banned. First-save check: the init has no probe on T6, so the
paired 200-battle read vs the generalist on T6 always runs and kills below
-5pp; reward invariant as before. Verdict: paired vs the generalist on T6 with
eval_mcB_20260920 and eval_mcB_20260913 as human arms; then the cross-team read
against the deployed T4 specialist. The deployed configuration is unchanged
until T6 earns it.

Ladder finding that motivated looking at the team (from `decisions.jsonl`, 50
games): the T4 specialist brought the SAME four every game -- Charizard +
Garchomp lead 50/50, Sylveon + Incineroar in the back; Venusaur (0.19 back-row
probability) and Toxapex (0.001) never reached the field. A matchup-blind
preview is the clearest next battle-logic lever. Human record of the exact T4
six: 33 sheets, 51.5%; Toxapex on any team 168 sheets, 42.9%; nobody runs that
core with Torkoal (0 of 60) -- my earlier "usual sixth is Torkoal" was wrong.

## PROMOTED by the user: brain v1 T4 specialist + T4 is the deployed configuration (2026-September 20, 20:05)

The user, 20:00: `yeah make it the official bot`. Promotion as a NEW artifact;
nothing was overwritten: `results_deployed/champion_mc_T4.zip` (verified copy of
`results_brainv1_spec/.../27525120.zip`, sha256 af4e78a5..., sidecar role
`production`) + `results_deployed/DEPLOYED.json` (checkpoint, sha, team
`teams/candidates_mc/T4.txt` + its sha, Reg M-C, the three opt-in guards, the
evidence, the previous deployed checkpoint and its sha). Tracked in git:
the manifest and the sidecar (weights stay local, as for the league champion).
`results_league/league_champion.zip` (deployed 2026-08-30 -> 2026-09-20, MB430)
stays immutable as a prior champion and the MB430 reference arm;
`teams/reg_mc/our_team.txt` is NOT repointed (every battery reference and the
league weights' zero entry mean MB430 by that name).

Evidence on record: ladder read 27-23 over 50 (Elo 1104 -> peak 1321, settling
~1250-1300; zero mechanical failures); screening PASS vs the generalist on T4
(weighted +3.47pp); cross-team vs the previous deployed configuration weighted
+6.33pp with no arm below the incumbent. NOT run: the n=5,000 promotion-tier
battery -- the promotion rests on the user's decision and the evidence above.

Team T4 (real top-player sheet, source MC302): Charizard @ Charizardite Y
(Weather Ball, Heat Wave, Ancient Power, Protect), Venusaur @ Life Orb
(Chlorophyll; Sludge Bomb, Leaf Storm, Earth Power, Protect), Sylveon @ Fairy
Feather (Pixilate; Hyper Voice, Quick Attack, Yawn, Detect), Garchomp @ Choice
Scarf (Rock Slide, Dragon Claw, Earthquake, Stomping Tantrum), Incineroar @
Sitrus Berry (Intimidate; Flare Blitz, Fake Out, Darkest Lariat, Parting
Shot), Toxapex @ Leftovers (Regenerator; Infestation, Toxic, Wide Guard,
Baneful Bunker).

Launchers: `tools/ladder_deployed.sh <n_total> [replay_dir]` (reads the
manifest, verifies the sha, serial, restart loop, refuses while heavy jobs
run); `exhibition_mode.sh` now defaults to the deployed checkpoint and team
(`CHECKPOINT` / `TEAM` env override; replay dir
`ladder_replays_exhibition_mc_T4`). The workspace CLAUDE.md names the new
deployed artifact and keeps all three champions immutable. Round spec2 keeps
training (step 27.8M at 20:02, shaped return 0.66); the bot is offline until
the user says ladder, and never while training runs.

## Fresh clones built; round spec2 training since 19:31 (2026-September 20, 19:35)

Refresh (`refresh_spec2_chain.log`): trajectories A 4,852 / B 4,940 from the
7,558-replay merged corpus (token 1019 = current; knowledge block populated,
the 8 threat floats are ZERO -- logs2trajs never computes them, so every human
clone is threat-blind and its zero-extended input weights stay zero: harmless
tonight, a future lever). Clones: `results_bc/mc_A_20260920` best epoch 2 =
**40.4%** top-1 on B (epochs 3-12 decay to 38.2%); `results_bc/eval_mcB_20260920`
best epoch 2 = **40.3%** on A (role eval_only). The 09-13 pool clone scores
39.2% on the new B: the field moved ~1pp for it, so the fresh-clone lever may
be modest. Smoke: the specialist beats the new eval clone 34/40.

The first spec2 launch died at the pool build, correctly: the specialist had
been stamped role `candidate` for the ladder read and `build_league.py` admitted
only production / seed / training_opponent. A candidate is learner lineage,
not an evaluation population -> `candidate` added to `ALLOWED_ROLES` (eval_only
and eval_opponent stay refused; holdouts also banned by content; new test;
5becc19). Relaunched 19:31:01 (`brainv1_spec2_chain.log`, training log
`brainv1_spec2_193109.log`): specialist (135 tensors copied) -> +8 intervals to
35,389,440 on T4, pool = specialist recipe with the 09-20 clone at stems
100/200; eval-only roots banned: eval_mcB, eval_mcB_20260913, eval_mcB_20260920.
Watcher kill line 0.79 (probe-matched) + paired clause + reward invariant.
Verdict: paired vs the specialist on T4, human arm = eval_mcB_20260920, extra
human arm eval_mcB_20260913 under `results_gate_battery_brainv1_spec2_human2/`.
Expected: last save ~10:00 on 09-21, verdict ~12:30.

## Specialist verdict complete: PASS on every arm; cross-team +6.3pp weighted over the deployed configuration (2026-September 20, 19:25)

The arms skipped for the fast path are in (`brainv1_spec_verdict_full.log`).
T4 specialist 27525120 paired vs the brain-v1 generalist on T4, n=1,000 per
arm: scripted **+8.3** (91.3 v 83.0), frozen +3.5, rotation1 +3.3, rotation2
+5.5, human clone +0.1 (84.6 v 84.5); weighted (human x2) **+3.47pp**, equal
+4.14pp -> **screening PASS**, the first since league 1 (baseline here = the
generalist on the same team). Diagnostics: pool clone mc_A +3.3, eval_D +3.9;
the mc_A-vs-eval_mcB divergence is 3.2pp (flag at 10): the human-side gain is
specific to the clone it trained against, which is tonight's single variable.
Cross-team against the deployed brain on MB430 (unpaired, n=1,000 each):
scripted **+4.9**, frozen +8.2, rotation1 +8.2, rotation2 +7.3, human **+4.7**;
weighted **+6.33pp**, equal +6.66pp -- no arm below the incumbent. The
scripted-arm breach that failed every M-C candidate (round 6, brain v1) was an
MB430 artifact: on its own team the new brain beats the scripted opponent more
often than the deployed brain does on MB430.

Overnight chain: merged corpus `battle_logs_top_mc_merged_20260920/` = 7,558
unique replays (09-13: 2,390); trajectories started 19:19.

## Ladder read complete: T4 specialist 27-23 over 50, Elo 1104 -> peak 1321, settling ~1250-1300; overnight chain running (2026-September 20, 18:40)

`ladder_replays_mc_brainv1spec_T4_20260920/` (serial, 16:27-18:35, one config):
block 1 **16-9** at mean opponent Elo 1195 (perf ~1295), block 2 **11-14** at
mean opponent Elo 1283 (perf ~1241), pooled **27-23 (54%, Wilson 40-67)**, mean
opponent Elo 1239, performance rating **~1267**. Pre-game Elo path 1104 -> 1300
after 25 -> peak 1321 -> 1267 before game 50 (a loss). Reference: the deployed
brain on MB430, 09-09, 13-12 at mean opponent Elo ~1110 (perf ~1125), account
left at ~1104. OUR forfeits/timeouts/parse errors: **0 in 50**. By opponent
rating: <1150 6-2, 1150-1250 13-10, 1250-1350 6-7, >=1350 2-4 -> a ~1250-1300
player in today's Reg M-C field. Opponent forfeits/timeouts 16 of the 27 wins
(5 at turn <= 2, all in block 1; without those 22-23). "Played to the end
11-23" is NOT a strength measure: the bot never forfeits, so every loss counts
while conceded wins do not. First faint ours 28/50 (won 10 = 36%); first faint
theirs 20 (won 15 = 75%): the opening still decides, and we lose first blood
more often than not. Trick Room games 5-2 (the historic TR hole does not show
on T4). Facing: Volcarona 8-2, Rillaboom 16-9, Basculegion 6-3, Indeedee-F 6-3,
Kingambit 6-4, Incineroar 9-7, Sneasler 10-10, Salamence 5-6, Raichu 4-5,
Milotic 4-4, Gardevoir 3-5, Garchomp 3-5. Bot offline 18:35.

Reading: the first real ladder gain since August. Same account, eleven days
apart: the old brain + MB430 held ~1100; brain v1 + T4 climbed ~150-200 points
and held. 50 games cannot separate 54% from 50%, but the rating path does not
need to: it is an independent measure of the same thing. **Promotion is the
user's decision** (recommended: make T4 + `results_brainv1_spec/.../27525120.zip`
the deployed configuration as a NEW artifact; `league_champion.zip` is never
overwritten). The MB430-anchored gate cannot judge a team change; proposal
stands: human-clone + PPO arms gate, scripted arm advisory, cross-team read
against the incumbent configuration.

**Overnight chain launched 18:36** (`training/refresh_clones_spec2_chain.sh`,
log `refresh_spec2_chain.log`; single variable vs the specialist round = the
human clone): deferred specialist arms (scripted + mc_A / eval_D) -> merged
corpus incl. the 09-20 scrape into `battle_logs_top_mc_merged_20260920/` ->
trajectories A/B `trajs_regmc_human_{A,B}_20260920` (current token length) ->
clones `results_bc/mc_A_20260920` (training_opponent) and
`results_bc/eval_mcB_20260920` (eval_only, banned by content), best epoch by
cross-bucket agreement -> smoke -> round **spec2** (`results_brainv1_spec2/`:
the specialist +8 intervals to 35,389,440 on T4 with the new clone in the
pool; first-save kill line 0.79 = the specialist's own probe 0.89 minus 0.10,
probe-matched this time, AND a paired 200-battle read below -5pp; reward
invariant) -> verdict paired vs the specialist on T4 with BOTH eval-only clones
as human arms. The battery's team pool and weights stay frozen (comparability).

## Ladder read, block 1: T4 specialist 16-9, Elo 1104 -> ~1290; block 2 pre-registered (2026-September 20, 17:35)

`ladder_replays_mc_brainv1spec_T4_20260920/`, serial, 16:27-17:24: canary 6-4
(audit clean: no crash, no parse errors, no forfeit or timeout on our side,
guards firing sanely) -> continued to 25 = **16-9 (64%, Wilson 95% 44.5-79.8)**.
Pre-game Elo 1104 -> 1282 before the last game (a win); mean opponent Elo 1195
(the deployed brain's 09-09 M-C read: 13-12 at mean opponent Elo ~1110, account
left at ~1104). Honest decomposition: 10 of the 16 wins ended on an opponent
forfeit/timeout (5 at turn <= 2); excluding turn <= 2 quits 11-9; games played
to the end 6-9; OUR forfeits/timeouts 0. First faint ours 12/25 (won 4 of
those; won 11 of the 12 where the opponent lost a Pokemon first). Trick Room 4
games (2-2). vs opponents >= 1200: 7-5; below: 9-4. Performance rating
~1295 (deployed 09-09 read ~1125), both +-~140: 25 games support
non-regression and mechanical soundness, not a rating claim. Bot offline 17:24.
Launcher fix: session logs are now timestamped (the 16:51 relaunch overwrote
the canary's session report; replays, decisions.jsonl and the numbers above
are intact; games 11-25 report kept as `..._block1_games11-25.log`).

**Block 2 (pre-registered 17:35, user-delegated judgement):** the same
checkpoint, team, guards and replay dir, 25 more games (total 50), fixed
length, stop only on a mechanical failure. No deployed-brain control block:
it would spend the account's rating to re-measure a configuration already read
at 13-12 near 1100. After block 2: bot offline; pooled tally; the deferred
specialist arms (scripted + mc_A / eval_D diagnostics); then the M1 data
refresh and an overnight training round, registered when block 2 is in.

## Specialist round done; ladder pick = T4 specialist 27525120; 10-game canary LIVE (2026-September 20, 16:30)

Specialist round (`results_brainv1_spec/`, save 7 of brain v1 -> +8 intervals
on T4 only) finished 15:24:51, exit 0, shaped return within [0.46, 0.94].
Probes (scripted / pool clone on T4, 100 battles, bare policy): 0.75/0.85,
0.81/0.82, 0.79/0.81, 0.84/0.84, 0.85/0.85, 0.83/0.88, 0.88/0.86, **0.89/0.87**
-> one finalist, the final save (best probe sum). Decision arms, paired vs the
generalist (brain-v1 save 7) on T4, n=1,000, seed 83 (fast path, 15:25-16:26):
human clone (eval_mcB) **84.6 v 84.5 = +0.1pp**, frozen 90.5 v 87.0 = **+3.5**,
rotation1 89.8 v 86.5 = **+3.3**, rotation2 90.8 v 85.3 = **+5.5**. Pre-
registered rule -> **PICK the specialist** (`ladder_pick.txt`). Reading:
specialising bought nothing against the human clone (the generalist already
held 84.5%) and 3-5pp against the PPO populations. Cross-team against the
deployed brain on MB430 (`tools/cross_team_read.py`, unpaired n=1,000 each):
human **+4.7 +- 1.7**, frozen +8.2, rotation1 +8.2, rotation2 +7.3; weighted
(human x2) **+6.62pp**. Scripted and diagnostic arms are deliberately pending
(filled in after the ladder read; scripted advisory on a non-MB430 team).

Ladder MEASUREMENT read (user-delegated; `league_champion.zip` stays the
deployed checkpoint): checkpoint stamped role `candidate` (sha af4e78a5),
`tools/ladder_read_loop.sh ... T4.txt 10 ladder_replays_mc_brainv1spec_T4_20260920`
launched 16:27:18; session log confirms Reg M-C, T4, the Reg M-C priors,
guards hard + the three opt-in guards, rerank and tempo on, serial. Canary =
10 games -> audit -> 25 only if clean with >= 2 wins.

## Fast path to the ladder at the user's word; the field moved toward T4 (2026-September 20, 14:55)

The user, 14:50: `yeah do that, get to ladder sooner`. The pre-registered rule
is unchanged; only its evaluation became lazy (`tools/ladder_pick.py`, 4 tests;
`evaluation/run_ladder_pick.sh`): the human arm of every finalist first, then
the three PPO arms of the finalist with the higher human rate, falling through
to the next on a disqualification, the generalist when none qualifies, nothing
below the cross-team bar (79.9%). A scratch waiter takes over at
`SPEC_VERDICT_START`, stops the chain's full battery and runs those arms into
the standard paths (`results_gate_battery_brainv1_spec/<stem>/screening/`), so
the scripted and diagnostic arms are filled in AFTER the ladder read by the
normal verdict script (`ARMS` filter added; completed arms are skipped). Pick
expected ~17:00 instead of ~18:50. Specialist probes so far (scripted / clone
on T4): 0.75/0.85, 0.81/0.82, 0.79/0.81, 0.84/0.84, 0.85/0.85, 0.83/0.88,
0.88/0.86; shaped return within [0.46, 0.94].

Fresh scrape 10:29 (`battle_logs_top_mc_20260920/`, 4,864 unique replays, 1,410
new single-game ones) and the new `tools/mc_meta_stats.py` (window vs
reference; `results_analysis/mc_meta_stats_20260920.json`), 3,351 games since
the 09-13 corpus: T4's pieces are UP (Charizard 15.3% of sheets +2.5 with a
52.1% sheet win rate, Garchomp +6.1, Sylveon +3.0, Venusaur +2.2, Incineroar
+1.2), rain -4.9, Trick Room 18.4% (-5.4); MB430's pieces are DOWN (Basculegion
-9.4, Kingambit -7.5, Floette -3.7, Whimsicott -3.6 at a 45.2% sheet win rate).
Risers: Gholdengo +12.0 (27.3%), Raichu +12.1 (20.0%), Rillaboom +6.1 (52.9%).
T4's sixth slot (Toxapex, 0.5% of sheets) is the archetype's odd piece; the
usual sixth is Torkoal -- a later lever, the sheet stays as trained.

## The user delegated the ladder-or-training call; decision rules PRE-REGISTERED before the specialist verdict (2026-September 20, 10:30)

The user, 10:30: `continue based on your judgement ill be back later` /
`whether its ladder or more training`. Judgement: both, in sequence -- the
specialist round finishes (5 of 8 saves done: probes 0.75/0.85, 0.81/0.82,
0.79/0.81, 0.84/0.84, 0.85/0.85 scripted/clone on T4), its verdict runs, then
a 25-game ladder MEASUREMENT read (not a promotion: `league_champion.zip`
stays the deployed checkpoint), then more training on refreshed data. Rules,
committed before any verdict arm exists:

1. **Ladder candidate.** A specialist finalist qualifies if, paired vs save 7
   on T4 (n=1,000 per arm): human arm (eval_mcB) delta >= -1.0pp AND no PPO
   arm (frozen, rotation1, rotation2) below -2.0pp. Among qualifiers the
   highest absolute human-arm rate goes; if none qualifies, save 7 (the
   generalist) goes on T4. The scripted arm is reported, advisory only.
2. **Cross-team sanity** for whichever brain goes: its human-arm rate on T4
   >= the deployed brain's on MB430 (79.7-79.9% at n=1,000).
3. **Ladder protocol.** Serial, `tools/ladder_read_loop.sh` (restart loop for
   dead sockets, refuses while heavy jobs run, credentials sourced shell-side
   only), fresh replay dir, three opt-in guards, checkpoint stamped role
   `candidate`. 10-game canary -> audit (no crash, no parse errors, no
   inactivity losses, guards firing sanely) -> continue to 25 only if the
   audit is clean and the canary has >= 2 wins (<= 1 of 10 has p ~ 1% under a
   50% bot: breakage, not noise). After 25: bot offline, tally with
   `tools/ladder_loss_profile.py`; 25 games support no claim beyond
   non-regression against the deployed brain's 13-12 (SE ~10pp).
4. **Then more training**, machine free again: M1 refresh from the 09-20
   scrape (running now, network-bound: `battle_logs_top_mc_20260920/`) into
   dated artifacts and a T4 continuation on the refreshed pool; registered in
   detail once the scrape and the ladder read are in.

Note on the specialist's first-save rule: its kill line (0.743) was derived
from the battery-style grid read (0.843, guards on), while the callback probe
runs the bare policy for 100 battles and reads ~9pp lower (brain-v1 save 7:
probe 0.67 vs grid 76.2 six-team). Save 1 read 0.75 and passed; the paired
clause was the safeguard. Future kill lines must use the probe's own baseline.

## Team pick: T4 (Charizard-Y sun); specialist round running (2026-September 20, 00:55)

Grid complete (save 7 vs the deployed brain, paired n=300 per cell, seed 83;
`results_team_grid/brainv1_19660800/`), with save 7's tournament read and the
pre-registered score (mean of the human-clone read and the tournament read):

| team | scripted: save 7 / deployed | human clone: save 7 / deployed | tournament | score |
|---|---|---|---|---|
| T0 (MB430) | 83.7 / 89.3 | 73.3 / 79.0 | 83.0 | 78.2 |
| T1 | 79.0 / 34.0 | 74.7 / 30.3 | 72.0 | 73.3 |
| T2 | 65.7 / 49.3 | 76.7 / 54.7 | 84.0 | 80.3 |
| T3 | 66.3 / 33.0 | 63.3 / 29.0 | 60.0 | 61.7 |
| **T4** | 84.3 / 39.3 | **82.0** / 38.3 | 82.0 | **82.0** |
| T5 | 78.0 / 40.0 | 63.0 / 43.7 | 71.7 | 67.3 |

Six-team means: scripted 76.2 vs 47.5, human clone 72.2 vs 45.8. Confirmation
reads (save 7 alone vs eval_mcB, n=1,000, fresh seed 8302): **T4 84.4**
[82.0, 86.5], T0 78.9, T2 77.3 -> the pre-registered rule picks **T4**
(confirmation lead +5.5pp over T0; `team_pick.json`). T4 =
`teams/candidates_mc/T4.txt` (real sheet, source MC302): Charizard-Mega-Y,
Venusaur (Life Orb), Sylveon, Garchomp (Choice Scarf), Incineroar, Toxapex.
Cross-team, before any specialising: the generalist on T4 reads 84.4% against
the human clone where the deployed brain on MB430 reads 79.7-79.9% (n=1,000
each, unpaired **+4.6 +- 1.7pp**): the first local evidence that new brain +
new team beats old brain + old team against human-like play.

Specialist round launched 00:51:55 (`training/brainv1_spec_chain.sh`; chain
log `brainv1_spec_chain.log`, training log `brainv1_spec_005155.log`): save 7
(135 tensors copied) -> +8 intervals to 27,525,120 (~14.5 h), our side T4
only, pool as pre-registered (clone x2, old champion, league-1 history,
deployed x2, save 8, save 7 x2 + resume); artifacts `results_brainv1_spec/`;
scratch watcher kill line 0.743 (save 7's scripted read on T4 minus 0.10) plus
the shaped-return invariant. After training: triage -> paired verdict vs save
7 on T4 (`results_gate_battery_brainv1_spec/`) -> `tools/cross_team_read.py`
against the deployed brain on MB430. Relaunching the chain with saves beyond
19660800 present keeps the pool and resumes. Tooling commits a495a95, 090d7bb.
Bot offline; ladder only with the user's word.

## Grid resumed; team-pick rule and the specialist round PRE-REGISTERED before T3-T5 are read (2026-September 19, 23:56)

Resumed at the user's word (`continue`). Pre-registration, committed before
any T3-T5 grid cell exists:

**Team pick.** (1) Score per team = mean of save 7's win rate against the
eval-only human clone in the grid (n=300) and save 7's tournament win rate
(n=300, vs itself on the weighted M-C pool): both opponents are competent
pilots of M-C teams, which the scripted opponent and the deployed brain are
not. Known so far: T0 (73.3 + 83.0)/2 = 78.2, T1 (74.7 + 72.0)/2 = 73.4,
T2 (76.7 + 84.0)/2 = 80.4. (2) The top three by score get a confirmation:
save 7 alone vs eval_mcB, n=1,000, fresh seed 8302 (SE ~1.4pp). The highest
confirmation read wins; if the top two are within 2pp, the higher step-1 score
wins. The scripted-opponent read is reported, never used for the pick.

**Specialist round (NEW_BRAIN_PLAN section 4 step 3).** Init = save 7
(`19660800.zip`); our side = the winning team only; every other training flag
as brain v1 (joint head, shaping 0.10/0.05, knowledge obs, hidden sheets 0.5,
lr 3e-5, n_epochs 3, target_kl 0.02); +8 intervals to 27,525,120. Pool: mc_A
clone x2, old champion, league-1 history, deployed x2, save 8 x1, save 7 x2 +
resume (one deployed copy swapped for save 8: the deployed brain is a weak
pilot of the pool's teams, the brain-v1 generalists are not). Artifacts in
`results_brainv1_spec/` (brain-v1 saves untouched). First-save kill rule:
eval/heuristic more than 10pp below save 7's own grid read on that team AND a
paired 200-battle read vs save 7 on that team below -5pp; shaped-return
invariant as before. Verdict: paired vs save 7 on the winning team, n=1,000
per arm (scripted, frozen, rotation1/2, human = eval_mcB; diagnostics mc_A,
eval_D); success = human arm >= +2pp with no PPO/human arm below -2pp
(scripted arm advisory on a non-MB430 team). The cross-team read against the
deployed brain on MB430 uses the same arms' absolute rates (deployed on MB430
at n=1,000: human 79.7-79.9, frozen 82.3-83.8, rotations 79.8-83.5, scripted
86.4-86.7). Ladder only with the user's word.

## Paused at the user's word during the team grid; T0-T2 read (2026-September 18, 13:13)

Stopped 13:12 (`pause it`): grid, watchdog and eval processes killed; nothing
running, bot offline. The cross-pilot grid (`evaluation/run_team_grid.sh`,
save 7 = `19660800.zip` vs the deployed brain, paired n=300 per cell, seed 83;
`results_team_grid/brainv1_19660800/`) completed three teams:

| team | vs scripted: save 7 / deployed | vs human clone: save 7 / deployed |
|---|---|---|
| T0 (MB430) | 83.7 / 89.3 (**-5.7**) | 73.3 / 79.0 (**-5.7**) |
| T1 (MC558) | 79.0 / 34.0 (+45.0) | 74.7 / 30.3 (+44.3) |
| T2 (MC588) | 65.7 / 49.3 (+16.3) | 76.7 / 54.7 (+22.0) |

Read so far: the generalist holds 73-77% against the human clone on every
team, where the deployed brain holds 79% on MB430 and 30-55% elsewhere; on
MB430 alone it trails the specialist by ~6pp on both opponents (the MB430
battery said -6.6 / -1.4 at n=1,000). T2 leads on the human read; T3-T5 are
pending (~35 min; completed cells are skipped on relaunch):
`HUMAN_BC=$(cat results_bc/eval_mcB_20260913/BEST.txt) PORT=7600 nohup caffeinate -is ./evaluation/run_team_grid.sh results_brainv1/saves_fp_hs_wt/reg_mc/seed1/19660800.zip results_team_grid/brainv1_19660800 300 >> team_grid_brainv1_19660800.log 2>&1 &`
Open decisions for the user: the team, whether the scripted-arm rule should
gate an M-C brain (every M-C-trained candidate breaches it while gaining on
the human and PPO arms), and any ladder read.

## Brain v1 VERDICT: both finalists FAIL the MB430 screening; the team-agnostic brain pilots every candidate team (2026-September 18, 12:45)

Battery (`run_brainv1_verdict_supervised.sh`, 07:59-12:03, port 7600; MB430,
paired n=1,000 per arm vs the deployed brain; human arm = eval_mcB_20260913
epoch 2): **save 7 (19660800)** heuristic **-6.6** (79.8 v 86.4, BREACH),
frozen +5.4, rotation1 +7.7, rotation2 +4.3, human -1.4 (confirm window);
weighted +1.33, equal +1.88; diagnostics mc_A +1.2, eval_D +2.1 -> **FAIL**.
**Save 8 (20643840)** heuristic **-9.5** (77.2 v 86.7, BREACH), frozen +2.2,
rotation1 +11.5, rotation2 +4.3, human **-4.7** (BREACH); weighted -0.15,
equal +0.76; mc_A +1.1, eval_D +1.8 -> **FAIL**. Same shape as round 6
(scripted arm down, PPO arms up); the human-clone gain round 6 had on MB430
(+5.0) is gone for the generalist, which saw MB430 in one sixth of its
episodes. No memorisation signal (clone diagnostics within 3 / 6pp).

Brain tournament (`run_team_tournament.sh`, n=300 per team, pilot vs itself on
the weighted M-C pool; `results_team_tournament/brainv1_<stem>/`): save 7 T2
84.0, T0 83.0, T4 82.0, T1 72.0, T5 71.7, T3 60.0 (mean 75.4); save 8 T4 86.7,
T0 85.0, T2 80.0, T1 69.7, T3 66.7, T5 66.7 (mean 75.8); the deployed brain
82.3 / 35.7 / 75.0 / 46.0 / 52.7 / 49.0 (mean 56.8); the clone 62.7 / 64.3 /
52.3 / 46.3 / 42.3 / 28.3. Both finalists pilot every team at 60-87% where the
deployed brain fell to 36-53% off MB430: the team-agnostic training did what
it was for. Caveat: the tournament's opponent side is the pilot itself, so it
ranks teams within a pilot and does not compare pilots
(`tools/compare_team_tournaments.py` prints the side-by-side). Cross-pilot
evidence = the six-team probe (save 7 / 8: 0.67 / 0.69 vs the deployed 0.463,
n=100 per team) and the **team grid now running** (`evaluation/run_team_grid.sh`:
save 7 and the deployed brain paired per team, n=300, vs the scripted opponent
and vs eval_mcB; `results_team_grid/brainv1_19660800/`, ~85 min): it picks the
team (M4) and gives the per-team human-clone delta the MB430 battery cannot.
Bot offline; nothing reaches ladder without the user's word.

## Brain v1 attempt 2 TRAINING COMPLETE (8 saves); battery running on saves 7 + 8 (2026-September 18, 08:00)

The resumed run finished at 07:59 (exit 0, final save `20643840.zip`; the
shaped return stayed within [0.46, 0.72] at every 30-minute read; ~153
steps/s throughout, no stall). Six-team probe reads (100 battles per arm,
averaged over T0-T5; deployed baseline 0.463 vs the scripted opponent):
save 5 0.64 / clone 0.76, save 6 0.63 / 0.73, save 7 **0.67 / 0.80**, save 8
**0.69** / 0.73 (saves 1-4: 0.61/0.70, 0.66/0.64, 0.66/0.67, 0.59/0.72).
Finalists by the triage rule (best probe sum + final): **19660800** and
**20643840**. The battery (`run_brainv1_verdict_supervised.sh`, port 7600;
MB430 arms paired vs the deployed brain at n=1,000: heuristic, frozen,
rotation1/2, human = eval_mcB_20260913; diagnostics mc_A + eval_D) started
07:59:48 -> `results_gate_battery_brainv1/<stem>/screening/`; the brain
tournament of each finalist on T0-T5 (n=300) follows automatically ->
`results_team_tournament/brainv1_<stem>/`. Caveat on the probe reads: they
are our six candidate teams vs the scripted opponent, not the MB430 battery
arm; the battery decides.

## Brain v1 attempt 2 resumed from save 4 (2026-September 18, 00:49)

At the user's word ("continue") the round resumed through the new
`training/resume_brainv1_chain.sh`: pool verified, fresh 7700, training
continues from `16711680.zip` (all 135 tensors copied, nothing zero-extended;
Adam restarts, as at the original launch) to 20,643,840 = four more intervals
(~7.5 h at 145-152 steps/s), then triage -> the supervised battery on the
finalists (MB430 arms vs the deployed brain, `run_brainv1_verdict_supervised.sh`)
-> the brain tournament of each finalist on T0-T5 at n=300
(`results_team_tournament/brainv1_<stem>/`). SB3 appends a second events file
to the same run directory on resume, so a log-only triage would count four
probes and mis-map the stems: `triage_league_log.py` gained
`--from-tensorboard`, which reads every events file of the run keyed by step
(a later file wins a replayed step); `unit_tests/test_triage_league_log.py`
(6 tests). Scratch watcher: per-save triage line, traceback grep, a stall
notice when the events file is silent for 20 min (the 09-16 sleep gap), and
the shaped-return invariant kill. Lid open, AC power.

## Brain v1 attempt 2 (valid reward) paused at 4 of 8 saves (2026-September 16, 09:40)

Relaunched 00:06 with the shaping fix; the shaped return stayed within
[0.02, 0.80] for all 1,352 rollouts (invariant enforced by the watcher).
Callback reads (100 battles, averaged over the six candidate teams; deployed
baseline 0.463 vs the scripted opponent): save 1 (01:54) heuristic **0.61**
/ clone 0.70; save 2 (03:44) **0.66** / 0.64; save 3 (05:33) **0.66** /
0.67; save 4 (07:21) **0.59** / 0.72. Throughput 145-152 steps/s while
running; progress stalled ~07:50-09:35 (72 rollouts in 2h18m: a sleep/lid
gap, not a crash). Paused at the user's word 09:39 at step 16,932,864;
nothing running. Resume = `brainv1_chain.sh` continues from save 4, or run
the battery on saves 2-4 now.

## Paused at the user's word; brain-v1 attempt 1 is INVALID -- the shaped reward did not telescope (2026-September 15, 11:35)

Stopped 11:22 (`pause it rn`): chain, watcher and training killed; nothing
running. Reads before the stop: save 1 (04:34) heuristic 0.51 / clone 0.58,
save 2 (06:18) 0.61 / 0.68, both averaged over the six candidate teams
(deployed baseline 0.46). Save 3 never came: throughput fell from 157 to 94
steps/s and, decisively, **`rollout/ep_rew_mean` climbed 0.6 -> 6.9 while
episodes lengthened 12 -> 20 turns** -- a shaped return that should telescope
to +-1 was being farmed. Cause found in `env.py`: the per-battle potential
was keyed by battle tag, but poke-env computes rewards for BOTH agents from
two battle views that share the tag, so each side overwrote the other's
potential. Fixed (key = tag + side; `test_both_sides_of_one_battle_telescope_
independently`). Saves 1-2 were trained on a corrupted signal: **discarded**;
the run is archived as `results_brainv1_attempt1_invalid/`, the valid
kill-line baselines kept in `results_brainv1/`. The relaunch (from the
deployed weights, same recipe) waits for the user's word. Standing rule
added to memory: a shaped run's mean episode return must stay within [-1, 1]
from the first rollouts on.

## Brain v1 merged and training team-agnostic on the six candidate teams (2026-September 15, 02:55)

- **Merged `brain-v1` into main** (e3071c8) between runs, as required; the
  one conflict (PROJECT_STATUS) resolved on main's log. Full suite on the
  merged code: **378 passed**, including the five exact-simulator tests the
  worktree could not run.
- **Loader fix found by the first evaluation (7792010):** SB3 rebuilds a
  stored checkpoint's network from the module constant for the token width,
  so the pre-merge deployed brain (1,197 floats per token) could not be
  loaded at all under the new width (1,205). `AttentionExtractor` now sizes
  `pokemon_proj` from the observation space it is given -- a stored
  checkpoint comes back at its own width and `upgrade_policy` extends it in
  memory (counter `policy_upgraded_obs_len`); the outcome-value and
  BC-agreement loaders upgrade too. Verified live: the deployed checkpoint
  loads, upgrades (1197 -> 1205, value bit-identical at 0.0988) and plays
  with the threat block active (52/52 states non-zero).
- **Kill-line baseline** (deployed brain vs the scripted opponent, 150
  battles per candidate team): T0 MB430 0.887, T1 0.273, T2 0.500, T3 0.300,
  T4 0.433, T5 0.387; mean **0.463**, kill line 0.363 (mean - 10pp). The
  team-agnostic eval averages over the six teams, so 0.46 is "no change".
- **Training launched 02:48** (`run_brainv1_training.sh`, log
  `brainv1_024826.log`): joint head on, shaping 0.10 faint / 0.05 HP,
  our side uniform over T0-T5, deployed-heavy pool (deployed x6 incl. resume,
  M-C clone x2, old champion, league-1 history), +8 intervals to 20,643,840.
  Resume upgraded in place (copied 127 tensors, zero-extended 3, added 5).
  First rollouts: **153 steps/s** (the threat block costs nothing
  measurable), approx_kl 0.026, clip 0.09, value loss 2.9 (the critic
  recalibrating to shaped returns). First save ~04:40; the watcher applies
  the first-save rule (kill line 0.363 AND paired -5pp on MB430).

## Round 6b verdict: both finalists FAIL on the scripted arm; the final save is +5.0 vs the M-C human clone and +2.9..+8.4 vs every PPO arm; brain tournament: the deployed brain is an MB430 specialist (2026-September 15, 02:00)

**Finalist 13762560 (save 1)** -- fresh-seed confirmation of the heuristic
arm (n=1,500, seed 8302): **-6.1 ± 1.2** (82.6 v 88.7), hardening the
screening read of -2.8; frozen +3.0, rotation1 +5.1, rotation2 +2.6, human
(eval_mcB_20260913) -0.3. Weighted +0.66. **FAIL.**

**Finalist 17694720 (save 5)** -- heuristic **-7.9 ± 1.5** (79.8 v 87.7,
BREACH), frozen **+5.7**, rotation1 **+8.4**, rotation2 **+2.9**, human
**+5.0 ± 1.7** (83.6 v 78.6). Weighted (human x2) **+3.18pp**, the largest
human-arm gain and the largest weighted gain of any candidate since league 1
-- and a FAIL by the pre-registered rule (no arm below -2pp). The shape of
rounds 3-5 again, sharper: everything that learns goes up, the scripted
opponent goes down. Diagnostics (eval_D = Reg M-B humans, mc_A = the pool
clone, 300) follow in `league6_verdict2.log`. Not deployable under the rule;
a ladder read of it would be measurement, not selection, and needs the
user's word.

**Brain tournament** (`results_team_tournament/deployed/`, the deployed brain
piloting each candidate vs the pool piloted by itself, 300 games): T0 MB430
82.3%, T2 TR/Psychic Terrain 75.0%, T4 sun 52.7%, T5 rain 49.0%, T3 46.0%,
**T1 consensus six 35.7%** -- versus the clone tournament's T1 64.3% / T0
62.7%. The deployed brain is an MB430 specialist and cannot pilot the meta's
own team; the team decision waits for the team-agnostic brain-v1 round
(chain armed: merge -> suite -> kill-line baseline -> pool -> training on
T0-T5 -> battery). Round 6 attempt 1 also showed the pool-softness trap
(see 02:20 entry); the brain-v1 pool copies round 6b's deployed-heavy shape.

## Paused for a machine restart (2026-September 13, 15:25)

All compute stopped at the user's word during finalist 1's heuristic
confirmation (n=1,500, seed 8302; not finished, reruns from scratch).
Resume = `scratchpad/verdict6_reordered.sh` (also under `training/` after
this commit): finalist-1 confirmation -> finalist-2 gate arms ->
diagnostics. Servers die with the restart; the runner restarts 7600 itself.
Pool from the 09-13 scrape (3,609 teams) committed here.

## Round 6b trained (five saves); finalist 1 screens CONFIRM with every PPO arm up; verdict reordered (2026-September 13, 15:20)

Training 02:14-11:28 at 148 steps/s, callback reads (100 battles each):
save 1 heuristic 0.89 / clone 0.86; save 2 0.85 / 0.81; save 3 0.68 / 0.87;
save 4 0.79 / 0.92; save 5 0.71 / 0.82. Triage finalists: save 1
(13,762,560) and the final save (17,694,720).

**Finalist 13762560 screening (M-C-anchored arms, n=1,000 paired, seed
83):** heuristic **-2.8 ± 1.5** (84.4 v 87.2, CONFIRM), frozen **+3.0**,
rotation1 **+5.1**, rotation2 **+2.6**, human (eval_mcB_20260913, stochastic)
**-0.3**; weighted (human x2) **+1.22pp**. The first candidate since league 1
to lift three PPO arms at once; the fresh-seed n=1,500 confirmation of the
heuristic arm decides.

The verdict chain was killed at 15:16 and relaunched reordered
(`verdict6_reordered.sh`, log `league6_verdict2.log`): the mcA diagnostic arm
(pool clone as opponent) had run 2h19m at ~79 decisions/min (the gate arms
run ~1,900/min) with the eval process CPU-bound and the server idle -- cause
not yet understood (the eval-only clone arm, same architecture, ran at full
speed). Order now: finalist-1 confirmation -> finalist-2 gate arms +
confirmations -> eval_D diagnostics -> mcA diagnostics bounded to 300.

## Chain resumed on refreshed data; round 6 attempt 1 killed at its first save; round 6b on a deployed-heavy pool (2026-September 13, 02:20)

- **Data refresh (00:03-00:14)**: scrape -> merged corpus of the three
  dated scrapes; pool rebuilt to **3,609** legal M-C teams; priors retrained
  (lead top-1 28.9%, move repertoire top-3 74.1%); trajectories A=2,066 /
  B=2,132; dated clones `mc_A_20260913` epoch 3 (39.2% agreement on B) and
  `eval_mcB_20260913` epoch 2 (39.9% on A, eval_only). The 09-10 clones stay
  as they were (the clone tournament's pilot keeps its provenance).
- **Battery smoke under the watchdog: 29/40 in 65 s** (deployed vs the M-C
  eval clone, M-C priors live). The 09-10 hang was a stale server.
- **Round 6 attempt 1 died at reset (00:15)**: the port-7700 server had run
  since Sep 7, before the simulator merge, so it did not know Reg M-C and all
  eight workers timed out ("Agent is not challenging"). Lesson recorded: every
  launcher restarts its server from the current build.
- **Round 6 attempt 1 (00:18-02:14), pool 40% clone: KILLED by the
  pre-registered first-save rule.** 148 steps/s, mean episode reward 0.90
  (the pool was too soft: an M-B-trained lineage piloting unfamiliar M-C
  teams plus a thin clone). Save 1 (13,762,560) read eval/heuristic **0.75**
  (kill line 0.80; every earlier round's first save read 0.82-0.89) and a
  paired 200-battle read on the M-C pool put it at **78.0% vs the deployed
  brain's 88.0%** on the same battles. Archived in `results_league6_attempt1/`.
- **Round 6b (02:20)**: same recipe, pool rebalanced to the deployed brain
  at 6 of 10 copies (incl. the resume point), M-C clone x2, old champion,
  league-1 history -- the deployed brain is the strongest Reg M-C pilot we
  have. The watcher now applies the first-save rule automatically (kill when
  eval/heuristic < 0.80 AND the paired read is worse than -5pp).

## Paused at the user's word; clone tournament read; venv repaired; exhibition mode online in Reg M-C (2026-September 10, 17:55)

- **01:20 pause** ("pause it rn"): the data layer had finished (scrape
  +1,392 replays -> merged corpus 1,615; pool 1,689 teams; M-C priors:
  preview lead top-3 52.5%, move repertoire top-3 72.5%, switch AUC 0.60;
  trajectories A=928 / B=998; clones `mc_A` epoch 4 (39.6% agreement on B)
  and `eval_mcB` epoch 3 (40.4% on A, role eval_only). **Round 6 never
  started**: the chain's battery smoke (not under the stall watchdog) sat
  25 min with an empty log; killed with the rest. On resume the smoke runs
  under `supervised_eval.sh`, then the round-6 build and launch.
- **Clone tournament complete** (`results_team_tournament/clone_mcA/`,
  `mc_A` piloting each candidate vs the weighted M-C pool played by itself,
  300 games/team): T1 consensus six 64.3% [58.8, 69.5]; T0 MB430 62.7%
  [57.1, 67.9]; T2 TR/Psychic Terrain 52.3%; T3 Psychic-terrain offence
  46.3%; T4 sun 42.3%; T5 rain 28.3%. T1 and T0 are inside each other's
  interval; the brain tournament decides. Pilot proxy only.
- **17:50 venv dead**: Homebrew's `python@3.13` had been removed during the
  day (3.12 and 3.14 remained), so `.venv/bin/python` was a dead symlink and
  every script failed. `brew install python@3.13` (3.13.15) restored it in
  place; torch/poke-env/SB3 import. Recorded in the runbook memory.
- **17:54 exhibition mode online** (`exhibition_mode.sh`, now Reg M-C with
  the three opt-in guards, unbuffered, yields to clone/tournament jobs):
  logged in as antonius1, format gen9championsvgc2026regmc, awaiting
  challenges; the M-C priors were picked up automatically through
  `utils.prior_path`. Replays -> `ladder_replays_exhibition_mc/`.

## First Reg M-C ladder read: 13-12 (52.0%), deployed brain + three guards, Reg M-B priors (2026-September 10, 00:50)

25 games 23:43-00:46 (`ladder_replays_mc_guards3_20260909/`, one session,
no dead sockets; the M-C queue matched in under a minute all night). Record
**13-12**, six of the wins by opponent forfeit; opponents averaged ~1110 Elo
(the M-C ladder is a day old; ours started at 1000, 1128 at the end). Not a
climb claim -- 25 games is a non-regression read -- but the bot is not
collapsing on a format it never trained on: loss profile
(`results_analysis/loss_profile_mc_read_20260910.*`) reads like the M-B one
in miniature: first faint ours -> 12.5% (1/8) vs 70.6%; >= 2 super-effective
hits -> 77.8% vs 37.5%; we set Tailwind by turn 2 -> 69.2% vs 33.3%;
opponents bringing Salamence 1/5, Indeedee-F 2/6, Sneasler 4/10, Rillaboom
5/11; losses average 4.0 faints for us vs 1.75 for them. Hit effectiveness:
our attacks 18% super-effective / 16% resisted vs opponents' 35% / 15% --
the coverage gap is the meta's, the resisted share is now the guards'.

**The guards fired for real.** 232 decisions: `resisted_target` promoted a
re-aimed twin 21 times (9% of decisions), `overkill_split` 11 times,
`dominated_weather_ball_weather` 2; our single-target attacks into a
resisting foe with a second foe on the field fell to **5/79 (6.3%)** from
15% in the August/September reads. No traceback, no stall, no illegal order.

Chain `after_ladder_mc.sh` took over at 00:47 (tally done; scrape running):
M-C data layer -> clones -> battery smoke -> round 6. The clone tournament
waits for the clone (`team_tournament_clone.log`); the round-6 first-save
watch is armed (`round6_watch.log`).

## Reg M-C program opened: the first M-C ladder read is running; design doc, loss profile, data-layer chain (2026-September 10, 00:05)

User decision (2026-09-09 evening): the bot becomes a Reg M-C bot -- train on
M-C battles only from here, build a new M-C team, and design a new brain from
the ladder losses without assuming the current team survives. Tonight:

- **M-C ladder read launched 23:43** (deployed brain + resisted_target /
  overkill_split / dominated_weather_ball_weather, `--reg mc`, MB430, Reg M-B
  priors as-is; `ladder_replays_mc_guards3_20260909/`, queue-resilient loop,
  25 games). Game 1 lost to a Rillaboom / Incineroar / Lucario / Kingambit /
  Palafin / Farigiraf team. Tally appended below when the read ends.
- **`NEW_BRAIN_PLAN.md`** -- the program: evidence, what the current brain
  cannot represent (slot 2 blind to slot 1's choice; one-sided knowledge; no
  memory; terminal-only reward; single-team training; M-B-only priors),
  ranked new-brain ideas (joint-action head, threat-symmetric knowledge,
  memory tokens, shaping + auxiliary heads + critic warm start, team-agnostic
  training, M-C clones; preview specialist and search-as-teacher parked),
  team candidates T0-T5 with a measured selection protocol, pipeline M0-M5.
- **`tools/ladder_loss_profile.py`** over all 159 deployed-brain ladder games
  (`results_analysis/loss_profile_deployed_mb.json`): 48.4% overall; we lose
  the opening exchange -- first faint ours 26.2% vs 63.8%; first KO on turn
  1-2 in 82% of games; behind on faints after turn 2 -> 24%; >= 2
  super-effective hits 61.1% vs 29.7%; opponent Trick Room 31.0%; opponent
  turn-1 Fake Out 33%; Basculegion + Whimsicott lead 35.3% (Charizard +
  Whimsicott 57.7%); Basculegion first faint 9.1%; losses average 4.0 vs 1.8
  faints (5 of 82 losses were close).
- **M-C meta** (`results_analysis/mc_meta_stats_20260909.json`, 565 games):
  Sneasler on 48% of teams, Rillaboom 40%, Basculegion 33%, Incineroar /
  Salamence / Kingambit 31%; consensus six Floette / Incineroar / Kingambit /
  Rillaboom / Salamence / Sneasler (14/22 as an exact roster); Trick Room in
  26% of games, setter side wins 60%. MB430's problems in M-C: Grassy Terrain
  halves Earthquake, Grassy Glide priority, Fake Out on three of the top four
  species, double Intimidate, Close Combat everywhere.
- **Regulation-aware priors**: `utils.prior_path(kind, reg)` picks
  `data/opponent_*_top500_reg<reg>.pt` when it exists (M-B fallback, path
  recorded); `ladder_ourteam.py` and `eval_counterfactual.py` default through
  it; `run_gate_battery.py` gained `--reg` / `--our-team` passthrough (both
  recorded in the scorecard). `top_500_rating_floor` now knows the M-C
  formats (floors 1140 bo1 / 1045 bo3 from the 09-09 snapshot) -- before this
  every M-C replay would have been dropped by the stale 1655 Reg M-B floor
  with no warning. `datagen/merge_battle_logs.py` de-duplicates dated
  scrapes. Tests: `unit_tests/test_regulation_priors.py` (+3, suite subset
  green).
- **Automatic chain after the read** (`after_ladder_mc.sh`): tally
  (record, loss profile, hit effectiveness, guard fire counts) -> re-scrape ->
  merged corpus -> team pool + weights -> three M-C priors -> trajectories
  A/B -> clones mc_A (epoch by agreement on B) and eval_mcB (eval_only) ->
  M-C battery smoke -> `training/league6_config.json` -> **round 6** (first
  M-C league round, `run_league6_training.sh`, +5 intervals) -> triage ->
  `run_league6_verdict_supervised.sh` (M-C-anchored arms + mc_A / eval_D
  diagnostics). Markers in `after_ladder_mc.log`.

## Reg M-C wired in: regulation id, a 625-team pool from open team sheets, opponent weights (2026-September 9, 10:14)

`format_map["mc"] = gen9championsvgc2026regmc` (`vgc_bench/src/utils.py`,
tests). `datagen/extract_ots_teams.py` turns the `|showteam|` lines of
open-team-sheet replays (every Bo3 game, opt-in Bo1) into Showdown export
teams -- species/item/ability/moves/nature/gender/level from the sheet,
spreads by a nature rule in Champions units (boosted offense + Speed at 32,
HP for Speed-lowering natures, leftover 2 to HP or Def; cap 32, budget 66)
-- and keeps only teams the merged simulator validates. From today's 565
M-C replays: 625 unique sheets, **625 valid M-C teams** in `teams/reg_mc/`
(+ `our_team.txt` = MB430). `data/team_weights_regmc.json` built from the
same replays' previews (uniform mix 50%). Early M-C meta by preview count
(bo1): Sneasler, Kingambit, Incineroar, **Rillaboom**, **Salamence**,
Basculegion, **Indeedee-F**, Farigiraf, Floette-Eternal, Garchomp,
**Golisopod**, Pelipper, Whimsicott, Archaludon, Gardevoir, Charizard --
three of the newly unlocked Pokémon are already top-11. Re-scrape and
rebuild as the format matures. Still M-B-only: the opponent priors
(preview/move/switch models), the human clones, the exploiter, and all
gate batteries.

Reproducibility note: `pokemon-showdown` is a submodule whose recorded
commit (branch `vgc-bench-mc`) exists only locally -- the submodule URL is
cameronangliss's fork. Publishing it needs a fork under the user's GitHub
and a `.gitmodules` URL change; flagged to the user.

Sanity eval on the merged simulator (10:14, `--reg mc`, M-C team weights,
200 battles, 8 workers): the deployed brain beats the scripted opponent
**86.5%** on M-C teams (vs ~90-92% on M-B teams), zero parse errors, no
odd counters, 64 s; opponents faced were led by Sneasler, Rillaboom,
Basculegion, Salamence, Kingambit, Incineroar, Indeedee-F. The whole
stack -- simulator, poke-env data, observations, guards -- works in Reg
M-C; the brain is merely a few points weaker against Pokémon it has never
seen. A first M-C ladder read (deployed + three guards, `--reg mc`,
M-B priors as-is) is prepared, not launched.

## Simulator updated for Reg M-C; MB430 validated locally; suite green (2026-September 9, 10:10)

`pokemon-showdown/` now sits on branch **`vgc-bench-mc`**: smogon master
as of 2026-09-09 (194 commits) merged into the fork branch (47 fork-only
commits kept: room-upkeep timeouts, batch validation, the champions
randomNormal fix, legacy VGC formats); conflicts were `config/formats.ts`
(took upstream, re-applied the fork's legacy entries, dropped the one
upstream now ships itself, VGC 2024 Reg G) and `package-lock.json` (took
upstream, `npm install`). `node build` ok. Local validator: **MB430 valid
in gen9championsvgc2026regmc and regmb** (control: Rillaboom rejected in
M-B, legal in M-C). A server from the new build on 7610 runs the whole
suite: **351 passed, 5 skipped**. The server logs three harmless REPL
socket "listen EINVAL" lines (unix-socket path too long under this
directory; battles unaffected -- same as before). The previous branch
(`vgc-bench`, 2026-06-20) remains for reproducing M-B artifacts. The M-C
corpus (565 replays) is in `battle_logs_top_mc_20260909/`.

## Reg M-C, what it is and what it takes (2026-September 9, 10:08)

Upstream definitions: Reg M-C = `mod: champions` (current data), ruleset
Flat Rules / VGC Timer / Open Team Sheets; Reg M-B is now `mod:
championsregmb`, a frozen snapshot, `searchShow: false`. Diff of the two
mods' `formats-data.ts`: **35 entries unlock in M-C, none removed** --
Wigglytuff, Persian(-Alola), Perrserker, Farfetch'd/Sirfetch'd, Mr. Mime,
Swalot, Gogoat, Golisopod (+Mega), Rillaboom, Cinderace, Inteleon,
Thievul, Toxtricity, Grapploct, Pincurchin, Indeedee(-F), Arboliva,
Baxcalibur (+Mega), Pawmot, Squawkabilly, Mabosstiff, Salamence (+Mega),
and Mega Z forms of Garchomp, Lucario, Absol. Our six species are unchanged
(same tiers) -- MB430 is legal. poke-env's data knows every new species, so
observations and parsing hold; the brain has simply never met them.

Groundwork: (1) the bundled simulator (fork branch 2026-06-20, 194 commits
behind smogon master, 47 fork-only commits: room upkeep, batch validation,
legacy VGC formats) is being merged with upstream on a new branch
`vgc-bench-mc` (conflicts: formats.ts, package-lock) -- needed before any
M-C training or local gate; then rebuild, test suite, and a local
validation of MB430 in M-C. (2) The scraper knows M-C; the top-500 M-C
corpus is 560 replays today (players 1045-1423, median ~1150; the format is
days old) -- scraped into `battle_logs_top_mc_20260909/`, too small for a
clone yet, enough for rough team weights; re-scrape daily. (3) VGCPastes has
no Champions M-C sheet yet, so an M-C team pool has to come from open-team-
sheet replays (a new extractor) until one appears. The team question is now
forced by the format, not chosen. Decision pending with the user: a first
M-C ladder read with the deployed brain + guards.

## FORMAT ROTATION: the Reg M-B ladder is closed; Reg M-C is live (found 2026-September 9, 10:04)

Ladder read A (deployed + three guards) queued for 78 minutes across two
sessions with zero games and two dead sockets. A search diagnostic with
full server messages (`search_test.py`, secrets filtered) returned the
cause: **`|popup|Error: Your format gen9championsvgc2026regmb is not
ladderable.`** The server's format list now flags Reg M-B `5c`
(challenge/tournament only) and **VGC 2026 Reg M-C `5e` (ladderable)**.
Sunday's 34 games were still Reg M-B; the rotation happened between Sept 6
and Sept 9. The keepalive deaths were idle connections with no search
running; login itself takes 1 s. The bundled Showdown copy (branch from
2026-06-20) predates Reg M-C, so the local sim cannot validate or train the
new format until it is updated. Consequences: no Reg M-B ladder read is
possible (Reg M-B remains playable by challenge, e.g. exhibition mode);
every pipeline asset is Reg M-B (team pool, priors, human clones,
exploiter, team weights); the account's Reg M-C rating starts fresh.
Bot offline; nothing queued.

## Ladder read A launched on the user's go: deployed brain + the three guards (2026-September 9, 08:44)

08:42: `ladder_ourteam.py --checkpoint results_league/league_champion.zip
--guards-extra resisted_target,overkill_split,dominated_weather_ball_weather
--n_games 25 --replay_dir ladder_replays_guards3_20260909` (run_config:
guards_extra recorded, profile hard, sha 8cc54b2b, mixing off, knowledge obs
on). Serial, credentials sourced in the launching shell. The read is a
non-regression check plus a look at whether the three targeting fixes show
up against real people; the account sits around 1000-1100 after Sunday's
collapse, so opponents are weaker than in August. Analysis after: record,
first-faint rate, `tools/hit_effectiveness.py`, per-guard firing counts
from decisions.jsonl. Promotion of the guards into HARD_GUARDS (the deployed
profile) only on a clean read and with the user's word.

## Round 5 VERDICT: both finalists FAIL; five fine-tuning rounds since league 1, zero passes -- the synthesis (2026-September 9, 04:34)

Supervised chain complete, every arm on its first attempt (no stall).
Cards vs the DEPLOYED champion, n=1,000 paired:

| arm | 15728640 (save 3) | 17694720 (final) |
|---|---|---|
| adversary (informative; no adversary in the pool) | +11.6 (52.2 v 40.6) | +14.6 (54.7 v 40.1) |
| human holdout eval_B (Aug clone) | -0.7 | -0.5 |
| heuristic | **-0.2** (clean) | -2.7 |
| frozen 64opp | -1.6 | **-3.3** |
| rotation 8opp | **-5.5** | **-5.5** |
| rotation tuned | -2.1 | -1.2 |
| weighted (human x2) | -1.8 | -2.3 |
| mix_A diagnostic | +1.1 (clean) | -0.7 (clean) |
| eval_D (Sep clone) diagnostic | **-1.6** | **-2.1** |

Both breach on the rotation 8opp arm beyond the confirmation window (the
final also on frozen); the fresh-seed confirmations were cancelled as
decision-irrelevant. No promotion, no ladder. The no-adversary hypothesis
was half right: the heuristic tax vanished for save 3 (-0.2) and softened
for the final (-2.7 vs -6.5 in round 4), but the PPO-arm tax stayed (-5.5)
and, decisively, neither finalist gained anything against September human
play (eval_D -1.6 / -2.1) despite training on September clones and
September teams. Both still beat the exploiter by +12..+15 without ever
seeing it -- generalization from the human clones, and a reminder that the
exploit meter is not a ladder proxy.

**Synthesis of the five rounds since the league-1 promotion** (2, 3, 3b,
4, 5; nine finalists): none passed. Every fine-tune from the deployed
weights at this learning rate moves the policy toward its pool and costs
2-7pp on some August population; the September-anchored diagnostics (round
4) showed those costs are real on September teams too; the only candidate
with a real September-human gain (round-4 save 1, +1.6..+4.7) paid -3.4 to
-3.7 on three arms. The deployed brain itself went 11-34 (32%) on the
September ladder. What IS measured and deployable: the three targeting
guards (`resisted_target` v3 every arm up, weighted +1.15; the bundle
weighted +1.2), all opt-in, zero regressions.

Decision brief for the user (nothing runs without their word):
(1) **Ladder read A -- deployed brain + the three guards, 25 games**: the
safe, measured change; script prepared. (2) Ladder read B -- round-4 save 1
+ guards, 25 games: the only "adapted" candidate; real local costs; an
experiment, not a promotion. (3) The strategic fork: fine-tuning from the
deployed weights looks plateaued; the remaining levers are the TEAM (cycle
complete, MB430 frozen since Aug 29; the analysis file shows Trick Room
rosters and mirror pieces dominating September), a from-scratch league on
September data (expensive; the PC port would make it feasible), or
accepting the current brain with the guards. Bot offline.

## Round 5 finalist 15728640 (save 3): FAIL -- heuristic tax gone, PPO-arm tax not; no September-human gain (2026-September 9, 02:10)

Supervised chain, every arm completed on the first attempt (no stall).
vs the DEPLOYED champion, n=1,000 paired: adversary **+11.6** (52.2 v 40.6;
informative -- no adversary was in the pool), heuristic **-0.2** (the first
candidate since league 1 with a clean scripted arm), frozen -1.6, rotation
8opp **-5.5** (breach), rotation tuned -2.1, human holdout eval_B -0.7 ->
weighted -1.8, FAIL. Diagnostics: mix_A +1.1 (divergence 1.8pp, clean);
**eval_D -1.6** (80.7 v 82.3) -- no gain against September human play
either. Reading: removing the adversary removed the heuristic tax, as
predicted, but the round did not buy anything on the human arms and paid
on a PPO arm. Final checkpoint 17694720's chain started 02:10; fresh-seed
confirmations for the flagged arms follow.

## Resumed (23:44): stall not reproducible; round-5 verdict relaunched under a stall watchdog (2026-September 8, 23:46)

Single-worker diagnostic of the stalled checkpoint (15728640 vs the
exploiter, 12 games per arm, seed 91, fresh server): no stall, no odd
counters, zero errors; candidate 9/12 vs deployed 3/12 (informative only).
The freeze is rare or concurrency-related, so the fix is structural rather
than a hunt: `evaluation/supervised_eval.sh` runs one eval under a
watchdog (kill + fresh server + retry when its log stops growing for 20
min, up to 2 retries; skips outputs that exist), and
`run_league5_verdict_supervised.sh` runs every arm that way -- exploit
re-measure (INFORMATIVE: the pool held no adversary), the five screening
arms, mix_A and eval_D -- for both finalists, printing the verdict per
finalist. Launched 23:45 (`league5_verdict_supervised.log`); ~2h15m per
finalist, done ~04:20. Bot offline.

## PAUSED at the user's order (10:14); the round-5 verdict chain had stalled silently (2026-September 8, 10:16)

Everything stopped: no training, evals, ladder or exhibition running. The
chain's state at the stop: the deployed arm of finalist 15728640's exploit
re-measure completed normally (406/1000 = 40.6%, 708 s), then the candidate
arm made ~4,000 decisions (~180 games' worth) and produced nothing further
for ~2 hours -- no error, warning or traceback in the log. That is the
frozen-battle stall class (a handler exception swallowed by poke-env; with
8 concurrent battles the arm freezes once all 8 are stuck). Cause unknown
until a single-worker diagnostic with full counters runs on that checkpoint;
NOT run while paused. Round-5 checkpoints are intact on disk. Resume plan
when the user says so: diagnostic (~10 games, 1 worker) -> fix or work
around -> relaunch `run_league5_verdict.sh` (with the exploit gate treated
as informative).

## Round 5 complete; verdict chain running on save 3 + final (2026-September 8, 07:40)

Five checkpoints, zero errors, 154-157 steps/s. Probes (heuristic / bc,
n=100): 0.89/0.78, 0.83/0.79, 0.78/0.90, 0.83/0.82, 0.77/0.75 -- no
adversary in the pool and the scripted-opponent probe never dropped the
way rounds 3-4 did; the human-clone probe reached 0.90 at save 3.
Finalists (tensorboard triage): **15728640** (save 3, best probe sum) and
**17694720** (final). `run_league5_verdict.sh` from 07:39: exploit
re-measure (informative here -- the pool held no adversary), then the
5-arm screening vs the DEPLOYED champion + mix_A diagnostic + eval_D arm.
Because the league-3 chain gates the battery on the exploit bar, a
fallback (`round5_fallback.sh`) runs the battery, mix_A and eval_D arms for
any finalist the chain skips. Verdict ~11:30-12:30. Bot offline.

## Round 5 save 1: clean, continue (2026-September 8, 00:31)

Save 1 (13,762,560): eval/heuristic **0.89** (kill line 0.80; the best
first-save heuristic probe of any round -- rounds 3/3b/4 opened at 0.71,
0.84, 0.82), eval/bc **0.78** (line 0.70); 157 steps/s, ep_rew +0.64..+0.66,
zero errors. Consistent with the round-5 hypothesis: without an adversary
in the pool the scripted-opponent probe does not drop. Original kill
criteria in force for saves 2-5; completion ~07:45, then `after_round5.sh`.

Save 2 (14,745,600, 02:15): eval/heuristic **0.83**, eval/bc **0.79**; fps 155.
Continue.

Save 3 (15,728,640, 04:00): eval/heuristic **0.78** (one-SE wobble after 0.89 /
0.83), eval/bc **0.90** (highest of any round); fps 155, ep_rew +0.80.
Written rule as in round 4: continue; kill only if save 4 also reads < 0.78.

Save 4 (16,711,680, 05:45): eval/heuristic **0.83**, eval/bc **0.82**; fps 154.
Rule satisfied; continue to save 5 (~07:30).

## Round 4 closed: both finalists fail on August AND September anchoring; round 5 launched -- the league-1 recipe on September data (2026-September 7, 22:45)

September-anchored diagnostic cards (opponents on September-weighted
teams, eval_D as the human arm; n=1,000 paired) beside the August ones:

| arm | 13762560 Aug / Sep | 17694720 Aug / Sep |
|---|---|---|
| heuristic | -3.8 / **-3.4** | -6.5 / **-6.4** |
| frozen 64opp | +0.5 / +0.7 | -2.0 / -0.6 |
| rotation 8opp | -3.9 / **-3.4** | -2.3 / **-3.8** |
| rotation tuned | -7.9 / **-3.7** | -1.5 / -1.9 |
| human (eval_B / eval_D) | -5.0 / **+1.6** (+4.7 on Aug teams) | -2.4 / -0.2 (+1.0 on Aug teams) |
| weighted | -4.2 / -1.1 | -2.9 / -2.2 |

Reading: the team re-weighting moves nothing; the scripted and PPO-arm
regressions are properties of the brains. Save 1 does carry a real gain
against September human play (+1.6 on September teams, +4.7 on August
teams) -- the adaptation is real but small, and it comes with -3.4 to -3.7
against three non-adaptive populations. Verdict: FAIL on both anchorings,
no promotion, no ladder. Save 1 is stamped `candidate` (sha e99ea7eb) and a
25-game ladder-read script exists for it (not launched; a user override).

Four adversary rounds (3, 3b, 4 x2) now say the same thing: the exploiter
in the pool closes its own exploit and costs 2-7pp against the scripted and
PPO populations, whatever the dose, whatever the anchoring. The one recipe
that ever transferred to ladder had no adversary: league 1 (human-BC at
37.5% + self-lineage). **Round 5 launched 23:0x** (`league5_*.log`,
`training/league5_config.json`): the league-1 recipe on September data --
mix_C x2, mix_AC x2, mix_A x2 (human 46%), old champion + league-1 history
x3 + deployed x2 (+ resume), NO exploiter, SEPTEMBER opponent team weights
with TR rosters x1.5, eval_B and eval_D banned. +5 intervals, ~9h; save-1
kill lines as before; `after_round5.sh` (tensorboard triage) ->
`run_league5_verdict.sh` (league-3 bars + eval_D arms). Bot offline.

Decision brief for the user: (1) ladder read of the DEPLOYED brain + the
three passed guards (25 games) is the lowest-risk ladder action available
now; (2) a ladder override for round-4 save 1 is possible but not
recommended (real -3.4..-3.7 arms, small human gain); (3) round 5's verdict
lands ~11:30 tomorrow.

## September-anchored diagnostic, final checkpoint: still a FAIL -- real regression, no human gain (2026-September 7, 21:15)

17694720 vs the deployed brain with opponents on SEPTEMBER-weighted teams
(`data/team_weights_regmb_sep.json`, 259 of 546 bundled teams matched
against 6,529 September previews) and eval_D as the human arm, n=1,000
paired (`results_gate_battery_league4/17694720_sep/`):

| arm | August-anchored | September-anchored |
|---|---|---|
| heuristic | -6.5 | **-6.4** |
| frozen 64opp | -2.0 | -0.6 |
| rotation 8opp | -2.3 | **-3.8** |
| rotation tuned | -1.5 | -1.9 |
| human (eval_B / eval_D) | -2.4 | **-0.2** (81.5 v 81.7) |
| weighted | -2.9 | -2.2 |

The team re-weighting changes almost nothing: the heuristic and PPO-arm
regressions are properties of the brain, not of the opponents' teams, and
against September human play the final checkpoint is level with the
deployed brain (-0.2 here, +1.0 on August teams). Verdict on either
anchoring: FAIL. 17694720 is out. Save 1 (13762560: +4.7 on eval_D, -5.0
on eval_B) is the only candidate with a real September-human gain; its
September-anchored battery runs next (~21:15-23:00).

## First current-meta reading: round-4 save 1 is +4.7 on the September clone while -5.0 on the August clone (2026-September 7, 19:31)

eval_D arm (September human clone, stochastic, n=1,000 paired):
**13762560 86.1% vs deployed 81.4% = +4.7pp** (first faint ours 23% vs
24%). The same checkpoint read **-5.0** against eval_B, the August clone --
a 9.7pp swing between two human-imitation arms that differ only in which
month's top-500 games they were cloned from. That is the adaptation
signature the August-anchored battery cannot see: the brain moved toward
September human play and away from August human play. Memorization checks
are clean for both finalists (mix_A divergence 1.9 / 0.4pp). The final
checkpoint's eval_D arm is running; then the September-anchored batteries
(September team weights + eval_D as the human arm) for the final
checkpoint and, queued after it, for save 1. If those show non-regression
on September populations, the gate itself is the thing that is stale, and
re-anchoring it is the user's decision.

Final checkpoint 17694720 vs eval_D (19:50): **81.0 vs 80.0 = +1.0pp**
(first faint ours 22% vs 23%) against -2.4 on the August clone -- the same
direction, a third of the size. Save 1 is the stronger September-human
candidate (86.1%) despite the worse August card. Verdict chain complete
(L4_VERDICT_COMPLETE 19:50); September-anchored batteries next: final
checkpoint first, then save 1.

## Round 4 VERDICT on the pre-registered battery: both finalists FAIL (2026-September 7, 17:16)

Full cards vs the DEPLOYED champion (n=1,000 paired, hidden sheets):

| arm | 13762560 (save 1) | 17694720 (final) |
|---|---|---|
| adversary (exploiter, stochastic) | **+22.5** (63.4 v 40.9) | **+23.0** (64.8 v 41.8) |
| human holdout eval_B (Aug clone) | **-5.0** (80.1 v 85.1) | -2.4 (84.3 v 86.7) |
| heuristic | **-3.8** (88.4 v 92.2) | **-6.5** (84.6 v 91.1) |
| frozen 64opp | +0.5 | -2.0 |
| rotation 8opp | **-3.9** | -2.3 |
| rotation tuned | **-7.9** | -1.5 |
| weighted (human x2) | -4.2 | -2.9 |
| mix_A diagnostic | -3.1 (vs eval_B -5.0: 1.9pp, clean) | pending |

Both breach the rule (13762560 on four arms; 17694720 on the heuristic,
-6.5, with the other four inside the confirmation window and all
negative). No promotion; no ladder. Mechanism marker: first faint ours vs
the heuristic 31-33% for the candidates vs 23-24% for the deployed brain --
the adversary-hardened opening is more committal and the scripted
max-damage opponent punishes it; vs the human clone the final checkpoint
loses its first mon LESS often (23% vs 25%) yet still loses more games.

Cross-round pattern, now four rounds deep (3, 3b, 4 with two finalists
each): every pool that contains the adversary closes the exploit (+15..+32)
and pays on the August-anchored arms. Round 4 -- the September clones at
38%, the exploiter capped at 15%, Trick Room rosters boosted -- paid the
MOST on the heuristic and the August human clone. Two readings: (a) the
candidates are simply worse; (b) the battery is anchored to August -- its
PPO arms and the heuristic pilot AUGUST-weighted opponent teams
(`data/team_weights_regmb.json`, built from the Aug-1 scrape) and its human
arm is the August clone -- so a brain adapted to the September field is
measured against the field it was moved away from. The two eval_D arms
(September clone; deployed baseline 80.9%) running now are the first
current-meta reading; a September-anchored diagnostic battery (opponent
team weights rebuilt from the 2026-09-06 scrape, eval_D as the human arm)
is being prepared as a DIAGNOSTIC, not a replacement for the pre-registered
gate -- changing the gate is the user's decision.

## Round 4 finalist 13762560 (save 1) FAILS broadly (2026-September 7, 15:32)

5-arm screening vs the DEPLOYED champion (n=1,000 paired): heuristic
**-3.8**, frozen +0.5, rotation 8opp **-3.9**, rotation tuned **-7.9**,
human holdout **-5.0** (80.1 v 85.1) -> weighted -4.2. Four breaches beyond
the confirmation window; the worst human-holdout reading of any finalist in
rounds 3-4, despite +22.5 vs the exploiter. mix_A diagnostic running for
the record; the final checkpoint 17694720's battery follows (~15:50-17:30)
and decides the round.

## Round 4: both finalists qualify on the exploit re-measure (2026-September 7, 14:07)

n=1,000 paired vs the final exploiter (stochastic), seed 83: **13762560
63.4% vs deployed 40.9% (+22.5)**, first faint ours 34% vs 58%; **17694720
64.8% vs 41.8% (+23.0)**, first faint ours 28% vs 57%. The 15% adversary
dose closes the exploit like the 31% and 10% doses did. Both go to the
5-arm screening battery vs the DEPLOYED champion + mix_A diagnostic
(13762560 first, from 14:06), then the eval_D (September clone) arms;
verdict ~19:00. Same pre-registered bars; tiebreak as written 09-05 (higher
weighted delta among passers; exploit margin breaks ties within 1pp).

## Round 4 complete; verdict chain running on two finalists (2026-September 7, 13:28)

Five checkpoints, zero errors, 155-157 steps/s throughout. Probes
(heuristic / bc, n=100): 0.82/0.77, 0.78/0.77, 0.85/0.70, 0.77/0.74,
0.78/0.75 -- flat, noisy around the 0.80 line, no shock and no trend.
Finalists: **13762560** (save 1, best probe sum) and **17694720** (final,
most exposure to the new pool). Incident: the training log is
stdout-buffered under nohup, so the SB3 tables never reached it and the
log-only triage found no finalists; the probes were taken from tensorboard
and the chain launched by hand at 13:28 (`triage_league_log.py` now falls
back to tensorboard, `--results-dir`). `run_league4_verdict.sh`: exploit
re-measure for both (qualify at >= 44.8%), then for qualifiers the 5-arm
screening vs the DEPLOYED champion + the mix_A memorization diagnostic +
the eval_D (September clone) diagnostic arm, baseline 80.9%. Same
pre-registered bars as rounds 3/3b (no arm below the deployed brain by
>2pp after the confirmation clause, weighted >= 0, mix_A divergence <=
10pp); ladder only on a pass and with the user's word.

## Round 4 save 2: heuristic probe 0.78 (line 0.80), bc 0.77 -- continue under a written rule (2026-September 7, 08:08)

Save 2 (14,745,600): eval/heuristic **0.78**, eval/bc **0.77**; fps 156,
ep_rew +0.50..+0.68, approx_kl on target, zero errors. The probes are n=100
(SE ~4pp): 0.78 after 0.82 is a one-SE wobble, not the catastrophic
forgetting the 0.80 line was written for (round 3 opened at 0.71/0.67).
Recorded deviation, same form as round 3's: continue; **kill if save 3
also reads below 0.78 on the heuristic probe or below 0.70 on bc.** The
battery, not the probes, decides anything that matters.

Save 3 (15,728,640, 09:55): eval/heuristic **0.85** (recovered), eval/bc
**0.70** (at the line, not below it); fps 156, ep_rew +0.62. Continue.

Save 4 (16,711,680, 11:40): eval/heuristic **0.77**, eval/bc **0.74**. Four
saves: 0.82 / 0.78 / 0.85 / 0.77 -- noise around the 0.80 line, no trend;
continue to save 5 (~13:25); the battery decides.

## Round 4 save 1: clean, continue (2026-September 7, 06:23)

Save 1 (13,762,560): eval/heuristic **0.82**, eval/bc **0.77** (kill lines
0.80 / 0.70), 156-157 steps/s, ep_rew +0.38..+0.72, approx_kl ~0.022, clip
~0.11, bc_opp_frac 0.875 (12 seeded files of 13), zero errors. No adaptation
shock at the 15% adversary / 38% human-clone mix (round 3's 31% dose opened
at 0.71/0.67). Original kill criteria stay in force for saves 2-5;
completion ~14:00, then `after_round4.sh` triages finalists from the probes
and runs `run_league4_verdict.sh`.

## Payoff matrix done: the pure equilibrium is 100% exploiter; round 4 launched 04:38 on a hardness-weighted pool (2026-September 7, 04:39)

The 3x10 matrix (`results_meta_game/round4/meta_game.json`, n=300/cell):
the final exploiter beats every one of our rows harder than anything else
(deployed 40%, r3b final 66%, old champion 53%), so the zero-sum
equilibrium is degenerate -- y* = 100% exploiter, the sparring-partner-only
mixture rounds 3 and 3b already showed overfits. After it, the hardest
columns for the deployed brain are mix_AC (81%), the r3/r3b finals (83-84%)
and everything else at 85-88%. Interesting aside: the ROW equilibrium is
100% r3b final -- the exploit-hardened checkpoint is the most robust of our
three against this population.

`make_league4_config.py` therefore uses a **hardness mixture** (weight =
1 - deployed win rate per column, every member kept), then the documented
overrides with mass redistributed: exploiter capped at 2 of 12 (its raw
weight was 31%), each September clone floored at 2. Final seeded pool (+
the deployed brain at the resume stem): exploiter 2, mix_C 2, mix_AC 2,
mix_A 1, deployed 1, r3 final 1, r3b final 1, old champion 1, league-1
11796480 1 -- 13 files: adversary 15%, human clones 38%, self-lineage 46%;
`tr_boost` 1.5 on the opponent team pool; eval_B and eval_D banned by
content. Verified; training launched **04:38** (`league4_043806.log`, +5
intervals to 17,694,720, ~9h). Kill criteria at save 1 as in round 3b
(eval/heuristic >= 0.80, eval/bc >= 0.70, no worker deaths). Recorded
incident: the auto-launcher's wait condition passed before the fix landed
and it trained ~2 minutes on the degenerate 5-file pool (04:36); killed,
pool wiped and rebuilt; the eval_D baseline (n=1,000) runs alongside the
first interval.

eval_D baseline (05:00, n=1,000 stochastic, seed 83): the deployed brain
beats the September human clone **80.9%** vs 85.6% against the August clone
(eval_B) -- the current-meta imitation is ~5pp harder for the deployed
brain, the same direction as the ladder collapse. `results_gate_battery_
evalD/deployed_vs_evalD_1000.json`; this is the reference for the round-4
finalists' eval_D diagnostic arm.

## `resisted_target` v3 (reranker revert fixed) PASSES clean: every arm up, no confirmation needed (2026-September 7, 03:23)

Deployed + `resisted_target` v3 vs deployed, screening n=1,000 per arm,
seed 83 (`results_guard_probe/resisted_target_v3/`):

| arm | delta |
|---|---|
| adversary (informative) | 0.0 (41.7 v 41.7) |
| human holdout eval_B | **+1.2** (86.1 v 84.9) |
| heuristic | +1.1 (92.2 v 91.1) |
| frozen 64opp | +1.3 (86.9 v 85.6) |
| rotation 8opp | **+2.0** (88.5 v 86.5) |
| rotation tuned | +0.1 (84.9 v 84.8) |
| weighted (human x2) / equal | **+1.15 / +1.14** |

PASS with every arm non-negative and nothing inside the confirmation
window -- the cleanest card of any inference-time rule in the project. With
promotions no longer reverted the guard fires 430-700 times per 1,000 games
(0.4-0.7 per game; ~93% promotions of a pair the policy had ranked, ~7%
injected), stands down ~100 times per 1,000 games because the resisted hit
already KOs, and 1-10 times because the calculator disagrees; zero errors.
Note the v2 card (+0.96 weighted, one confirmation) was measured with most
corrections silently undone by the reranker; v3 is the real effect.

Three opt-in guards now hold screening passes: `resisted_target` (v3),
`overkill_split`, `dominated_weather_ball_weather`. They were measured as
two separate candidates; the combined set is what a ladder read would carry
(`--guards-extra resisted_target,overkill_split,dominated_weather_ball_weather`).
They enter HARD_GUARDS (the deployed profile) only after such a read, with
the user's word. Chain continues: payoff matrix (cell 5 of 30 at 03:20),
eval_D baseline, round 4.

## Bundle `overkill_split` + `dominated_weather_ball_weather` PASSES after three fresh-seed confirmations (2026-September 7, 01:40)

Deployed + both guards vs deployed, screening n=1,000 then the adopted
confirmation clause (seed 8302, n=1,500) on every arm inside the window
(`results_guard_probe/bundle_overkill_wbw/`):

| arm | screening | confirmation |
|---|---|---|
| adversary (informative) | 0.0 (40.5 v 40.5) | -- |
| heuristic | +1.0 | clear |
| frozen 64opp | +0.4 | clear |
| rotation 8opp | -1.1 | **+0.3** |
| rotation tuned | -2.1 | **+1.7** |
| human holdout eval_B | -1.6 | **+1.9** |
| weighted (human x2) / equal | -0.8 | **+1.2 / +1.1** |

Verdict PASS under the rule as written (the fresh reading decides). Honest
pooled picture across both seeds: about neutral to slightly positive on
every arm -- three screening dips all flipped on fresh seeds, which is what
noise around zero looks like. `overkill_split` fires ~0.5 times per game
(455-522 per 1,000; nearly all promotions of a split pair the policy had
ranked), `dominated_weather_ball_weather` changes the pick 3-11 times per
1,000 games. Neither costs anything measurable locally; their value is the
specific ladder blunders they remove. Both stay opt-in until a ladder read;
next in the chain: the `resisted_target` v3 re-measure (started 01:39), the
payoff matrix, round 4.

## Ladder stopped at the user's order ("it's bleeding rating, fix it"); diagnosis; round 4 auto-launches tonight (2026-September 6, 22:37)

The v3 read went 0-4 (the first file in its dir is the v2 game the login
rejoined mid-battle and lost on the timer -- excluded) and the user stopped
it at 22:33; the bot is offline and stays offline until something passes.
The guard never fired in those games: 0 of our 16 single-target attacks hit
a resisted foe with a second foe up, so the situation did not arise; zero
errors.

Diagnosis over all 34 of today's games (`tools/analyze_ladder_previews.py
ladder`, `results_analysis/ladder_preview_analysis.json`) vs the August 125:

| | August (125) | today (34) |
|---|---|---|
| win rate | 53% | 32% |
| opponent set Trick Room | 16% of games, won 35% | 26%, won 22% |
| no Trick Room | won 56% | won 36% |
| first faint ours | 39% | 47% |

Not one blunder class: the ordinary games are lost too, and the field has
shifted (Raichu 7 of 34 rosters; Kingambit, Whimsicott, Basculegion,
Sneasler everywhere; more Trick Room). The brain is the August brain
against a September field; the September human clones scored the August
clone 5pp less faithful. The targeting guards are real but small; the fix
has to reach the weights.

Program tonight, all local, chained and monitored: (1) battery for the
bundle overkill_split + dominated_weather_ball_weather, then the
resisted_target v3 re-measure (each with the confirmation clause); (2) the
payoff matrix, trimmed to three rows (deployed, r3b final, old champion) x
10 columns so it finishes in ~2h; (3) **round 4 auto-launch**:
`training/make_league4_config.py` turns the column equilibrium y* into 12
seeded copies (floors: each September clone >= 1; cap: exploiter <= 2 --
rounds 3/3b's dose lesson), `tr_boost` 1.5 on the opponent team pool, eval_B
AND eval_D banned by content, deployed brain at the resume stem; then
`build_league.py`, verify, fresh 7700 server, `run_league4_training.sh`
(+5 intervals, ~9h). First-save kill criteria as before. Then the battery;
ladder only on a pass and with the user's word.

## Two more findings from the live read: the reranker was undoing the guard; Ice Weather Ball; ladder read restarted on v3 (2026-September 6, 22:14)

Game 3 of the read (loss vs H4irashi), from the audit. **Turn 2:** the guard
FIRED (sun still up at decision time: Fire Weather Ball and Moonblast both
resisted by Talonflame, neutral on Oranguru) and promoted the re-aimed pair
-- then the opponent reranker, which runs after the guards, put the original
pair back. Its score is log(prob / top prob) + tactical terms, and a
promoted twin kept its own 7% policy probability against the original's
28%, so the probability term alone reverted it. 93% of the guard's firings
in the battery were promotions, so the battery's +0.96 was measured with
most corrections silently undone. Fix: a promoted twin now inherits the
probability of the pair it corrects (`match.prob = max(match.prob,
top.prob)`), so tactical evidence can still outrank it but the policy's
preference for the resisted target cannot. Same fix in `overkill_split`.
**Turn 4:** Ice Weather Ball (snow) into Abomasnow, neutral, while Heat Wave
was 4x. G12 (`dominated_weather_ball`) returns early under ANY active
weather by design (it targets the 50 BP no-weather case). New opt-in guard
`dominated_weather_ball_weather`: under active non-sun weather, demote
Weather Ball when Heat Wave's calculator-expected damage on the same target
is >= 1.5x (Wide Guard stand-down kept; sun left to G12's mirror rule).
Tests: 67 across the guard files; suite green.

Consequence: the 22:02 ladder read (v2: promotions reverted) was stopped at
1-2 and restarted as v3 with `--guards-extra resisted_target` only -- the
rule the user approved, now actually taking effect -- in a fresh dir
`ladder_replays_guard_v3_20260906`. The two newer guards go through the
battery first (queued after the read as one bundle, then a v3 re-measure of
`resisted_target`), and reach ladder only on a pass and with the user's
word. Payoff matrix resumes after those.

## Ladder read with `resisted_target` running (user's go, 22:02); `overkill_split` guard built for the next A/B (2026-September 6, 22:10)

The user gave the go at 22:01: 25 serial games, deployed brain +
`--guards-extra resisted_target`, fresh dir `ladder_replays_guard_20260906`
(run_config records guards_extra=resisted_target, profile hard, knowledge
obs on, mixing off). The payoff matrix was paused (4 cells on disk, resumes
after the ladder). First game won (rating 1026 -> 1064); the guard fired
once in the first two games.

The user then spotted the next blunder class in that won game (turn 7):
Rock Tomb AND Last Respects both aimed at a 1%-HP Whimsicott (policy 18%);
it fainted to Rough Skin before either moved, both redirected to Oranguru,
and Last Respects (Ghost) did nothing into a Normal type. The split -- Rock
Tomb finishing Whimsicott, Wave Crash into Oranguru -- was ranked fourth at
12%. Frequency: both our single-target attacks aim at the same foe about
once per game (128 of 161 double-attack turns in the August 125; 21 of 29
today), the KO-guaranteed subset is smaller.

**`overkill_split`** (opt-in, not in HARD_GUARDS; 6 tests incl. the exact
position): when the played pair stacks two single-target attacks on one foe
and either alone is a calculator-verified guaranteed KO (min roll), keep the
finishing action and send the other slot's damage into the other live foe --
the best-ranked such pair the policy considered (single-target or spread,
not immune), else the same move re-aimed when it does damage there, else
stand down (`:no_alternative`). Counted `:promoted` / `:injected` /
`:twin_mismatch`. Its A/B (same launcher, `run_guard_probe.sh
overkill_split` + the confirmation clause) is queued right after the ladder
read, before the payoff matrix resumes. Suite 341 passed; Ruff/Pyright at
baseline.

## `resisted_target` PASSES the screening battery; ladder read awaits the user's word (2026-September 6, 21:49)

Deployed + `resisted_target` vs the deployed brain, paired, hidden sheets,
seed 83 (`results_guard_probe/resisted_target/`):

| arm | delta |
|---|---|
| adversary (exploiter, stochastic; informative) | +2.4 (40.3 v 37.9); v1 read -1.5 |
| human holdout eval_B | **+1.9** (86.4 v 84.5) |
| heuristic | +0.8 (92.1 v 91.3) |
| frozen 64opp | screening -2.3 -> fresh-seed n=1,500 (seed 8302): **-0.5** (85.8 v 86.3) |
| rotation 8opp | -0.1 (87.8 v 87.9) |
| rotation tuned | +1.8 (86.2 v 84.4) |
| weighted (human x2) / equal | **+0.96 / +0.77** |

Verdict under the adopted tier rule: **PASS** -- no confirmed breach,
weighted >= 0, human holdout +1.9. The confirmation clause worked in the
candidate's favour this time: the frozen arm's -2.3 at n=1,000 read -0.5 on
a fresh seed (exactly the ~1-in-4 false-flag rate the SEs predict). The
guard fired ~1,000 times per 1,000 games vs the exploiter and ~400 vs the
heuristic (fewer resisted-target situations), 6-8% of firings injected a
pair the policy had not ranked, the KO exception stood down 57 times per
1,000 heuristic games, zero errors, zero pairing mismatches. First-faint
markers unchanged on every arm: this is a mid-game targeting fix, not an
opening fix -- the 61% first-faint collapse on ladder is a separate problem.

Reading: the first gate pass of an inference-time rule since the guard era,
and a modest one (+0.4..+1.0 weighted) against local populations that play
the type chart far less sharply than humans do (opponents on ladder land
super-effective hits 40% of the time). The rule's value is expected to be
larger on ladder than locally, and only a ladder read can show it. Next per
the plan: 25 serial ladder games, deployed brain + `--guards-extra
resisted_target`, fresh replay dir `ladder_replays_guard_20260906`, same
instrumentation -- **not launched**: the ladder is stopped at the user's
request and restarts only on their word. Deployed configuration unchanged
(the guard stays opt-in until a ladder read backs it).

## Ladder batch stopped at 7-16; the resisted-target blunder class; `resisted_target` guard built and its A/B pre-registered (2026-September 6, 19:19)

The 25-game deployed-brain batch was stopped by the user after 23 games:
**7-16 (30%)**, first faint ours **14/23 (61%)** vs 34% in the August
100-game run, the account sliding from ~1250 toward ~1090 against
1000-1130 opponents. Far below the 55/100 record; the opening exchange is
collapsing. Replays are in `ladder_replays_league_20260906/` (a fresh
single-config dir: the run_config guard refused the 125-game dir because
the mixing flags -- even at their off defaults -- changed the recorded
configuration; the interrupted 24th game counts as a timer loss).

The blunder the user watched (game vs underwaterficus, turn 5): Charizard-
Mega-Y in sun sent Weather Ball into Rotom-Wash (resisted) beside a Mega
Meganium (super-effective, likely a KO). The audit: the policy ranked the
Rotom target first at 28% and the Meganium target fifth at 4%; no guard or
reranker considered the matchup. Same game, turns 1-2: Basculegion's Wave
Crash into Altaria (resisted, twice) beside Sneasler (neutral, frail, and the
mon that then took the game). This is the August finding -- our super-
effective hit rate 10.4% vs humans' 21.9% -- with a face.

**`resisted_target` guard** (`vgc_bench/src/guards.py`, registered, opt-in,
NOT in HARD_GUARDS): when the played pair sends a single-target damaging
move into a foe whose current typing resists it while the other live foe
takes strictly more (neutral or better), is not known immune
(`deals_no_damage`), and -- when the calculator can evaluate both -- takes
more damage, the twin pair (same actions, other target) is promoted if the
policy ranked it or injected with the top pair's weight. Weather Ball
follows the weather; Tera Blast is skipped; the twin is verified by decoding
it back through poke-env before use; every stand-down is counted
(`:other_immune`, `:calc_disagrees`, `:twin_mismatch`). Per-player
`guard_overrides` (new) let an A/B enable it on one arm only:
`eval_counterfactual.py --candidate-guards resisted_target`,
`run_gate_battery.py --candidate-guards`, `ladder_ourteam.py --guards-extra`
(recorded in run_config). 10 unit tests including the exact Rotom/Meganium
and Altaria/Sneasler positions; suite 335 passed. Smoke test, 12 local games
vs the exploiter: fired 5 times in 134 decisions (2 injected, 3 promoted),
zero errors, baseline arm untouched.

**Pre-registered A/B** (`evaluation/run_guard_probe.sh resisted_target`,
`results_guard_probe/resisted_target/`): deployed + guard vs deployed, paired,
seed 83. Exploit meter n=1,000 (informative). The decision is the 5-arm
screening battery under the adopted tier rule: **advance = verdict PASS (no
confirmed breach, weighted >= 0) with the human holdout >= 0**; first-faint-
ours is the mechanism marker. On a pass, the next step is a 25-game ladder
read with `--guards-extra resisted_target` -- with the user's word, since
the ladder is stopped at their request. The deployed configuration is
unchanged until then.

Firing budget, measured (`tools/hit_effectiveness.py`): 15% of our
single-target attacks go into a foe that resists them while a second foe is
on the field -- 13/85 in today's 23 games, 69/444 in the August 125 -- and
opponents land super-effective hits on 40% of their attacks vs our 19%
(today) / 26% (August). A rule that fires ~5 times a game can be judged by
win rate; a neutral-to-super-effective preference is the obvious follow-up
if this factual version passes.

v1 exploit reading (kept as `exploit_1000_v1_noKOexception.json`): -1.5pp
(38.9 v 40.4), 1,088 firings in 1,000 games, 81 injected / 1,007 promoted,
zero errors -- the policy usually RANKS the better target and still prefers
the resisted one. Before any battery data a defect was found in the rule: it
ignored whether the resisted hit finishes a low-HP foe (a resisted KO beats
a bigger non-KO). v2 adds the `guaranteed_ko` exception (`:current_ko`
counted; 11th test) and the probe was relaunched from the exploit step at
19:39 (`guard_probe_v2.log`).

## Every-turn mixing is worse even against the exploiter; mixing lever closed; ladder batch running (2026-September 6, 18:06)

The one pre-registered secondary variant (`always`, top-3, T=1) on the
exploit meter, n=1,000 paired: **39.1% vs 41.3% = -2.2pp**
(mixing_ran 10,410, changed picks 3,769, zero mismatches). Dose-response:
opening-only +2.6, every-turn -2.2 -- more sampling hurts, against the one
opponent it was meant to confuse. The policy's argmax beats its own
sampled play essentially everywhere; unpredictability at the action level
is not where the brain's exploitability lives. The mixing code stays
(default off, tested, useful for future A/Bs), the lever is closed, and
the answer to the poker question is now measured rather than argued: the
equilibrium has to be found in training (step 2), not applied at decision
time on top of a fixed policy. `results_mixing_probe/always_k3_t1.0_l2/
exploit_1000.json`. Ladder batch started 18:06: deployed brain, 25 serial
games into the 125-game corpus dir; exhibition mode follows, then the
round-4 meta-game overnight.

## Mixing probe VERDICT: opening mixing PARKED -- small exploit gain, real local cost (2026-September 6, 17:47)

Deployed + opening mixing (top-3, T=1, preview + turns 1-2) vs the deployed
brain, n=1,000 paired per arm, hidden sheets, seed 83
(`results_mixing_probe/opening_k3_t1.0_l2/`):

| arm | mixed vs deployed |
|---|---|
| adversary (exploiter, stochastic) | **+2.6** (43.3 v 40.7) -- bar was +5 |
| human holdout eval_B | 0.0 (85.7 v 85.7) |
| heuristic | -2.2 (89.8 v 92.0) |
| frozen 64opp | -1.1 (83.1 v 84.2) |
| rotation 8opp | **-3.1** (85.0 v 88.1) -- breach |
| rotation tuned | -1.2 (84.0 v 85.2) |
| weighted (human x2) / equal | -1.3 / -1.5 |

Mechanism verified on every arm (mixing_ran ~4,700 per arm, ~1,530 changed
picks, zero pairing mismatches). The marker tells the story: first faint
ours rises on EVERY arm for the mixed brain (rotation 8opp 25% vs 20%,
human 29% vs 25%, frozen 28% vs 25%) -- a sampled opening loses the first
mon more often than the argmax opening. Verdict by the pre-registered
rules: primary bar missed (+2.6 < +5) and the cost side breached
(rotation1 -3.1, beyond the confirmation window), so no confirmation runs
are spent; **the opening-mixing lever is PARKED**, deployed brain
unchanged.

What it teaches, cheaply: the exploiter's 60% is not about predicting which
of three near-equivalent picks the brain makes at the opening. Against
non-adaptive opponents the policy's argmax is simply its best action and
sampling is noise (-1 to -3pp); against the adversary the unpredictability
buys only a third of the bar. The holes are in WHAT the policy values, so
the fix has to reach the weights -- step 2 (Nash-weighted league round 4
against the new human clones and the exploiter). The every-turn variant's
exploit meter is running now as the one pre-registered follow-up
(`always_k3_t1.0_l2/`); its battery runs only on a +5pp pass. Then the
25-game deployed-brain ladder batch and exhibition mode.

## eval_D: an eval-only human clone of the CURRENT meta; ban list covers both holdouts (2026-September 6, 16:57)

`results_bc/eval_D/saves_bc/seed2/4.zip` (sha 99ffe1b5, role eval_only)
trained from holdout D (buckets 5-9 of the 2026-09-06 scrape), 8 epochs,
scored on its opposite bucket set C: epoch 4 = 41.1% top-1 / 56.5% top-3
(epochs 1-5: 39.7-41.1). It is to the September meta what eval_B is to the
August one: a human-style population no training pool may contain.
`training/build_league.py` now bans SEVERAL eval-only roots by content
(`eval_only_roots: ["results_bc/eval_B", "results_bc/eval_D"]` in a league
config; every .zip under each root; an empty root refuses to build) --
15 league-guard tests, the 3b pool re-verifies. Use as a diagnostic arm via
`run_gate_battery.py --arms human_bc --human-bc results_bc/eval_D/saves_bc/
seed2/4.zip`; the pre-registered battery weighting is unchanged (eval_B x2)
until a round is pre-registered with the new arm. Queued after tonight's
meta-game: the deployed brain vs eval_D, n=1,000, as that arm's baseline.

## Two new human-BC opponents; the Aug clone is 5pp behind the current meta (2026-September 6, 16:54)

Both behavior clones trained in minutes from the foundation checkpoint
(30 epochs, `--div_frac 0.1`), scored by held-out human-action agreement
(800 trajectories, per-slot top-1 / top-3), epochs swept:

| policy | data | best epoch | on holdout B (Aug meta) | on holdout D (Sep meta) |
|---|---|---|---|---|
| mix_A (reference) | Aug corpus, buckets 0-4 | 30 (as shipped) | 39.0 / 54.1 | **37.0 / 50.8** |
| **mix_C** | 2026-09-06 scrape only, 3,604 trajs | 3 (peak 2-4) | 40.1 / 57.2 | 41.5-42.2 / 58-59 |
| **mix_AC** | union of every Reg M-B log, 8,990 trajs | 5 (peak 1-5) | **41.2 / 57.8** | 41.6-41.8 / 58.7-59.0 |

Both peak within the first five epochs (the foundation init already knows
the game; small corpora overfit past that) and beat mix_A on the Aug
holdout by 1-2pp top-1 and 3-4pp top-3. The finding that matters: on the
NEW holdout (buckets 5-9 of the fresh scrape, 3,748 trajectories, players
1675-1923) mix_A drops to 37.0% while mix_C/mix_AC hold 41.5-42.2% -- the
meta moved in five weeks and the Aug clone no longer represents current
human play as well. Stamped `training_opponent`: `results_bc/mix_C/saves_bc/
seed1/3.zip` (sha c94edf52) and `results_bc/mix_AC/saves_bc/seed1/5.zip`
(sha d933847f); `obs_len` in the sidecar is null for pretrain.py zips (no
launcher reads it). Added as stochastic columns to
`training/meta_game_config.json` (now 5x10 cells, 15,000 battles). A
third clone, `eval_D`, is training from holdout D for an eval-only current-
meta human arm (to be banned from every pool by content, like eval_B).

## Mixing probe step 1: +2.6pp on the exploit meter, below the +5pp bar (2026-September 6, 16:23)

Deployed + opening mixing (top-3, T=1, preview + turns 1-2) vs deployed,
n=1,000 paired vs the final exploiter: **43.3% vs 40.7% = +2.6pp** (paired
SE ~1.5pp). The mechanism fired as designed -- mixing_ran 4,609 (4.6 per
battle), changed the played pair 1,524 times (33% of mixed decisions), zero
pairing mismatches -- so the reading is real, and small. Sampling among the
policy's own top-3 at the opening moves the exploiter only a third of the
way to the bar: the adversary's edge is mostly WHAT the brain does in the
exchange, not WHICH of three near-equivalent picks it makes. The cost
battery (five arms) is running for the record.

Pre-registered follow-up, written before its data: exactly ONE secondary
variant -- mixing on every turn (`always`, k=3, T=1) -- exploit meter only,
`results_mixing_probe/always_k3_t1.0_l2/`. Rule: >= +5pp -> its screening
battery; otherwise the mixing lever is PARKED (recorded as a small positive,
not deployed) and step 2 proceeds. Then the 25-game deployed-brain ladder
batch and exhibition mode, as planned.

Step-2 data, in parallel: the fresh top-500 scrape wrote 7,716 bo1 + 3,138
bo3 Reg M-B logs (players 1675-1923, median 1714) to
`battle_logs_top_20260906/`. The Aug corpus split is reconstructed as
crc32(tag)%10 buckets 0-4 = train (A), 5-9 = eval-only (B) -- matched by the
per-bucket game counts -- so the new data uses the same split:
`trajs_regmb_human_C` (buckets 0-4, 3,604 trajectories; bo1 top games are
92% sheet-less, as before) and `trajs_regmb_human_D` (5-9, 3,748, eval-only).
Two BCs from the foundation checkpoint, 30 epochs each: mix_C (new games
only) and mix_AC (deduplicated union of every Reg M-B log, buckets 0-4),
scored by held-out agreement on B against mix_A's 38.7%. `pretrain.py`'s
default `--div_frac 0.01` crashes on corpora this size (chunks of ~313
transitions < the 1,024 BC batch); 0.1 used.

## Step 2 tooling: meta-game payoff matrix + Nash opponent weights (2026-September 6, 16:11)

Upstream vgc-bench already carries a double-oracle mode (`LearningStyle.
DOUBLE_ORACLE`: `update_payoff_matrix` plays each new checkpoint vs every
pool file at n=100 and samples opponents from `nashpy.Game(...).linear_program()[0]`)
but it assumes a symmetric self-play population and a matrix that starts
at 1x1 -- unusable for a league resumed from a seeded pool. So the
PSRO-style weighting is done OFFLINE and auditable instead:
`training/meta_game.py` (6 tests) runs every row-vs-column cell through
`eval_counterfactual.py --baseline-only` (new flag: one policy vs one
opponent, half the cost of a paired run), assembles the payoff matrix (rows
pilot MB430, columns pilot the pool teams, BC columns sampled), solves the
zero-sum game with two HiGHS linear programs, and prints the COLUMN
equilibrium y* -- the opponent mixture that is hardest for the whole row
population, i.e. the meta-strategy the next best response should train
against -- plus integer copy counts for an FP league (uniform file
sampling, so multiplicity = weight). Cells resume from disk; `--solve-only`
re-solves.

`training/meta_game_config.json` (round 4): rows = deployed, r3 final, r3b
final, old champion, league-1 mid; columns = bc_mix_A (stochastic),
exploiter final, deployed, r3/r3b finals, old champion, league-1 history
(8847360, 11796480); n=300/cell, 40 cells = 12,000 battles (~3h). The
frozen eval PPOs stay out so every battery arm remains an unfit
population. Runs after tonight's ladder batch. Then: a fresh top-player
scrape for a second human-BC (`scrape_top_players.py --measure` sizing it
now), and league round 4 with the y*-derived copy table.

## Mixed-strategy play: built, gated, probe launched (2026-September 6, 16:02)

The user chose the program "make the brain unexploitable, then change the
team" (options and recommendation given 14:30). Step 1 = stop being
deterministic at the mind-game decisions. The finding that prompted it: the
production stack always plays `cands[0]` after guards and rerankers
(`_guarded_action`), and preview is two argmax calls -- the deployed brain
never mixes. In a simultaneous-move game that is the most exploitable policy
shape; the exploiter's 60% and the 1300+ opening punishes are its symptom.

Built (default off, counted, audited): `PolicyPlayer(mixing_mode=off|opening|
always, mixing_top_k=3, mixing_temperature=1.0, mixing_last_turn=2,
mixing_seed)`. At an active decision (opening = team preview + turns <= 2)
the stack samples the played pair among the top-k eligible candidates with
weights = policy probability^(1/T), renormalised; guard-demoted and
strategic-only pairs are never sampled, so mixing cannot reintroduce a known
blunder; the chosen pair rotates to the front so audit, counters and caller
agree; any internal error counts and leaves the ranking untouched. Counters
`mixing_ran`, `mixing_changed_pick`, `mixing_skipped:*`, `mixing_error:*`;
the decision JSONL gains a `mixing` block. Flags: `eval_counterfactual.py
--candidate-mixing` (candidate arm only, sampler seeded from --seed),
`run_gate_battery.py --candidate-mixing` (forwarded to every arm, recorded in
the scorecard), `ladder_ourteam.py --mixing` (recorded in run_config.json).
Smoke test, 8 local battles with a replay dir: mixed arm mixing_ran 40,
changed_pick 11, audit blocks on turns 0/1/2 with weights; baseline arm
untouched. 7 unit tests.

Gate fix (the 13:55 finding, adopted with the user's go):
`evaluation/scorecard_verdict.py` reads a battery directory, computes each
arm's paired SE from the per-battle records, and applies the tier's rule --
screening ADVISORY (an arm within +/-1pp of the -2pp bar is CONFIRM -> a
fresh-seed n=1,500 re-run of that arm decides; a --confirm reading replaces
the screening reading and is judged hard), promotion HARD. It reproduces the
3b verdict (FAIL after confirmation; neutral false-fail 31% at those SEs).
7 unit tests. FUTURE_BOT_PLAN's measurement rule updated.

**Pre-registered probe** (`evaluation/run_mixing_probe.sh opening 3 1.0 2`,
evidence in `results_mixing_probe/opening_k3_t1.0_l2/`): deployed + mixing
vs deployed, paired, hidden sheets, seed 83, restarted eval server. Step 1,
the exploitability meter: n=1,000 vs the final exploiter (stochastic);
deployed reads ~40%; **success = mixed arm >= deployed + 5pp**. Step 2, the
cost: the 5-arm screening battery, judged by scorecard_verdict (advisory +
confirmation clause); **acceptable = no confirmed breach and weighted >= 0**.
Both must hold to take mixing to a 25-game ladder read; the deployed brain is
not touched either way. Before reading any number: the mixed arm's telemetry
must show mixing_ran > 0 and mixing_changed_pick > 0. Expected: the exploiter
loses part of its edge (it was trained against a deterministic target); the
non-adaptive arms (heuristic, frozen PPOs) may pay a little -- the price of
unpredictability, and the reason the human holdout and the adversary carry
the decision. Secondary variants (always; k=2/5; T=0.7) only if the primary
shows signal, one at a time. After the probe the machine runs a 25-game
ladder batch of the DEPLOYED brain (corpus 125 -> 150; no mixing), then
exhibition mode.

Suite 318 passed, 5 skipped; Ruff 14 (baseline), Pyright 23 (baseline).

## League 3b VERDICT: both finalists fail; the heuristic breach confirmed at n=1,500 (-4.1pp) (September 6, 13:55)

Full cards vs the DEPLOYED champion (n=1,000 paired, hidden sheets, zero
preview-pairing mismatches on every arm):

| arm | 14745600 | 17694720 (final) |
|---|---|---|
| adversary (exploiter, stochastic) | **+15.4 (55.6 v 40.2)** | **+27.4 (67.4 v 40.0)** |
| human holdout eval_B | -0.1 (84.9 v 85.0) | +0.9 (86.5 v 85.6) |
| heuristic | -0.4 (91.3 v 91.7) | **-2.7 (88.2 v 90.9)** |
| frozen 64opp | **-3.0 (83.3 v 86.3)** | -0.1 (86.5 v 86.6) |
| rotation 8opp | **-2.6 (84.8 v 87.4)** | **+2.2 (87.2 v 85.0)** |
| rotation tuned | -0.4 (85.3 v 85.7) | **+3.4 (89.5 v 86.1)** |
| weighted (human x2) / equal | -1.1 / -1.3 | +0.8 / +0.7 |
| mix_A diagnostic (vs eval_B) | +3.8 (-0.1) -> 3.9pp, clean | +2.9 (+0.9) -> 2.0pp, clean |

**14745600 fails** as pre-registered: two PPO arms below the deployed brain
by more than 2pp (frozen -3.0, rotation 8opp -2.6), weighted -1.1. Its
exploit closure (+15.4) is also the weakest of the four adversary-trained
finalists across rounds 3 and 3b.

**17694720 (final checkpoint):** the lighter dose did what round 3b was run
to test. Against the same final checkpoint in round 3 (31% adversary share)
the three PPO arms read -3.5 / -5.6 / -4.6; at ~10% they read -0.1 / +2.2 /
+3.4, the exploit fix held (+27.4 vs +32.3), the human holdout stayed
positive (+0.9 vs +3.8) and the memorization check is clean (mix_A +2.9 vs
eval_B +0.9). The one arm below the line is the heuristic at -2.7 (88.2 v
90.9). The 13:00 pre-registered confirmation ran on a fresh seed (8301,
n=1,500 paired, restarted server, zero mismatches): candidate **87.8%**
(1317/1500) vs deployed **91.9%** (1379/1500) = **-4.1pp**, paired SE 1.1pp
(about four standard errors below zero); pooled with screening (n=2,500)
-3.6pp. Rule as written: breach CONFIRMED, **17694720 FAILS**, no further
re-measurement. The mechanism replicates: first faint ours vs the heuristic
35% for the candidate vs 25% for the deployed brain on both seeds, while
vs the adversary it fell 57% -> 30%. The candidate's opening is tuned
toward PPO/human-style openers and away from the heuristic's max-damage
play -- a real style shift, not noise. The confirmation clause hardened
the verdict rather than rescuing the candidate, which is what it is for.

**Gate-design finding (recorded for the checklist; this round's verdict
does not depend on it):** from the per-battle records, the paired standard
error per arm at n=1,000 is 1.2-1.6pp. Under the screening rule "no arm
below the deployed brain by >2pp" a truly NEUTRAL candidate fails on at
least one of five arms 33-35% of the time; a candidate truly -1pp on every
arm fails ~75% of the time; at the promotion tier's n=5,000 the neutral
false-fail rate is 0.4-0.6%. Proposal (user's call, changes the checklist):
screening non-regression becomes advisory with the +/-1pp fresh-seed
confirmation clause; the hard "no arm < -2pp" rule lives at 5,000/arm.

Neither finalist passes, so no ladder game (rule, and the user's standing
instruction). Deployed brain unchanged: `results_league/league_champion.zip`
(55-45 over 100; 52.8% over 125). Reading across rounds 3 and 3b (four
adversary-trained finalists): the exploit closes robustly at any dose
(+15..+32) but every candidate pays 2-6pp on some local arm, and WHICH arm
moves with the pool -- a narrow adversary teaches a narrow fix with a
diffuse cost. The one lever that ever transferred to ladder was a broadly
human-like opponent (league 1: every arm up, 44.9% -> 55%). Open decisions
for the user: (1) next round direction -- recommended: a fresh
human-imitation opponent from a different slice of the mined human games
(pipeline check first), with the exploiter at <= 10% as a regularizer,
gated at the promotion tier; (2) the gate proposal above; (3) a ladder-read
override for a specialist -- not recommended (a confirmed -3.6pp arm, and
25 games cannot resolve a 3pp question; the deployed brain keeps its slot);
(4) the PC port (WSL2 + CUDA) as a second training node; (5) the team
question, now that the league-1 cycle is complete.

Corpus: `results_gate_battery_league3b/*/screening/scorecard.json`,
`*/exploit_remeasure_1000.json`,
`17694720/confirm_heuristic_1500_seed8301.json`.

## League 3b finalist 2 at four of five arms; confirmation rule written BEFORE the human holdout reads (September 6, 13:00)

17694720 vs the DEPLOYED champion (n=1,000 paired, hidden sheets):
heuristic **-2.7** (88.2 v 90.9), frozen -0.1, rotation 8opp **+2.2**,
rotation tuned **+3.4**; human holdout and mix_A pending (~13:15 / ~13:30).
Exploit re-measure +27.4. Finalist 1 (14745600) completed and FAILS:
heuristic -0.4, frozen **-3.0**, rotation1 **-2.6**, rotation2 -0.4, human
-0.1 (weighted -1.1, equal -1.3; mix_A +3.8 vs eval_B -0.1 = 3.9pp
divergence, clean). Zero preview-pairing mismatches on every arm.

By the letter of the 3b bars (no arm below the deployed brain by >2pp),
finalist 2 already fails on the heuristic arm whatever the human holdout
says. Recorded now, before that arm lands, so it cannot be a post-hoc
move: the heuristic reading is one arm at about two standard errors
(n=1,000 paired) and inside 1pp of the bar -- the zone where the
counterfactual track's pre-registered rule (FUTURE_BOT_PLAN: within +/-1pp
of its bar -> fresh-seed n=1,500 confirmation before accept/reject)
refuses to call a verdict at n=1,000. That rule was written for the +2.0pp
promotion bar; applying it to a regression bar is an EXTENSION and is
labelled as such. Plan, fixed now: after the chain completes, one
fresh-seed (seed 8301) n=1,500 heuristic-arm re-run of finalist 2 vs the
deployed champion on a restarted eval server. Decision rule: the fresh
reading decides the arm (the screening reading stays in the record);
>= -2.0pp = non-breach, and the candidate becomes a CONDITIONAL pass for
the user's decision (with the weighted delta, the human holdout and the
mix_A divergence as they read); < -2.0pp = breach confirmed, verdict FAIL,
no further re-measurement. No ladder game is played on this candidate
without the user's word in either case: a pass here would rest on a rule
extension the user has not reviewed. Round-3 precedent is consistent with
this: every earlier breaching finalist had two or more breaching arms,
where a single-arm confirmation could not have changed the verdict.

## League 3b: both finalists qualify on the exploit re-measure (September 6, 10:15)

n=1,000 paired vs the final exploiter (stochastic): 14745600 **55.6% vs
deployed 40.2% (+15.4)**, first faint ours 59->36%, long games 40->56%;
17694720 **67.4% vs 40.0% (+27.4)**, first faint 57->30%, long games 39->62%.
Round-3-level closure at the lighter dose for the final checkpoint. The
decisive test is the 5-arm non-regression battery vs the deployed champion
(14745600 first, then 17694720) + mix_A diagnostics; tiebreak rule as
pre-registered on 09-05 (higher weighted delta among passers; exploit margin
breaks ties within 1pp).

## League 3b complete; verdict chain running (September 6, 09:35)

Five checkpoints, zero errors. Probes (heuristic / bc): 0.84/0.81,
**0.89/0.82**, 0.87/0.80, 0.76/0.77, 0.83/0.83 -- no round-3-style shock,
band stable. Finalists: 14745600 (best probes) and 17694720 (final).
Verdict chain (VERDICT_ROOT=results_gate_battery_league3b): exploit
re-measure for both, then the 5-arm screening vs the deployed champion and
the mix_A diagnostic for qualifiers; same pre-registered bars.

## League 3b save 1: clean, no adaptation shock (September 6, 02:40)

Save 1 (13,762,560): eval/heuristic **0.84**, eval/bc **0.81** (round 3 opened
at 0.71/0.67 under the same probes); ep_rew +0.56..+0.78, approx_kl ~0.02,
clip 0.08-0.09, 156 steps/s, zero errors. Continue under the original kill
criteria; completion ~09:45.

## League 3b launched: adversary dose cut to one copy (September 6, 00:45)

Pool (`training/league3b_config.json`): deployed champion at resume
(12,779,520) + league-1 history (4) + old champion + bc_mix_A x3 + ONE copy
of the final exploiter (stem 400; ~10% initial, decaying). League v1 team
weights, same hyperparameters, +5 intervals to 17,694,720 on a fresh 7700
server. Bars unchanged from round 3 (pre-registered): exploit re-measure
>= +5pp over the deployed brain's ~39%; 5-arm screening vs the deployed
brain with no arm below by >2pp and weighted >= 0; mix_A divergence <= 10pp;
first-save kill criteria; ladder 25 only on a full pass. Hypothesis: a
lighter dose keeps most of the exploit fix without the 2-6pp PPO-arm tax.

## League 3 VERDICT: exploit closed, general bar failed by both finalists; round 3b prepared (September 5, 23:05)

Full cards vs the DEPLOYED champion (n=1,000 paired, hidden sheets):

| arm | 14745600 | 17694720 (final) |
|---|---|---|
| adversary (exploiter, stochastic) | **+27.7** (67.0 v 39.3) | **+32.3** (71.4 v 39.1) |
| human holdout eval_B | -2.3 | **+3.8** (87.5 v 83.7) |
| heuristic | -3.2 | +0.4 |
| frozen 64opp | -1.1 | -3.5 |
| rotation 8opp | 0.0 | -5.6 |
| rotation tuned | -2.0 | -4.6 |
| weighted (human x2) | -1.8 | -0.95 |
| mix_A diagnostic (vs eval_B) | +0.3 (-2.3) clean | +1.0 (+3.8) clean |

**Both fail the pre-registered non-regression rule** (no arm below the
deployed brain by >2pp): 14745600 breaches on the heuristic and the human
holdout; 17694720 breaches on all three PPO arms. No promotion, no ladder
test (rule as written). Memorization checks are clean for both -- the costs
are real style shifts, not clone-memorization.

Two readings, both recorded: (1) the adversary at 31% of the pool
overfits the sparring partner -- the classic PSRO failure mode -- so the
next experiment is dose, not direction: **round 3b** (one exploiter copy,
~10% initial) is prepared in `training/league3b_config.json` +
`run_league3b_training.sh`, NOT launched. (2) 17694720's shape -- better vs
human-like play (+3.8 holdout) and the adversary, worse vs the old PPO
lineage -- is the direction the ladder rewards, and a 25-game ladder read
would be informative; that override is the user's call, not the pipeline's.

Corpus of round-3 evidence: `results_gate_battery_league3/*/screening/
scorecard.json` and `*/exploit_remeasure_1000.json`.

## League 3 finalist 14745600 FAILS screening: a specialist trade (September 5, 21:00)

5-arm screening vs the DEPLOYED champion (n=1,000 paired): heuristic
**-3.2**, frozen -1.1, rotation 8opp 0.0, rotation tuned -2.0, human holdout
**-2.3** -> weighted -1.8pp (promotion weighting) / -1.7 (equal). Two arms
breach the pre-registered "no arm below by >2pp"; four of five point down.
Read: the 31% adversary share bought +27.7pp against the exploiter at a
~2pp general cost -- PSRO-style overfitting to the sparring partner.
Candidate NOT promotable as-is. mix_A diagnostic running for the record;
finalist 17694720's battery follows. If it shows the same shape, round 3's
lesson is "lighter exploiter share (1-2 copies) + keep the human/self mix",
i.e. a round 3b, not a verdict against the approach.

## League 3 finalist 17694720: 71.4% vs the exploiter; both finalists qualify (September 5, 19:35)

Final checkpoint, same paired protocol: deployed **391/1000 = 39.1%**,
finalist **714/1000 = 71.4%** (+32.3pp); first faint ours 58% -> 30%; 7+
turn games won 39% -> 69%. Both finalists now face the 5-arm screening
battery vs the DEPLOYED champion (14745600 first, then 17694720) and the
mix_A diagnostic. **Tiebreak, written before any battery result:** among
finalists that pass (no arm below the deployed brain by >2pp, weighted >= 0,
mix_A divergence <= 10pp), promote the one with the higher weighted battery
delta; exploit margin breaks a tie within 1pp. Promotion tier (5,000/arm)
runs only if the screening weighted delta is >= +1pp or the human holdout
is >= +2pp; otherwise the candidate is a "specialist" and goes straight to
the 25-game ladder test as a non-regression check with the exploit fix as
the claim.

## League 3 finalist 14745600: exploit flipped -- 67.0% vs the exploiter (September 5, 19:12)

Exploit re-measure (n=1,000, stochastic final exploiter, paired seed 83):
deployed brain **393/1000 = 39.3%**, finalist **670/1000 = 67.0%** (+27.7pp
vs the +5pp qualifying bar). Mechanism markers moved as designed: first
faint ours 56% -> 38%; 7+ turn games (598 of 1,000) won 38% -> 63%. The two
adversary-verified holes are closed against the adversary. Screening battery
vs the deployed champion + mix_A diagnostic follow (the generalization
question); second finalist 17694720 re-measuring now.

## League 3 complete; verdict chain running (September 5, 18:53)

Five checkpoints, zero errors. Probes (heuristic / bc, n=100): 0.71/0.67,
**0.84/0.85**, 0.79/0.82, 0.84/0.77, 0.81/0.73. Finalists: 14745600 (best
probes) and 17694720 (final, most adversary exposure).
`evaluation/run_league3_verdict.sh` runs the exploit re-measure for both
first (n=1,000 vs the final exploiter; qualify at >= 44.8%, i.e. +5pp over
the deployed brain's 39.8%), then the 5-arm screening battery vs the
DEPLOYED champion and the mix_A diagnostic for qualifiers only. Ladder 25
only on a full pass.

## League 3 save 2: recovered -- run continues (September 5, 13:40)

Save 2 (14,745,600): eval/heuristic 0.71 -> **0.84**, eval/bc 0.67 ->
**0.85** (rule was >= 0.78 / >= 0.70). The save-1 dip was adaptation shock
from the adversary-heavy pool, as read; ep_rew +0.52..+0.70, 150 steps/s,
zero errors. Original kill criteria back in force for saves 3-5; completion
~19:00, then `evaluation/run_league3_gates.sh` on the triaged finalists.

## League 3 first save: probes under the kill lines; continuing ONE interval under a written rule (September 5, 11:45)

Save 1 (13,762,560): eval/heuristic **0.71** (kill line 0.80), eval/bc
**0.67** (line 0.70); optimizer healthy (ep_rew +0.36..+0.60, approx_kl
0.015-0.020, clip 0.08-0.10, 150 steps/s, zero errors, bc_opp_frac
0.375-0.5 = the seeded 7/13 share). Reading: adaptation shock from a much
harder pool (31% adversary) -- round 2 also opened under its lines and
recovered by interval 3 -- and the probes are n=100 (SE ~4-5pp). Recorded
deviation from the kill rule: continue exactly one interval. **Save-2 rule,
written now:** eval/heuristic >= 0.78 AND eval/bc >= 0.70 (recovering), else
kill and record "exploiter share too high; retry at 2/13". The final gates
(exploit re-measure >= +5pp, non-regression battery, memorization) are
untouched.

## League round 3 launched: the exploiter joins the pool (September 5)

Pool (`training/league3_config.json`): deployed champion resumes at
12,779,520; its league-1 history (4) + the old champion; bc_mix_A x3 (23%);
**exploiter final checkpoint x3 + interval-4 x1 (31%)** at stems 400-700 so
they can neither collide with new saves nor be mistaken for the resume point.
Team weights: league v1 (our_team zeroed, no TR boost). +5 intervals to
17,694,720, port 7700 (server restarted first), same hyperparameters.

**Pre-registered bars (written before any result):**
1. Exploit re-measure, n=1,000 stochastic exploiter (final ckpt) vs the
   candidate: champion's 39.8% must rise by >= +5pp (the direct target).
2. Standard 5-arm battery vs the DEPLOYED brain at screening n (1,000/arm):
   no arm below baseline by >2pp, weighted >= 0 (non-regression), memorization
   diagnostic clean (mix_A vs eval_B divergence <= 10pp). Promotion tier only
   if screening also shows weighted >= +1pp or human holdout >= +2pp.
3. First-save kill criteria as before (eval/heuristic < 0.80, eval/bc < 0.70,
   ep_rew <= 0, worker deaths, throughput < 120 steps/s).
4. Ladder: 25 games ONLY on a pass (the user's standing rule).

## Search re-gate VERDICT: a valid tie -- search FAILS the +3pp bar; no ladder test (September 5, 05:20)

On the clean server the gate ran end to end in 4h with zero errors and
perfect pairing (300/300 opponent previews replayed, 0 mismatches; no policy
repairs fired -- the earlier "policy None before load" and ledger-drift
symptoms were stale-room artifacts too). Paired n=300, hidden sheets,
held-out human opponent (bc_eval_B stochastic), v2h evaluator:

| arm | wins | search decisions | fallbacks | truncations | p50 / p90 / max |
|---|---|---|---|---|---|
| champion (no search) | 252/300 = 84.0% | -- | -- | -- | -- |
| live exact search | 254/300 = 84.7% | 1,415 | 951 | 850 | 7.4 / 7.6 / 7.9 s |

**+0.7pp vs the pre-registered +3.0pp bar: FAIL.** Mechanically sound
(in-budget, serial, no stalls) but no strength: search reaches a decision on
only ~1/3 of its turns inside 8s (the rest fall back to the policy+guards)
and those decisions are a wash. This matches the Aug-22 ties on the old
champion. With v3h rejected, the evaluator remains the bottleneck (v2h is
preview-blind and trained on the old bot's games). Per the user's standing
order the 25-game ladder test does NOT run.

`results_search_v3h/search_regate_hidden300.json` is the record. What the
night actually bought: four stall classes fixed for good (asserts, load
race, ledger drift, stale server rooms) -- the eval harness is now robust to
serial search arms and learned opponents, which it never was.

Next lever, from the exploiter's leak map (60.2% vs the champion, via the
opening exchange and long games): **league round 3 with the exploiter in the
FP pool** -- the proven training loop, aimed at the measured leaks -- rather
than more search work until an evaluator beats 0.2087 on ladder replays.

## Fourth stall source: stale server rooms after a killed run; gate relaunched clean (September 5, 01:13)

After the ledger fix the serial search arm still froze silently at its second
decision, on a poke-env CRITICAL "You cannot reject open team sheets after
Team Preview" (unknown server errors are logged, never retried). Two causes
addressed: (1) paired arms now share concurrency -- a 1-slot search player
vs an 8-slot opponent opened extra rooms on the foe's side; (2) the decisive
one: every relaunch tonight followed a mid-battle kill on the same eval
server, which kept the dead run's rooms and fed the next run's identically
named players stale mid-battle messages (the champion arm itself broke at
startup once this way). `run_search_regate.sh` now restarts the eval server
before launching. Relaunched 01:13 on a clean server: champion arm started
with zero errors.

## Third stall class fixed: paired-preview ledger drift (September 5, 00:52)

The relaunched search arm then froze seven battles on
`RuntimeError: preview pairing mismatch` -- the PreviewLedger replays the
control arm's opponent drafts by matchup fingerprint, and a serial search
arm vs an 8-way control arm with a stochastic opponent does not reproduce
the matchup order. Raising inside a battle handler is the same stall class
as the asserts. The ledger now returns None (player's own preview) and
reports `mismatched` per arm; tests updated. Gate relaunched 00:52 --
three stall classes fixed tonight, all of the "exception in a poke-env
handler = frozen battle" family; the shim, the policy repair, and the ledger
fallback now cover every path that fired.

## Search re-gate: two stall bugs found and fixed before it could run clean (September 5, 00:47)

The gate was re-pointed at the held-out human opponent (the no-search
champion sits at ~91% vs the heuristic, leaving no room for a +3pp effect;
bc_eval_B baseline ~80-87% across runs). Its first serial search arm then
stalled on a **bare `assert isinstance(self.policy, MaskedActorCriticPolicy)`
in a decision path** -- six such asserts existed (choose_move, team preview,
the batch inference loop); all now repair-and-count (`_repair_policy`) with
regression tests. Diagnostics then showed the real trigger: a
**construction-order race** -- poke-env's listener thread delivered battle
requests to the search-arm player while `set_policy` was still inside
`PPO.load` (+ the outcome-evaluator load on search arms), so the policy read
None. Decisions now wait for a loading policy (`policy_wait_s`, 30s in eval,
6s on ladder) and count `policy_waited`. Gate relaunched 00:47 with the fixes;
pre-registered bar unchanged (beat no-search by >=3pp at n=300 paired, live
decisions, p90 < 9s). Ladder test only on a pass, per the user's order.

## Evaluator v3h FAILS the ladder gate; search re-gate armed with the incumbent (September 5, 00:20)

Resumed. First finding: the "incumbent calibration" from 09-03 was a
traceback -- the 08-31 reorg left six scripts anchoring ROOT to their own
folder (`Path(__file__).resolve().parent`), so calibrate_vs_ladder looked for
`tools/results_analysis/...`. All six repointed to `parents[1]` (including
evaluate_tactical_gate, whose fixture paths were silently broken too).

Calibration duel on the refreshed 446-game audit (7,571 scored states from
434 battles, identical corpus for both nets):

| net | ladder Brier | ECE | preview | turns 1-3 | 4-6 | 7+ |
|---|---|---|---|---|---|---|
| v2h incumbent | **0.2087** | 0.1029 | 0.3466 | 0.2207 | 0.1339 | 0.1132 |
| v3h candidate | 0.2145 | 0.1124 | 0.3608 | 0.2242 | 0.1342 | 0.1243 |

**v3h loses in every bucket** despite passing the data, per-style and
historical-holdout gates -- the sim-to-ladder gap once more; plausibly the
exploiter-heavy historical rotation pulled the label distribution away from
human play. Standing gate holds: the v2h net remains the evaluator. Both nets
stay preview-blind (mean p 0.06-0.08 vs 41% actual), the known open problem.

The search question is untouched by this: the promoted brain has never been
searched with ANY evaluator. `evaluation/run_search_regate.sh` is armed with
the v2h net (pre-registered bar unchanged: beat no-search by >=3pp at n=300
paired, live decisions, p90 < 9s), gated on AC power (machine was on
battery). Per the user's order, the 25-game ladder test runs ONLY on a pass.

## PAUSED by the user mid-calibration (September 3, ~10:25)

v3h trained and saved (`results_outcome_v3h/outcome_value.zip`, selected
epoch 1, temperature 1.118, data/style/holdout gates all TRUE). The
incumbent's ladder-replay pass completed; the v3h pass was stopped by the
pause. Resume = rerun `tools/calibrate_vs_ladder.py value --outcome-value
results_outcome_v3h/outcome_value.zip` (~5 min), compare to the incumbent
file, then `evaluation/run_search_regate.sh` on a pass. Do not resume
without the user's word.

## Overnight stall found and fixed; evaluator v3h training resumed (September 3, 10:15)

The chain generated the full dataset (19,902 games / 149,271 states / 98
failures = 0.5%) by ~01:15 and then died silently: generate_outcome_dataset
exited nonzero on ANY failed game, `set -e` obeyed it, and the SystemExit
printed no traceback for the watcher to catch. ~9 idle hours. Fixes shipped
with tests: the generator now budgets isolated failures (--max-failed-games,
default 1% of --games) and both chain launchers trap ERR and print
CHAIN_FAILED / GATE_FAILED. Training restarted 10:15 on the finished dataset
(holdout style historical: 29,348 rows removed from training, 36,907
reserved for the style gate); calibration vs the v2h incumbent follows.

## Exploiter verdict: beats the deployed champion 60.2% -- and wins the way 1300+ humans do (September 3, 00:35)

The corrected exploiter (target locked to our roster via --opponent_team;
learner from the pool, initialized from the champion) finished 5 intervals
with a monotonic climb (mean ep_rew per interval -0.11, -0.12, -0.02, +0.05,
+0.11). Measured at n=1,000 (hidden sheets, stochastic exploiter, seed 83):
**champion 398/1000 = 39.8%** (`results_exploiter/exploit_eval_1000.json`).

The exploit's shape matches the ladder loss analysis of the 1264-1407 band
almost exactly: the exploiter takes the opening exchange in 58% of games
(champion recovers only 21% of those) and drags 58% of games to turn 7+,
where the champion wins 39%. An automated adversary independently
rediscovered the two human-observed holes -- opening prediction and the
endgame. Design brief for league 3: exploiter checkpoint(s) in the FP pool.

Evaluator v3h chain launched 00:35 from `training/run_evaluator_v3h.sh`:
20k-game outcome dataset on the promoted brain with the old champion AND the
exploiter in the historical rotation (positions vs the exploiter are the
leaky ones -- exactly what search needs to value correctly), then the
holdout-style trainer, then the ladder-replay Brier instrument vs the v2h
incumbent (0.2218 on the current 421-game corpus). Search re-gate
(`evaluation/run_search_regate.sh`, pre-registered: beat no-search by >=3pp
at n=300 with live decisions, p90 < 9s) follows a pass; the 25-game ladder
test runs ONLY if search clears that gate (user's standing order).

## Exploiter probe PAUSED by the user at 4/5 checkpoints (September 1, ~10:20)

The corrected exploiter (target = champion locked to our roster via the new
--opponent_team; learner from the pool) crossed from losing (-0.1..-0.2) to
consistently beating the champion at home (ep_rew +0.06..+0.22, ~55-60% WR)
by interval 4 -- **exploitable leaks confirmed in-training**. Checkpoints
13762560..16711680 banked in results_exploiter/saves_ex_hs_wt/reg_mb/seed1/.
Paused ~713k steps short of completion. Resume = relaunch
training/run_exploiter_training.sh (resumes from 16711680, ~2h); the banked
checkpoints are already measurable meanwhile. Do not resume without the
user's word. Queued after resume/verdict: the n=1,000 exploit measurement,
exploiter replay analysis, and the evaluator retrain.

Also recorded from the aborted first attempt: with pool teams on BOTH sides
the champion reads ~51% vs the heuristic (vs ~90% on its own team) -- the
policy is far from team-general; relevant to the any-team end-goal.

## Ladder corpus at 125 games: 66-59 (52.8%), new Elo peak 1407 (August 31, ~05:45)

The standing-order 25-game batch (deployed champion) ran 11-14 -- a cold
batch after a 7-3 start (last 15: 4-11). Cumulative: **66-59 = 52.8%,
Wilson95 [44.1, 61.3], one-sided p=0.038 vs the 44.9%/321 baseline** --
still above, softened from the n=100 read (55.0%); the true rate is
plausibly low-50s. **New all-time Elo peak: 1407** (mid-batch; prior peak
1365, old-bot lifetime 1318); ended the batch ~1281 after the cold stretch.

Mechanical: zero timer losses, zero parse errors (the Zoroark shim's first
live outing -- nothing to swallow this batch). First-faint-ours crept to
39% cumulative (34% at n=100; still far below the old 52.5%). TR-set split
now 7/20 = 35% -- the hole round 2 failed to fix locally persists on
ladder, at ~16% incidence.

Next: repo reorganization + exhibition (challenge-listener) mode per user
requests, both queued for this idle window.

## League round 2: NULL -- both candidates rejected at screening; deployed brain unchanged (August 31, 01:00)

Verdict by the pre-registered bars, baseline = the deployed league champion:

- 17694720 (final ckpt): general arms -2.0 / -0.4 / -1.2 / -0.6 / +0.9
  (weighted -0.7), memorization clean (-1.9 vs mix_A), and the decisive
  **TR-pool arm +0.9pp vs the +4.0 advance bar**. Fails.
- 16711680: all arms negative, rotation-tuned -6.7pp breaches the -4
  tolerance outright. Fails.

Reading: the TR curriculum at x3 boost (72% of training mass) bought no
measurable TR-pool gain and taxed general play slightly -- consistent with
round 1 having harvested the big distribution-shift win and the marginal
return now being small. One caveat recorded for future rounds: the TR-pool
arm is a PPO opponent piloting TR teams, which sets and exploits the room
far less than humans do, so it is a weak local proxy for the ladder TR-set
split (6/19 = 32%); any future TR attempt should gate on a better instrument
(e.g. a TR-executing scripted arm, or directly on ladder TR-set games).

Cost of the null: one overnight of idle compute + one battery night. The
deployed brain was never at risk. Per the user's standing order, a 25-game
ladder batch launches on the deployed champion when the chain drains, and
the full analysis digest follows.

## League 2 complete; screening running vs the DEPLOYED brain (August 30, 21:20)

Run finished on schedule (5 checkpoints to 17,694,720; 8 absorbed known-class
parse crashes, both classes now shim-fixed for future runs). The curriculum
visibly bit: eval/heuristic on the TR-heavy slate rose 0.79 -> 0.90 (ckpt 3)
and eval/bc finished at its best (0.82). Candidates: 17694720 (0.85/0.82) and
16711680 (0.86/0.81).

**Baseline for all round-2 gates is `results_league/league_champion.zip`**
(the deployed brain), not the old champion. Screening chain launched 21:20:
both candidates x 5 arms (n=1,000 paired), mix_A memorization diagnostic, and
the pre-registered **TR-pool diagnostic** (64opp frozen opponent with the
TR-boosted league-2 team weights -- the slice this round exists to improve).

**Pre-registered bars (written before any result):**
- Screening advance: no arm below baseline by >4pp AND (weighted >= +3pp OR
  TR-pool delta >= +4pp).
- Promotion (5,000/arm): no arm below baseline by >2pp, TR-pool delta >= +3pp,
  weighted >= 0 -- a TR-specialist upgrade is acceptable ONLY if nothing else
  pays for it.
- Ladder confirmation focus: the TR-set split (was 6/19 = 32%) plus overall
  non-regression vs 55/100.

## Post-promotion loss analysis -> Zoroark fix shipped + league round 2 (TR curriculum) launched (August 30, midday)

Full-instrument pass over the new brain's 100 games:

- **Mechanical blunders are gone.** Immunity-blocked moves: 15 across 55 wins
  vs 18 across 45 losses -- near-even and rare; zero timer losses; guards and
  rerankers in normal ranges. Nothing left to hand-patch at the move level.
- **Zoroark parse crash FIXED** (`vgc_bench/src/pokeenv_patches.py`): the
  shim now swallows exactly the Illusion team-overflow ValueError (counted,
  reported); all other ValueErrors still propagate. Scope test extended
  (`test_pokeenv_patches.py`). This was the last known battle-stalling bug.
- **Config-split of the canary corpus:** TR-likely matchups are FIXED --
  49% (23/47) vs the old bot's 36.6%, at human level (48.2%). The residual
  strategic hole is games where TR actually gets set: 32% (6/19), vs humans'
  ~42% in the same spot. The brain also reorganized its own preview: it now
  leads charizard+garchomp half the time (57% WR) where the old bot favored
  charizard+floetteeternal.
- **Confidence is still anti-calibrated** (chosen-action probability HIGHER
  in losses; AUC 0.445-0.454) -- unchanged pathology, still no free lunch
  from confidence-gated interventions.
- Value-net ladder Brier drifted to 0.2218 on the mixed-config corpus (it
  models the old champion; not deployed; retrain before any future
  search/counterfactual use).

**League round 2 launched 12:10** from the promoted checkpoint with a
property-derived TR curriculum: `build_league.py --config league2_config.json`
boosts P(TR)>=0.6 team weights x3 (72% of training mass; no species
hardcoding). Pool: promoted brain (resume at 12,779,520) + its round-1
history + the OLD champion + bc_mix_A x3; +5 intervals to 17,694,720. Same
battery gates; all five arms remain never-trained-against. Pre-registered
expectation: the TR-pool diagnostic and TR-set ladder split improve without
any arm regressing.

## PROMOTED ON LADDER: 55-45 over 100 audited games -- the first deployed-behavior improvement in project history (August 30)

The league fine-tune candidate completed the pre-registered 100-game ladder
read: **55-45 (55.0%), Wilson95 [45.2, 64.4], z=2.03 vs the 44.9%/321
baseline (one-sided p=0.021)**. Mechanically spotless across all 100: zero
timer losses, zero poke-env parse errors, one Zoroark-roster game without
incident. **First faint ours: 34/100 (34%) vs the historical 52.5%** -- the
opening-exchange fix the whole diagnosis pointed at, confirmed at scale on
real ladder.

Honest calibration: the CI lower bound (45.2) sits just above the old point
estimate, so this is a pass of the pre-registered bar, not an overwhelming
one; continued accumulation tightens it and the corpus now feeds the
calibration instruments (421 total audited games across both configs).

**Deployment: `results_league/league_champion.zip` is now the ladder
checkpoint** (explicit `--checkpoint` invocation; sha 8cc54b2b, stamped
production_candidate -> production). `results_repaired/champion.zip` remains
untouched as the immutable prior champion and local baseline arm.

The arc that got here, in one paragraph: the 2026-08 diagnosis said the
~40pp sim-to-ladder gap was opponent distribution -- the policy had never
trained against human-like play. The fix was exactly that: fictitious-play
fine-tune of the champion's own weights against a pool seeded 3/8 with the
human-BC policy (bc_mix_A), gated by a 5-arm battery of never-trained-against
populations (screening n=1,000, promotion n=5,000, memorization diagnostic),
then the standard ladder rollout. Every stage passed on the first attempt.

**Next:** keep accumulating ladder games (target 150+ for a tighter read and
per-Elo-bin analysis); rerun `tools/analyze_ladder_previews.py` and
`calibrate_vs_ladder.py` on the new corpus; fix the Zoroark parse
vulnerability (chip open); then the last unchecked goal -- generalization
beyond the fixed team -- and/or a second league round from the new
checkpoint (the loop is now proven and repeatable).

## Ladder rollout at 35 games: 20-15 (57.1%), clean; running to the 100-game threshold (August 30, morning)

Canary 5-5 plus extension 15-10 = **20-15 over 35 audited serial games**
(+12.2pp over the 44.9%/321 baseline; one-sided p ~= 0.07 -- suggestive, not
yet a claim; the pre-registered threshold is ~100-150 games). Between-batch
review clean again: zero parse errors, zero timer losses on our side, one
Zoroark-roster game passed without incident (normal loss, no parse damage).
**First faint ours: 11/35 (31%) vs the historical 52.5%** -- the
opening-exchange transformation holds at the larger sample. A 65-game batch
is running to land the corpus at exactly 100 games for the first
claim-grade read.

## Ladder canary: 5-5, mechanically CLEAN, first-faint rate transformed; 25-game extension running (August 30, 00:51)

The promoted league candidate played its first 10 real ladder games
(serial, audited, `ladder_replays_league_canary_20260830/`, run_config.json
verified -- the Stage-A instrumentation's first live corpus). Record 5-5.

**Mechanical review: clean on every check.** Zero poke-env parse errors
(patch report), zero tracebacks, zero Zoroark encounters (the known parse
bug remains live-untested; fix chip open), no timer losses (the two
inactivity flags were OPPONENTS forfeiting -- both wins for us), normal
guard/reranker activity (91 decisions, guaranteed_ko x7, rerankers engaged).

**Loss shape moved exactly where the diagnosis predicted.** First faint was
ours in only 2/10 games vs the historical 52.5% -- the opening exchange,
identified in the 2026-08 diagnosis as where games are decided and where the
stack was thinnest, is the thing the league fine-tune visibly changed.
Conversion after taking the first KO matched history (5/8 = 62.5% vs 63.6%),
and both first-faint-ours games were losses (historical 25.6%). n=10 decides
nothing about win rate; the canary's job was "did anything break" and the
answer is no.

**Extension to 25 more games launched 00:51** (same config, same dir; 35
total when done). Per the standing protocol the next read happens there;
~100-150 games needed for any claim against the 44.9%/321 baseline.

## PROMOTION: league candidate 12779520 passes the full battery (August 30, 00:09)

**First promoted policy candidate in project history.** 25,000 paired battles
(5,000/arm, seed 83, hidden sheets) against five populations neither policy
ever trained on:

| arm | champion | candidate | delta |
|---|---|---|---|
| heuristic | 86.4% | 91.0% | +4.6pp |
| frozen 64opp | 78.8% | 85.5% | +6.8pp |
| rotation 8opp | 82.6% | 86.9% | +4.3pp |
| rotation tuned | 81.4% | 85.5% | +4.2pp |
| human holdout eval_B | 81.4% | 83.9% | +2.5pp |

All four pre-registered conditions pass: zero arms below champion (bar: none
< -2pp), weighted +4.15pp (bar >= +2.0), delta-human_bc +2.5pp standalone
(bar >= +2.0), memorization diagnostic clean (2.2pp divergence vs 10pp flag).
Every screening delta replicated at 5x the sample -- the signature of a real
effect, not selection noise. Artifact: `results_league/league_champion.zip`
(copy of the 12779520 checkpoint; sha 8cc54b2b...; stamped
role=production_candidate). `results_repaired/champion.zip` remains deployed
and untouched.

**Next per the standing rollout protocol: a 10-game audited ladder canary**
(user-run, serial) with `--checkpoint results_league/league_champion.zip`,
then loss review, then 25 more, then ~100-150 games for a real claim against
the 44.9%/321 baseline. The 2026-08 diagnosis said the sim-to-ladder gap is
opponent distribution; this candidate was built by closing exactly that gap
in training, and the ladder canary is the hypothesis' first live test.

## League fine-tune: BOTH candidates pass screening -- first gate passes in project history (August 29, evening)

The overnight league run (champion weights continued for +4.9M steps vs a pool
with bc_mix_A at 3/8 decaying share) completed cleanly: 5 checkpoints, 152-155
steps/s throughout, first-save kill criteria all passed (eval/heuristic 0.82
flag-band, eval/bc 0.82, ep_rew positive), four absorbed poke-env Zoroark
parse crashes (known bug, one pool team; chip filed for the ladder-side fix
since live play shares the parse path).

Screening batteries (n=1,000/arm, paired seed 83, hidden sheets, champion
baseline; every arm a population NEITHER policy trained against):

| arm | 11796480 | 12779520 (final ckpt) |
|---|---|---|
| heuristic | +5.3pp (91.9 v 86.6) | +7.9pp (93.2 v 85.3) |
| frozen 64opp | +9.4pp (86.9 v 77.5) | +6.8pp (84.3 v 77.5) |
| rotation 8opp | -2.6pp (81.8 v 84.4) | +4.2pp (88.2 v 84.0) |
| rotation tuned | +4.7pp (84.8 v 80.1) | +4.2pp (86.1 v 81.9) |
| human_bc eval_B (stoch) | +4.4pp (85.9 v 81.5) | +3.5pp (84.7 v 81.2) |
| promotion-weighted | +4.3pp | **+5.0pp** |

Both pass the pre-registered screening bar (no arm < -4pp AND (delta-human_bc
>= +4 OR weighted >= +3)). The mix_A memorization diagnostic is CLEAN on both:
delta vs the actual sparring partner exceeds delta vs the held-out sibling by
only 2.1pp / 2.2pp against a 10pp flag -- the gains generalize to human-style
play rather than memorizing mix_A's quirks. Every prior artifact in this
project gained only on populations it was fit to; this is the first to gain on
four-to-five populations it never saw.

**12779520 (clean sweep, highest weighted) advances to the promotion tier**
(5,000/arm, ~8h20m, launched 17:14; bar: no arm -2pp, weighted >= +2pp, AND
delta-human_bc >= +2pp standalone). 11796480 is held as backup. If promotion
passes, next is the ladder rollout: 10 audited canary games -> review -> 25 ->
~100-150 for a claim vs the 44.9%/321 baseline. The champion stays deployed
and untouched throughout.

## Round 2 REJECTED -- counterfactual track closed; league fine-tune pulled forward (August 29, 04:30)

The resumed aggregation round ran clean end-to-end (700/700 games, zero
failures, 4,424 new positions; combined dataset 11,697 train / 2,936
validation; tactical gate 18/18) and **every candidate failed the per-mode
regression gate**. Champion-paired n=500 results (epoch = validation rank):

- epoch 7 (top pick): open 420v437 (**-3.4pp**), hidden 438v432 (+1.2pp),
  population 390v384 (+1.2pp) -> weighted +0.05pp, dead on the open mode.
- epoch 2: open -0.6pp, hidden +0.4pp, population 390v407 (**-3.4pp**) ->
  weighted -1.75pp, dead.
- epoch 5: open +0.6pp, hidden +1.2pp, population 377v409 (**-6.4pp**) ->
  weighted -2.75pp, dead.

Nothing came within the +/-1pp confirmation window of the bar, so the new
n=1,500 confirmation tier never triggered. Reading both rounds together: six
battle-evaluated candidates cluster within eval noise of zero (champion
same-seed spread alone is 3.6-5pp at n=500); the only consistent sliver is
hidden-mode +1.2-1.6pp, well below promotion. On-policy aggregation (half the
round-2 rollouts steered by the round-1 residual) did not help and plausibly
hurt the population mode. **Per the climb plan: the counterfactual track is
closed** -- no round 3 unless the league changes the picture. The residual
recipe's honest legacy: the v2h value net (still the promoted leaf evaluator)
and the generation/validation hardening.

With the machine free 12 hours early, the league fine-tune moved up: smoke
run at ~04:35, real launch immediately after (5 intervals to 12,779,520,
port 7700, kill criteria at first save per the plan).

## The climb plan: round-2 resumed, league fine-tune infrastructure built (August 29)

A fresh strategic review (full plan approved by the user) started from the
uncomfortable truth: deployed ladder behavior has not changed since the Aug-4
champion weights -- every promotion since was measurement- or search-side, and
search is not deployed. Ladder truth stands at 144-177 (44.9%) over 321 games,
Elo median 1125, newest replay Aug 22. Three tracks now run:

**Track 1 -- round 2 resumed (tonight).** The paused aggregation round
restarted at 00:18; the 10 stale pre-fix failure records self-heal on resume
(errored games re-play and overwrite -- verified in
`generate_counterfactuals.py` before launch; error count was down to 3 within
two minutes). New measurement rule to kill the 0.05pp absurdity: the n=500
pipeline verdict is advisory; any candidate within +/-1pp of the +2.0pp bar
gets a fresh-seed n=1,500/mode confirmation on that single candidate before
accept/reject (champion same-config spread at n=500 was 3.6pp across the three
v5h eval runs -- verdicts at that n are coin flips at the margin).

**Track 2 -- league fine-tune of the champion (the never-pulled lever).** The
diagnosis says the ~40pp sim-to-ladder gap is opponent distribution, and the
policy has never trained against human-like play. Infrastructure landed today:

- `build_league.py` seeds `results_league/saves_fp_hs_wt/reg_mb/seed1/` with
  the champion at its own stem 7864320 (resume point), its four lineage
  checkpoints, and **bc_mix_A at stems 100/200/300** -- a 3/8 human-BC share
  decaying as self-saves join the FP pool. Stem 100 triggers callback.py's
  per-interval `eval/bc*` telemetry by file presence (the `--behavior_clone`
  flag must NOT be passed; it would change the method dir).
- Role quarantine is now enforced by CONTENT, not filename: integer-stem
  naming strips sidecars, so `league_manifest.json` (at the league root) pins
  every seeded stem to a sha256 and bans every checkpoint under
  `results_bc/eval_B/` by hash. `verify_league_dir()` (utils.py) runs inside
  `vgc_bench.train` before every launch; `callback.py` opponent sampling now
  filters to integer-stem zips (a stray .DS_Store previously crashed
  `int(p.stem)` mid-run), calls `refuse_eval_only_checkpoint` on selections,
  and logs `train/bc_opp_frac`. Negative test performed on the real pool: a
  hand-copied eval_B checkpoint under an innocent stem is refused by hash.
- `data/team_weights_regmb_league.json` zeroes `our_team.txt` (it is
  byte-identical to MB430.txt and BOTH sat in the sampled pool, silently
  double-weighting the mirror during training; explicit 0.0 because a missing
  key defaults to weight 1.0). Eval batteries keep the old weights file for
  baseline comparability.
- `run_league_training.sh`: champion flags except the pool (single-variable
  discipline), port 7700, `--total_steps 12779520` = +5 intervals (~9h
  overnight, 5 candidates). All three frozen PPOs stay OUT of the league so
  every gate-battery arm remains a never-trained-against population.
- Gates pre-registered: screening (1,000/arm paired) advance iff no arm worse
  than champion by >4pp AND (delta-human_bc >= +4pp OR weighted >= +3pp), plus
  a non-gating mix_A-vs-eval_B divergence diagnostic (>10pp gap = learned
  mix_A's quirks); promotion (5,000/arm) needs no arm -2pp, weighted >= +2pp,
  AND delta-human_bc >= +2pp on its own; then the standard ladder rollout.
  Kill criteria at first save: eval/heuristic < 0.80, eval/bc < 0.70,
  ep_rew_mean <= 0, worker deaths, or throughput < ~120 steps/s.

**Track 3 -- restart live evidence.** Zero ladder games exist since Aug 22 and
none have ever flowed through the Stage-A instrumentation. A 25-50 game batch
with the current champion config (user-run, any machine-free window) is the
highest information-per-effort action open; calibrate_vs_ladder and the
preview analyzer rerun after.

Team question resolved with the user: MB430 stays frozen through this cycle;
revisit with league results in hand. Suite 293 passed / 5 skipped; Ruff at the
16-error baseline; Pyright at the 23-error baseline (both untouched).

## v5h verdict: rejected at +1.95pp vs the +2.0pp bar -- best counterfactual round ever (August 25)

**The residual (epoch 6) improved every mode and missed promotion by half a
battle.** Weighted score +1.95pp against the pre-registered >=+2.0pp bar:
population +3.0pp (402/500 vs 387/500), hidden +1.6pp, open +0.2pp, no
regression anywhere. Every previous round was flat-to-negative and reliably
LOST 3-7pp on fresh populations; this is the first counterfactual artifact to
GAIN on the population mode -- exactly where human-grounded labels were
supposed to help. Training wiring all held: 10,437 usable positions, 2,089
held-out validation (the new >=2,000 gate), +0.8pp validation rank gain,
corrections applied on 42% of positions. The two other top-validation epochs
(8 and 4) were negative in battles -- validation rank ordering does not track
battle strength at this scale. The counterfactual-preview side candidate lost
-5.95pp and is dead, consistent with every other preview-model result.

Five distinct failures were fixed en route, each with a regression test:
corrupt Ditto particle ("nothing" placeholder move, also silently poisoning
LIVE hidden-world search vs Ditto teams), zero-candidate positions crashing
the NPZ writer, poke-env's [from]move override KeyError (Transform/echo),
the new validation-power gate correctly refusing an underpowered 1,660-row
split, and ragged candidate widths across chunks crashing collate.
Generation itself finished 1,750/1,750 games with ZERO failed games.

**Next: aggregation round 2** (the pipeline's designed loop, not a re-roll):
~700 new games with half the rollouts steered by the round-1 residual
(on-policy data for the corrected policy), round-1 data retained for
training, same bar, fresh eval seeds. The pre-registered bc_eval_B
(stochastic human holdout) check runs on whatever candidate faces promotion.

## Counterfactual retry v5h launched: human-grounded labels at scale (August 25)

The rematch of the project's most important failed experiment, with both
autopsied causes fixed. The four rejected rounds (v1-v4) generated labels with
`outcome_value: None` -- planner leaves scored by the champion's own overfit
critic -- while the champion's own adapter ranked the OPPONENT's branches
(blended 60/40 with the replay predictors), so every label was the champion
grading positions against its own idea of the opponent. And they validated on
~240-376 held-out positions, far below what the 0.5pp rank-gain gate can
resolve.

The v5h round changes exactly three things:

1. **Leaves: the v2h net** (`--outcome-value`), the only artifact here that has
   beaten a gate on a population it was not fit to.
2. **Opponent branches and rollout choices: the human-imitation policy.** New
   `--opponent-base-checkpoint` (plumbed through the pipeline) builds a
   `SplitBranchPrior`: our candidates stay champion-ranked (that is the policy
   being improved), the opponent's branches AND its sampled trajectory choices
   come from `bc_mix_A` blended with the replay move/switch predictors. The
   eval-only quarantine is enforced at load (`refuse_eval_only_checkpoint`).
   Side benefit: visited states now follow human-like opponent lines instead
   of champion-like ones.
3. **Power: >=2,000 held-out validation positions, enforced.** The trainer
   hard-fails before spending any training time when the split is smaller
   (`--minimum-validation-positions`, recorded in metrics), validation
   fraction raised to 0.20, and the pipeline's minimum usable positions raised
   to 8,000 (a ~6x scale-up over v4's 1,605).

Probe-calibrated scale: ~81s/game, ~7.7 positions/game single-worker at v4's
proven search shape (depth 2, root 8, opp 6, chance 1, budget 20s); the run
targets 1,400 games / ~10k positions on 6 workers (~5-7h generation), then
residual training, the tactical gate, and the standard 3-mode x 500-battle
promotion gates, all under the existing resumable pipeline
(`results_counterfactual_v5h/`). Pre-registered extra check before believing
any pass: the selected residual also faces `bc_eval_B` (stochastic) --
the anti-overfitting comparator the old rounds never had.

## Exact-preview repair: the 8/24 re-gate was invalid; planner repaired (August 24-25)

**The Stage-E "exact preview ties champion" result was an artifact.** The
n=300 re-gate's arm telemetry shows `exact_preview_truncated: 300` and zero
`exact_preview` decisions: every one of the 300 preview searches blew its
8-second budget, and `_planned_teampreview` silently fell back to champion
preview on every battle. The run measured champion-vs-champion (hence the
"tie"). Root cause: the eval harness forces serial play for wall-clock move
search (`effective_workers = 1 if move_search`) but not for the preview-search
arm, so 8 concurrent searches shared one policy-inference lock and all timed
out. The ORIGINAL hidden-60 gate (44/60 vs 53/60) was valid -- its telemetry
shows 59/60 searches deciding -- so "the old net loses by 15 points" stands,
but "the v2h net closes the deficit" was never actually measured. The previous
status entry's re-gate paragraph is superseded by this one.

**Harness fixed so this cannot silently recur:** preview-search arms now run
serial like move-search arms, and every arm's JSON carries a first-class
`preview_search` block (budget, determinizations, decisions used, truncated
fallbacks, errors) plus a loud WARNING when an exact-preview arm made zero
decisions. `--preview-determinizations` added to both entry points.

**The planner itself was repaired (`live_preview.py` rewritten):**

1. **Multi-determinization.** The planner sampled ONE hidden-set world; every
   ranking was hostage to that sample's items/spreads. It now runs up to 8
   mass-weighted worlds sequentially and merges rankings with the same
   60/30/10 risk blend the move planner uses (`aggregate_plans`, extracted
   from `ExactDeterminizationPlanner` into a shared module function). Only
   cleanly-completed worlds vote; truncated worlds are dropped, and if no
   world completes the decision reports `truncated` and the champion path
   plays (>= 1 clean world is never worse than the old single-world planner).
2. **Champion-pick injection.** Candidates were the preview predictor's
   top-12 plans; the champion policy's own pick was invisible to the search
   whenever the predictor ranked it 13th or lower. A `ChampionInjectedPrior`
   wrapper now guarantees the champion's plan a candidate slot in every world
   (`_champion_preview_order` recomputes the champion's two-stage pick
   side-effect-free, audit rows suppressed), so the search can only override
   the champion after actually evaluating the champion's plan. Decision logs
   record `champion_rank` and `override_margin` per preview, and guard
   counters split `exact_preview_agrees_champion` / `_overrides_champion`.
3. **Budget corrected to the actual rules.** The 8s preview budget was
   inherited from move-turn thinking; the VGC Timer grants **90 seconds at
   Team Preview** against a 420s bank (`Timer Max First Turn = 90`,
   `data/rulesets.ts`). Preview budgets up to 60s are now allowed (each world
   still capped at 9s), and the planner runs off the event loop
   (`asyncio.to_thread`) in both player classes because budgets beyond ~15s
   would break the 20s websocket keepalive that the old in-loop call relied
   on implicitly.

Team-agnostic throughout: worlds come from the belief over whatever roster the
opponent shows; injection uses whatever the champion picks; nothing references
MB430. 17 new unit tests (`test_live_preview_repair.py`); suite 277 passed;
Ruff clean on changed files; Pyright unchanged at the 23 pre-existing errors.

**Second finding: 8 seconds stopped being enough for even ONE world.** The
serial baseline re-gate (`baseline_serial_hidden300.json`) STILL truncated
300/300 -- concurrency was necessary but not sufficient. Offline profiling
with the production stack (champion on mps, knowledge obs on, v2h evaluator)
measured **25.5s to complete one 12x6 preview world** (195 nodes); the
hidden-60 era completed searches in ~2-3s, so per-node cost grew roughly
tenfold as the stack gained knowledge embeddings and the outcome evaluator.
Every "exact preview" number produced at an 8s budget since that cost growth
was champion preview wearing a costume. `PlannerConfig` now admits budgets up
to 60s (move-turn entry points still enforce <= 9s themselves), and per-node
cost reduction (embed caching, batched child ranks/leaf evals) is the queued
lever that would buy k=3+ worlds later.

**Deployable shape chosen: k=2 worlds x 56s total** (28s slices, each fitting
the 25.5s need with margin; 52s of the 90s preview allowance; bank use
trivial). The 6-battle smoke confirmed the full loop live: 5/6 searches
decided (the 6th truncated and stood down to champion), all deciding searches
completed 2/2 worlds in 41-54s, the champion's plan appeared in every
ranking (injection working; ranks 1,1,4,4,6), 3 overrides with margins
0.097-0.149, agreements report margin exactly 0.

**Gates:** (1) baseline serial n=300 -- DONE, invalid-as-search but a clean
second champion reference (268/300 = 89.3% vs champion arm 265/300 = 88.3%,
pure noise, confirming the costume effect); (2) repaired-planner gate
(k=2 x 56s, injection on) n=300 seed 303 -- DONE (`repaired_w2_hidden300.json`).

**Gate verdict: NOT PROMOTED -- the repaired search's overrides are exactly
neutral.** Topline 259/300 (86.3%) vs paired champion arm 252/300 (84.0%)
looks like +2.3pp, but the decomposition kills it. The search decided 203/300
previews (97 stood down at the 28s slices; no wall-clock drift -- first/second
half medians 49.8s/49.9s; 174 decisions completed both worlds) and OVERRODE
the champion's pick in 134. In those 134 treated battles: exact won 114, the
champion arm won 115 of the very same battles (delta -1). The +7 net came
entirely from the untreated subsets (agree +4, fallback +4), where both arms
played identical previews and only downstream server RNG differed. Discordant
battles overall: 35 vs 28, two-sided p = 0.45. Override margins were large in
the planner's own units (median 0.159, max 0.458) and converted to nothing --
**at this depth (one exact first turn, 1x1 continuation) and this evaluator,
the preview search's perceived margins carry no real win-probability signal.**

The honest arc: the OLD valid gate showed overrides actively hurting (-15pp,
blind evaluator); the v2h evaluator + repair brought overrides to exact
parity, not superiority. Champion preview -- a frozen PPO argmax -- currently
equals ~50 seconds of exact simulation across two hidden worlds per decision
against this opponent population. Champion preview stays deployed; the search
preview returns to candidate status with its remaining levers (deeper
first-turn continuation, per-node cost reduction to afford k=3+ and bigger n)
explicitly deprioritized behind the counterfactual/residual retry, per the
discipline against polishing a null.

Also worth recording: the SAME champion config, same seed, read 90.0% /
88.3% / 84.0% across three runs (server battle RNG) -- a 6pp same-config
spread at n=300. Only within-run paired comparisons mean anything at this
sample size; cross-run comparisons of 300-battle win rates are storytelling.

## Stage E complete: the human-grounded value net passes every gate (August 24)

**Dataset (`outcome_data_v2h`):** 19,918 games / 145,345 states, generated with
the new opponent mix (35% champion-free human prior over the BC base, 25% raw
human BC, 25% frozen-PPO rotation across all three checkpoints, 15% model
prior, 0% uniform), every game labeled at its Team Preview state (the v1
blindness fixed at source), 24.5% of games with OUR side drawn from the team
pool, and a healthier 67.4% label base rate (v1: 78.4%). 82 failures (0.4%).

**The v2h outcome net (`results_outcome_v2h/outcome_value.zip`) passed every
gate it faces:** pooled data gate; per-style gate on all five styles; the
holdout-style gate (trained with `historical` excluded, still beats the
champion critic on it -- the first artifact in this project to improve on a
population it was not fit to); tactical fixtures 18/18; and the standing
ladder-replay instrument, **Brier 0.2037 vs the incumbent's 0.2141**, with the
late-game bucket collapsing from 0.1502 to 0.0992 and mean prediction tracking
the true base rate (0.420 vs 0.430). Honest caveat: the ladder-replay PREVIEW
bucket improved only 0.354 -> 0.332 (mean p 0.05 -> 0.08) even though locally
generated preview states are now well calibrated (Brier 0.198, mean p 0.646) --
either the replay instrument reconstructs preview inputs imperfectly or the
preview competence transfers weakly; the functional test below says the truth
is closer to "it works". The v2h net is now the default `--outcome_value` in
both entry points; the v1 net remains on disk for comparison.

**The exact-preview deficit was the blind evaluator, confirmed functionally.**
Re-gated at n=300 hidden with the new net: live exact preview 266/300 (88.7%)
vs champion preview 270/300 (90.0%) -- a statistical tie, up from 73.3% vs
88.3% with the old net. The 15-point deficit was the evaluator, not the
planner. A tie does not pass the "must beat" promotion bar, so champion
preview stays deployed; the exact preview teacher is a live candidate again
(multi-determinization and candidate widening remain unexplored levers).

**The Garchomp bring signal is confounding, settled causally.** The generic
`forced_bench_species` mechanism (team-agnostic, stands down safely, 1,000/1,000
preview firings in the A/B) forced the bench in paired 500-battle runs:
free-draft 79.0% vs benched 77.4% against the learned population, and 81.0% vs
74.8% against the human-imitation arm. Forcing the bench HURTS -- the policy
benches Garchomp exactly when the matchup is already favorable, as suspected.
Per the pre-registered rule, no ladder games are spent on this experiment.

Verification: 260 repository tests pass, Ruff clean. No ladder or training
process is running; the local eval server on port 7600 is left running.

## Stage D closed as a documented null; preview improvement rerouted (August 24)

Stage D's mandate was preview rules with content taken from data, never
intuition. The deeper mining (`tools/mine_tr_denial.py`, 1,910 human games
against dedicated Trick Room setters -- per-species set rate >= 0.9, the
trigger that separates real setter teams from the half-the-meta roster
aggregate) refuted every candidate rule, including the denial direction the
Stage-A aggregate suggested:

- teams that even HAVE a denial holder (Encore/Taunt/Fake Out/Imprison): 48.3%
  vs 53.6% without; LEADING one: 46.4% vs 50.8%;
- denial leads do suppress TR (set rate 38.8% vs 44.3%) but the suppression
  does not convert (TR-never-set with denial lead 49.7% vs 51.8% without);
- own-TR flip leads 45.3%; two pure-attacker leads 44.8% vs 50.7% (with no
  TR-suppression effect at all); Tailwind null in every cut; bring-slowest and
  frail-fast already refuted in Stage A.

Our own 71 dedicated-setter ladder games agree in direction (led our Encore
holder: 4/14; TR was set MORE often when we did). The consistent story:
winners against Trick Room are the sides whose normal game plan does not need
to bend at preview. Composition-level anti-TR rules have no supported content,
so none ship -- the rule engine's mandate ("adopt only what the mining
supports") yields the empty set, and building it anyway would be the
ship-on-plausibility failure the replan exists to prevent.

Where the preview gap actually lives: the ladder-calibration instrument showed
the deployed outcome net predicts ~5% win probability at every Team Preview
state (it never trained on one), which mechanistically explains the exact
preview teacher's rejected 44/60 gate -- it ranks preview plans with a blind
evaluator. Preview improvement therefore reroutes through Stage E: retrain the
outcome net WITH turn-0 states and human-grounded opponents, then re-gate the
repaired exact preview teacher against champion preview. The mining tools and
the null tables stay in results_analysis/ as the record.

## Stage B complete, Stage C implemented (August 23-24, overnight gates running)

Stage B built the two sim-to-real instruments the replan called for.

**Human-imitation arm.** `logs2trajs.py` now prefilters with named, counted skip
reasons (the top-500 bo1 file was 94% sheet-less -- previously 3,068 anonymous
"failed traj reads"), parses player lines tolerantly, and takes `--logs`,
`--out_dir`, and crc32 `--buckets`. The sheeted Reg M-B human corpus (both bo3
files plus both bo1 files, deduplicated) converted into two battle-disjoint
pools: `trajs_regmb_human_A` (6,712 trajectories, training-eligible) and
`trajs_regmb_human_B` (7,092, **eval-only**). Two behavior-cloned policies were
trained from the converted foundation checkpoint (`pretrain.py` gained
`--trajs_dir/--output_dir/--eval_every`); held-out human-action agreement is
38.7%/38.6% per-slot top-1 (above the 37% shallow-predictor baseline) and ~54%
top-3, with the two independent halves agreeing almost exactly. `bc_eval_B` is
stamped `role: eval_only` and `refuse_eval_only_checkpoint` hard-fails in both
data-generation pipelines if it ever enters a training mix. The eval harness
gained `--opponent-stochastic` (a deterministic BC collapses to one line per
matchup) plus sidecar sha verification, and `run_gate_battery.py` runs the
standard four-opponent battery at powered tiers (screening 1,000/arm, promotion
5,000/arm).

Characterization, 1,000 battles per arm, hidden sheets, stochastic human-BC
opponent: **champion 809/1000 (80.9%)**, pre-fine-tune seed converted_v4
610/1000 (61.0%). The registered prediction that the champion's local dominance
would collapse was wrong; instead the arm cleanly separates policies it was
never fit to by ~20 points with tight intervals -- a valid anti-overfitting
comparator, though not a ladder-difficulty proxy.

**Ladder calibration instruments** (`calibrate_vs_ladder.py`, rerun after every
batch). Confidence pass: the policy's chosen-action probability carries no
outcome information on real games -- AUC 0.492 at preview, 0.465 in battle
(n=97 joined battles). Value pass (replays all stored ladder games through the
deployed outcome net by injecting our known team as a synthetic |showteam|):
**ladder Brier 0.2141 vs the local test's 0.1035; ECE 0.1368 vs 0.0222** over
5,459 states from 312 battles. Per-turn: the net predicts a 5.0% win chance at
Team Preview (actual 39%, Brier 0.354 -- worse than a constant), recovers
mid-game (turns 4-6: 0.146), and turns overconfident late (predicts 58%,
actual 38%). The generation pipeline never sampled preview states, so exact
search has been leaning on the value net exactly where it is blindest. Standing
gate: no value net ships unless its ladder-replay Brier beats 0.2141, and Stage
E's retrain must include turn-0 states.

**Stage C implemented, default-off, gates queued overnight:**

- Turn-1/2 reliability floor: `_moveset_prior` now reports the renormalized
  posterior mass of its surviving set, and prediction reliability is
  `max(revealed/4, min(cap, posterior))` under `VGC_PRIOR_RELIABILITY_FLOOR`
  (default 0 = historical behavior byte-for-byte; per-arm override
  `--candidate-reliability-floor` for paired A/Bs). Awakens the opponent and
  tempo layers on the turns that decide the format; the >=0.999 switch-evidence
  gate is untouched.
- KO-promotion hardening: `stats_were_synthesized` (stateless recompute-and-
  compare detection of ensure_stats' from-absent spread), `robust_ko_scale`
  (min-roll damage scaled to a max-EV, boosting-nature defender), and
  `VGC_KO_PROMOTION_MODE` off/robust/skip with a hidden-item margin. Applies
  only to the promoting `guaranteed_ko` path; two_on_one compares both options
  against the same fabricated defender, which cancels. First counting evidence
  from stored logs: past `safe_spread_ko` promotions realized an opposing KO on
  only 4/7 of their turns. `tools/count_ko_promotions.py` measures realized-KO
  rates from the overnight off-vs-robust runs before any default changes.

### Overnight gate results (August 24, early)

**The reliability floor FAILS its pre-registered gate and stays off.** Five
paired 300-battle runs, floor arm vs revealed-moves-only baseline: heuristic
0.35 floor 253 vs 266 (-4.3 pts, beyond the -2 bar); population 0.35 tied
233/233; population 0.20 -2.7; population 0.50 -2.3; and first-faint-ours was
worse in all five floor arms. The mechanism itself verified working (turn-1/2
opponent-reranker activity rose ~20%, tempo influence nonzero), so the result
is informative rather than a wiring failure: bounded, posterior-scaled prior
evidence on the opening turns makes decisions worse against these populations
-- the same direction as every previous attempt to treat priors as facts.
`VGC_PRIOR_RELIABILITY_FLOOR` remains 0.0 and the per-arm A/B flag remains for
future experiments. (Arm noise at n=300 is ~2.7 pts, but zero of five
configurations showed any positive signal, so nothing merits escalation to a
powered tier.)

One instrumentation gap was found and fixed during the KO runs: the eval
harness only wrote decision audits for search arms, so promotion turns could
not be joined to replays. Decision logs are now written beside kept replays for
every arm, and the two KO-counting runs were repeated with logging.

**The KO promotion bound also stays off.** Counted over the repeated logged
runs (four arms, ~14,000 decisions): mode off fired 201/198 promotions per arm
with realized-KO rates 58.7%/54.0%; mode robust fired 168/155 (a 19% cut) at
58.3%/58.1%. The bound prunes claims but the pruned claims were not
disproportionately false, and win rates are within noise (246+239 vs 247+244 of
300). The pre-registered criterion -- a materially higher realized rate --
is not met, so `VGC_KO_PROMOTION_MODE` remains off. Caveat recorded: the
"opposing faint that turn" proxy is diluted by switches and Protects, so the
~57% realized band overstates the false-claim rate; a claimed-target-specific
counter is future work if ladder audits ever surface promoted-KO whiffs.

Stage C therefore closes with both mechanisms implemented, instrumented, and
correctly kept OUT of production by their own gates in a single night --
exactly the failure mode (ship-on-plausibility) the replan was built to end.
The per-arm floor flag, KO modes, and counting tools remain for future
experiments. Next: Stage D, denial-oriented preview rules from the mined
human-response tables.

Verification: 256 repository tests pass (18 new this stage), Ruff clean, no new
Pyright errors. Overnight queue: floor 0.35/0.20/0.50 paired 300-battle gates
(heuristic + population + open-sheet no-op check) and the two logged KO-counting
runs. No ladder process is running.

## Replan and Stage A: hardening + instrumentation (August 23)

A full replan replaced the prior promotion track after a three-way investigation
(321 parsed ladder replays, 1,072 decision audits, the production pipeline traced,
all training artifacts read). Diagnosis, in priority order: (1) every local gate
measures against two opponent policies while ladder play is against humans -- the
~40-point local-to-ladder gap is opponent distribution, and every learned artifact
that improved against the opponents it was fit on regressed 3-7 points against any
other population; (2) games are decided at Team Preview and turns 1-2, exactly
where the stack is thinnest (preview is raw two-stage policy argmax; the opponent
and tempo rerankers are inert until moves are revealed); (3) the losses are the
policy's own confident choices -- guards changed a played action zero times in
1,072 audited decisions. Standing decisions: the team stays frozen, every NEW
component must be team-agnostic (derive from roster properties at runtime, never
hardcode species), the search-arm ladder gate is dropped, and ladder batches are
run on demand as measurement, never as candidate selection.

Stage A landed:

- `--knowledge_obs` can no longer default silently. `ladder_ourteam.py` resolves
  explicit flag -> checkpoint sidecar metadata -> hard fail, verifies the sidecar
  sha256, and `stamp_checkpoint_metadata.py` stamped the six production artifacts
  (champion, converted_v4, three frozen opponents, outcome value; obs_len 12132).
- Every ladder run appends its resolved configuration to
  `<replay_dir>/run_config.json` and refuses a directory whose recorded material
  config differs, so each replay directory stays single-config and auditable.
  `eval_counterfactual.py` arms now embed a `resolved_flags` block.
- Team Preview shadow logging: both players write one consolidated turn-0 record
  per game (chosen leads/bring, predictor top-5 plans for both sides, opponent
  per-species Trick Room rates) to the decision log. The wasted our-plan
  computation in `_learned_teampreview` was removed.
- New team-agnostic `vgc_bench/src/preview_rules.py` (Trick Room set rates from
  the counted joint sets; roster-level probability).
- `eval_counterfactual.py` battle results now record `first_faint_side` and the
  opponent's Trick Room probability, with a per-arm `loss_shape` summary --
  first-KO-exchange rate is a first-class gate metric (ladder: 25.6% win rate
  after losing the first Pokemon vs 63.6% after taking it).
- New `tools/analyze_ladder_previews.py`. Its `ladder` pass reproduces the audit
  from disk (144/321 = 44.9%; per-batch records; Garchomp appeared 37.4% vs
  absent 69.3%; TR-set games 25.5%). Its `humans` pass mined 10,548 scraped
  human games (5,231 clean TR-vs-non-TR pairings) and corrected a planned rule
  before it was written: humans do NOT beat Trick Room by bringing their slowest
  Pokemon (46.0% when they did vs 51.0% when they did not); the winning response
  is denial -- when TR never got set the non-TR side scored 50.0% vs 42.1% when
  it did. The anti-TR preview work therefore targets denial and setter pressure,
  not slowing down. Humans hold 48.2% against TR-likely rosters overall; our
  36.6% on the same class marks a ~12-point recoverable gap.

Verification: 231 repository tests pass (12 new), five optional integration tests
skip, Ruff format and lint clean on every touched file, and no new Pyright errors
(23 pre-existing ones from the earlier uncommitted tree are flagged separately).
No ladder or training process is running.


The repaired PPO champion at `results_repaired/champion.zip` remains untouched and
deployed. Exact live search and its distilled residual are candidate components only;
neither reaches ladder until the complete local promotion and rollout gates pass.

## Latest planner repair

The local planner now repairs the failure patterns found in the last two exact-search
losses. Every move family still reaches root screening, but a useful move can no
longer be represented only beside a dramatically worse partner. Deep search adds RNG
samples only around inaccurate moves, and exact branches explicitly penalize a chosen
action that disappears because its Pokemon is knocked out first. Across hidden worlds,
the planner now prefers the strongest action that actually reached the configured
future-depth coverage instead of falling all the way back when an unsearched shallow
row narrowly tops the aggregate.

Live reconciliation was also extended for Encore into Fake Out/Struggle, Trick item
transfers, trapping/disabled request flags, and remembered charging-move targets.
Verification is 198 tests, 18/18 tactical fixtures, and 2,000/2,000 parity states over
two seeds. The ladder-default eight-second configuration passed hidden and open
latency gates with p90 7.73s/7.66s and maximum 7.75s/7.68s. Serial learned-opponent
local A/Bs tied champion 6/8 hidden and scored 6/8 versus 5/8 open; the sample is a
non-regression check, not proof of a win-rate gain. No ladder or training process is
running.

## Current decision after the rejected 25-game ladder batch

The 7-16 ladder batch was stopped. It exposed three structural bugs: spent Mega
Evolution state was missing from reconciled simulator shadows, the hidden-sheet clock
was divided so broadly that most roots never reached the second move turn, and
background pondering almost never matched the opponent's actual continuation. The
bot therefore looked future-aware in configuration while often acting on shallow or
impossible branches.

Those defects are repaired locally. Side-wide mechanics are synchronized, all eight
belief worlds are retained but only two representative worlds consume foreground
deep-search time, the best four root actions are deepened, and a result must reach at
least 50% weighted future-depth coverage or fall back to champion plus hard guards.
Production uses one shared RNG sample and pondering is off. A fresh hidden-sheet gate
passed with 11/12 useful-depth searches, one safe fallback, zero illegal actions,
7.37s p50, 8.61s p90, and 8.88s maximum latency. The repaired search then tied champion
10/12 in a small production-shaped hidden-sheet A/B; this proves no local regression,
not a win-rate gain.

A terminal-outcome Team Preview experiment was also rejected. It scored 83.0% versus
78.7% on one familiar learned opponent and 85.0% versus 83.0% against heuristic play,
but collapsed to 72.3% versus 79.7% against a separate learned population. That is
opponent-policy overfitting from observational outcomes. Champion Team Preview stays
active, and the candidate cannot be promoted. Verification is 189 passing tests,
18/18 tactical fixtures, and 1,000/1,000 exact snapshot parity. No ladder or training
process is running.

## Selective chess-style fixed-team system (August 22)

The candidate move planner now thinks in a chess-like cycle. On important positions it
searches complete simultaneous turns through the bundled Pokemon Showdown Champions
simulator. It submits its move, then uses the opponent's thinking time to expand likely
replies in an isolated simulator. On the next request it reuses that work only when the
observed opponent action, public state, legal actions, and hidden-world consensus still
match. Quiet or safely matched positions avoid a redundant fresh search; changed or
uncertain positions are searched again. Late background work is cancelled and can
never delay submission or mutate the live battle.

Foreground search screens every legal move family, deepens the strongest lines for two
move turns, samples likely opponent replies and shared RNG, and scores leaves with a calibrated
terminal-win network plus a 10% independent mechanics value. Hidden-world outcomes are
aggregated as 60% expectation, 30% lower-tail value, and 10% worst case. Team Preview
uses a separate exact teacher that looks through each bring/lead plan into the first
complete move turn.

The hard compatibility gates now pass:

- exact live snapshot parity: 1,000/1,000 states, zero mismatches;
- generated Showdown choices round-trip by Pokemon identity rather than mutable party
  index, including forced-pass/replacement turns;
- top-eight hidden-set coverage: 87.4% over all 165 team-pool species, with no missing
  species;
- post-ladder selective live-search timing: 7.82s p50, 8.09s p90, and 8.54s maximum
  over 30 mixed open/hidden decisions, with zero genuine planner fallbacks or missed
  submissions;
- permanent tactical ordering: 18/18 ladder-derived fixtures;
- repository verification: 185 tests passing and five optional integration tests
  skipped.

The terminal-outcome evaluator was trained from 10,000 completed games and 64,167
states split by opponent team. Its held-out Brier score is 0.1035, log loss 0.3304,
and calibration error 0.0222, compared with 0.1342/0.4571/0.0680 for the old critic.
It passes all 12 Earthquake, weather, Trick Room, Encore, Yawn, sacrifice, switching,
and endgame orderings.

Live exact search is wired behind `--search`; selective search is the default and
background pondering is disabled. Each battle maintains up to eight
set/bring worlds, conditions them on public moves/items/abilities/speed/damage, uses
one shared RNG outcome in production, and falls back to champion plus hard
guards on any incompatibility. Open sheets remove set uncertainty but retain up to six
possible back-pair roots because sheets do not reveal which four were selected. A
50-decision serial production gate completed with exact audits and no missed
submission. Three decisions directly reused background expansions in 0.09-0.32s. A
paired 25-game open-sheet test used approximately 25% fewer fresh searches and scored
22/25 versus 18/25 for search-every-turn; the hidden-sheet pair scored 20/25 versus
18/25. These local heuristic samples support a small serial ladder gate, not a proven
ladder win-rate gain. Team Preview remains champion-controlled unless a separately
trained preview candidate obtains at least 1,500 genuine planner labels and passes
evaluation.

### First serial ladder gate and repairs

The first ten audited ladder games finished 5-5. No turn timed out and four searched
continuations were reused immediately, proving that the chess-style lifecycle works
against real opponents. The candidate was still rejected: total decision latency
reached 10.08s p90/11.02s maximum, repeated Tailwind wasted turns in two losses, and
hidden brought-four worlds were discarded on new reserve reveals until one decision
fell back to champion plus guards.

The post-gate repair makes eight seconds the live search budget, runs every exact
ranking through the same production hard guards as the champion, and strictly rejects
a side-condition move that would mechanically fail. Hidden worlds are rebuilt around
every revealed Pokemon instead of merely shrinking; stable nickname identity handles
Mega Evolution, Transform, and custom nicknames. Expected elimination of impossible
worlds is now separate from a genuine action fallback in telemetry. A fresh
production-shaped local gate retained all eight hidden worlds and measured 7.82s p50,
8.09s p90, and 8.54s maximum over 30 decisions with zero genuine fallbacks or missed
  submissions. This gate preceded the later rejected 7-16 ladder batch described at
  the top of this file.

### Encore and no-weather Weather Ball repair

The audited `supergrokmax999` game exposed two independent final-decision failures.
On Turn 2, selective scheduling judged the position close enough to a searched branch
to skip a fresh search, so the champion policy chose Charizard Protect without valuing
the next-turn Prankster Encore lock. On Turn 8, exact search itself selected a
no-weather, 50-BP Normal Weather Ball into Whimsicott even though the champion prior
assigned that line only 0.00047% probability and Heat Wave was the stronger attack.

Both paths now share two final safeguards. A first Protect is demoted when Encore is
revealed and known to move first on the following turn, unless the partner guarantees
that the Encore user is removed immediately. The check respects Dark-type Prankster
immunity, Psychic Terrain, priority-blocking abilities, Trick Room, Tailwind, and
already-active Encore. No-weather Weather Ball is demoted only when Heat Wave is
currently legal and offers at least 50% more expected damage; it stands down for close
damage comparisons, active weather, and revealed Wide Guard. Exact continuations use
the same checks, so cached thinking cannot bypass them.

The expanded repository suite passes 185 tests, and the permanent tactical gate is
18/18. A 12-decision mixed open/hidden production-budget timing run measured 7.66s
p50, 7.95s p90, and 7.97s maximum. Two hidden turns safely used the champion fallback
because all sampled exact roots failed before producing a result; neither failure was
caused by the new guards, and neither threatened the turn timer. No ladder process is
running.

### Free Shell Smash and repeated-Protect repair

The `aadrisntgggg` audit showed that the bot already had the necessary hidden-set
evidence but valued it incorrectly. Shell Smash occurs in 420 of 473 recorded
Blastoise joint sets (88.8%), and the live move model assigned Shell Smash 30.4% on
Turn 1 beside Sneasler's 78.5% Fake Out prediction. Exact search nevertheless scored
Double Protect at +0.488, including a +0.389 worst modeled branch, because the leaf
value overvalued taking zero immediate damage and undervalued Blastoise reaching +2
offense and +2 Speed. Search completed only one move turn and therefore did not see
the following Water Spout double KO directly.

The final-decision layer now rejects Double Protect when the evidence-conditioned set
posterior gives at least 70% probability to a Shell Smash-class snowball move and a
currently legal attack can meaningfully damage that user. It stands down while
concretely stalling asymmetric Tailwind or Trick Room, and it never deletes the move
from legality. The same replay's final Charizard also repeated Protect with no reserve
or speed-control objective; the endgame rule now chooses a policy-supported damaging
move in that exact 1v2 pattern.

Verification is 185 passing repository tests and 18/18 permanent tactical fixtures.
A fresh 12-decision mixed timing run measured 7.65s p50, 7.84s p90, and 7.88s maximum.
One turn safely used the champion after an unrelated root-reconciliation budget
failure; no new guard caused a timeout or planner error. No ladder process is running.

### Single-target Weather Ball and 2-on-1 repair

The audited `Dux67` game separated three issues. Turn 2's second Tailwind was a
mechanically failing action already covered by the current redundant-side-condition
rejection. On Turn 6, exact search ranked Rock Tomb plus Heat Wave at `+0.826` but
Rock Tomb plus sun-powered Weather Ball at only `+0.164`; that was a leaf-evaluator
error, not a Showdown simulation error. The final live layer now compares the actual
current-weather damage of both moves and demotes Heat Wave when only one foe remains
and legal Weather Ball offers at least 10% more expected damage. The calculation
includes Weather Ball's dynamic type, doubled base power, weather boost, STAB,
accuracy, Utility Umbrella, and Bulletproof, including its hidden-sheet fallback.
Cached continuations use the same rejection and cannot bypass it.

On Turn 7, Protect plus Solar Beam beat the best two-attack lines by only `0.0005` in
the learned value. The existing 2-on-1 focus rule now has an exact Delphox regression:
when two attackers can make safe progress into the lone foe, that microscopic value
edge cannot turn one action into an unnecessary Protect gamble. Together these raise
verification to 185 passing repository tests and 18/18 tactical fixtures.

The post-repair 12-decision production-budget timing gate used six open-sheet and six
hidden-sheet decisions. It measured 7.81s p50, 7.91s p90, and 8.03s maximum, with no
planner fallbacks or missed submissions, and passed the latency gate. A few sampled
hidden roots failed identity reconciliation; the surviving roots completed safely and
that pre-existing diagnostic is separate from the new constant-time comparison.

Conservative training freezes every champion parameter and learns only a
confidence-gated joint-action residual. Four CPU simulator workers generate exact
labels while one MPS process trains for at most eight epochs; rounds are sequential to
avoid Apple-GPU contention. Complete root screens remain valid training labels when
optional deepening reaches its anytime budget. Failed generation games abort the run
and remain retryable. The first residual attempt in `results_iterative_v2/round_01`
did not improve held-out action ranking and was rejected; no candidate was promoted
and no training process is active.

Evaluation arms are now literal: champion policy, distilled policy (champion plus
residual), optional preview policy, and live exact search. The best three residual
epochs receive matched open-sheet, hidden-sheet, and learned-population evaluations.
Promotion requires at least a two-point weighted gain, no mode worse by more than two
points, tactical success, and acceptable latency.

`run_rollout_gate.py` implements the final pre-ladder gate: 500 paired-seed battles in
each of open, hidden, and learned-population modes with up to eight concurrent local
battles. It saves every battle result and replay, extracts every exact fallback and
timeout, reruns tactical tests, checks the ten-second cap, and writes a new deployment
manifest. It never overwrites the repaired champion. Only a passing manifest may
proceed to ten serial audited ladder games and then, after loss review, 25 more.

## Opponent-aware planning layer (August 4)

The last live ladder batch is stopped at 99 saved replays. Its compact audit found a
46-53 record and no broad type-chart regression. The apparent Dragon-into-Fairy error
was a Dragon Claw aimed at Staraptor while the opponent switched Sylveon in. That is
an opponent-intent failure, so further blind PPO/ladder volume is paused.

Three small replay-trained priors now supply the missing hidden-state model:

- `opponent_preview_top500_regmb.pt` jointly scores lead-two/bring-four plans. On
  held-out battles it reaches 33.3% exact lead, 59.0% lead top-three, 31.6% exact
  bring-four, and 55.7% bring top-three. It conditions its distribution on every
  observed switch-in during a battle.
- `opponent_switch_top500_regmb.pt` estimates voluntary-switch probability and the
  incoming Pokemon. Switch discrimination is modest (ROC AUC 0.63), so it is an
  advisory search prior, never a hard guard. Conditional replacement accuracy is
  47% top-one and 87% top-three.
- `opponent_move_top500_regmb.pt` ranks moves and targets from the active matchup,
  rosters, HP, turn, slot, and explicit move semantics. Held-out accuracy is 37.4%
  exact move, 76.6% move top-three, and 62.2% target class.

Training filters each replay side against the actual Elo floors recorded by the
top-500 scrape (1655 Reg M-B, 1432 M-B Bo3). The earlier all-sides preview model was
contaminated by lower-rated opponents from the same replay and is not the production
prior.

The learned selector itself remains experimental: in a matched 300-battle local A/B,
learned preview scored 86.7%, the specialized PPO preview 89.0%, random preview 77.7%,
and random-preview heuristic play 47.7%. Therefore the PPO still chooses our four and
leads; the learned preview network predicts the opponent. Runtime validation over 20
local battles found no predictor exceptions and about 0.9 ms median / 1.1 ms p90 for
the combined opponent models.

The priors are now connected to final turn selection through a conservative tactical
reranker. It can only reorder near-tied, non-vetoed policy actions. It adds the value
change caused by a predicted switch and the defensive value of our post-choice board
against predicted incoming moves; it deliberately does not reward raw immediate
damage, which the PPO already knows and which displaced support moves in the first
version. Hidden-sheet switch guesses stand down until actual movesets are known, while
revealed moves can still contribute defensive evidence.

Final 300-battle gates with deterministic learned opposition were 84.3% aware versus
82.7% baseline with open sheets (36 changed decisions), and 83.0% aware versus 79.7%
baseline with hidden sheets (5 changed decisions). Against the static heuristic it
scored 84.3% versus 83.0% (22 changed decisions). These samples support deploying the
layer conservatively; they do not establish a statistically certain ladder gain.

## Speed-control and Encore repair (August 5)

A won ladder battle exposed a separate tactical gap: the policy stacked Tailwind and
Trick Room, then double-Protected on the final Trick Room turn even though the healthy
Sand Rush Excadrill was faster outside room. The observation said that both field
effects existed but never computed the resulting move order or their remaining turns.

`tempo_reranker.py` now computes effective Speed with stages, Tailwind, paralysis,
known speed items, weather abilities, Surge Surfer, Quick Feet, Slow Start, and
Protosynthesis/Quark Drive. Trick Room reverses only equal-priority speed order;
priority itself remains dominant. Active comparisons are threat-weighted by HP, so a
5% Tyranitar does not cancel the relevance of a healthy Excadrill. Hidden opponent
EVs, nature, item, and abilities are represented as speed intervals; overlapping
ranges stand down rather than treating an imputed set as fact.

The same near-tie reranker now recognizes three joint/timed ideas:

- double Protect is penalized when the final Trick Room turn robustly favors us;
- one Protect plus continued progress receives modest credit when Trick Room
  robustly favors the opponent, including a protected ally beside Earthquake;
- Encore scores the move that will actually be locked after priority and speed order.
  In the reported Turn 2, +2 Rage Powder acts before +1 Prankster Encore, so Encore
  locks Rage Powder and receives no false credit for cancelling the previous Trick
  Room.

The first generic Protect-plus-spread bonus failed its local gate (79.0% versus 81.7%
over 300 open-sheet learned-opponent games) and was rejected. After restricting it to
harmful Trick Room, the 300-game open-sheet arm scored 85.0%; only four decisions had
different tempo evidence. A hidden-sheet interval check changed zero final actions in
100 games, as intended. These asynchronous local samples are directional rather than
statistically decisive, so the layer remains soft and near-tie-only.

Every ladder decision is now written to `<replay_dir>/decisions.jsonl`, including the
policy's top alternatives, hard demotions, predicted-opponent score, Trick Room and
Tailwind duration, speed-order confidence, and the exact tempo factors that changed
the choice. This avoids trying to infer the bot's reasoning from replay text alone.

## Retarget, priority, and safe-spread repair (August 5)

A later ladder replay exposed three deterministic inference failures. When Raichu
fainted in the left opposing slot, Showdown automatically retargeted Basculegion's
Last Respects into the sole surviving Farigiraf. The immunity mask had inspected the
now-empty requested slot instead, so it allowed a Ghost move that could only resolve
into an immune Normal type. Target resolution now mirrors that one-foe auto-retarget
for the type mask and every target-based guard.

The same replay revealed Armor Tail through a `|cant|` protocol message. Upstream
poke-env discarded the revealed ability and retained only the fact that Kingambit
could not act, causing Sucker Punch to be repeated. The parser shim now preserves the
revealed ability as battle state, and the factual guard blocks positive-priority moves
into revealed Armor Tail, Dazzling, Queenly Majesty, or grounded targets under Psychic
Terrain. Hidden-sheet games also use a separately attributed soft demotion when the
conditioned replay prior exceeds 99%; Farigiraf is Armor Tail in 99.7% of the current
Reg M-B set data, so the bot now avoids even the first Sucker Punch in that matchup.
It does not permanently blacklist every failed Sucker Punch because failure against a
status move is transient; it preserves the actual mechanic that caused the persistent
failure.

Charizard's sand-boosted Weather Ball into Tyranitar was not itself a type error: it
became a 100-BP Rock move and was comparable to resisted spread Heat Wave. The bad
half of the pair was Garchomp choosing Dragon Claw instead of a safe Earthquake beside
Flying Charizard. A narrow safe-spread rule can now promote Earthquake when its ally
Protects or is mechanically immune and the pair adds a guaranteed opposing KO. It
does not apply a broad spread-move bonus, which previously regressed evaluation.

The decision audit now records guard stages, demotion counts, and vetoed action pairs
alongside the opponent/tempo reranker evidence. Focused regressions cover all three
reported failures. A 50-battle production-path validation finished 43-7 (86%), with
no parser or decision fallback errors; the safe-spread rule changed 22 of 397 turns.

Training reward is still sparse terminal win/loss (`+1/-1`, with PPO `gamma=1`). PPO's
critic/advantage calculation does not literally reward every action in a win equally,
but credit assignment is weak enough that a poor move inside a win may not be
corrected quickly. The current repair is inference-time and regression-tested; the
next training phase should append explicit speed-control features and distill these
counterfactual candidate rankings rather than adding a large hand-written reward that
the policy can exploit.

## Resource, switching, and endgame repair (August 5)

The next ten-game ladder audit exposed five planning failures that the policy's
ordinary top-six action prefix could not repair: remaining in with a healthy Yawned
Pokemon, retaining a -2 Attack physical attacker, protecting one slot in a clean 2v1,
spending Mega Evolution on Floette before a visible Pelipper + Swampert rain line,
and choosing resisted Dragon Claw/Aqua Jet over a policy-supported guaranteed KO.

Candidate generation now preserves one legal switch and one ordinary non-Mega move
per slot. These additions are marked `strategic_only`: generic opponent/tempo
reranking cannot promote them unless a specific guard has already put one first.
That isolation matters. The first broad implementation let low-probability rescued
actions enter ordinary reranking, changed too many turns, and scored 248/300 (82.7%)
versus the previous planner's 265/300 (88.3%), so it was rejected.

The accepted rules are deliberately narrow:

- switch a healthy Yawned Pokemon unless staying guarantees the end of the battle;
- offer a switch to reset a healthy physical attacker at -2 Attack or worse unless
  its current line guarantees a KO;
- in a clean 2v1, prefer two policy-supported attacks to a one-Protect gamble when
  both attacks make more progress;
- preserve Charizardite Y when rain is active, or when a visible Pelipper/Politoed
  plus active Swampert makes the imminent rain-speed plan concrete, if declining the
  other Mega loses no guaranteed KO;
- allow ally-safe Earthquake to cross a large policy gap, but allow an ordinary
  guaranteed-KO promotion only when the policy still gives it at least 48% of the
  top line's probability and the top line spends an attack into a resistance.

The local damage wrapper also repairs a poke-env omission: Last Respects now uses
`50 * (1 + fainted allies)` base power (without mutating the Pokemon's shared Move
object). This fixes the audited comparison where two fainted allies made it 150 BP
but the planner evaluated it at 50 BP beside rain-boosted Aqua Jet.

After narrowing, the same 300-battle seed scored 266/300 (88.7%) against the static
heuristic, effectively level with the previous 265/300 result while retaining the
targeted corrections. This is an acceptance gate against broad collateral damage,
not proof of a ladder win-rate increase.

## What the review found

- The model did learn useful team preview. Against a historical learned-policy
  population, learned preview scored 65% while random preview scored 58% under the
  factual guard profile.
- The old static-heuristic benchmark overstated strength and was too noisy to decide
  whether training helped. It previously reported strong local results while ladder
  play remained near 40%.
- The full nine-guard stack overcorrected. In matched 100-battle population runs,
  factual guards scored 65%, no guards scored 63%, and all guards scored 59%.
- Type ignorance is now handled independently of the learned policy. The action mask
  forbids known damaging immunities even when team sheets are hidden.
- Approximate one-ply search was not a faithful Pokemon simulator. It is disabled and
  both ladder/evaluation scripts refuse `--search` until live poke-env snapshots can
  be synchronized with the exact Showdown simulator.

## Loss audit and retest (August 6)

The 14-11 ladder batch exposed three repeated-looking patterns: early Basculegion
losses without an immediate KO, Mega Floette consuming the team's one Mega before
Charizard Y could contest rain, and no-progress Protect lines in a reserve-less final
2v2. Only the latter two survived testing.

An opponent-prior survival intervention was rejected twice. The first version fired
123 times per 100 hidden-sheet battles and scored 81/100 versus 90/100 before the
change. Restricting it to early Basculegion turns reduced firing to 13 times in 300,
but still scored 249/300 versus 268/300. It is disabled by default. This was the
correct rejection: in the next ladder batch Basculegion fainted by Turn 2 in 7/10
wins but only 5/15 losses, so preserving it was not a reliable objective. Profitable
trades, not survival, are what matter.

The accepted changes make rain-Mega reservation persist when Floette's best ordinary
move changes, recognize active Archaludon and other concrete rain abusers, and reject
zero-progress final 2v2 Protect lines when no asymmetric speed-control turn is being
stalled. Against a deterministic historical learned-policy population they scored
226/300 versus 227/300 before the changes, effectively level. The result is saved in
`results_repaired/eval_accepted_loss_repairs_population_300.json`.

The fresh 25-game ladder retest finished 10-15 (40%) with 249 audited decisions, 21
team-sheet timeouts, and no parser or decision fallback errors. Neither newly
accepted rule changed a move in that batch, so the decline from the previous 14-11
sample cannot be attributed to those rules. Combined, the two batches are 24-26
(48%), which is the honest current live estimate.

Across the two batches, Whimsicott + Basculegion leads were 16-9 while other completed
leads were 7-17. That correlation did not establish causation: a direct 100-battle
population A/B scored adaptive preview 71/100 and a forced Whimsicott/Basculegion
lead 70/100. The fixed lead remains available through `--stable-lead` for experiments
but is not the ladder default.

The conclusion is now stronger than another list of guards: exact mechanics failures
are mostly contained, while live losses remain broad strategic valuation errors. More
ladder volume on the same policy will measure that weakness but will not teach it.
The next useful model change is counterfactual planning data: generate plausible
opponent responses, evaluate several successor boards, and train the policy/value
model to rank those outcomes rather than learning only from terminal win/loss reward.

## Production inference profile

The ladder default is now:

- corrected knowledge observation enabled for the v4 checkpoint;
- hard type-immunity action mask enabled;
- evidence-conditioned moveset priors when sheets are hidden;
- factual guards plus narrow regression-tested planning: zero damage,
  first-turn-only moves, known redirection, status immunities, priority denial,
  redundant side conditions, safe-spread/resisted guaranteed KOs, Yawn/debuff
  switches, 2v1 focus-fire, and rain-Mega resource preservation;
- learned opponent bring/move/switch priors and conservative near-tie reranking;
- a 20-second Team Preview cap when an opponent leaves the Open Team Sheets prompt
  unanswered; normal turns are unaffected;
- approximate search disabled.

The experimental ally-damage, setup-into-KO, repeated-Protect, and KO-tiebreak rules
remain available for A/B tests but are not production defaults.

## Repaired training distribution

The next run is configured to train the fixed ladder team against all 546 Reg M-B
teams. Sampling is a 50/50 mixture of uniform full-pool coverage and archetype
frequency from 4,819 matched top-player replay previews. This gives roughly 133
effective teams instead of concentrating almost all training into about 50.

Half of training battles hide Open Team Sheets. Both simulator clients reject sheets
in those battles, which creates the correct hidden information without poke-env's
accept/reject race. Unknown sets are filled from evidence-conditioned usage priors.

Training uses fictitious play against saved historical policies, a fixed team on our
side, PPO KL protection, and the corrected observation. It cannot start accidentally:
`run_repaired_training.sh` requires `results_repaired/training_gate.pass`. The gate
passed with 64% open-sheet, 73% hidden-sheet, and 65% learned-population win rates;
the learned-population 95% lower bound was 55.3%.

## Verification

- 76 unit tests pass when the local test servers are running; five server-dependent
  cases skip when their ports are absent.
- Integration pipeline tests pass; local server-dependent cases are exercised in CI.
- Eight standalone bot regressions pass, including exact Showdown state cloning,
  hidden-sheet sampling, guard wiring/composition, observation repairs, and immunity
  behavior.
- Checkpoint conversion preserves every existing policy tensor and produces exactly
  identical projection output when the newly appended features are zero.
- Ruff formatting/lint and Pyright pass.

## Remaining blocker

An exact Showdown state bridge now exists and advances cloned Reg M-B battles, but live
search remains blocked on a parity-tested poke-env-to-Showdown state synchronizer. It
must cover switches, targeting, Protect, Mega Evolution, weather, abilities, items,
and hidden-information determinizations before search is safe for ladder decisions.
The opponent candidate ordering is now learned rather than highest-damage-only, but
it is intentionally dormant behind this exact-simulation gate. Once parity passes,
search can branch over the predicted top-three moves and likely switches, evaluate
their exact successors with the critic, and feed the improved decisions back into
self-play/distillation.
