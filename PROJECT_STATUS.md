# VGC Bot Project Status

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
