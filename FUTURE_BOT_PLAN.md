# Future-Looking VGC Bot Checklist

## Replan (August 23) — active track

The promotion track below is superseded by a diagnosis-driven replan (approved
2026-08-23; full text in the session plan). Standing rules: the team stays frozen
but every new component must be team-agnostic; ladder batches are on-demand
measurement, never candidate selection; local gates must be powered (screening
1,000 battles/arm, promotion 5,000/arm) and every learned artifact validated
against a population it was not fit on.

- [x] **Stage A — hardening + instrumentation**: knowledge_obs fail-safe with
  stamped checkpoint sidecars; per-directory `run_config.json`; Team Preview
  shadow logging; team-agnostic `preview_rules.py` (Trick Room rates);
  first-faint/TR metrics in eval battle results;
  `tools/analyze_ladder_previews.py` reproducing the ladder audit from disk and
  mining 10,548 human games (finding: anti-TR play is denial, not slowing down).
- [x] **Stage B — human-imitation eval arm + calibration instrument**: fix
  `logs2trajs.py` (per-reason skip counters, showteam prefilter, tolerant rating
  parse, parameterized I/O); convert the ~7,100-game sheeted Reg M-B human corpus
  (mostly bo3) into disjoint A/B trajectory pools; train `bc_mix_A` (training
  side) and `bc_eval_B` (quarantined eval arm); wire a 4-arm gate battery with a
  stochastic learned opponent; characterize champion vs `bc_eval_B` x1,000; new
  `calibrate_vs_ladder.py` (policy-confidence AUC from decision logs; ladder
  Brier for the value net) as the standing sim-to-real instrument.
- [x] **Stage C — opening-turns fixes**: BOTH mechanisms implemented,
  gated, and REJECTED for production by pre-registered criteria (2026-08-24).
  Reliability floor: 0/5 paired 300-battle runs positive, first-faint worse in
  all five -> `VGC_PRIOR_RELIABILITY_FLOOR` stays 0. KO promotion bound: 19%
  fewer promotions but realized-KO rate flat (~58%) and win rate within noise
  -> `VGC_KO_PROMOTION_MODE` stays off. Machinery (per-arm floor flag, KO
  modes, `tools/count_ko_promotions.py`) retained for future experiments.
- [x] **Stage D — preview rules: closed as a documented NULL (2026-08-24).**
  The dedicated-setter mining (1,910 human games) refuted every candidate rule
  -- denial holders/leads, TR flips, attacker pressure, Tailwind, slow brings
  all score at or below baseline -- so no rule content ships. The preview gap
  traces to the outcome net being blind at turn-0 states (predicts 5% win
  at every preview); preview improvement reroutes through the Stage-E value
  retrain, then re-gating the exact preview teacher.
- [x] **Stage E — value-net retrain + bring-selection experiment (2026-08-24)**:
  the v2h net passed EVERY gate (pooled, per-style, holdout-style, tactical,
  ladder Brier 0.2037 < 0.2141) and is the new default leaf evaluator; the
  Garchomp signal was settled causally as confounding (forced bench: -1.6 pts
  vs population, -6.2 pts vs the human arm -> no ladder spend).
  ~~The exact-preview "tie" with the evaluator fix~~ was INVALIDATED on
  2026-08-25: all 300 re-gate searches truncated and silently played champion
  preview (see PROJECT_STATUS). The evaluator's effect on exact preview was
  never measured.
- [x] **Exact-preview repair (2026-08-25): gated, NOT promoted -- overrides
  exactly neutral.** Harness now forces serial play for preview-search arms
  and reports a per-arm `preview_search` block (zero-decision arms WARN);
  planner rewritten with multi-world determinization (mass-weighted,
  clean-worlds-only voting, champion fallback when no world completes),
  champion-pick injection (`champion_rank` / `override_margin` logged per
  decision), off-loop execution, and budgets matched to the real VGC Timer
  (90s at preview; one world measured at 25.5s with the production stack ->
  shape k=2 x 56s). Gate n=300 hidden seed 303
  (`results_preview_repair/repaired_w2_hidden300.json`): 203/300 searches
  decided, 134 overrode champion, and the treated subset went 114 wins vs the
  champion arm's 115 in the same battles (delta -1; overall discordance 35 vs
  28, p=0.45). Perceived margins (median 0.159) carried no real signal.
  Champion preview stays. Remaining levers (deeper continuation, per-node
  cost reduction for k=3+ and powered n) parked BEHIND the counterfactual
  retry, per the no-polishing-nulls rule.

## League fine-tune (August 29) — the climb plan's new lever

Continue PPO from the champion's own weights against a pool that finally
includes human-like play (bc_mix_A at a 3/8 decaying share). Single-variable
discipline: champion flags unchanged except the opponent pool and the league
team-weights file (our_team.txt zeroed — it double-counted the MB430 mirror).
All three frozen PPOs stay out so every battery arm remains a
never-trained-against population. Measurement reform now standing: any
counterfactual candidate within ±1pp of its bar gets a fresh-seed
n=1,500/mode confirmation before accept/reject.

- [x] Safety layer: `verify_league_dir()` content-hash verification inside
  `vgc_bench.train`; callback opponent sampling hardened (integer-stem filter,
  eval-only refusal, `train/bc_opp_frac` telemetry); `build_league.py` with
  sha/role manifest + eval_B banned by content (all 31 epochs); negative test
  performed against the real pool; 13 unit tests.
- [x] League built and verified: `results_league/saves_fp_hs_wt/reg_mb/seed1/`
  (champion at 7864320 + 4 lineage + bc_mix_A ×3), `league_manifest.json`,
  `data/team_weights_regmb_league.json`, `run_league_training.sh` (port 7700,
  +5 intervals to 12,779,520 steps).
- [x] Live smoke proved the wiring (155 steps/s, bc_opp_frac live), then the
  run completed 2026-08-29: 5 checkpoints, first-save criteria passed, four
  Zoroark parse crashes absorbed without a stall.
- [x] Screening battery: BOTH candidates passed on five never-trained-against
  arms (12779520 swept all five, weighted +5.0pp; 11796480 +4.3pp);
  memorization divergence 2.2pp / 2.1pp vs the 10pp flag — clean.
- [x] **Promotion battery PASSED (2026-08-30, 25,000 paired battles):**
  12779520 vs champion — heuristic +4.6, frozen +6.8, rotation1 +4.3,
  rotation2 +4.2, human holdout +2.5 (standalone bar +2.0). Zero regressing
  arms; weighted +4.15pp vs the +2.0 bar. Every screening delta replicated at
  5x the sample. Artifact: `results_league/league_champion.zip`
  (sha 8cc54b2b…, role production_candidate). First promoted policy candidate
  in project history; the deployed champion remains untouched.
- [ ] Ladder rollout per the standing protocol: 10 audited canary games
  (user-run) → review → 25 more → extend toward ~100-150 for a claim vs
  44.9%/321. Local evidence has never been this strong, but ladder remains
  the only arbiter.


The repaired champion is immutable. A candidate becomes deployable only after every
gate below passes; a failed stage remains resumable and cannot start ladder play.

## Exact-planner repair (August 22, final local pass)

- [x] Compact root coverage now adds a strong joint partner whenever set-cover would
  represent a move only through a dramatically weaker pairing.
- [x] Inaccurate moves receive adaptive shared-RNG sampling during deep search without
  multiplying every deterministic branch.
- [x] Branch values penalize selected actions lost when their Pokemon is knocked out
  before moving.
- [x] Hidden-world aggregation selects the best action whose future was searched over
  the required posterior mass instead of discarding the whole plan.
- [x] Snapshot parity covers Encore-to-Struggle, Trick item swaps, trapping, disabled
  moves, and the private target of our own charging move.
- [x] Error audits use a fresh `error_fallback` schedule rather than inheriting the
  previous turn's successful schedule.

Final gates: 198 repository tests, 18/18 tactical orderings, 2,000/2,000 parity
states across two seeds, and accepted 12-decision production latency in both modes.
At the ladder-default eight-second search budget, hidden timing was 7.64s p50,
7.73s p90, 7.75s max; open timing was 7.53s p50, 7.66s p90, 7.68s max. There were no
illegal or missed submissions. Fresh serial learned-opponent A/Bs tied champion 6/8
hidden and improved 5/8 to 6/8 open; these samples establish no regression, not a
statistically reliable win-rate gain. No model training or ladder run was started.

## Post-25-game ladder repair (August 22)

The 7-16 ladder batch was stopped and rejected. Its main failure was not simply a
weak learned value: the live/exact bridge forgot that Mega Evolution is a side-wide
spent resource, so shadow worlds offered impossible second Megas; the planner spread
its clock across eight hidden worlds and usually never completed the future turn; and
background pondering matched only four of 189 observed continuations. Team Preview
also remained specialized and brittle. These are pipeline failures, so that batch is
not training evidence and no model receives reward from it.

The repaired candidate now clears the following local gates:

- 1,000/1,000 divergent-shadow live snapshots reconcile exactly, including spent
  Mega/Z/Dynamax/Tera resources;
- a 12-decision hidden-sheet production gate reaches required future depth on 11/12
  decisions, safely falls back on the remaining decision, and records zero illegal
  actions or legality-driven choice changes;
- search timing is 7.37s p50, 8.61s p90, and 8.88s maximum with no missed submission;
- repaired selective search ties champion 10/12 in a production-shaped hidden-sheet
  local A/B, with zero illegal actions and p90 7.86s;
- 189 repository tests and all 18 permanent tactical fixtures pass.

The observational terminal-outcome Team Preview candidate is rejected. It improved
two familiar hidden-sheet modes (+4.3 and +2.0 points) but regressed from 79.7% to
72.3% against a separate learned population. It learned opponent-policy correlations,
not a robust causal preview ranking. Champion Team Preview remains active; open-sheet
battles always fall back to it, and the rejected model is opt-in only.

## Priority 0 — Measurement and action compatibility

- [x] Identity-based Showdown-to-poke-env move/switch mapping
- [x] Mandatory legal-candidate round trips; generation aborts on any incompatibility
- [x] Literal champion, distilled, preview, and live-exact evaluation arms
- [x] Preview arm disabled unless a new preview model was actually trained
- [x] Search configuration, latency, truncation, root failure, and fallback audits
- [x] Permanent fixtures for the reported ladder mistakes

Acceptance: 100% action compatibility in the generation smoke and 189 repository
tests passing (five integration tests skipped when their optional services are absent).

## Priority 1 — Terminal-outcome win evaluator

- [x] 10,000 games and 64,167 labeled states, with at most eight states per game
- [x] Fixed team versus the full Reg M-B pool, 50% hidden sheets
- [x] Opponent-team-disjoint train/validation/test split
- [x] Frozen champion actor and standalone outcome network
- [x] Temperature-calibrated probabilities and provenance
- [x] 90% learned outcome value plus 10% mechanics safety value at leaves
- [x] Earthquake, weather, Trick Room, Encore, Yawn, sacrifice, switching, and
  endgame fixtures

Acceptance: Brier 0.1035, log loss 0.3304, ECE 0.0222, and 12/12 tactical
orderings. See `results_parity/outcome_value_metrics.json`.

## Priority 2 — Hidden sets, RNG, and live exact parity

- [x] Up to 12 weighted particles per species
- [x] Legal team determinizations respecting Item Clause and Mega constraints
- [x] Evidence conditioning from moves, items, abilities, move order, and damage
- [x] One open-sheet set world; up to eight hidden-sheet worlds
- [x] Up to four shared exact-Showdown RNG samples; production uses one so useful
  future depth fits the turn clock
- [x] Public live-snapshot synchronizer
- [x] 1,000 sampled-state parity gate
- [x] Two-second move-family screen, depth-two deepening, eight-second live stop, and
  champion-plus-guards fallback
- [x] 60/30/10 expectation/downside/worst-case aggregation

Acceptance: 1,000/1,000 parity, 87.4% top-eight particle coverage, 8.61s p90 and
8.88s maximum hidden-sheet live-search latency, with no illegal action or missed
submission.

## Priority 2.5 — Selective search and chess-style pondering

- [x] Save next actions and successor positions from important foreground searches
- [x] Reuse a continuation only when opponent action, public state, legality, and
  hidden-world consensus still match
- [x] Skip a repeated search when the current position matches an acceptable searched
  branch; otherwise retain champion-plus-guards or fresh-search fallback
- [x] Keep background pondering implemented as an opt-in experiment
- [x] Disable pondering in production after only four of 189 jobs matched the next
  observed continuation and only three were reusable
- [x] Audit starts, partial/completed jobs, branch matches, errors, and rejection causes
- [x] Run a 50-decision mixed open/hidden timing and natural-match benchmark
- [x] Run paired local battles against search-every-turn and champion controls
- [x] Run a controlled ten-game serial ladder gate with full audits

Acceptance: p90 normal-turn latency remains at most nine seconds, no missed
submission, shallow searches fall back explicitly, and no material local battle
regression versus champion.

Measured August 22:

- Final serial production-shape gate: 50 mixed open/hidden decisions, 8.74s p50,
  8.93s p90, 9.17s maximum, zero fallbacks, and zero missed submissions. Three
  positions directly matched completed background work and returned in 0.09-0.32s.
- Paired 25-game open-sheet test: selective search 22/25 versus 18/25 for
  search-every-turn, with approximately 25% fewer fresh searches.
- Paired 25-game hidden-sheet test: selective search 20/25 versus 18/25.
- Production-budget 10-game checks also favored selective search (7/10 versus 6/10
  open; 8/10 versus 6/10 hidden), but these samples are too small to estimate a true
  win-rate gain.
- Four-game concurrent hidden-sheet stress: 6/8 with no planner fallback or policy
  inference race after serialization. Its 10.74s p90 is a local multi-game CPU
  contention result and is not ladder-safe; ladder remains one game at a time.
- First serial ladder gate: 5-5. There were no timer losses, but the candidate failed
  promotion because end-to-end p90/max reached 10.08/11.02s, repeated Tailwind was
  selected while Tailwind was active, and hidden worlds could collapse on reserve
  reveals. No additional ladder games were started.
- Post-ladder repairs put every exact ranking through the production hard guards,
  strictly reject redundant side conditions, rebuild hidden worlds around all revealed
  identities (including Mega and Transform), and use an eight-second search budget.
  The current 30-decision mixed gate measured 7.82s p50, 8.09s p90, 8.54s maximum,
  zero genuine fallbacks, zero missed submissions, and eight hidden worlds throughout.
- The audited Encore/Weather Ball game is now permanent: first-use Protect is rejected
  when a revealed faster Encore user can lock it and the partner cannot remove that
  user, while no-weather Weather Ball is rejected only when legal Heat Wave offers at
  least 50% more expected damage and no revealed Wide Guard justifies the single-target
  line. Active-weather Weather Ball also replaces Heat Wave when only one foe remains
  and its current-mechanics expected damage is at least 10% higher. The tactical gate
  is now 18/18 and all 189 repository tests pass.
- The hidden Blastoise regression now combines the set posterior with the board: its
  current top-player sets contain Shell Smash 88.8% of the time, so Double Protect is
  rejected when an active attack can contest that setup. It remains valid for concrete
  Trick Room or Tailwind stalling. A final lone Pokemon also cannot repeat Protect
  without a speed-control stall objective when a damaging move is available.

## Priority 3 — Conservative iterative training

- [x] Frozen champion plus confidence-gated residual joint-action ranker
- [x] Four concurrent CPU generators and one sequential MPS trainer
- [x] 50% champion / 50% latest-candidate trajectories after round one
- [x] 50% hidden sheets and a 90-minute generation cap
- [x] Maximum eight epochs with validation after each epoch
- [x] Cumulative historical data and top-three checkpoint battle evaluation
- [x] Preview remains unchanged below 1,500 genuine labels
- [x] Two-point weighted promotion and two-point per-mode regression gates
- [x] ~~A production aggregation round passes its ranking and battle gates~~
  **Track closed 2026-08-29 after round 2 REJECTED** (best round-1 candidate
  +1.95pp vs the +2.0pp bar; every round-2 candidate breached the per-mode
  regression gate — see PROJECT_STATUS). Six candidates over two
  properly-powered rounds cluster within eval noise of zero; per the climb
  plan, no further counterfactual rounds unless the league track changes the
  picture.

The first residual attempt in `results_iterative_v2/round_01` was rejected because it
did not improve held-out action ranking. A later battle evaluation was paused at
389/500 games. No candidate was promoted and no training process is active.

## Rollout

- [x] Automated 500-game open/hidden/population rollout driver
- [x] Up to eight concurrent local battles; ladder remains serial
- [x] Per-battle results/replays and per-turn exact audits
- [x] Automatic loss, fallback, timeout, tactical, and latency review artifact
- [x] New deployment manifest that never overwrites the repaired champion
- [x] Selected candidate passes 500 paired-seed battles in all three modes
  (superseded by the stronger 5-arm/25,000-battle league promotion battery,
  2026-08-30)
- [x] Ten serial ladder games with full audits
- [x] Every ladder loss/fallback/timeout reviewed
- [x] Twenty-five additional serial ladder games (extended to 90 more)
- [x] **Fixed-team candidate PROMOTED (2026-08-30): 55-45 over 100 audited
  ladder games** (Wilson95 [45.2, 64.4]; z=2.03 vs the 44.9%/321 baseline,
  one-sided p=0.021; first-faint-ours 34% vs 52.5%; zero timer losses, zero
  parse errors). Deployed ladder checkpoint:
  `results_league/league_champion.zip` via explicit `--checkpoint`;
  `results_repaired/champion.zip` remains the immutable prior champion.
- [x] ~~Search on the promoted brain~~ **re-gated 2026-09-05: valid tie
  (+0.7pp vs +3.0 bar, 1,415 in-budget decisions, 2/3 of searches truncated)
  -- not deployed; ladder test skipped per the user's rule.** Evaluator v3h
  also rejected (ladder Brier 0.2145 vs 0.2087). Exploiter probe: 60.2% vs
  the champion via opening exchange + long games -> league 3 input.
- [ ] Generalization beyond the fixed team begins
