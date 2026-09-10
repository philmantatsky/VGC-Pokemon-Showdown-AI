# Reg M-C program: a new brain, a new team (2026-09-10)

Decision (user, 2026-09-09 evening): the bot becomes a Reg M-C bot. After the
first M-C ladder read, training uses M-C battles only (new Pokémon and items
included), a new M-C team is built, and the main goal stays battle logic --
a new brain designed from the ladder losses, without assuming the current
team survives.

This document is the design and the argument. The checklist lives in
`FUTURE_BOT_PLAN.md`; the dated log in `PROJECT_STATUS.md`.

## 1. What the ladder losses say (159 games, deployed brain, Reg M-B)

`tools/ladder_loss_profile.py` over every game the deployed brain played on
ladder (Aug 30 canary 125, Sep 6 batch 23, guard reads 11); pooled 77-82
(48.4%). Full tables: `results_analysis/loss_profile_deployed_mb.json`.

| signal | when it holds | otherwise |
|---|---|---|
| we suffer the first faint | 26.2% (17/65) | 63.8% (60/94) |
| we are behind on faints after turn 2 | 24% (11/45) | -- |
| first KO of the game lands on turn 1 | 81/159 games | on turn 1-2: 130/159 (82%) |
| opponent Fake Out on turn 1 | 33% (8/24) | -- |
| opponent sets Trick Room | 31.0% (9/29) | 52.3% (68/130) |
| we land >= 2 super-effective hits | 61.1% (58/95) | 29.7% (19/64) |
| our lead Basculegion + Whimsicott | 35.3% (18/51) | Charizard + Garchomp 52.6%, Charizard + Whimsicott 57.7% |
| Basculegion faints first | 9.1% (1/11) | Charizard first 21.1%, Garchomp 26.7% |
| Wave Crash on turn 1 | 65% of those games lost (43) | Heat Wave turn 1: 38% lost (48) |
| opponent rating >= 1200 | 44.2% (38/86) | 53.4% (39/73) |

Reading: **games are decided in the first two turns**, and they are decided
by (a) who lands the super-effective damage and (b) who eats the disruption
(Fake Out, Intimidate, Trick Room). Losses average 4.0 faints for us vs 1.8
for them; only 5 of 82 losses were close (one opposing Pokémon left). The bot
does not lose long, grinding games -- it loses the opening exchange and
never recovers. This is the same diagnosis as August ("broad strategic
valuation errors at preview/turns 1-2"), now with numbers.

Two more facts matter for the design:

- The three factual targeting guards (resisted target, overkill split,
  weather-ball type) were the only measured gain in five fine-tuning rounds.
  Target selection IS the super-effective-hit statistic above. A brain that
  cannot see "my move X into their slot B is a KO, into slot A is resisted"
  loses the opening exchange no matter how it is trained.
- Five league rounds from the deployed weights (2, 3, 3b, 4, 5) all failed the
  gates. Fine-tuning the same architecture on the same signal is plateaued.

## 2. What the current brain cannot represent (code facts)

Verified in `vgc_bench/src/policy.py`, `policy_player.py`, `vgc_knowledge.py`,
`env.py`, `train.py`, `callback.py`:

1. **Slot 2 does not know what slot 1 chose.** The policy emits one logit
   vector for both slots; the second slot's distribution is the first slot's
   logits under a legality mask. Double-targeting for a KO, "spread + Protect",
   "Fake Out + set up" are only expressible as marginal habits, never as a
   conditional plan.
2. **Knowledge is one-sided.** The 24-float knowledge block (damage fraction,
   guaranteed-KO flag, type multiplier per move × foe) is computed for OUR two
   actives only. The network is never told what the opponent's moves do to us,
   who moves first under Tailwind/Trick Room/priority, or that Fake Out is
   available (first turn on the field) on their side.
3. **No memory.** The observation is a snapshot: no record of what the
   opponent did on previous turns beyond `is_last_used` per move. The bot cannot
   notice "this player Protects every other turn" or "they will set Trick Room
   next".
4. **Sparse signal.** Reward is terminal ±1 with gamma 1; ~10 decisions per
   game per side; matchup luck dominates. A PPO update sees ~300 games worth
   of noise per gradient step. The value head has to learn material,
   tempo and speed control from win/loss alone.
5. **One team.** The deployed brain was trained with our side fixed to MB430.
   It can pilot other teams (tokens are property-based, no species ids) but has
   never practised doing so, so any team tournament it plays is biased toward
   MB430-like teams.
6. **Opponent models live outside the network** (preview/move/switch priors,
   rerankers) and are trained on M-B replays; they say nothing about the 35
   new species.

## 3. The new brain: ranked ideas

Each item names the mechanism, the loss it targets, the cost, whether old
checkpoints survive, and the gate it must pass. Ranking is by expected value
per unit of compute on this machine (mps, ~150 steps/s ≈ 1.85 h per 1M
steps).

### 3.1 Joint-action head (autoregressive slot 2) -- highest value, cheap

Condition slot 2's logits on slot 1's chosen action: embed the chosen action
(107-way) and add a learned projection into the slot-2 latent before
`action_net`. Initialise the projection to zero so the converted checkpoint
plays bit-identically on day one; PPO then learns coordination.
Targets: super-effective double-targets, KO + cover, Fake Out + setup.
Cost: ~150 lines in `policy.py` + `convert_checkpoint.py` + tests; no extra
inference latency worth measuring.
Gate: paired battery vs the same weights without the head after one league
round; plus a coordination counter (share of turns where both slots attack
the same foe when only the pair is a KO) from the decision audit.

### 3.2 Threat-symmetric knowledge (tail-append) -- high value, cheap

Append to every ACTIVE token, ours and theirs: best incoming damage fraction
and KO flag from each opposing active (their revealed or prior-most-likely
moves), a speed-order bit per pair that respects Tailwind/Trick Room/priority
(the `tempo_reranker` already computes this), and "Fake Out available"
(first turn on the field + move known or prior >= 0.3). Old checkpoints
zero-extend (`convert_checkpoint.py`, tail-append invariant).
Targets: the opening exchange (who moves first, what kills what), Fake Out
disruption, Trick Room reads.
Cost: `vgc_knowledge.py` + `utils.py` lengths + one converter run + tests;
~+15% embed time (calc calls double; memoised per state).
Gate: same battery as 3.1; knowledge-block ablation (zeros vs filled) at
equal weights must show the filled brain ahead after training.

### 3.3 Opponent-action memory tokens -- medium value, moderate cost

Add K=4 history tokens to the transformer input: for each of the last four
turns, the opponent's two revealed actions (move id embedding, target class,
switch flag, Protect flag) and our own. New tokens, not new features:
old checkpoints keep their 12 Pokémon tokens; the history projection starts
at zero. Targets: within-game adaptation (Protect rhythm, TR timing,
switch patterns) -- the thing rerankers approximate with static priors.
Cost: observation plumbing in `policy_player.py` (a per-battle ring buffer
already exists for the audit), `AttentionExtractor`, converter, tests.
Gate: battery; plus a "Protect prediction" probe -- does the value/policy
change when the opponent's previous action changes?

### 3.4 Dense learning signal -- medium value, cheap to try

Three complementary pieces, all leaving the objective unchanged:
- **Potential-based shaping** on material: reward += Φ(s') − Φ(s) with
  Φ = (their faints − our faints + HP-fraction differential)/k. With gamma 1
  this telescopes to a constant per episode, so the optimal policy is unchanged
  but every turn now carries gradient.
- **Auxiliary heads** on the shared features: predict next-turn faint events
  for all four actives and the opponent's next action class (self-supervised
  from rollouts). Representation learning for exactly the quantities in
  section 1.
- **Critic warm start** from the calibrated outcome value net (v2h, Brier
  0.1035) by distillation before PPO resumes, so the value head starts near a
  known-good evaluator instead of relearning it from ±1.
Cost: `env.py` reward (shaping), `policy.py` heads + loss terms in a small
PPO subclass, one distillation script. Gate: sample efficiency -- reach the
league-1 eval marks in fewer steps; battery as always.

### 3.5 Team-agnostic training (our side drawn from a pool) -- required

Train with our side sampled from the M-C candidate pool (section 4) plus
MB430, weighted toward the finalists. The architecture supports it
(`our_team_paths` accepts many teams; tokens carry no species ids). This is
what makes a team tournament fair and what protects the program from the
team question being decided later.
Cost: none in code; harder optimisation (the brain must learn several
teams), so budget 1.5-2x the steps of a single-team round.
Gate: per-team win rates vs the M-C pool; no team may lag the pool median by
more than 5pp before specialisation.

### 3.6 Population: M-C human clones + mixture -- required, existing infra

Behaviour-cloned humans from the M-C corpus (bucket split A=train, B=eval-only,
by content-hash as before), plus self-lineage; hardness mixture with
floors/caps (`training/make_league4_config.py`), no adversary in the first
round (four adversary rounds all overfit). Re-scrape daily; the corpus is
565 games today and grows with the format.

### 3.7 Preview specialist -- worth one round, parked behind 3.1-3.4

Lead choice is a measurable lever (Basculegion + Whimsicott 35% vs 58% for
Charizard + Whimsicott). The observational preview model overfit once
(2026-08); the fix is to train the preview head on the new brain's own
self-play outcomes vs the human population (on-policy labels, not human
labels), with the same bring/lead action space. Parked until the turn-1/2
mechanics above are in, because a better lead into the same bad exchange
gains little.

### 3.8 Search as a teacher -- parked

Exact search ties the champion at play time and four residual rounds failed.
Expert iteration (search targets distilled into the policy) is the last
lever on this list, to be revisited only if 3.1-3.4 stall on the battery.

### Bigger model?

d_model 256 / 3 layers / feed-forward 256 is small. Widening to 384 with 4
layers is a from-scratch retrain (no conversion path) and multiplies compute
per step by ~2.3x. Not in v1: every item above preserves the deployed
weights as the starting point, which is worth more than capacity while the
opponent distribution is still moving.

### Implementation status (2026-09-10, branch `brain-v1`)

Built behind flags, unit-tested (30 tests), converted checkpoint verified
bit-identical, live smoke on M-C teams clean (guards and rerankers unchanged):

- **3.1 joint-action head**: `MaskedActorCriticPolicy(joint_head=True)`;
  `logits_with_latent` / `conditioned_logits`; `build_candidates` and the
  search path use the conditional; `train.py --joint_head`;
  `convert_checkpoint.py --joint-head` adds the zero-ended head.
- **3.2 threat block**: `threat_obs_len = 8` at the token tail for BOTH
  sides' actives (`vgc_knowledge.threat_knowledge`, memoised in
  `PolicyPlayer._threat_for`, counters `threat_obs_computed/nonzero`);
  observation 1197 -> 1205 floats per token; every older checkpoint converts.
- **3.4 shaping**: `train.py --shaping_faint W --shaping_hp W`
  (potential-based, telescopes to the terminal +-1; both 0 = unchanged).
  Auxiliary heads and the critic warm start are NOT built yet.
- **3.3 memory tokens**: not built (changes the token count; v1.1).
- **Upgrade on load** (`policy.upgrade_policy`, `load_state_dict_upgraded`):
  every consumer -- the live player, `PPO.load` paths, training resume, BC
  init -- brings an older checkpoint to the current token length in memory
  (zero-extended tail columns, zero-ended head), counted as
  `policy_upgraded_obs_len`. The deployed brain, the clones and the lineage
  keep loading unchanged and play bit-identically; nothing on disk is
  rewritten.

Merging this branch changes the observation length for every process that
imports the package, so it merges only between runs (never under a training
or evaluation chain). BC trajectories on disk carry the old token and must be
regenerated before the next clone build; the pool checkpoints need no
conversion.

## 4. The team

MB430 is legal in M-C and still played (Basculegion is on 33% of M-C teams,
Kingambit 31%, Floette 22%, Whimsicott 15%, Charizard 13%, Garchomp 12%),
but the format moved under it. From 565 top-player M-C games
(`results_analysis/mc_meta_stats_20260909.json`):

- **The core**: Sneasler on 48% of teams, Rillaboom 40%, Incineroar 31%,
  Salamence (Mega) 31%, Kingambit 31%, Basculegion 33%. Pair cores:
  Rillaboom+Sneasler 255 teams, Rillaboom+Salamence 230, Salamence+Sneasler
  227. The consensus six -- Floette / Incineroar / Kingambit / Rillaboom /
  Salamence / Sneasler -- went 14/22 as an exact roster.
- **What it does to MB430**: Grassy Terrain halves Earthquake (Garchomp's
  main move); Grassy Glide is priority under it; Fake Out is on Rillaboom
  (67% of sets), Incineroar (100%), Sneasler (35%); Intimidate ×2 (Salamence,
  Incineroar) on our physical attackers; Close Combat everywhere for
  Kingambit; Salamence's Tailwind + Hyper Voice outpaces our Tailwind plan.
- **Trick Room** is set in 26% of games (149/565) and the setter's side wins
  60% of them. Our bot wins 31% against TR.
- Best-performing rosters with n >= 10 are single-player teams (one pilot),
  so their rates are pilot skill as much as team strength.

Candidates (all exist as legal export files in `teams/reg_mc/`, built from
real open team sheets, so spreads and items are what top players use):

| id | roster | why |
|---|---|---|
| T0 | MB430 (control) | comparability with every prior number |
| T1 | Floette-Mega, Incineroar, Kingambit, Rillaboom, Salamence-Mega, Sneasler | the consensus six; most human data to learn from and against |
| T2 | Indeedee-F, Farigiraf, Hatterene, Torkoal, Gallade-Mega, Kingambit | anti-meta: Psychic Terrain + Armor Tail switch off the meta's priority (Fake Out, Grassy Glide, Sucker Punch, Aqua Jet); TR punishes Sneasler/Salamence speed; Eruption / Expanding Force / Dazzling Gleam are SPREAD moves, so the plan depends least on the bot's weakest skill (single-target choice) |
| T3 | Indeedee-F, Gardevoir-Mega, Sneasler (Psychic Seed), Basculegion, Kingambit, Salamence-Mega | Psychic-terrain offence; 10/16 and 17/28 rosters in the corpus |
| T4 | Torkoal, Charizard-Y, Venusaur, Incineroar, Sylveon, Garchomp | sun spread (Eruption, Heat Wave, Hyper Voice); 13/14 roster; keeps our Charizard knowledge |
| T5 | Pelipper, Archaludon, Golisopod-Mega, Swampert, Grimmsnarl, Venusaur | rain; Electro Shot/Hurricane/Muddy Water; 7/9 roster; the least represented archetype (an edge if the meta ignores it) |

Selection is measured, not argued:

1. **Clone tournament** (cheap, team-agnostic pilot): the M-C human clone
   pilots each candidate vs the weighted M-C pool piloted by the same clone,
   n=300 per team. This ranks teams by "how well a competent generic pilot
   does with it in this meta". Candidates within 5pp of the leader survive.
2. **Brain tournament** after the team-agnostic round (3.5): the new brain
   pilots each survivor vs the pool (clone + self-lineage), n=500. The team
   the BRAIN plays best wins, which is the only criterion that matters.
3. **Specialise**: one league round with our side fixed to the winner; gates;
   ladder read.

The bot's measured strengths and weaknesses feed the candidate list: it wins
short games on super-effective damage and loses to disruption and single-
target mistakes. Teams whose plan is spread damage under a field effect
(T2, T4) or straightforward priority-immune offence (T2, T3) fit that profile
better than a Tailwind/Encore control team. T1 is the safe default. T5 is the
exploration pick.

## 5. Pipeline (what runs, in order)

**M0 -- ladder read (running)**: 25 games, deployed brain + three guards,
`--reg mc`, MB430. Tally: record, first-faint, `hit_effectiveness`, guard
fire counts, `ladder_loss_profile`. Baseline for everything below.

**M1 -- M-C data layer** (automatic after M0, `after_ladder_mc.sh`):
1. Re-scrape both M-C formats (bo1 + bo3) into a dated dir; merge corpora.
2. Rebuild the team pool from open team sheets (`extract_ots_teams.py`) and
   the opponent weights (`build_team_weights.py`), M-C only.
3. Retrain the opponent priors on M-C replays: preview, move, switch models
   → `data/*_regmc.pt`; wire `--reg mc` defaults in `ladder_ourteam.py` and
   `eval_counterfactual.py` to pick them.
4. Trajectories: buckets 0-4 → `trajs_regmc_human_A` (train), 5-9 →
   `trajs_regmc_human_B` (eval-only, banned by content from every pool).
5. Clones: `results_bc/mc_A` (30 epochs from the foundation checkpoint,
   div_frac 0.1, pick the epoch by agreement on B), `results_bc/eval_mcB`
   (role eval_only, run_id 2).
6. Battery anchored to M-C: `run_gate_battery.py --reg mc --team-weights
   data/team_weights_regmc.json --human-bc results_bc/eval_mcB/...`
   (scripted, frozen PPO ×3, human clone). Champion-vs-itself sanity first.

**M2 -- round 6, the M-C baseline** (automatic after M1): league fine-tune
of the deployed weights on M-C data only -- pool = deployed ×2 + resume,
league-1 history ×2, old champion, `mc_A` clone ×4; M-C team weights; our
side MB430 for continuity (single variable: the data). +5 intervals
(~9 h). Screening battery on the M-C arms. This is the cheapest test of
"does the plateau lift when the data changes?".

**M3 -- brain v1** (code while M2 trains; training after): 3.1 joint-action
head, 3.2 threat-symmetric knowledge, 3.4 shaping + auxiliary heads, behind
flags, converted from the round-6 finalist (or the deployed weights if round
6 fails), trained team-agnostic (3.5) on T0-T5 + the finalist weighting.
Kill/continue at the first save as always.

**M4 -- team tournament** (clone tournament runs during M2 as soon as the
clone exists; brain tournament after M3). Then specialise on the winner.

**M5 -- gates and ladder**: screening → promotion → 25 audited ladder games
with the user's word, serial, bot offline otherwise.

## 6. Compute and calendar

| step | wall clock | machine |
|---|---|---|
| M0 ladder read | ~2-3 h | light |
| M1 data layer | ~2 h (logs2trajs + clones dominate) | CPU + mps |
| M2 round 6 | ~9 h + 2 h battery | full |
| M3 brain v1 training | ~12-16 h (team-agnostic, +8 intervals) | full |
| M4 tournaments | ~3 h | full |
| M5 batteries | ~2 h screening, ~8 h promotion | full |

Nothing heavy overlaps the ladder. Each stage writes its verdict to
`PROJECT_STATUS.md` and the checklist in `FUTURE_BOT_PLAN.md`.

## 7. What would kill each idea

- 3.1/3.2: no battery movement after a full round with the counters showing
  the head/features are used → the bottleneck is the training signal, go to
  3.4 harder.
- 3.4 shaping: a brain that farms KOs and loses (shaped-return up, win rate
  flat) → drop the HP term, keep faints only, or drop shaping.
- 3.5: per-team win rates diverge by > 10pp → specialise earlier.
- Team: if the clone tournament and the brain tournament disagree by more
  than 10pp on the winner, trust the brain tournament and record the
  disagreement -- the clone is a pilot proxy, not the pilot.
- The M-C corpus staying small (< 1,500 games) → clones stay thin; the
  self-lineage and the scripted arm carry the pool, and the human arm is
  advisory until the corpus grows.
