# Future-Looking VGC Bot Checklist

## Opponent predictor — side experiment (October 4); not part of the deployed bot

Source of truth: `OPPONENT_PREDICTOR.md`. Gates a predictor-driven change must pass before
ladder, in order:

- [x] Pre-registered offline readings (147001a4). R1 quality: passes (-0.205 nats [-0.234,
  -0.177] against the count-table bar on our own ladder opponents). R2 Elo: does not stay
  (0.002-0.005 nats against 0.02) -> the Elo-blind model. R4 oracle: the opponent reranker
  cannot use a predictor (perfect foresight flips 4.0% [3.3, 4.8] of decisions).
- [x] Train/serve parity: runtime == offline path on 2,615 of 2,615 turns of our saved games.
- [x] Retrain on the fixed trainer and the rebuilt dataset (10-05): R1 stands (-0.196 on the
  old data); on 1.8x the battles fine NLL on our ladder opponents 1.672 -> 1.583. Learning
  curve running; a final retrain waits for the download.
- [x] Calibration maps built (10-05); not demonstrated on our own opponents, so not deployed.
  Fit them from the shadow log instead.
- [x] **Shadow mode ON at the user's word (2026-10-05 03:43):** the forecast is logged with
  every move decision and read by none (`DEPLOYED.json` `opponent_forecast`;
  `checks/forecast_shadow_live.py` passed).
- [ ] Read the shadow log on real ladder games (accuracy and calibration by sheet state)
  before anything acts on it.
- [ ] First decision-changing use: an **opt-in guard**, with thresholds fixed from the
  scorecard's decision-relevance tables beforehand, then the usual guard A/B (mirror for
  non-regression only: the mirror opponent is our own bot, which a human-move predictor is not
  expected to predict) and a ladder trial at the user's word.

## Deployed configuration: T6 + human openings (September 24; T6 from September 23)

- [x] Codex's T6 work committed (82db6ba).
- [x] **T6 PROMOTED by the user 2026-09-23** (`results_deployed/champion_mc_T6.zip`
  + `teams/candidates_mc/T6.txt`, `DEPLOYED.json`). Locally NOT better than T4
  (83.8% vs 86.2%); promoted on the user's decision. T4 is a prior champion.
- [x] Set-data side-by-side on T4 (`results_set_prior_ablation_T4`): T4 is better
  with its Reg M-B data (M-C -1.97pp, 95% [-3.66, -0.32], hidden-sheet games);
  T6 vs T4-with-its-data -4.30pp [-7.90, -0.89]. Set data now recorded per brain
  in DEPLOYED.json and applied by the launchers.
- [x] T6 ladder read (2026-09-23, user-delegated): 21-29, Elo 1253 -> ~1100; fixed
  opening diagnosed (`tools/ladder_opening_audit.py`).
- [x] Preview-entropy cycle (`results_brainv1_t6_preview1`): no clear change on
  either save; back row adapts, lead does not.
- [x] Decision for the user: switch back to T4? **No** (user, 2026-09-23: "switching
  doesnt solve the problem"); T6 stays deployed and learns matchup-dependent
  previews instead.
- [x] Human-trained preview model for our kind of team
  (`data/preview_t6_focus_20260923.pt`; gankyburner 15-4 with 7 lead pairs).
- [x] Human-opening cycle (pre-registered 0765aad; replaces the lead curriculum),
  2026-09-24: control arm no clear change (-2.51); **both trained saves BETTER**
  than the deployed brain (+4.93 [+1.22, +8.82]; **+7.08 [+3.69, +10.51]** for
  21,135,360, the pre-declared pick); practice adds +9.6 on identical openings;
  3 leads / 5 fours by matchup. Remaining weakness: rain matchups with the
  Farigiraf + Incineroar / Charizard + Venusaur openings.
- [x] **PROMOTED by the user 2026-09-24** ("make it the official bot and run 25
  ladder games"): `results_deployed/champion_mc_T6hp.zip` (save 21,135,360, sha
  c5e92237) + the model preview (a9de37c); T6 is now an immutable prior champion.
- [x] Ladder read started 11:27, stopped by the user at 2-1 after spotting
  move-choice mistakes; `dominated_attack` built, A/B locally neutral (+0.05pp),
  **mirror 71.5% [69.5, 73.4]** (wins close games); DEPLOYED at the user's word.
- [x] `trick_room_direction` (user's Trick Room lesson): HURT where it acted
  (-9.8pp); stopped; off. Redesign must judge by our whole brought four / plan.
- [x] Ladder read with the attack check: 22 games 11-11; the 25-game read 13-12
  (T6: 21-29, T4: 27-23). Loss causes in PROJECT_STATUS (2026-09-25 21:10).
- [x] Four fixes built at the user's word ("yes do all 4 fixes"), opt-in guards
  with tests on the real ladder positions: `dominated_throat_chop`,
  `focus_boosted`, `keep_our_weather`, `trick_room_counter` (12 of 571 logged
  ladder decisions change, 11 games). Two claims corrected (games 11/19 blocks
  were unshown; weather overwrite 1 loss vs 2 wins).
- [x] Their mirrors (pre-registered 2026-09-25): Throat Chop 50.4% then 49.7%
  with focus_boosted (pooled 50.05% [48.5, 51.6], neutral) -> **DEPLOYED**
  (replay_tag T6_humanpreview1_throatchop); Trick Room counter **39.1%, loses**
  -> off (contesting their room beats yielding it); weather **unmeasured** (3
  changes in 2,000 games) -> off, the user decides.
- [x] 15-game ladder read with Throat Chop (2026-09-26): 8-7, 1144 -> ~1170.
- [x] New user lessons -> `wide_guard` + `focus_boosted` (held-out A/B pooled
  +0.53 [-0.08, +1.19], deploy-eligible) and `dominated_spread` (mirror **56.4%
  [54.2, 58.6]**, wins close games): all DEPLOYED (replay_tag
  T6_humanpreview1_wideguard). Near-tie mixing (the user's wheel) LOSES its
  mirror (41.8%): off. `keep_our_weather` off (user).
- [x] Training cycle `brainv1_t6_contexts1` (35% human-clone opponents, from
  T6hp): both saves **no clear change** (pooled +1.02 [-0.05, +2.14], +1.27
  [-0.16, +2.68]); gains against the held-out human clones with open sheets.
  Not promoted.
- [x] Ladder read of the 8-guard stack (T6hp): 8-7, 1175 -> peak 1239 -> ~1218.
- [x] Head-to-head (the user asked): save 22,118,400 beats T6hp 59.3% [57.1,
  61.4] -> **PROMOTED by the user 2026-09-26** as `champion_mc_T6ctx.zip`
  (replay_tag T6ctx; guards, team, preview unchanged).
- [x] Ladder read of T6ctx (user, 2026-09-26): 6-9, 1219 -> peak 1282 -> ~1150.
  Mistake review of wins and losses: wrong attack/target is the largest class,
  as common in wins; the network's own pick corrected in 45% of decisions
  (T6hp 47%); the ladder reranker undid 6 of 37 guard corrections.
- [x] Tactical fine-tune (my decision at the user's delegation, pre-registered in
  PROJECT_STATUS 2026-09-26 18:05): `results_tactical1/sft/tactical_e4.zip` **wins
  its mirror 58.0% [55.8, 60.2]**; battery -0.40 [-1.61, +0.84] (no clear change,
  no population below -3pp); guards correct it 31% less. Promotion: the user.
- [x] T6ctx reference arm for guard/brain A/Bs: `results_brain_ab_deployed_T6ctx`
  (`run_guard_ab.py --baseline`).
- [x] **PROMOTED by the user 2026-09-26 evening** ("turn on the sticky fix and make
  tactical official then more ladder"): `results_deployed/champion_mc_T6tac.zip`
  (sha 05e82218) + `sticky_guard_corrections: true`; replay_tag T6tac; T6ctx is now
  an immutable prior champion.
- [x] Ladder read of T6tac (15 games, the user's word): 7-8, 1140 -> ~1105; guards
  corrected its own pick 35% (T6ctx 45%); sticky fired 0 times (all closed-sheet
  games).
- [x] Guard fix: `resisted_target` no longer retargets Fake Out (the flinch is the
  point; ladder 2026-09-26 T6tac game 8, and two earlier reviews) -- 2026-09-27,
  with the playbook wiring.
- [x] Rain/sand curriculum cycle (`training/run_t6_weather_trial.py`) from
  `tactical_e4` (= T6tac): mirrors 35.45% / 35.3% (lose), battery -0.58 [-1.97,
  +0.81]; the pre-registered weather-router reading on the 14 rain/sand rosters
  -0.92 [-3.03, +1.08] -> **failed, no router**; T6tac stays alone.
- [x] Our own playbook (the user's handoff idea, 2026-09-27): `PLAYBOOK_T6.md` /
  `data/playbook_t6.json` (draft 1.1), planner `vgc_bench/src/playbook.py` at team
  preview (the card, its reasons, the cards passed over), turn-1 script guard
  `playbook_opening`, wired into PolicyPlayer, the harness and every launcher
  (`--playbook`, DEPLOYED.json `playbook`, `tools/ladder_trial.sh`).
- [x] Playbook evaluation (pre-registered 2026-09-27): mirror **30.6% [28.6, 32.6],
  loses** (Support Room vs our own team 13.6%, Water Room 47.5%); battery **-8.03
  [-11.04, -4.96]**, every population negative; per card Support Room -14.6, Sun
  Room -5.2, Water Room -2.7; the turn-1 script alone about neutral (-1.1 where the
  plan matched T6tac's own). The loss is the unpractised plan choice.
- [x] 15-game ladder TRIAL (the user's word), Water Room only (per-card rule):
  **11-4**, 10-4 without the one free win (09-28: the other forfeits came from
  behind and count; T6tac alone 7-8); Indeedee /
  Psychic Terrain leads 4-0; one loss from a hard-guard bug (fixed:
  `severe_attack_drop_switch` vs Fake Out). A read, not proof; promotion is the
  user's call.
- [x] Practise the playbook (2026-09-27): mirror **54.4% (wins)**, battery
  **-4.38 [-7.38, -1.42]** (not deploy-eligible -> no ladder); practice effect
  +3.64 [+1.93, +5.48]; per card Water Room -0.3, Sun Room -4.1, Support Room -8.6.
- [x] The practised brain + Water Room + turn-1 script: battery -0.06 [-2.68, +2.63]
  (holds up) -> ladder (the user's word) **6-9** (5-9 without a free win). Without
  the script:
  battery -1.81 [-3.93, +0.66], heuristic -6.0, mirror 44.5% -> no ladder; the script
  is worth ~+1.75pp to the practised brain.
- [x] The doomed-Pokemon lesson (the user chose it, 2026-09-27): tactical fine-tune 2
  (`results_tactical2/sft/tactical_e4.zip`) learned it (wasted-move mass 0.77 ->
  0.53) -> mirror 48.25% inconclusive (open 55.8 / hidden 40.7), battery +0.35
  [-0.64, +1.37] (deploy-eligible, not better; hidden sheets +1.06). Ladder /
  promotion: the user's call.
- [x] The user's ladder review (2026-09-28) -> three opt-in guards `drop_free_finish`,
  `fake_out_partner_acts`, `switch_the_crippled`: battery +0.26 [-0.53, +1.03]
  (deploy-eligible), ladder T6tac + guards 23-17 over 40, 20-17 without free wins
  (they fired 3 times).
  DEPLOYED 2026-10-01 at the user's word ("add the guards to the bot"): 11 guards,
  replay tag T6tac_guards11.
- [x] Tactical fine-tunes 3 / 4 from T6tac: mirrors 41.2 / 47.4% (lose), batteries
  +0.34 / +0.61 -> a second tactical round on the brain's own games does not beat it.
- [x] Draft 2 of the playbook -- dropped 2026-09-28: re-checked on the 83 later
  ladder games, its Psychic Terrain premise did not replicate (Indeedee leads 5-11
  -> 9-8) and Support Room lost -14.6 locally. Sand is the hole that holds
  (Tyranitar leads 1-12) -> the T6m route in `TEAM_REVIEW_T6.md` (the user's call).
- [ ] Blastoise (2026-09-28, `results_analysis/blastoise_set_20260928/`): keep Fake
  Out (54 of 64 flinched) and Ice Beam (our Garchomp / Salamence KOs); Aura Sphere over
  Water Pulse scores best in our games (KOs +40 / -20: Kingambit, Archaludon,
  Tyranitar) -> a third team (T6m + Aura Sphere) in T6m's clone tournament (the
  user's call).
- [x] T6 set-variant clone tournament (pre-registered 2026-09-28 11:02, run
  22:36-23:59): sand pool T6 24.8%, T6m 36.0% (+11.2pp [+7.2, +15.2]), T6mAS
  30.2%; full pool even (T6 46.7, T6m 48.3, T6mAS 47.7) -> T6m qualifies.
- [x] T6m practice cycle (the user's word 2026-10-01, pre-registered 23:50): battery
  +0.34pp [-0.98, +1.69] (deploy-eligible; rain rosters +4.2pp, the 3 sand rosters
  -1.0pp), head-to-head vs T6tac on T6 LOST 43.8% [41.6, 46.0] -> gate HOLD, no
  ladder. Diagnostics: T6tac itself on T6m 41.0%, the candidate on T6 41.0% -> each
  brain is best on its practised team and T6 still wins the close games.
  Candidate `results_tactical_t6m1/sft/tactical_e4.zip` kept.
- [x] T6m ladder trial at the user's word (2026-10-02): 6-9 over 15 (7-9 over 16, the
  restart added one game); Charizard never brought. No case for T6m over T6.
- [x] threat_first (2026-10-03, the user's word): mirror 49.7%, battery -0.10pp
  [-0.68, +0.48] -> ladder trial 18-22 (16-22 w/o free wins); fired once in 332
  ladder decisions (correctly). Correct but rare; promotion is the user's call.
- [x] Round 2: threat_first2 PASSES (mirror 49.9%, battery +0.44pp [-0.39, +1.26],
  +1.49pp where it fired) -> ladder 22-18 over 40 (3 changes in 345 decisions);
  doomed_switch FAILS (battery -2.97pp [-4.37, -1.74]) -> dropped.
- [x] threat_first2 added to `guards_extra` at the user's word (2026-10-03; 12 guards,
  replay tag T6tac_guards12).
- [x] Turn-1 play (the early-loss lever): the Water Room turn-1 script alone on our
  usual preview (`playbook_script_only`, 2026-10-03): battery pooled -0.26pp [-1.43,
  +0.97], human_new -3.58 -> HOLD by the pre-registered floor, no ladder A/B; changed
  turn 1 in 53% of games, neutral where it acted (-0.11). Not recommended; a turn-1
  plan needs practice, not a guard (the "real handoff" item below).
- [x] **T6tac practice cycle** (the user, 2026-10-03: "do that next training"): RL save
  35.2%, tactical re-fit 37.2% head-to-head vs the deployed bot; battery +0.03pp -> HOLD.
  Practice after the tactical fine-tune washes its lessons out; next cycle should keep them
  inside training (auxiliary loss toward the tactical teacher), the user's call.
- [x] The user's 2026-10-03 game review: three guards (together HOLD -- the broad HP-move guard
  hurt), the rare trio PASS (wasted_fake_out + throat_chop_main_threat recommended, the HP-move
  guards off); open-sheet preview + late re-plan PASS (non-regression; changes ~4% of
  open-sheet plans); T6e HOLD (head-to-head 43.4%, battery +0.50pp). Deployments: the user's call.
- [x] **Deployed at the user's word** (2026-10-03 night: "keep the guards, keep the team sheet
  stuff its gonna be useful later, and keep earth power"): team T6e (Torkoal Earth Power), 14
  guards (+ wasted_fake_out, throat_chop_main_threat), `sheet_preview: true`; replay tag
  T6tac_T6e_guards14. T6e went in over its HOLD; each piece was measured alone against the
  12-guard bot, the combination is unmeasured.
- [x] Challenge listener reconnects after a lost server connection (exit 75 +
  `tools/reconnect_loop.sh`, rejoins an unfinished battle; live `checks/disconnect_live.py`);
  it had sat deaf from 17:52 on 2026-10-03.
- [ ] Ladder read of T6tac_T6e_guards14 (the user's sessions): watch the hidden-sheet games,
  where the unpractised Earth Power lost the head-to-head (36-39%).
- [ ] T6e practice inside a cycle that keeps the tactical lessons (auxiliary loss toward the
  tactical teacher), so the brain practises Earth Power instead of only playing it.
  2026-10-04 (the user: "start the earth power practice training ... just use ep when its super
  effective ... and it does better damage than the rest of the moves and theres no better switch
  in"): first the Earth Power lesson -- 4,000 practice games on T6e + the focus fine-tune
  (only Earth Power positions taught, the rest anchored), pre-registered GO / NEUTRAL / HOLD
  (`training/t6e_ep_chain.sh`, `training/t6e_ep_gate.py`). **GO** (04:05): head-to-head 52.8%
  [50.6, 54.9] vs the deployed bot, battery -0.66pp [-1.48, +0.15]; Earth Power's share when
  it is the best attack 31% -> 49%. Ladder read (the user's word): **6-9** over 15. Earth Power
  was clicked 3 of the 4 times it was the best attack and never when worse. The losses were the
  old ones: losing a Pokemon first (8 of 9) and the Venusaur + Torkoal lead (0-3).
- [ ] Lead selection: the learned preview's Venusaur + Torkoal lead went 0-3 here.
- [x] KL-anchored practice on T6e from the Earth Power candidate (`training/t6e_anchor_chain.sh`,
  pre-registered 03:40): **HOLD**. Drift 0.098 / 0.068 vs 0.52 / 0.37 unanchored; head-to-head
  51.7% [49.6, 53.9] (open 55.7 / hidden 47.8) vs 35.2% for plain practice; battery -0.34pp.
  The anchor keeps the lessons, but the practice adds nothing over its start.
- [x] **T6ep PROMOTED by the user 2026-10-04** ("make the earth power brain official and start
  the matrix search"): `results_deployed/champion_mc_T6ep.zip` (sha f92248c6), replay tag
  T6ep_guards14; T6tac is now an immutable prior champion.
- [ ] One-turn matrix-game search with mixed strategies on the exact bridge (`simulate_batch`), the
  shared ingredient of the two bots that topped Reg M-C (`RESEARCH_TOP_BOTS.md`).
  - [x] Built (`PlannerConfig(solution="nash")`, critic leaf). V1 38.8%, V2 39.1%, V3 34.9%
    vs the deployed bot over 800 games each: **loses** -- but those runs did not measure it
    (next line).
  - [x] 2026-10-04: the search stack was playing a different game (twelve defects: lost
    guards, a rebuilt view that never matched the live one, hidden opponents without items,
    un-evolved Megas, a hard-coded seat, noise-sized overrides). Fixed; a null search now
    plays the deployed bot's pair in 140 of 140 decisions. Details in PROJECT_STATUS.
  - [x] V4 on the fixed stack (anchored game, the bot's own pick as the default): anchor 0.2
    **50.7% [46.7, 54.7]** over 600 games, anchor 0.07 **54.5% [49.6, 59.3]** over 400. It no
    longer loses; no detectable gain.
  - [x] 2026-10-04 21:30: the search ran its networks on mps, where one-position calls are
    4-15x slower than on cpu (85% of a searched decision was network overhead; one in five
    ran out of time), and 41% of its decisions compared candidates on a single random
    stream. `--device cpu`, `--a-search-streams`. Every earlier search number measured that
    throttled configuration.
  - [x] V5 (four cpu arms, launched 21:31) ended after one round as a pilot: the search's
    copies of the battle read a zeroed threat block in 30% of lookups (the threat cache's key
    had no object in it). Fixed; the encoder is a third faster with the same bytes
    (`unit_tests/test_observation_identity.py`).
  - [ ] **V6 (pre-registered 22:08, `../vgc-bench-v6` at 19445d0e):** the same 2 x 2 (leaf raw
    / calibrated x streams one / at least 4), anchor 0.07, six shards. **Stopped by the user
    at 23:39 after 800 games** (an interim read, not the clock stop): 56.0 / 57.0 / 55.5 /
    61.5%, together 57.5% [54.0, 60.9]; no arm separated from another. Resumable. Gate for a
    ladder trial (with `--device cpu`): pooled lower bound > 50% at the planned size, the
    rosters below, then the user's word.
  - [ ] **Against other teams** (`evaluation/search_roster_ab.py`,
    `tools/search_roster_arms.sh`): the deployed bot with and without the search against the
    battery's six opponents on its 47 held-out rosters, roster-paired. The mirror is one
    matchup; this is the gate between a mirror win and a ladder trial. **Pre-registered
    10-05 01:00 at the user's word; starts when V6 ends** (6,204 games an arm, 7-9 hours).
  - [ ] Ladder trial at the user's word: `TRIAL_SEARCH="..."` in `tools/ladder_trial.sh`
    (flags for the measured configuration are in its header; `--device cpu`).
  - [ ] Next search changes (V7, after V6 is read): keep the bot's own replacement at
    forced switches (the search's values there are noise, sd 0.21 between equivalent
    placements); an override must clear its own standard error; cheaper KO cells (17% of
    the streams arms' decisions still reach the budget); the opposing gender from preview.
  - [ ] Oracle arm (real opponent sets in the worlds) once there is an effect to explain.
- [ ] Opposing Megas keep their pre-Mega stats (2026-10-04; every forme change, open and
  hidden sheets): a stale opposing forme is on the field in 71% of our ladder games and 35% of
  turns; the stored line understates what it does to us by a quarter (real / predicted 1.27;
  1.01 with the right line, 1.01 for ordinary foes). Details and the audit in PROJECT_STATUS.
  - [x] Reproduced (`unit_tests/test_forme_stats.py`) and measured before any change
    (`evaluation/forme_stats_audit.py`, `results_analysis/forme_stats_20261004/`): with the
    right numbers the 14-guard stack changes 36 of 2,641 logged picks (1.4%, 0.09 per game).
  - [x] Opt-in guard-profile entry `forme_stats` (guards only; the stored line is back after
    the stack, so the observation is as trained; `vgc_knowledge.py` untouched). Default off.
  - [ ] Its gate (pre-registered 21:00, amended 21:20; the opt-in guard rule): mirror not
    lost (upper bound >= 50%) AND the held-out battery deploy-eligible (pooled upper >= 0, no
    population below -3pp, errors <= 1%): `evaluation/guard_ladder_gate.py go
    results_guard_ab_forme_stats results_mirror_forme_stats2 forme_stats`. Both run from
    `tools/forme_stats_chain.sh` (about 85 minutes; NOT running: its scheduled start was
    cancelled on 10-04 23:42 after the user stopped the search run, it waits for the user's
    word); the verdict lands in `results_analysis/forme_stats_20261004/chain.log`. At one changed pick
    in eleven ladder games neither can show a gain -- only that it is not broken.
    - The first mirror (10-04 evening) showed the entry did nothing with open sheets (0
      changed picks in 1,000 games: an open-sheet line carries the sheet's nature and was not
      recognised). Fixed the same evening. That mirror: 49.6% [47.4, 51.7] over 2,000, not
      lost; its hidden-sheet half, which stands, 48.9% [45.8, 52.0] with 0.29 changed picks
      per game.
  - [ ] The user's decision after the gate: `guards_extra += forme_stats` (a ladder trial first
    through `tools/ladder_trial.sh`).
  - [ ] The larger half, the brain's own inputs: the knowledge / threat blocks would move in
    98% of exposed decisions. Correcting them needs a fine-tune on the corrected observation and
    an ablation on the deployed brain first (the 09-23 rule). The user's call.
- [ ] Gates versus the ladder bot's last stage (found 2026-10-04): the held-out battery and
  every chained mirror run without the opponent / tempo reranker; `eval_counterfactual.py` (gate
  battery, rollout gate, team tournament) runs both rerankers and cannot take sticky
  corrections; `exhibition_mode.sh` drops sticky for a non-deployed checkpoint or team. The
  user's call whether the gates should play the deployed last stage.
- [ ] T6m, the user's call: another practice round (+983,040 steps; beat the
  unpractised 41.0% baseline and reach 50%) or drop T6m.
- [ ] The real handoff: plan features at the token tail (card, roles, targets) and a
  training cycle that practises each card's opening, so the brain plays the plan
  instead of a guard forcing turn 1.
- [ ] Then, if it holds: RL practice for the reading mistakes (doomed mon left in,
  predictable Protects, Trick Room vs Trick Room).
- [x] Guards that stick on ladder (`sticky_guard_corrections`, opt-in): **mirror with
  the rerankers on 53.4% [51.2, 55.5], wins close games** (open sheets 57.7%, where
  it fires ~1/game); deploy-eligible. Ladder use: the user's call.
- [ ] Sun lead vs rain (training target).
- [ ] Next training target (after the ladder read): the two openings still below
  the old Trick Room line on their matchups, rain first.
- [ ] Opening research (`OPENINGS_RESEARCH_T6.md`, 2026-09-27): playbook draft 2
  proposals -- Support Room vs Psychic Terrain with a Psychic attacker, turn-2 Rain
  Dance vs sand, two new openings to practise (Charizard + Farigiraf, Charizard +
  Incineroar), turn-1 facts for the tactical teacher. (Its Sun Room vs Tailwind and
  Fake Out reachability flags are already in draft 1.1, `c7b76b2`.)
- [ ] Team / moveset review (`TEAM_REVIEW_T6.md`, 2026-09-27 evening): clone team
  tournament with T6, T6m (Torkoal Earth Power over Heat Wave, Charizard Weather
  Ball over Solar Beam -- the two tweaks our own ladder logs support, 09-28 audit;
  Incineroar keeps Passho, Venusaur keeps Leaf Storm), T4 and T0-T5 -- T6 has never
  been in one; then, only if T6m wins there, a practice cycle on T6m (the user's word).
- [x] Deployment plumbing for a learned preview (in use since 2026-09-24):
  `DEPLOYED.json` fields `learned_preview: true`, `preview_model`,
  `preview_model_sha256`, optional `replay_tag` (fresh replay dirs); all launchers
  read the manifest through `tools/deployed_config.py` (sha-verified). One model
  serves our preview and the opponent-plan belief: on the 50 T6 ladder opponents
  the focus model predicts their plans as well as today's top-500 model.

## Current priority — T6 diagnosis-first repair (September 20)

- [x] User-authorized follow-through heartbeat `continue-vgc-validation`
  (September 21): check every 30 minutes, review confirmation, perform one bounded
  local follow-on cycle, then pause/report. No ladder or promotion.
  Keep the same Codex model; if usage runs out, wait for the normal reset.
  No alternate-model fallback/delegation, reset credits, or credit purchases.

Supersedes the long spec2/T6 runs below. User paused Claude and authorized this
bounded experiment. T4 remains deployed; no ladder or automatic promotion.

- [x] Add a separate hash-pinned deployed Reg M-C direct-challenge listener.
  Challenges remain serial and unranked; rated ladder launching is unchanged.
- [x] Add one-room reconnect/recovery for an interrupted Reg M-C challenge;
  Showdown room history rebuilds state before the bot continues choosing.

- [x] Confirm the previous run is stopped; preserve all champions/checkpoints.
- [x] Replace M-B-only hidden-set evidence with format-aware current M-C training
  evidence across observation/guard/preview/search consumers, with provenance.
- [x] Append explicit unknown/inferred/revealed threat evidence; preserve old
  checkpoint projections with zero extension and regression tests.
- [x] Add fixed human-clone population share (20%) instead of dilution by saves.
- [x] Run a controlled opening pilot: 240 local games, six opposing rosters,
  both sheet modes and five opening arms. T6 preview is not collapsed. Tiny
  per-cell counts are diagnostic only; no opening override is deployed.
- [x] Build/verify an isolated short-run league, round-held-out rosters, and
  automatic failure/timeout stops. 421 tests pass; 29 baseline Pyright issues
  remain, reproduced unchanged at HEAD.
- [x] Finish the bounded +983,040-step T6 trial (`results_brainv1_t6_repair1`).
  Both saves completed within the time ceiling; no forced opening labels.
- [x] Compare init and both saves against the quarantined human clone on the
  same reserved rosters, hidden/open sheets. Independent battle RNG; not paired
  outcome differences. These rosters were not withheld from generalist history.
  Results: 63.3% init, 75.0% earlier save, 65.7% final save (300 games each).
- [x] Broader confirmation COMPLETE (`results_t6_confirmation_v2`): earlier save
  vs init, 47 additional roster-disjoint teams, 1,034 games/model/population,
  two human clones + three PPO opponents + scripted opponent, hidden/open.
  No search/reranker additions; same guard stack on both models. Saves opponent
  previews, replay/decision samples, cluster-aware scorecard and tactical flags.
  Candidate 84.6% vs generalist-on-T6 56.6%; all populations/modes improve.
  Integrity verified; see `results_t6_confirmation_v2/REVIEW.md` for limitations.
- [x] Review preview collapse and representative losses: identical leads/four
  across 6,204 candidate games. Turn-one setter loss is a concrete vulnerability,
  not proof of a superior alternative preview. Context flags checked against
  replay and decision logs; no blanket weather/HP rule warranted.
- [x] ONE bounded follow-on COMPLETE: `results_t6_vs_deployed_v1`, same six
  populations/47 rosters, candidate T6 versus deployed T4 configuration, 12,408
  games. Opponents choose preview naturally for each own team; independent RNG.
  Smoke complete, 435 tests pass. No policy changes, no additional training.
  T4 86.2% vs T6 83.8%; T6 improvement not established, frozen opponent -8.3 pp.
- [x] Review follow-on results/audits; pause `continue-vgc-validation` and
  report recommendation. No further automatic experiments. Keep T4 deployed.
  Review: `results_t6_vs_deployed_v1/REVIEW.md`. Next proposed work is controlled
  opening-survival comparisons on training-side rosters, not more blind PPO.
- [ ] Confirm matchup weaknesses on more rosters before any targeted curriculum
  or learned preview selector. Keep complete games and hard earlier examples.
- [ ] Only advance a useful candidate to the full multi-population gate; no
  inference of ladder strength from this diagnostic and no automatic deployment.

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

## Reg M-C program (September 10) — active track

Decision 2026-09-09: the bot is a Reg M-C bot from here on. Design and the
evidence behind it: `NEW_BRAIN_PLAN.md` (loss profile of 159 ladder games,
the six things the current brain cannot represent, ranked new-brain ideas,
the team candidates and how the team is chosen). Standing rules unchanged:
team-agnostic components, powered gates, learned artifacts validated on a
population they were not fit on, ladder only with the user's word.

- [x] **M0 — first M-C ladder read** launched 2026-09-09 23:43 (deployed
  brain + three targeting guards, `--reg mc`, MB430, M-B priors as-is;
  `ladder_replays_mc_guards3_20260909/`). Tally in PROJECT_STATUS when done.
- [x] **M1 — M-C data layer** (done 2026-09-13: dated artifacts `*_20260913`, M-C pool 3,609 teams, clones mc_A epoch 3 / eval_mcB epoch 2) (`after_ladder_mc.sh`, automatic after M0):
  re-scrape both M-C formats → `battle_logs_top_mc_merged/`
  (`datagen/merge_battle_logs.py`); rebuild `teams/reg_mc/` and
  `data/team_weights_regmc.json`; M-C opponent priors
  `data/opponent_{preview,move,switch}_top500_regmc.pt` (Elo floors per
  format, `utils.prior_path` resolves by `--reg`); trajectories A (buckets
  0-4) / B (5-9); clones `results_bc/mc_A` (training_opponent) and
  `results_bc/eval_mcB` (eval_only, banned by content); battery smoke on M-C
  arms (`run_gate_battery.py --reg/--our-team`).
- [x] **M2 — round 6, the M-C baseline** (2026-09-15: both finalists FAIL on the scripted arm only; final save heuristic −7.9, human clones +5.0/+5.3, PPO arms +2.9..+8.4, weighted +3.18): league fine-tune of the deployed
  weights with M-C data only (`training/league6_config.json`,
  `run_league6_training.sh`, verdict `run_league6_verdict_supervised.sh` on
  M-C-anchored arms + mc_A / eval_D diagnostics). Answers "does the plateau
  lift when the data changes?".
- [ ] **M3 — brain v1** (NEW_BRAIN_PLAN §3): joint-action head, threat-
  symmetric knowledge, memory tokens, dense signal (shaping + auxiliary
  heads + critic warm start), behind flags with checkpoint conversion;
  trained team-agnostic on the candidate pool.
  - [x] Code on branch `brain-v1` (2026-09-10): joint head, threat block
    (+8 floats per token), potential-based shaping, converter, live-reload
    tolerance, 30 unit tests, live smoke clean. Merged into main 2026-09-16
    (e3071c8; upgrade-on-load for older checkpoints).
  - [ ] Auxiliary heads, critic warm start, memory tokens (v1.1).
  - [x] Training round from the deployed weights, our side = T0-T5,
    deployed-heavy pool, valid shaping (attempt 2, 2026-09-16/18; 8 saves).
    MB430 screening: save 7 heuristic −6.6 / human −1.4 / PPO +4.3..+7.7
    (weighted +1.33), save 8 heuristic −9.5 / human −4.7 / PPO +2.2..+11.5
    (−0.15) → both FAIL the gate as written; six-team probe 0.67/0.69 vs
    the deployed 0.463.
- [ ] **M4 — team tournament**: clone tournament on T0-T5 (n=300/team), then
  the brain tournament, then specialise on the winner.
  - [x] Clone tournament (2026-09-13): T1 64.3%, T0 62.7%, T2 52.3%.
  - [x] Deployed-brain tournament (2026-09-15): T0 82.3%, T2 75.0%, T1 35.7%
    — an MB430 specialist.
  - [x] Brain-v1 tournament (2026-09-18, n=300, both finalists): every team
    60-87% (save 7 T2 84.0 / T0 83.0 / T4 82.0; save 8 T4 86.7 / T0 85.0 /
    T2 80.0). Caveat: the opponent side is the pilot itself — ranks teams
    within a pilot, does not compare pilots (`tools/compare_team_tournaments.py`).
  - [x] Cross-pilot team grid (`evaluation/run_team_grid.sh`, 2026-09-20):
    save 7 beats the deployed brain by +16..+45pp on every non-MB430 team
    (−5.7 on MB430); pre-registered pick (`tools/pick_team_from_grid.py`,
    confirmation n=1,000) = **T4** (Charizard-Y sun): 84.4% vs the human
    clone, against the deployed brain's 79.7-79.9% on MB430.
  - [x] Specialise on the winner (2026-09-20): `results_brainv1_spec/.../27525120.zip`;
    paired vs the generalist on T4: human +0.1 (84.6%), PPO +3.3..+5.5;
    cross-team vs the deployed brain on MB430: human +4.7, PPO +7.3..+8.2,
    weighted +6.6pp.
- [ ] **M5 — gates and ladder**: screening → promotion → 25 audited games
  with the user's word.
  - [x] Ladder measurement read (user-delegated, 2026-09-20): T4 specialist
    **27-23 over 50**, Elo 1104 → peak 1321, settling ~1250-1300 (deployed
    brain 09-09: 13-12 near 1100); zero mechanical failures.
  - [x] **PROMOTED 2026-09-20 by the user** (`yeah make it the official bot`):
    `results_deployed/champion_mc_T4.zip` + `teams/candidates_mc/T4.txt`,
    recorded in `results_deployed/DEPLOYED.json`; launch with
    `tools/ladder_deployed.sh`. The n=5,000 promotion tier was not run.
  - [ ] Gate re-anchoring (pending the user): future candidates are judged against
    the deployed configuration ON T4; human-clone + PPO arms gate, scripted arm
    advisory; a team change uses the cross-team read.
  - [ ] Round spec2 stopped for the user's T6 team change; long T6 run then
    paused by user. Replaced by the bounded diagnosis-first trial above.

## League fine-tune (August 29) — the climb plan's new lever

Continue PPO from the champion's own weights against a pool that finally
includes human-like play (bc_mix_A at a 3/8 decaying share). Single-variable
discipline: champion flags unchanged except the opponent pool and the league
team-weights file (our_team.txt zeroed — it double-counted the MB430 mirror).
All three frozen PPOs stay out so every battery arm remains a
never-trained-against population. Measurement reform now standing: any
counterfactual candidate within ±1pp of its bar gets a fresh-seed
n=1,500/mode confirmation before accept/reject.
**Extended 2026-09-06 to every gate battery** (`evaluation/scorecard_verdict.py`):
the screening tier's non-regression rule is ADVISORY -- an arm within ±1pp of
the −2pp bar goes to a fresh-seed n=1,500 confirmation of that arm, and the
fresh reading decides; the hard rule lives at the promotion tier (n=5,000),
where a neutral candidate's false-fail rate is ~0.5% instead of ~33% at
n=1,000 (paired SE 1.2-1.6pp per arm).

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
- [x] ~~League 3 (exploiter in the pool)~~ **2026-09-05: exploit closed
  (+28 / +32pp vs the adversary; finalist 2 also +3.8 on the human holdout)
  but both finalists breach the non-regression rule on PPO arms (-3.5..-5.6)
  -- overfit to a 31% adversary share. Not promoted; round 3b (one exploiter
  copy) run 2026-09-06, next item.**
- [x] ~~League 3b (one exploiter copy, ~10%)~~ **2026-09-06: PPO-arm tax
  gone (-0.1 / +2.2 / +3.4), exploit held (+27.4), human holdout +0.9 --
  but the heuristic arm breached (-2.7 screening, CONFIRMED -4.1 on a
  fresh-seed n=1,500). Both finalists fail; not promoted; no ladder.
  Gate finding: at n=1,000 (paired SE 1.2-1.6pp/arm) the "no arm < -2pp"
  rule fails a neutral candidate ~1 time in 3 across five arms (0.5% at
  n=5,000). ADOPTED 2026-09-06 with the user's go: hard non-regression at
  the promotion tier only; screening gets the +/-1pp confirmation clause.**
- [x] ~~Mixed-strategy play (the poker lever)~~ **PARKED 2026-09-06: opening
  mixing +2.6 vs the exploiter (bar +5) and -1..-3pp on every non-adaptive arm
  (rotation 8opp -3.1 breach); first-faint-ours up on every arm. The holes are
  in what the policy values, not in which near-tie it picks.** Original entry:
  the deployed brain never mixes (top pick after guards; preview = two
  argmax calls), the most exploitable shape in a simultaneous-move game.
  `--mixing opening` (top-3, T=1, preview + turns 1-2; guard-demoted pairs
  never sampled). Pre-registered bars: exploit meter (n=1,000 vs the final
  exploiter) >= deployed + 5pp AND screening battery with no confirmed
  breach and weighted >= 0 -> 25-game ladder read. Program agreed with the
  user: make the brain unexploitable (mixing, then a Nash-weighted league
  round with a new human-like opponent), then revisit the team.
- [x] **`resisted_target` guard (2026-09-06): retarget a known-resisted
  single-target attack to the foe it hits harder (facts only; KO exception;
  twin verified by decoding; opt-in, per-player overrides). Screening PASS:
  human +1.9, heuristic +0.8, rotations -0.1/+1.8, frozen -0.5 on a fresh-seed
  n=1,500 confirmation; weighted +0.96 -- measured with the reranker reverting
  most promotions. v3 (promoted twins inherit the corrected pair's prob):
  every arm up (human +1.2, rotations +2.0/+0.1, frozen +1.3, heuristic +1.1;
  weighted +1.15), no confirmation needed. Combined ladder read with all three
  opt-in guards pending the user's word; HARD_GUARDS promotion after it.**
- [x] **`overkill_split` + `dominated_weather_ball_weather` (2026-09-06/07):
  split two attacks off a foe one of them already KOs; Weather Ball vs Heat
  Wave under non-sun weather. Screening PASS after fresh-seed confirmations
  (human +1.9, rotations +0.3/+1.7, heuristic +1.0, frozen +0.4; weighted
  +1.2; pooled ~neutral). Opt-in until a ladder read.**
- [ ] Neutral-to-super-effective target preference: the follow-up candidate if
  the ladder read backs the factual rule (opponents 40% SE hits vs our 19-26%).
- [x] ~~Round 4 (September clones + capped exploiter, hardness-weighted, TR
  boost)~~ **2026-09-07: exploit closed (+22.5 / +23.0) but both finalists
  fail the August-anchored battery (final: heuristic -6.5, every arm
  negative; save 1: human -5.0, rotation tuned -7.9). Fourth adversary
  round with the same shape. Open question for the user: the battery's
  opponents pilot August-weighted teams and its human arm is the August
  clone -- a September-anchored diagnostic (Sep team weights + eval_D) is
  running to tell adaptation from degradation. Not promoted; no ladder.**
- [x] ~~Round 5 (league-1 recipe on September data, no adversary)~~
  **2026-09-09: both finalists FAIL (rotation 8opp -5.5 each; heuristic tax
  gone for save 3 at -0.2; eval_D -1.6 / -2.1: no September-human gain).
  Five rounds since league 1, zero passes -- fine-tuning from the deployed
  weights is plateaued. Deployable now: the three opt-in guards. Decisions:
  ladder read of deployed + guards; the team; a from-scratch September
  league (needs compute).**
- [ ] ~~Round 5 (launched 2026-09-07 23:00): league-1 recipe on September data
  -- human clones 46%, self-lineage, NO adversary, September team weights, TR
  x1.5. Verdict ~2026-09-08 11:30 (August bars + eval_D arms), then a
  September-anchored diagnostic if needed.**
- [ ] **Step 2, Nash-weighted league round 4:** `training/meta_game.py` +
  `meta_game_config.json` (5 rows x 8 columns, n=300) -> column equilibrium
  y* -> copy table for `build_league.py`; second human-BC from a fresh
  top-player scrape (A-bucket only; eval_B untouched). Gate at the promotion
  tier per the adopted rule.
- [ ] Generalization beyond the fixed team begins
