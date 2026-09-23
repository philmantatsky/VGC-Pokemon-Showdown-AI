# Deployed T4 versus candidate T6 — final local review, September 21

## Decision

Keep T4 deployed. T6 has not demonstrated an improvement over the deployed
configuration. The authorized follow-on cycle is complete; pause automatic work.
No promotion, ladder session, or additional training was started.

| Opponent | Deployed T4 | Candidate T6 | T6 minus T4 | Roster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| New held-out human clone | 82.0% | 84.3% | +2.3 pp | -6.9 to +10.9 |
| Frozen PPO | 91.4% | 83.1% | -8.3 pp | -15.1 to -2.4 |
| Rotation 1 | 82.3% | 84.0% | +1.7 pp | -7.4 to +10.8 |
| Rotation 2 | 87.4% | 84.8% | -2.6 pp | -10.3 to +4.4 |
| Previous held-out human clone | 84.2% | 78.1% | -6.1 pp | -14.8 to +2.1 |
| Scripted heuristic | 89.7% | 88.6% | -1.1 pp | -4.6 to +2.5 |

1,034 games/configuration/population; 6,204 per configuration, 12,408 total.
Pooled descriptive scores: T4 5,346/6,204 (86.2%); T6 5,201/6,204 (83.8%).
Equal-population pooled delta -2.34 pp; bootstrapping the 47 whole rosters across
all populations together gives 95% interval [-5.75, +0.93] pp (10,000 resamples,
seed 20923). Thus the pooled result does not establish T6 is strictly worse,
but supplies no evidence of the required improvement. Frozen-PPO regression is
the clearest individual signal; intervals are not multiple-testing-adjusted.

Hidden: T4 85.91%, T6 84.66%, difference -1.26 pp, interval [-5.06,+2.45].
Open: T4 86.43%, T6 83.01%, difference -3.42 pp, interval [-8.25,+1.19].
Exploratory category rates suggest TR rosters are the largest gap (T4 85.53%,
T6 77.65%), but this is a post-hoc subgroup, not a tuned curriculum prescription.

This measures different weights AND different teams. Opponents selected preview
naturally against each actual team. Battle RNG and opponent preview choices are
not paired. Opponent rosters/modes/repeats are matched. These rosters were excluded
from the T6 fine-tune/initial checkpoint selection, not necessarily historical
generalist or T4 training. Do not call this unseen-team generalization or predict
ladder performance from the absolute local win rates.

## Integrity and termination

All source/data/model hashes in manifest.json match disk, including the deployed
weights, team and deployment manifest. All 12 arms pass validate_arm: 94 complete
cells each, 11 games/cell, 47 rosters, 517 games per sheet mode, unique battle IDs,
correct own-team membership, complete telemetry, zero recorded guard errors or
fallback replacements. Preview-pairing mismatch counters are zero but are not a
pairing claim: cross-team preview pairing was intentionally disabled. All six
populations are complete; status is complete_review_required; no Python evaluation
or training process remains. Local Showdown servers were left untouched.

## Representative replay and decision-log review

Candidate still selected the same lead and four in all 6,204 games. No matchup-
aware preview learning has been demonstrated. Audits are the first roster in each
category/mode, not random loss samples. Each arm has 110 audited games.

- Human candidate losses 3443628 (open) and 3443618 (hidden), MC2198: Torkoal
  Protect + Farigiraf Trick Room, then Wave Crash + a critical Glaive Rush KOs
  Farigiraf before setup. Logs confirm the intended setup choice, not a missing
  command. Despite taking two KOs turn two, the candidate loses speed control.
- Frozen candidate loss 3445453, MC210 open: Garchomp Earthquake + Charizard-Y
  Weather Ball KOs Farigiraf before Trick Room with no crit required. Torkoal's
  Protect + Trick Room joint choice retains 97.8% probability. Turn-two Fake Out
  arrives only after the setter has already fainted; the slow attackers then
  take heavy damage before Eruption/Water Spout can act.
- Frozen candidate loss 3445688, MC2198 hidden: Whimsicott Tailwind + Kingambit
  Kowtow Cleave removes Farigiraf on turn one. Torkoal Protects; Charizard then
  faints before attacking. This repeats the earlier confirmation vulnerability.
- Across candidate audits, setter loss before turn-one Trick Room appears in
  both human losses and 24/25 frozen losses, but ALSO 4 human wins and 22 frozen
  wins. This is a strong diagnostic pattern in these few sampled rosters, not
  proof that a specific alternative lead wins or a general causal loss rate.
- Frozen T4 loss 3444419, MC210 open: Charizard falls to Rock Slide before its
  chosen Heat Wave; Garchomp/Charizard lead is also vulnerable. T4 is not free of
  tactical weaknesses. Raw weather/HP flags are naturally absent from its
  different movesets, so do not treat flag totals as a fair skill comparison.

Earlier reviewed weather/HP flags remain contextual, not proven automatic
mistakes; see ../results_t6_confirmation_v2/REVIEW.md for full-turn analysis.
No blanket weather, Protect, or current-HP guard was added to fit these outcomes.

## Recommended next work (not started)

Keep the improved T6 checkpoint as a research candidate and preserve T4 as the
deployed control. The next useful experiment is an opening-survival study on
TRAINING-side rosters: compare current Torkoal/Farigiraf with supported Trick Room
(Blastoise/Farigiraf or Incineroar/Farigiraf) and a sun attack opening, holding
the turn policy fixed. Measure both full-game wins and whether the setter gets
its move, including opponents that double-target or OHKO Farigiraf. Estimate
counterfactual outcomes over repeats rather than forcing a new rule from one
replay. Only train a matchup-conditioned selector if those alternatives show
repeatable benefits, then validate on fresh reserved rosters/populations.

Do not repeat broad PPO training unchanged, rerun null experiments until they
pass, or automatically use this reviewed holdout as a training curriculum.
This recommendation needs the user's next authorization; the heartbeat is paused.
