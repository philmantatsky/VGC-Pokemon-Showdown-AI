# T6 confirmation review — September 21

## Result and integrity

Completed 12,408 local games: 1,034 per model per opponent population, evenly
split open/hidden (517 each), 47 rosters, 11 repeats per roster/mode.
Candidate checkpoint 20,152,320 compared with generalist 19,660,800 **both on
T6**. This is not a comparison with the deployed T4 configuration.

| Opponent | Generalist T6 | Candidate T6 | Gain | Roster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| New held-out human clone | 55.6% | 86.0% | +30.4 pp | +19.8 to +40.2 |
| Frozen PPO | 50.9% | 84.2% | +33.4 pp | +26.0 to +40.9 |
| Rotation 1 | 61.7% | 85.6% | +23.9 pp | +15.8 to +32.3 |
| Rotation 2 | 54.0% | 84.8% | +30.9 pp | +20.6 to +40.3 |
| Previous held-out human clone | 62.5% | 80.0% | +17.5 pp | +10.7 to +24.7 |
| Scripted heuristic | 54.7% | 86.9% | +32.2 pp | +27.1 to +37.2 |

Pooled descriptive rates: 5,248/6,204 (84.6%) versus 3,509/6,204 (56.6%).
Every population/mode improved; all reported roster-bootstrap intervals are
positive. These intervals are individual, not simultaneous multiple-comparison
guarantees. Opponent rosters repeat across populations, so do not calculate a
naive pooled independent-game confidence interval.

Revalidated all 12 result files: 94 complete cells each, 11 games/cell, no duplicate
battle IDs, 94 telemetry cells each, zero recorded guard errors or replaced
fallbacks, zero opponent-preview mismatches. Opponent preview fingerprint/choice
multisets match between models for each roster/mode. Every source/data/checkpoint
hash in the final confirmation manifest matches disk. Resume-related runner and
audit parser revisions remain documented in the original sidecars. Deployment
weights and manifest were not changed.

Opponent previews were paired; battle RNG was independent. The selected rosters
were excluded from this fine-tune and first checkpoint-selection comparison,
not necessarily from the generalist's historical training. No independent sun or
psychic-terrain bucket is represented under the mutually exclusive categorizer.
This is local evidence, not an 85% ladder win-rate prediction.

## Preview and loss review

Across all 6,204 candidate games the lead is Torkoal/Farigiraf and the selected
four are Torkoal/Farigiraf/Blastoise/Charizard. Only bench ordering varies (869
versus 165 games per population). The gain therefore does **not** demonstrate
matchup-aware selection of four or leads. Nor does fixed selection alone prove
it is weaker than a variable preview.

Audits sample the first roster per category/mode, not random games. The resumed
human baseline contains 198 audit files versus 110 per other audited arm because
sampling restarts; do not compare raw flag totals as balanced error rates.

Reviewed the raw replay protocol and chosen/candidate decision logs for all
three sampled human-clone candidate losses, plus frozen-policy losses against
Garchomp/Charizard, Whimsicott/Kingambit, and the Gallade/Camerupt endgame.

- Human losses `3431207`, `3431189`, `3430982`: Torkoal selects Protect and
  Farigiraf selects Trick Room. The setter is knocked out before acting; each
  has a critical hit in the first-turn attack sequence. Subsequent slow attacks
  and low-HP Eruption are symptoms of the failed setup. This does not prove a
  different opening wins, and a critical hit is not automatically a planning bug.
- Frozen loss `3433025`: opposing Garchomp Earthquake + Charizard-Y Weather Ball
  knocks out Farigiraf turn one without a critical hit. The chosen setup has
  97.8% joint-policy probability. The replacement Blastoise uses Fake Out next
  turn, too late to protect the setter. This is a concrete opening vulnerability,
  not evidence of a parser or action-mapping failure.
- Frozen loss `3433260`: Kingambit KOs Farigiraf turn one while Torkoal Protects;
  the bot never gets Trick Room. It remains capable of taking KOs afterwards,
  but loses control of move order.
- These patterns cover all 3 sampled human losses and 25/26 frozen losses:
  Farigiraf fainted turn one before Trick Room. Nineteen frozen sample wins
  also lost the setter that way. This is correlation in a deliberately narrow
  audit sample, not a causal claim or an overall loss-rate estimate.

## Context flags are not blanket rules

- `3433025` turn 2: Torkoal was full HP when Eruption was chosen and fell to
  67/177 before using it. The post-move low-HP flag cannot establish that the bot
  chose Eruption while already at 38% HP. Forecasting incoming damage is the
  relevant weakness; a current-HP-only prohibition would not address this case.
- `3430982` turn 3: Eruption was selected at 91/177 HP, then used at 25/177
  after Dazzling Gleam. Heat Wave was a candidate, but no counterfactual replay
  establishes the winning line. Keep this as a tactical review case.
- `3432959` turn 6: no-weather Weather Ball hits Gallade, whose **revealed Wide
  Guard** justifies the existing spread-move guard standing down. Heat Wave has
  more raw damage, but should not be forced solely from this hindsight outcome.
- `3430914` turn 10: no-weather Weather Ball targets a 2%-HP Pelipper. Water
  Spout KOs it first and Weather Ball redirects into a protected Volcarona.
  This is not the original neutral-full-HP Weather Ball mistake.
- `3430891` turn 4: Water Spout in sun KOs both Hatterene and four-times-water-weak
  Camerupt, winning the battle. A blanket water-in-sun prohibition would be wrong.
- `3433014` turn 5: Water Spout in sun KOs Incineroar and damages Garchomp while
  Charizard Protects. Again, the weather flag alone is not evidence of a mistake.

## Next bounded cycle

No demonstrated mechanics or pipeline defect was established by this review, so
do not patch general rules to fit these few outcomes. Retain the opening
vulnerability as a follow-up priority. First measure whether unchanged candidate
T6 improves on the **deployed T4 configuration**, not merely the original T6
generalist. Use all six populations, the same 47 reserved rosters and both modes,
1,034 games/configuration/population, fresh independent games. Opponents choose
their previews naturally for each different team; forcing T4 previews onto T6
would bias that comparison. Report configuration differences, not weight-only
causation. Stop after this one local cycle, review, then pause the heartbeat.

No new training, ladder games, promotion, credential access, commits or pushes.
