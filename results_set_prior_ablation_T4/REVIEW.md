# T4 brain: Reg M-C vs Reg M-B opponent set data -- review, September 23

## Result

The T4 brain plays better with the Reg M-B set data it was trained with and
went 27-23 on ladder with. Codex's Reg M-C data costs it **-1.97 pp** pooled
(equal-population mean over six populations; whole-roster bootstrap 95%
**[-3.66, -0.32]**, 47 rosters, 10,000 resamples). Delta = new (M-C) minus old (M-B).

| Opponent | Old M-B data | New M-C data | New minus old | 95% interval | Hidden sheets | Open sheets |
|---|---:|---:|---:|---:|---:|---:|
| New held-out human clone | 86.0% | 82.0% | -4.0 | [-8.1, -0.0] | -8.5 | +0.6 |
| Frozen PPO | 90.7% | 91.4% | +0.7 | [-3.0, +4.4] | +1.2 | +0.2 |
| Rotation 1 | 87.3% | 82.3% | -5.0 | [-10.6, +0.4] | -7.7 | -2.3 |
| Rotation 2 | 91.2% | 87.4% | -3.8 | [-7.4, -0.5] | -6.6 | -1.0 |
| Previous held-out human clone | 84.4% | 84.2% | -0.2 | [-4.4, +3.9] | -1.7 | +1.4 |
| Scripted heuristic | 89.2% | 89.7% | +0.5 | [-1.9, +2.9] | +0.8 | +0.2 |

Equal-population means: old 88.1%, new 86.2%. 1,034 games per arm per
population (47 held-out rosters x 11 repeats x both sheet modes).

## Where the loss is

Hidden-sheet games carry it (-6.6 to -8.5 against three populations); open-sheet
deltas stay within +-2.3. That is where the mechanism acts: with open sheets the
true sets replace the guesses, so the set data barely matters. A brain's weights
learned to read the guesses it trained with; "more accurate" inputs it never
trained on make it worse. Codex's data fix is right for brains trained with it
(T6), wrong for T4. Recorded in `results_deployed/DEPLOYED.json`: T4 needs
`VGC_SET_PRIOR_REG=mb`; T6 (deployed) keeps the Reg M-C data. The launchers apply
the recorded setting, and `ladder_ourteam.py` now records it per replay dir.

## Consequence for the T6-vs-T4 comparison

`results_t6_vs_deployed_v1` ran T4 with the Reg M-C data (its arm manifests:
`set_prior_reg: mc`), i.e. handicapped. Against T4 with the data it laddered
with, on the same rosters, modes and repeats (`t6_vs_t4_old_data.json`):

| Opponent | T4, old data | T6 | T6 minus T4 |
|---|---:|---:|---:|
| New held-out human clone | 86.0% | 84.3% | -1.6 |
| Frozen PPO | 90.7% | 83.1% | -7.6 |
| Rotation 1 | 87.3% | 84.0% | -3.3 |
| Rotation 2 | 91.2% | 84.8% | -6.4 |
| Previous held-out human clone | 84.4% | 78.1% | -6.3 |
| Scripted heuristic | 89.2% | 88.6% | -0.6 |

Pooled **-4.30 pp, 95% [-7.90, -0.89]**: locally T6 is measurably weaker than
T4 as it actually played on ladder (Codex's -2.34 [-5.75, +0.93] understated the
gap). Different teams and weights; opponents pick previews naturally; battle
RNG independent; not a ladder forecast. T6 stays deployed on the user's
decision (2026-09-23); switching back is a `DEPLOYED.json` change.

## Integrity

New-data arm = the T4 arm of `results_t6_vs_deployed_v1`; all 105 files that
study pinned hashed the same before the run and between populations (except
`DEPLOYED.json`, which the arms never read). Old-data arm: 6 x 1,034 games,
`validate_arm` passed (cell counts, no duplicates, correct own roster, no guard
errors, no preview-pairing mismatches); every arm manifest records its set data
(`mb` d99f2622... vs `mc` a4ef12f1...). Launched 00:18, complete 01:00, one MPS
process, up to 8 local battles, Showdown on 7610 (stopped after the run). No
training, promotion or ladder games.
