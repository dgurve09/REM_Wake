# Multi-Dataset PSG Stage-First Transition Protocol v0.1

**Date frozen:** 2026-10-25
**Precondition:** Corrected PSG decision commit `652a883`
**Result status at freeze:** No stage model, stage probability, derived alarm, or stage-first event result exists
**Evidence status:** Exploratory mechanism comparison on previously inspected datasets; not independent confirmation

## 1. Question

Does predicting the five sleep stages and then deriving REM-to-Wake boundaries improve event performance over the corrected direct `ADV3-CHANGE` detector?

The corrected direct detector reached pooled external F1 0.2290 and recall 0.2564. A stage-first comparison is scientifically necessary because sleep staging is an established alternative formulation and because the direct target contains only 482 positives across 196 nights. This experiment tests problem formulation, not a new clinical application.

## 2. Fixed cohort and features

Use the corrected 196-night PSG cohort:

| Dataset | Nights | Participant groups | Corrected primary labels |
|---|---:|---:|---|
| BOAS | 128 | 100 | Published human consensus |
| gx36m | 58 | 20 | Strict 2-of-3 known-stage majority; other epochs unknown |
| ISRUC group 3 | 10 | 10 | Two-expert agreement; disagreement unknown |

Reuse the verified `ADV3-CHANGE` feature caches containing 444 causal values at 15-second decisions. Do not regenerate, select, prune, or alter features after results.

## 3. Causal stage target

At a decision time `t`, assign the stage of the latest 30-second epoch whose signal has begun and whose available portion ends at `t`. At exact epoch boundaries, retain the preceding epoch; at the midpoint, use the current epoch. Algebraically, use `ceil(t / 30) - 1`.

This prevents the target at an exact boundary from naming a stage before any signal from that stage is available. Unknown epochs are excluded from stage fitting and stage metrics. Never bridge an unknown interval or a gap when deriving an event.

## 4. Fixed stage model

Fit one five-class logistic model per held-out dataset:

- `StandardScaler` fitted only on the two training datasets;
- L2 multinomial logistic regression;
- `C=0.1`;
- `solver=lbfgs`;
- balanced stage weights calculated from the training rows;
- `max_iter=1000`, `tol=1e-6`, and seed `20261025`; and
- no dataset indicator, resampling, probability calibration, feature selection, or hyperparameter search.

Report accuracy, balanced accuracy, macro F1, Cohen kappa, and per-stage precision, recall, and F1. These stage metrics are diagnostic; event metrics remain primary.

## 5. Fixed causal event derivations

Apply the model at every supported 15-second decision and compare two predeclared deterministic derivations:

1. `STAGE-HARD`: emit an alarm when two consecutive supported predictions change from `REM` to `W`.
2. `STAGE-CONFIRM2`: emit an alarm at the second consecutive `W` prediction after at least two consecutive `REM` predictions. All four predictions must be 15 seconds apart.

`STAGE-CONFIRM2` is primary because it requires two 15-second-spaced decisions for each state and reports 15 seconds after the nominal scored-event midpoint when the predictions follow the scored boundary exactly. The rule is causal and uses no future decision. Because adjacent feature windows overlap, these decisions must not be described as independent 30-second evidence intervals. `STAGE-HARD` is a diagnostic for stage-prediction chatter.

Use the same primary references, ignored-boundary handling, participant aggregation, supported-hour denominator, and match interval as corrected PSG v0.2. Do not tune an event threshold, refractory interval, tolerance, or minimum run length.

## 6. Evaluation

Retain the same three leave-one-dataset-out roles:

1. hold out BOAS after training on gx36m plus ISRUC;
2. hold out gx36m after training on BOAS plus ISRUC; and
3. hold out ISRUC after training on BOAS plus gx36m.

gx36m and ISRUC remain the primary external datasets. BOAS remains descriptive because it informed earlier development. Retain scorer-specific gx36m and ISRUC sensitivity using safe out-of-vocabulary parsing.

Use 2,000 participant bootstrap resamples stratified by external dataset to compare each stage-first derivation with the frozen corrected `ADV3-CHANGE` participant result. Seed the bootstrap with `20261025`.

## 7. Hypothesis and gate

`H-STAGE-FIRST-XDATA`: a causal stage-first formulation provides more transportable REM-to-Wake event structure than the direct rare-event formulation.

`STAGE-CONFIRM2` passes the exploratory point gate only if all conditions hold:

1. pooled external F1 is at least 0.2590, exceeding corrected direct F1 by at least 0.03;
2. pooled external recall is at least 0.25;
3. pooled external false alarms do not exceed 0.30/hour;
4. neither gx36m nor ISRUC F1 is more than 0.03 below corrected direct `ADV3-CHANGE` on that dataset;
5. all fits converge and all stage probabilities are finite; and
6. scorer sensitivity is reported without selecting a favorable expert.

Bootstrap intervals determine evidence strength. Even if the point gate passes, this reused-dataset experiment cannot provide independent confirmation.

## 8. Stop rules

- Do not revise the class weighting, feature set, causal target alignment, run confirmation, or matching rule after results.
- If `STAGE-CONFIRM2` fails, preserve the formulation failure and do not try a longer confirmation run on these held-out results.
- If it passes, retain it only as a mechanism candidate for a separately frozen paired wearable study.
- Do not describe PSG-only stage or event performance as wearable validation, diagnosis, sleep-paralysis detection, or narcolepsy detection.

## 9. References

1. Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: end-to-end hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2019;27:400-410. https://doi.org/10.1109/TNSRE.2019.2896659
2. Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum PJ, Igel C. U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*. 2021;4:72. https://doi.org/10.1038/s41746-021-00440-5
3. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9(1):81-87. https://doi.org/10.5664/jcsm.2350
4. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
