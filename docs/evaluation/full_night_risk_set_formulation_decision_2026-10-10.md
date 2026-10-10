# Full-Night Risk-Set Formulation Decision

**Decision date:** 2026-10-10

**Protocol commits:** `72a37eb`, `5d035ca`

**Result code commit:** `a131072`

**Primary result commit:** `afb1d2c`

**Sensitivity code commit:** `9ab54e3`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Stop balanced full-risk-set fitting; retain natural-prior full-risk-set fitting as a developmental formulation for a separately predeclared confirmation, without claiming a reliable detector

## Question

The earlier detector used 2,563 review-sampled negatives that covered only 3.488% of the regenerated eligible negative pool and overrepresented early-night and uncommon stage-pair strata. It was uncertain whether this sample omitted essential full-night diversity and whether balanced class weighting distorted probabilities relative to the natural REM-to-Wake prevalence.

The protocol separated these mechanisms. `SAMP-BAL` reused the exact sampled-negative balanced-loss control. `RISK-BAL` used every eligible full-risk-set negative with balanced loss. `RISK-NAT` used the same full risk set without class weighting.

## Fixed design

The experiment retained wearable `HB_1/HB_2`, the reviewed 360-value enriched context, fold-local correlation pruning, standard scaling, elastic-net logistic regression, `C=0.1`, five frozen participant folds, inner-only threshold selection, one-to-one event matching, and a primary alarm budget of 0.25 false alarms/hour. The 0.10, 0.50, and 1.00/hour budgets were predeclared sensitivity analyses.

The full labelled risk set contained 73,656 rows: 180 primary events and 73,476 eligible negatives. Event predictions covered all 75,539 supported full-night boundaries. Validation and current-test partitions remained closed.

## Primary event result

Primary membership, +/-15-second tolerance, and inner-selected 0.25/hour thresholds produced:

| Candidate | True positive | False positive | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
| SAMP-BAL | 9 | 162 | 0.0526 | 0.0500 | 0.0513 | 0.2574 |
| RISK-BAL | 0 | 0 | undefined | 0.0000 | 0.0000 | 0.0000 |
| RISK-NAT | 29 | 167 | 0.1480 | 0.1611 | 0.1543 | 0.2653 |

`RISK-NAT` improved F1 by `+0.1030` and recall by `+0.1111` relative to `SAMP-BAL` at a similar held-out false-alarm rate. Absolute performance remained low: approximately 85% of emitted alarms were false positives.

## Risk-set result

| Candidate | Average precision | Brier score | Log loss |
|---|---:|---:|---:|
| SAMP-BAL | 0.0242 | 0.061054 | 0.786660 |
| RISK-BAL | 0.0492 | 0.035599 | 0.482461 |
| RISK-NAT | 0.0719 | 0.002708 | 0.012729 |

The natural-prior formulation reduced Brier score by 92.39% relative to `RISK-BAL`. It also increased average precision, although average precision remained only 0.0719 at a prevalence of 0.2444%.

## Hypothesis decisions

`H-RISK` failed. `RISK-BAL - SAMP-BAL` recall was `-0.0500`, below the required `+0.05`. Its participant-bootstrap recall interval was `-0.0958` to `-0.0173`, and its F1 interval was `-0.0886` to `-0.0199`. Balanced full-risk-set fitting stops at v0.1.

`H-PRIOR` passed its frozen point gate. The Brier reduction was 92.39%, exceeding the required 10%, and recall increased by 0.1611 rather than decreasing by more than 0.05. For `RISK-NAT - RISK-BAL`, the participant-bootstrap intervals were:

- recall: `+0.1046` to `+0.2314`;
- F1: `+0.1043` to `+0.2057`;
- false alarms/hour: `+0.1873` to `+0.3552`;
- average precision: `+0.0095` to `+0.0684`; and
- Brier score: `-0.0395` to `-0.0270`.

The favorable discrimination and calibration directions therefore came with a higher alarm burden than the zero-alarm `RISK-BAL` operating point.

## Alarm-budget sensitivity

| Budget (/hour) | SAMP-BAL F1 | RISK-BAL F1 | RISK-NAT F1 | RISK-NAT false alarms/hour |
|---:|---:|---:|---:|---:|
| 0.10 | 0.0000 | 0.0000 | 0.0972 | 0.0874 |
| 0.25 | 0.0513 | 0.0000 | 0.1543 | 0.2653 |
| 0.50 | 0.1535 | 0.0854 | 0.1748 | 0.4607 |
| 1.00 | 0.1822 | 0.1435 | 0.1716 | 0.9325 |

`RISK-NAT` led at the three stricter budgets but not at 1.00/hour. Its best observed F1 was 0.1748 at the 0.50/hour selection budget. The result is therefore a formulation improvement under constrained alarm burdens, not evidence that the detection problem has been solved.

## Convergence and retained failures

Forty-four of 50 new fits reached the fixed 3,000-iteration limit. Twenty-four of 25 `RISK-BAL` fits and 20 of 25 `RISK-NAT` fits were nonconverged; only both outer-final fits for fold 2 converged. No solver, tolerance, iteration limit, or coefficient was changed after results were inspected. The point decision is retained, but coefficient and threshold stability remain unresolved.

The execution record also retains:

1. an initial sequential runtime estimate of approximately nine hours, resolved by parallelizing independent fits without changing settings;
2. a thread-unsafe warning-capture attempt whose outputs were rejected and archived;
3. exact probability saturation at 1.0, resolved by a protocol amendment adding a threshold strictly greater than 1 as the no-alarm sentinel; and
4. a stopped sensitivity run that confused the 73,656 labelled risk rows with 75,539 supported prediction boundaries; no output was written before correction.

## Verification

- The primary validator passed twice over 73,656 risk rows, 75 fit records, and 127 external artifacts.
- The sensitivity validator passed twice over 48 metric rows, 3,072 participant rows, 3,182 predicted events, and 17 source hashes.
- The 0.25/hour sensitivity rows exactly reproduced the validated primary event table.
- All score, model, and risk-table artifacts remain outside Git; compact reviewed outputs and their hashes are retained.
- Validation and current-test data remained closed.

## Interpretation and next decision

The previous review-sampled, balanced-loss formulation was not an adequate proxy for the natural full-night event problem. Increasing negative coverage alone did not help when balanced loss was retained. Natural-prior fitting materially improved calibration and constrained-budget event detection, showing that training-prior distortion was the more important tested mechanism.

This does not establish clinical, real-time, or independent performance. The context remains retrospective through +90 seconds, F1 remains below 0.18 across the predeclared budgets, and most new fits did not converge.

Do not tune negative sampling, class weights, alarm budgets, or thresholds on these outer results. Block 10 should predeclare a numerically stable hard-label comparator and one interval-aware label-uncertainty mechanism while retaining the natural full-night risk set, participant grouping, alarm-burden reporting, and closed validation/test partitions. Any solver stabilization must be defined before outcome inspection and reported separately from the label-uncertainty effect.

## Method references

1. King G, Zeng L. Logistic regression in rare events data. *Political Analysis*. 2001;9(2):137-163. https://doi.org/10.1093/oxfordjournals.pan.a004868
2. Fithian W, Hastie T. Local case-control sampling: efficient subsampling in imbalanced data sets. *The Annals of Statistics*. 2014;42(5):1693-1724. https://doi.org/10.1214/14-AOS1220
3. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
