# Interval-Aware REM-to-Wake Localization Protocol v0.1

**Created:** 2026-10-10

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technical uncertainty

The PSG hypnogram assigns one stage to each 30-second epoch. A derived REM-to-Wake boundary is therefore known only within the stored `nominal_boundary_sec +/- 15 s` interval, not at an exact physiological instant. Earlier train-only error analysis showed that widening event matching from +/-15 to +/-45 seconds recovered nine events and increased F1, but tolerance widening cannot determine whether explicitly representing the label interval improves localization.

The preceding full-night formulation experiment also found that natural-prior fitting was preferable to balanced loss under constrained alarm budgets. That result reached F1 0.1543 at the primary budget, but 44/50 new `saga` fits reached the iteration limit. Block 10 therefore requires a numerically stable hard-label comparator before the label-uncertainty mechanism can be interpreted.

## 2. Pre-result support audit

A train-only grid audit used existing reviewed feature onsets without fitting a model. It found:

- 75,539 supported boundaries on the original 30-second phase;
- 75,428 supported boundaries on a new +15-second phase;
- 150,967 unique 15-second-grid boundaries over 629.0292 supported hours;
- 147,055 eligible negative boundaries after the frozen 135-second REM/Wake exclusion; and
- all 180 primary train events supported at nominal -15, nominal, and nominal +15 seconds.

These counts establish feasibility only. They contain no model score or outcome comparison.

## 3. Fixed signal representation

Use wearable `H2` only: `HB_1` and `HB_2`.

Retain the reviewed enriched representation:

- 45 features per 30-second signal window;
- eight windows at relative starts `[-120, -90, -60, -30, 0, +30, +60, +90]` seconds;
- 360 values per candidate;
- the frozen raw-signal filter, 128 Hz resampling, and train-derived robust channel scaling;
- fitting-subset variance removal at or below `1e-12`; and
- fitting-subset greedy absolute-correlation pruning at `0.95`.

The original phase reuses the reviewed feature cache. The +15-second phase must be computed from raw wearable EDF samples with the same feature functions; averaging adjacent feature rows is prohibited because spectral, entropy, coherence, and Hjorth features are nonlinear. A shifted 30-second window is retained only when it spans two contiguous valid PSG-scored epochs. Eight contiguous shifted windows are required for a context.

The resulting context includes signal through candidate time +120 seconds. This is retrospective localization, not prospective or real-time detection.

## 4. Fixed target formulations

Both candidates use the same 147,055 eligible negative boundaries and natural class prior.

| Candidate | Positive rows per event | Positive row weight | Total positive weight per event | Purpose |
|---|---:|---:|---:|---|
| `HARD15` | nominal only | 1 | 1 | Numerically stable hard-label comparator |
| `INT15` | nominal -15, nominal, nominal +15 s | 1/3 | 1 | Uniform interval-aware label distribution |

The interval distribution is uniform because the hypnogram provides no evidence favoring one location inside the 30-second interval. The three rows are not independent events. Dividing their weights by three keeps total event contribution and the natural training prior equal between candidates.

Negative boundaries must satisfy the frozen 240-second context rules and be at least 135 seconds from every REM-to-Wake or Wake-to-REM nominal boundary. Candidates inside the exclusion zone that are not assigned a positive label are omitted from fitting but remain in full-night scoring.

## 5. Numerical-stability control

Both candidates use `SGDClassifier` with:

- `loss="log_loss"`;
- `penalty="elasticnet"`;
- `l1_ratio=0.5`;
- `alpha = 1 / (0.1 * fitting_sample_weight_sum)` to retain the previous `C=0.1` regularization scale under mean-loss optimization;
- `learning_rate="optimal"`;
- `average=True`;
- `shuffle=True`;
- `tol=1e-4`;
- `n_iter_no_change=10`;
- `max_iter=2000`; and
- `random_state=20261010`.

Pruning and standard scaling are fitted only on fitting participants. Sample weights are passed to the loss after pruning and scaling. The two candidates have identical fitting-weight sums in every fold.

The numerical-stability control passes only if all 50 fits stop before 2,000 iterations, produce finite probabilities and coefficients, and complete all required score rows. If any fit reaches the limit or produces nonfinite output, the interval hypothesis cannot advance in v0.1; preserve the failure without changing solver settings.

## 6. Participant-grouped nested execution

Reuse the exact five frozen train `pid` folds. For each outer fold:

1. hold out the outer-fold participants;
2. rotate each remaining fold once as the inner-held-out fold;
3. fit `HARD15` and `INT15` on the other three folds;
4. score every supported 15-second-grid boundary in the inner-held-out recordings;
5. select one threshold per candidate from concatenated inner out-of-fold scores;
6. refit each candidate on all four outer-training folds; and
7. score the outer-held-out recordings once.

Outer labels cannot influence pruning, scaling, fitting, thresholds, model selection, or solver settings.

## 7. Alarm selection and event evaluation

Reuse the full-night threshold grid from the formulation experiment, including the explicit no-alarm sentinel. The primary inner alarm budget is 0.25 false alarms per supported hour. Also report the predeclared 0.10, 0.50, and 1.00/hour budgets.

Contiguous above-threshold 15-second candidates form one alarm. The highest probability represents the run; exact ties use the earliest candidate.

Use the unchanged reference events and one-to-one matcher. Primary outcomes use primary-quality events and nominal +/-15-second matching. Report:

- precision, recall, F1, and false alarms/hour;
- true-positive, false-positive, false-negative, and ignored-prediction counts;
- median and maximum absolute timing error;
- matched predictions at -15, 0, and +15 seconds relative to the nominal boundary;
- +/-45-second and expanded-quality sensitivity results; and
- participant-cluster bootstrap intervals with 2,000 resamples and seed `20261010`.

## 8. Frozen hypothesis and decision gate

### H-INT: interval-aware localization

`INT15` passes the v0.1 point gate only if, at the primary 0.25/hour operating condition:

1. F1 exceeds `HARD15` by at least `0.03`; and
2. false alarms/hour increase by no more than `0.05`.

Report recall and timing-error differences as secondary mechanisms, not additional pass conditions. The direction is participant-supported only if the F1-difference interval is entirely above zero and the false-alarm-difference interval upper bound is at most `0.05`. Full material-size support requires the F1 lower bound to reach `+0.03`.

The point decision is valid only if the numerical-stability control passes.

## 9. Required controls

1. Confirm exactly 82 train recordings and 64 train `pid` groups.
2. Confirm all 180 primary events have three supported interval candidates.
3. Confirm identical negative rows and total fitting weights for `HARD15` and `INT15` in every fit.
4. Confirm no validation or current-test input is read.
5. Verify original-phase enriched features against the reviewed cache and validate finite shifted features and feature bounds.
6. Verify participant disjointness in every fit.
7. Record fit iterations, coefficients, retained features, thresholds, support, and convergence without changing settings.
8. Hash shifted features, models, and score artifacts outside Git.
9. Retain only compact reviewed tables and hashes in Git.
10. Run an independent read-only validator twice.

## 10. Stop rule

Do not revise the stride, interval points, row weights, negative exclusions, features, solver, regularization, alarm budgets, thresholds, or gate after outer results are available.

If the convergence control fails, record an inconclusive numerical result and stop v0.1. If convergence passes but H-INT fails, stop this uniform interval-label mechanism and close Block 10 without trying alternate smoothing widths on the same outer results. A pass retains the mechanism only for a genuinely new locked or scientifically matched cohort; it does not authorize validation/test reopening or a reliability claim.

## 11. Method references

1. Bottou L. Large-scale machine learning with stochastic gradient descent. In: *Proceedings of COMPSTAT 2010*. 2010:177-186. https://doi.org/10.1007/978-3-7908-2604-3_16
2. Bach F. Adaptivity of averaged stochastic gradient descent to local strong convexity for logistic regression. *Journal of Machine Learning Research*. 2014;15:595-627. https://www.jmlr.org/papers/v15/bach14a.html
3. Geng X. Label distribution learning. *IEEE Transactions on Knowledge and Data Engineering*. 2016;28(7):1734-1748. https://doi.org/10.1109/TKDE.2016.2545658
4. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
