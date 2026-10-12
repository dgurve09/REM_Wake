# Causal Bidirectional REM/Wake Transition Protocol v0.1

**Created:** 2026-10-11

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technical uncertainty

The existing direct detector uses eight 30-second windows centred around a candidate boundary. Four windows occur after that candidate, so its result is retrospective localization and cannot establish real-time feasibility.

It is also uncertain whether REM-to-Wake should be learned alone or whether both REM-to-Wake and Wake-to-REM boundaries share a detectable state-change signature. Adding Wake-to-REM events could improve transition detection by increasing the number of positive boundary examples, but it could also make the target less specific and reduce REM-to-Wake performance.

This experiment therefore separates two questions:

1. can a detector using only signal available at the decision time identify a REM/Wake transition; and
2. if a transition is present, can its direction be distinguished as REM-to-Wake or Wake-to-REM?

This is a new train-only developmental endpoint. It does not reopen Block 10, revise its stopped interval-label result, or provide independent confirmation.

## 2. Causal decision point

The hypnogram-derived nominal boundary has a 30-second uncertainty interval from `nominal - 15 s` to `nominal + 15 s`. The positive decision point is fixed at `nominal + 15 s`, when the complete uncertainty interval has elapsed.

Candidate decisions occur every 15 seconds. Each candidate uses eight contiguous 30-second feature windows ending at the candidate time. Relative to the decision time, window starts are:

`[-240, -210, -180, -150, -120, -90, -60, -30] seconds`.

No sample after the decision time is used. An alarm is counted as a timely detection only from `nominal + 15 s` through `nominal + 30 s`. Earlier alarms are not credited because the annotated uncertainty interval had not fully elapsed.

The earlier uniform `nominal - 15, nominal, nominal + 15 s` target is not reused: its first two points require information that is still in the future under this causal definition.

## 3. Fixed cohort and event membership

Reuse the frozen train partition and five participant-grouped folds.

Primary events must:

- belong to a train recording;
- be labelled `REM_to_Wake` or `Wake_to_REM` in transition labels v0.1;
- pass `primary_analysis_eligible` in quality membership v0.1; and
- have a complete causal context at `nominal + 15 s`.

The pre-result membership expectation is 180 REM-to-Wake and 62 Wake-to-REM primary events before the causal-context check. Expanded-quality events are used only as an ignored-event sensitivity set, not as fitting positives.

Negative candidates use the same train recordings and must be at least 135 seconds from every labelled REM-to-Wake or Wake-to-REM nominal boundary. Near-event candidates that are not assigned a positive label are omitted from fitting but remain in full-night scoring.

## 4. Fixed representation

Use wearable `H2` only: `HB_1` and `HB_2`.

For each 30-second window, retain the five log band-power features per channel (delta, theta, alpha, sigma, beta). Eight causal windows produce 80 input values. The existing reviewed 30-second and +15-second feature caches, filter, 128 Hz resampling, and train-derived channel scaling are reused without recomputing a different representation.

Within each fit, `StandardScaler` is fitted on fitting participants only. No feature selection is performed.

## 5. Stage A: transition detection

Compare two fixed targets using identical negative candidates:

| Candidate | Positive fitting events | Purpose |
|---|---|---|
| `CAUSAL-RW` | REM-to-Wake only | Direction-specific causal comparator |
| `CAUSAL-BI` | REM-to-Wake and Wake-to-REM | Test whether a shared boundary target improves detection |

Both candidates use natural class prevalence and one positive row per event at `nominal + 15 s`.

Use L2 logistic regression with:

- `solver="lbfgs"`;
- `C=0.1`;
- no class weighting;
- `max_iter=1000`;
- `tol=1e-6`; and
- `random_state=20261011`.

This fixed solver replaces neither Block 10 candidate. It is selected before this run as a simple deterministic stability control for the new causal target.

## 6. Stage B: transition direction

At primary event decision points only, fit a separate L2 logistic model to distinguish:

- positive direction class: REM-to-Wake; and
- negative direction class: Wake-to-REM.

Use the same 80 causal features and solver settings, with `class_weight="balanced"` because this diagnostic compares two event types rather than estimating their natural full-night prevalence. Report participant-grouped outer out-of-fold macro F1, balanced accuracy, class recalls, and confusion counts at the fixed 0.5 decision threshold.

Stage B is an oracle-event direction diagnostic. It is not evidence that a full-night detector can first find the event.

## 7. Participant-grouped execution

For each of the five outer folds:

1. hold out all recordings from the outer-fold participants;
2. rotate each remaining fold once as the inner-held-out fold;
3. fit each Stage A candidate on the other three folds;
4. concatenate inner out-of-fold full-night scores;
5. select the threshold with maximum recall, then precision, then F1, then threshold, subject to at most 0.25 false alarms per supported hour;
6. refit each Stage A candidate on all four outer-training folds and score every supported candidate in the outer fold once; and
7. fit Stage B on outer-training primary event checkpoints and score outer-held-out primary event checkpoints.

Outer labels cannot influence scaling, fitting, or Stage A threshold selection.

## 8. Evaluation

Contiguous above-threshold 15-second candidates form one alarm. The highest score represents the run; exact ties use the earliest candidate.

Primary Stage A evaluation uses causal matching: an alarm matches an event only if it occurs from 0 to 15 seconds after the `nominal + 15 s` decision point. Matching is one-to-one. Predictions in the corresponding interval of an excluded or non-target transition are ignored before remaining predictions are counted as false alarms.

Report for both candidates:

- precision, recall, F1, false alarms/hour, TP, FP, FN, and ignored alarms;
- all-transition performance;
- REM-to-Wake-only performance; and
- Wake-to-REM-only performance.

The primary comparison is REM-to-Wake performance for `CAUSAL-BI` minus `CAUSAL-RW` at 0.25 false alarms/hour.

## 9. Numerical gate and hypothesis decision

The run is numerically interpretable only if every fit:

- completes before 1,000 iterations without a convergence warning;
- has finite coefficients, intercepts, and probabilities;
- has fewer than 1% exact probability endpoints (`0` or `1`) on its held-out scores; and
- produces more than 100 unique held-out Stage A probabilities.

`H-BI` passes its point gate only if, for primary REM-to-Wake events:

1. `CAUSAL-BI` F1 exceeds `CAUSAL-RW` F1 by at least 0.03; and
2. false alarms/hour increase by no more than 0.05.

All-transition performance and Stage B direction performance are secondary. A point-gate pass retains the formulation only for new confirmation; it is not a reliability claim.

## 10. Required controls

1. Confirm 82 train recordings and 64 train `pid` groups.
2. Confirm participant disjointness in every fit.
3. Confirm no validation or current-test input is read.
4. Confirm every positive feature window ends at or before its decision time.
5. Confirm both Stage A candidates use identical negatives.
6. Confirm primary event counts by direction after the causal-context check.
7. Record iterations, convergence, intercepts, score ranges, unique-score counts, and endpoint fractions.
8. Preserve compact reviewed tables in Git and hash large scores/models stored outside Git.
9. Validate reviewed outputs read-only.

## 11. Stop rule

Do not change the causal decision point, context, event membership, feature set, solver, regularization, class weighting, negative exclusion, threshold rule, alarm budget, or gate after outer results are available.

If the numerical gate fails, preserve the failure and stop v0.1. If `H-BI` fails, do not tune the bidirectional target on these outer results. Any subsequent causal formulation requires a new protocol and a new confirmation boundary.

## 12. Method references

1. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
2. Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
3. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
