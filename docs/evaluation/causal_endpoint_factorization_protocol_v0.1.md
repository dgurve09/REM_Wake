# Causal Endpoint Factorization Protocol v0.1

**Created:** 2026-10-15

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Schedule and evidence boundary

Block 10 finished earlier than its October 18 endpoint. This supplemental train-only experiment was requested on October 15 and does not start Block 11, which remains scheduled for October 19. It tests one new localization mechanism after the causal bidirectional experiment identified a specific insufficiency.

The same train participants have already supported several method-development experiments. Nested out-of-fold estimates reduce participant leakage but do not create a new confirmation cohort. Any favorable result is therefore hypothesis-generating and requires a newly locked or external wearable cohort before an advancement claim.

## 2. Technical uncertainty

The causal REM-to-Wake detector learned a rare event directly from 180 positive checkpoints and approximately 147,000 full-night negatives. It reached F1 0.0517 at 0.2528 false alarms per hour. Adding Wake-to-REM positives did not improve it.

At known transition checkpoints, however, the separate direction classifier reached macro F1 0.8290 and balanced accuracy 0.8272. This suggests that the wearable signal contains useful state-direction information once a checkpoint is supplied, while rare full-night checkpoint localization remains unresolved.

It is uncertain whether direct rare-event fitting is discarding the abundant state supervision already present in the hypnogram. A factorized model can learn two more common targets across the full night and combine them explicitly:

1. evidence that the state immediately before a candidate boundary is REM; and
2. evidence that the state immediately after that boundary is Wake.

## 3. Relationship to prior methods

This is not ordinary five-stage sleep classification. The output remains an event score on a 15-second decision grid.

The August `DE-D` endpoint experiment also multiplied REM-before and Wake-after probabilities, but it used 80 features centred on the boundary from -120 through +90 seconds, reviewed sampled backgrounds, validation-selected thresholds, and signal occurring after the decision point. The present experiment differs in four fixed respects:

- all feature windows end by the causal decision time;
- state-head labels are generated for the full train hypnogram rather than only reviewed event/background rows;
- thresholds are selected inside participant-grouped nested folds; and
- validation and current test remain closed.

The comparison therefore tests whether abundant full-night endpoint supervision resolves causal localization, not whether another architecture improves the same rare-event objective.

## 4. Causal grid and endpoint labels

Reuse the reviewed causal grid of 150,967 decisions over the 82 train recordings. Each decision uses eight contiguous 30-second wearable feature windows ending at the decision time. Window starts relative to the decision are `-240, -210, ..., -30` seconds. No later signal is used.

For a decision at time `d`, define the candidate hypnogram boundary as `b = d - 15 s`. This permits 15 seconds of the putative new state to be observed while remaining causal.

Using the half-open 30-second human consensus hypnogram intervals:

- `rem_before = 1` if the valid stage immediately before `b` is REM 4;
- `wake_after = 1` if the valid stage immediately at or after `b` is Wake 0.

At a 30-second scoring boundary, the labels describe the adjacent epochs. At a midpoint inside one scored epoch, both sides use that same epoch label. A row is excluded from state-head fitting if either side is missing, disconnected, outside the scored hypnogram, or not one of stages 0 through 4. Hypnogram labels are fitting targets only and are never model inputs.

## 5. Fixed representation and models

Use wearable `H2` only: `HB_1` and `HB_2`.

Each of the eight causal windows contributes log band power in delta, theta, alpha, sigma, and beta for both channels, producing 80 input values. Reuse the reviewed causal feature caches and the frozen preprocessing path.

Fit two independent pipelines within every fold:

1. `StandardScaler` fitted on fitting participants only;
2. L2 `LogisticRegression` with `C=0.1`, `solver="lbfgs"`, `class_weight="balanced"`, `max_iter=1000`, `tol=1e-6`, and `random_state=20261015`.

The event score is frozen as:

`CEF = P(rem_before) * P(wake_after)`

The product is a deterministic conjunction score, not a calibrated joint-probability claim. No head weighting, alternate combination, feature selection, architecture search, class-weight search, or regularization search is permitted in v0.1.

## 6. Participant-grouped execution

Reuse the five frozen train `pid` folds. For every outer fold:

1. hold out all recordings from outer-fold participants;
2. rotate each remaining fold once as the inner-held-out fold;
3. fit both state heads on the other three folds using all label-supported decisions;
4. concatenate inner out-of-fold `CEF` scores;
5. select one event threshold under the fixed 0.25 false-alarms/hour budget;
6. refit both heads on all four outer-training folds; and
7. score every supported outer-fold decision once.

Threshold selection maximizes recall, then precision, then F1, then threshold among points at or below the false-alarm budget. Outer labels cannot influence scaling, fitting, or threshold selection.

## 7. Event evaluation

Contiguous above-threshold 15-second decisions form one alarm. The highest score represents the run; exact ties use the earliest decision.

Primary references are the 180 primary-quality REM-to-Wake events supported by the causal grid. An alarm is timely from `nominal + 15 s` through `nominal + 30 s`, matching the prior causal experiment. Matching is one-to-one. Predictions in the corresponding intervals of excluded or non-target transitions are ignored before remaining predictions are counted as false alarms.

Report precision, recall, F1, false alarms/hour, TP, FP, FN, ignored alarms, and participant-level results. Use 2,000 participant-cluster bootstrap resamples with seed `20261015` for `CEF - CAUSAL-RW` differences.

Secondary mechanism diagnostics report outer out-of-fold average precision and ROC AUC for each state head on label-supported decisions. These diagnostics do not replace event-level performance.

## 8. Numerical gate

The result is numerically interpretable only if every state-head fit:

- finishes before 1,000 iterations without a convergence warning;
- has finite coefficients, intercepts, and probabilities;
- has fewer than 1% exact probability endpoints on held-out scores; and
- produces more than 100 unique held-out probabilities.

The combined `CEF` score must also have more than 100 unique outer out-of-fold values and fewer than 1% exact endpoints.

## 9. Advancement gate

Compare `CEF` with the frozen `CAUSAL-RW` train nested out-of-fold result at the same causal timing rule and alarm budget.

Retain the factorized mechanism only for future confirmation if:

1. `CEF - CAUSAL-RW` event F1 is at least +0.03;
2. false alarms/hour increase by no more than 0.05; and
3. the numerical gate passes.

A passing point gate is not evidence of deployment reliability and does not authorize opening validation or current test. Failure stops v0.1 without changing labels, combination rule, model settings, threshold rule, timing, or alarm budget after the result is known.

## 10. Required controls

1. Confirm 82 train recordings and 64 train `pid` groups.
2. Confirm no validation or current-test input is read.
3. Confirm no feature window extends beyond its decision time.
4. Confirm endpoint labels are derived only from valid human hypnogram intervals.
5. Confirm participant disjointness in every fit.
6. Reproduce the frozen `CAUSAL-RW` comparator from its reviewed participant table.
7. Record label prevalence, iterations, convergence, intercepts, probability ranges, unique counts, and endpoint fractions.
8. Store large scores and models outside Git with hashes; retain compact reviewed tables in Git.
9. Validate reviewed outputs read-only.

## 11. Method references

1. Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: end-to-end hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2019;27:400-410. https://doi.org/10.1109/TNSRE.2019.2896659
2. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
3. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
4. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
5. BOAS dataset, OpenNeuro `ds005555`, snapshot `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
