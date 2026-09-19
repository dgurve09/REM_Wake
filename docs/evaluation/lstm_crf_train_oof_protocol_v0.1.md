# Train-Only LSTM-CRF Temporal-Representation Protocol v0.1

**Created:** 2026-09-19  
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`  
**Authorized partition:** Train only, 82 recordings from 64 `pid` groups  
**Validation partition authorized:** No  
**Current test partition authorized:** No  
**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological Uncertainty

The frozen direct wearable detector represents eight consecutive 30-second epochs as 80 unrelated bandpower columns and applies logistic regression. Its clean validation event F1 was 0.1123 with 1.4558 false alarms per supported hour. Train-only noise augmentation corrected one induced channel failure but did not improve clean event detection. Fixed cross-view agreement reduced alarm burden, but required PSG at inference.

The unresolved question is whether wearable-only temporal ordering and an explicit transition-state constraint can improve participant-grouped REM-to-Wake event separation beyond the current flattened representation. A positive result is not assumed: only 180 reviewed primary train transitions are available, the boundary is derived from 30-second hypnogram epochs, and a recurrent model may overfit participant or position regularities.

## 2. Known Method and Limitation

Linear-chain conditional random fields model a conditional distribution over complete label sequences and learn dependencies between adjacent output states [1]. Bidirectional LSTM-CRF models combine past and future input context with a structured output layer [2]. These are established sequence-labeling methods; applying them does not itself establish new knowledge or clinical utility.

The existing `H2-D` logistic method already uses the same eight epochs but cannot learn recurrent temporal dynamics or constrain an event to contain an ordered pre-boundary and post-boundary state. This protocol tests only whether those two changes improve the direct event endpoint on the available train participants.

## 3. Research Hypothesis

For a complete context with relative epoch onsets `[-120, -90, -60, -30, 0, +30, +60, +90]` seconds, an eligible REM-to-Wake candidate is encoded as:

`[background, background, background, pre-boundary, post-boundary, background, background, background]`

An eligible reviewed background candidate is encoded as eight `background` states. The `pre-boundary` and `post-boundary` states are event-endpoint labels derived from the reviewed transition candidate; they are not general sleep-stage targets.

**H-LC1:** Learning temporal emissions and the ordered endpoint sequence will increase train out-of-fold primary event F1 by at least 0.05 relative to an otherwise matched flattened logistic comparator, without increasing false alarms per supported hour.

The proposed mechanism is that recurrent emissions can distinguish the evolution across the context while the CRF penalizes locally high but sequence-inconsistent endpoint evidence. The main failure alternative is that the feature and label information is too weak or scarce for the additional parameters.

## 4. Fixed Inputs

Use only:

- the 82 train recordings and 64 train `pid` groups in `pid_split_assignments_v0.1.tsv`;
- the previously generated clean `HB-2` epoch features from `block7_feature_generation_validation_v0.1`;
- five Welch log-bandpower features for each of `HB_1` and `HB_2`, in their stored order;
- primary-analysis-eligible reviewed REM-to-Wake transitions as positive candidates;
- primary-analysis-eligible reviewed background windows as negative candidates; and
- the exact five-fold `pid` allocation in `train_oof_fold_assignments_v0.1.tsv` from the frozen Block 8 experiment.

The expected labeled set is 2,743 candidates: 180 transitions and 2,563 backgrounds. A candidate is removed only if its exact complete eight-epoch context is unavailable. The construction report must record every retained and removed row.

Human stage codes may define the reviewed transition labels and valid temporal support. They are not model inputs and are not prediction targets.

## 5. Fixed LSTM-CRF Configuration

The sole new model, `LC-1`, uses:

- per-feature `StandardScaler` fit on all epoch positions from outer-fold training candidates only;
- one bidirectional LSTM layer with input size 10 and hidden size 16 in each direction;
- one linear projection from 32 hidden values to three state emissions;
- trainable start, transition, and end scores in a linear-chain CRF;
- exact conditional log-likelihood computed with the forward log-sum-exp algorithm;
- positive-sequence loss weight equal to `negative_count / positive_count` within each outer training fold;
- Adam with learning rate `0.003`, weight decay `0.0001`, batch size 128, and gradient-norm clipping at 5;
- 60 fixed epochs, without early stopping or validation monitoring;
- fold seed `20260919 + fold`; and
- CPU execution, one PyTorch thread, deterministic algorithms, and no architecture or hyperparameter search.

No external CRF package is introduced. The implementation must expose the path-energy and log-partition calculations directly and verify them against exhaustive enumeration on short synthetic sequences before any data fit.

For each complete full-night context, the event score is:

`sigmoid(energy(fixed endpoint path) - energy(all-background path))`

The CRF partition function cancels in this two-path log-odds contrast. This score tests the fixed event sequence against the fixed null sequence; it is not a sleep-stage probability.

## 6. Matched Comparator

`LR-OOF` uses the identical labeled candidates, outer folds, full-night contexts, and event evaluator. Each eight-by-ten context is flattened to 80 columns. Within each outer fold it fits:

- `StandardScaler` on outer-fold training rows only; and
- `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, tol=1e-4, random_state=20260919)`.

This comparator measures the contribution of learned sequence structure rather than a split, label, feature, or support change. It is a new train-only out-of-fold estimate and must not be substituted with the earlier validation-selected `H2-D` result.

## 7. Train-Only Out-of-Fold Evaluation

For each of the five frozen folds:

1. fit both scalers and models using candidates from the other train `pid` groups;
2. score every complete full-night context for the held-out train groups;
3. verify that no held-out `pid` contributed to scaling or fitting; and
4. retain external full score artifacts and compact fold/accounting records.

After every train participant has been scored exactly once out of fold, select a separate operating threshold for each model from `0.01` through `0.99` in increments of `0.01`. Use primary membership and +/-15-second matching. Select maximum event F1, then minimum false alarms/hour, then maximum recall, then maximum threshold.

Contiguous above-threshold 30-second candidates form one alarm. The candidate with the highest score represents the run; an exact score tie uses the earliest time. Event matching, ignored-reference handling, and support accounting remain identical to the existing project evaluator.

Report primary and expanded membership at +/-15 and +/-45 seconds. Primary membership at +/-15 seconds is the sole decision endpoint.

## 8. Decision Rules

`LC-1` passes the developmental advancement gate only if, at its separately selected train out-of-fold threshold:

1. primary +/-15-second F1 is at least `LR-OOF F1 + 0.05`; and
2. false alarms per supported hour are no greater than `LR-OOF`.

Use a paired participant-cluster bootstrap with 2,000 resamples and seed `20260919` to report 95% percentile intervals for the `LC-1 - LR-OOF` F1 and false-alarm-rate differences. The gate is a fixed point-estimate rule. Evidence is described as participant-supported only if the full F1 interval is above zero and the full false-alarm interval is at or below zero.

If either advancement condition fails, stop `LC-1` at v0.1 and retain the failure. Do not revise the state path, class weight, threshold objective, hidden size, context, or candidate set in response to the result. A passing train-development result only makes the method eligible for a separately committed evaluation on a new locked or external wearable cohort.

## 9. Required Controls

Before accepting a result, verify:

1. exact 82-recording and 64-`pid` train membership;
2. zero validation or current-test subject, feature, score, model, or path access;
3. exact reproduction of the frozen five-fold `pid` allocation;
4. every train participant held out exactly once;
5. no candidate identity shared across fit and held-out data;
6. exact feature-name order and finite eight-by-ten inputs;
7. exact positive and background candidate accounting;
8. scaler fitting restricted to each outer training fold;
9. exhaustive synthetic agreement for CRF path energy, partition, and negative log-likelihood;
10. deterministic duplicate fit agreement on one fold;
11. finite loss, gradients, model parameters, and full-night scores;
12. all five fits complete the fixed 60 epochs;
13. 99 threshold rows per model and exact decision-rule reconstruction;
14. paired participant support for both models;
15. external artifact sizes and SHA-256 values; and
16. identical reviewed outputs on an immutable rerun.

## 10. Execution Order

1. Commit this protocol.
2. Commit the result-producing implementation and an independent read-only validator.
3. Run implementation-level CRF synthetic checks.
4. Construct the train candidates and verify fold isolation.
5. Fit and score the five outer folds for both models.
6. Select train-only thresholds and calculate fixed endpoints.
7. Retain the result regardless of direction.
8. Run the independent validator twice.
9. Perform one immutable rerun.
10. Record the decision and update the dated weekly record.

## 11. Storage Boundary

Fold model states, scalers, and full-night out-of-fold score tables remain under `REM_W_data/derived/lstm_crf_train_oof_v0.1`, outside Git. Git retains the protocol, code, compact construction and fold records, threshold curves, event metrics, participant results, bootstrap intervals, integrity checks, artifact hashes, and decision note. Raw EDF, binary feature arrays, full scores, and model weights must not enter Git.

## 12. Interpretation Boundary

This is a train-only developmental comparison on reused reviewed labels. Thresholds are selected and reported on the same pooled out-of-fold predictions, so the estimates are not independent confirmation. The experiment cannot establish clinical performance, prospective alarm utility, external generalization, superiority to modern raw-signal architectures, or a wearable-method advance.

This standalone temporal-representation screen does not start or replace Block 9. The scheduled external-PSG compatibility audit remains no earlier than 2026-09-21, and the interval-aware boundary analysis remains Block 10 work. The current test partition stays closed.

## References

1. Lafferty J, McCallum A, Pereira F. Conditional random fields: Probabilistic models for segmenting and labeling sequence data. *Proceedings of the 18th International Conference on Machine Learning*. 2001:282-289. https://repository.upenn.edu/bitstreams/4905e2c0-e9d5-4961-804b-973de8bdfc7c/download
2. Huang Z, Xu W, Yu K. Bidirectional LSTM-CRF models for sequence tagging. *arXiv*. 2015;1508.01991. https://doi.org/10.48550/arXiv.1508.01991
