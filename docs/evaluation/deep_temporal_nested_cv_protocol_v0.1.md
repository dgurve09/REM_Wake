# Nested Train-Only Deep Temporal Exploration Protocol v0.1

**Created:** 2026-09-22

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological Uncertainty

The train-only `LC-1` LSTM-CRF improved primary out-of-fold event F1 from 0.0982 to 0.1965 and reduced false alarms from 1.4679 to 0.5306/hour relative to a matched flattened logistic comparator. It retained 56 primary true positives versus 57 for logistic regression while reducing false positives from 924 to 334. The gain therefore came from temporal specificity rather than greater sensitivity.

Two uncertainties remain:

1. whether the specificity gain depends on the LSTM encoder, the CRF state constraint, or their interaction; and
2. whether another fixed temporal encoder can materially improve event F1 without restoring false alarms.

The earlier `LC-1` threshold was selected at 0.99, the upper edge of the old probability grid. This indicates that threshold resolution and model selection must be handled inside a nested procedure rather than revised against the already observed out-of-fold result.

## 2. Known Methods and Limitation

Bidirectional LSTM-CRF is an established sequence-labeling architecture [1]. Gated recurrent units provide a lower-parameter recurrent alternative [2]. Temporal convolutional networks provide a convolutional sequence-modeling alternative with dilated residual receptive fields [3]. These known architectures do not establish new knowledge by themselves.

The project-specific uncertainty is whether their inductive biases change participant-held-out REM-to-Wake event specificity when inputs, labels, folds, support, and event scoring are held constant. An unrestricted architecture or hyperparameter search would overfit 180 train transitions and is prohibited.

## 3. Fixed Candidate Family

All candidates use the same eight consecutive 30-second epochs, ten stored `HB_1`/`HB_2` log-bandpower features per epoch, 2,743 reviewed train candidates, positive/background labels, per-feature training-fold scaler, participant grouping, and full-night event evaluator.

### C1: `BLSTM-CRF`

Exact architectural reproduction of `LC-1`:

- one bidirectional LSTM layer;
- hidden size 16 per direction;
- linear projection from 32 values to three CRF emissions; and
- fixed endpoint path `[0, 0, 0, 1, 2, 0, 0, 0]` versus the all-background path.

### C2: `BGRU-CRF`

- one bidirectional GRU layer;
- hidden size 16 per direction;
- the same three-emission projection and CRF; and
- the same fixed endpoint and background paths.

This tests whether a lower-parameter recurrent gate improves generalization under limited positive events.

### C3: `TCN-CRF`

- one 1x1 projection from 10 to 32 channels;
- two residual temporal blocks with kernel size 3 and dilations 1 and 2;
- one convolution, ReLU, and fixed 0.10 dropout per block;
- sequence-length-preserving symmetric padding;
- linear projection from 32 values to three CRF emissions; and
- the same fixed endpoint and background paths.

This tests a convolutional temporal representation with a complete eight-epoch receptive field.

### C4: `BLSTM-2H`

- the same bidirectional LSTM encoder as C1;
- one binary `pre-boundary` head at relative epoch -30 seconds;
- one binary `post-boundary` head at relative epoch 0 seconds;
- weighted binary cross-entropy for each head; and
- event score equal to the product of the two sigmoid probabilities.

This tests whether the earlier endpoint-factorization mechanism is more useful than a CRF path constraint after a learned temporal encoder.

No candidate is added, removed, widened, deepened, or rerun with a changed configuration after results are available.

## 4. Fixed Optimization

Every candidate uses:

- Adam with learning rate `0.003` and weight decay `0.0001`;
- batch size 128;
- gradient-norm clipping at 5;
- 60 fixed epochs;
- no early stopping, scheduler, pretrained weight, or augmentation;
- positive sequence weight `negative_count / positive_count` in the fitting subset;
- deterministic CPU algorithms and one PyTorch thread; and
- seed `20260922 + 100 * outer_fold + 10 * inner_fold`, with inner fold 9 reserved for the outer-final fit.

All candidates receive the same seed within a fold. Parameter counts are recorded and used only as a final tie-breaker.

## 5. Nested Participant-Grouped Design

Reuse the exact five outer `pid` folds from the prior train-only experiment. For outer fold `k`:

1. hold outer fold `k` completely closed;
2. use each of the other four frozen fold identifiers once as an inner held-out fold;
3. fit all four candidates on the remaining three fold groups;
4. score every complete full-night context for the inner held-out groups;
5. concatenate the four inner held-out score sets separately for each candidate;
6. select one candidate-specific threshold using only those inner out-of-fold scores;
7. select one architecture using only the inner event results;
8. fit the selected architecture on all four outer-training fold groups; and
9. apply its frozen inner threshold once to the closed outer fold.

Also fit C1 on all four outer-training groups and apply its own inner-selected threshold to the same outer fold. If C1 is selected, reuse the fitted model. This creates a nested `BLSTM-CRF` comparator under the same threshold procedure.

After all five outer folds, concatenate the one-time outer predictions. Outer scores do not influence architecture, threshold, epoch, or hyperparameter selection.

## 6. Threshold Grid and Selection

Use this fixed probability grid:

- `0.01` through `0.95` in increments of `0.05`; and
- `sigmoid(x)` for logits `3.0` through `14.0` in increments of `0.25`.

Remove exact duplicates and sort ascending. The logit tail extends threshold resolution above 0.99 without selecting a grid after seeing new candidate scores.

For each candidate, select maximum primary +/-15-second event F1, then minimum false alarms/hour, maximum recall, and maximum threshold.

Select the architecture with maximum inner event F1, then minimum false alarms/hour, maximum recall, fewer trainable parameters, and candidate name. The final candidate recommended for future confirmation is the architecture with the best mean inner rank across outer folds, then the same ordered tie-breakers. Outer results are not used to choose that recommendation.

## 7. Event Evaluation

Contiguous above-threshold 30-second candidates form one alarm. The highest-score candidate represents the run; exact ties use the earliest time. Matching and ignored-reference handling remain unchanged.

Primary membership at +/-15 seconds is the sole decision endpoint. Report fixed sensitivities for:

- primary membership at +/-45 seconds;
- expanded membership at +/-15 seconds; and
- expanded membership at +/-45 seconds.

Fold-specific inner-selected thresholds are applied before outer predictions are pooled. No pooled outer threshold is selected.

## 8. Hypothesis and Decision Rule

**H-DT1:** Nested architecture selection will improve primary +/-15-second event F1 by at least 0.05 relative to nested `BLSTM-CRF`, without increasing false alarms per supported hour.

Use a paired participant-cluster bootstrap with 2,000 resamples and seed `20260922` for selected-pipeline minus nested-`BLSTM-CRF` F1 and false-alarm differences.

The exploration passes the point gate only if both H-DT1 conditions pass. The direction is participant-supported only if the F1 interval is entirely above zero and the false-alarm interval is entirely at or below zero. Full material-size support requires the F1 interval lower bound to reach +0.05.

If the point gate fails, stop this four-candidate exploration at v0.1. If it passes, freeze the inner-ranked candidate for a separately committed new-cohort evaluation. No result authorizes validation or current-test use.

## 9. Required Controls

Before accepting a result, verify:

1. exact 82-recording and 64-`pid` train membership;
2. no validation or test participant, feature, score, model, or result access;
3. exact reuse of the five frozen outer folds;
4. every outer participant held out exactly once;
5. four complete inner fold identifiers within every outer fold;
6. no participant overlap across any fit/inner-heldout or outer-fit/outer-heldout boundary;
7. exact common feature schema, candidate identities, context, and support;
8. scaler fitting restricted to each fitting subset;
9. four and only four frozen candidates;
10. fixed 60-epoch completion with finite losses, gradients, parameters, and probabilities;
11. exact model parameter-count accounting;
12. candidate-specific inner threshold selection on the fixed grid;
13. architecture selection without outer-score access;
14. one selected-pipeline and one nested-baseline outer prediction per supported boundary;
15. paired support for all 64 participants;
16. deterministic duplicate agreement for one inner fit per candidate;
17. external artifact size and SHA-256 verification; and
18. identical reviewed outputs on a full immutable rerun.

## 10. Execution Order

1. Commit this protocol.
2. Commit the result-producing implementation and independent read-only validator.
3. Run synthetic model-shape, CRF, loss, and receptive-field checks.
4. Complete all inner fits, full-night scores, thresholds, and architecture selections.
5. Freeze each selected outer architecture and threshold before outer scoring.
6. Complete the nested baseline and selected-pipeline outer evaluation.
7. Retain all results regardless of direction.
8. Run the independent validator twice.
9. Perform one immutable full rerun.
10. Record the decision and weekly chronology.

## 11. Storage Boundary

Fold model states, scalers, inner and outer full-night score tables, and training tensors remain under `REM_W_data/derived/deep_temporal_nested_cv_v0.1`, outside Git. Git retains compact fold membership, model configuration, parameter counts, training summaries, threshold selections, architecture selections, event outputs, participant results, bootstrap intervals, decisions, integrity checks, hashes, and the failed result if applicable.

## 12. Interpretation Boundary

This is bounded model development on a reused train cohort. Nested selection prevents direct outer-fold tuning within this experiment, but the candidate family was motivated by prior results from the same participants. It is not independent confirmation and cannot establish clinical performance, prospective alarm utility, or deployment readiness.

This experiment does not start or replace Block 9. External PSG compatibility remains separate work, and external PSG is not a wearable confirmation cohort. The already used validation and current test partitions remain closed.

## References

1. Huang Z, Xu W, Yu K. Bidirectional LSTM-CRF models for sequence tagging. *arXiv*. 2015;1508.01991. https://doi.org/10.48550/arXiv.1508.01991
2. Cho K, van Merrienboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of EMNLP*. 2014:1724-1734. https://doi.org/10.3115/v1/D14-1179
3. Bai S, Kolter JZ, Koltun V. An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv*. 2018;1803.01271. https://doi.org/10.48550/arXiv.1803.01271
