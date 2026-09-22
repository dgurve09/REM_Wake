# Train-Only LSTM-CRF Temporal-Representation Decision

**Decision date:** 2026-09-22

**Protocol commit:** `2f9d961`

**Initial implementation commit:** `2db3c91`

**Path-check correction commit:** `7f5df50`

**Accepted result commit:** `cc6dff0`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Eligible for evaluation on a new locked or external wearable cohort; no current detector replacement

## Question

Can a small wearable-only bidirectional LSTM with a three-state linear-chain CRF improve participant-grouped REM-to-Wake event separation beyond the same eight epochs flattened into a balanced logistic classifier?

## Fixed Comparison

Both models used the same 2,743 reviewed train candidates, ten stored two-channel bandpower features per epoch, eight-epoch contexts, five frozen participant folds, full-night held-out scoring, event consolidation, quality membership, support, and event matching.

`LR-OOF` flattened each context to 80 columns. `LC-1` used one bidirectional LSTM layer with 16 hidden values per direction and a CRF over `background`, `pre-boundary`, and `post-boundary` states. The endpoint path was fixed as `[0, 0, 0, 1, 2, 0, 0, 0]`; it was an event-label sequence, not a sleep-stage target.

Every participant was held out exactly once. Each model selected its threshold from its own pooled train out-of-fold predictions under the frozen event-level rule.

## Primary Result

| Model | Threshold | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|
| LR-OOF | 0.98 | 0.0581 | 0.3167 | 0.0982 | 1.4679 |
| LC-1 | 0.99 | 0.1436 | 0.3111 | 0.1965 | 0.5306 |

At primary membership and +/-15 seconds, `LC-1 - LR-OOF` was:

- `+0.0983` event F1, exceeding the prespecified `+0.05` point gate; and
- `-0.9373` false alarms per supported hour, satisfying the no-increase gate.

The gain came mainly from fewer false positives: 334 for `LC-1` versus 924 for `LR-OOF`. True positives were similar at 56 versus 57. This supports the proposed temporal-specificity mechanism rather than improved event sensitivity.

## Sensitivity Results

| Membership | Tolerance | LR-OOF F1/FAR | LC-1 F1/FAR |
|---|---:|---:|---:|
| Primary | +/-45 s | 0.1164 / 1.4361 | 0.2575 / 0.4988 |
| Expanded | +/-15 s | 0.1330 / 1.4679 | 0.2105 / 0.5306 |
| Expanded | +/-45 s | 0.1640 / 1.4361 | 0.2743 / 0.4988 |

The direction was consistent across the three fixed sensitivity settings. These settings were not used for the decision gate.

## Participant Uncertainty

The paired 2,000-resample participant bootstrap gave:

| Contrast | Point | 95% interval |
|---|---:|---:|
| Event-F1 difference | +0.0983 | +0.0485 to +0.1367 |
| False alarms/hour difference | -0.9373 | -1.4804 to -0.4089 |

Both intervals exclude zero in the favorable direction. The F1 lower bound is slightly below the prespecified `+0.05` materiality threshold. Therefore, the improvement direction is participant-supported, while the full material-size gate is supported by the aggregate point only.

## Failed First Run and Correction

The first completed run was rejected because its path-isolation predicate matched the word `validation` inside the historical train-feature directory name `block7_feature_generation_validation_v0.1`. All other controls passed, and the manifest contained only train feature files.

The failed outputs were retained. Commit `7f5df50` narrowed the predicate to explicit validation/test path segments without changing the scientific method. The accepted run reproduced the same models, scores, thresholds, and metrics.

## Reproducibility

- All 14/14 in-run controls passed.
- CRF forward partition and negative log-likelihood matched exhaustive enumeration within `4.77e-7`.
- A duplicate fold-1 fit reproduced the model-state hash, scaler, and probabilities exactly.
- All five LSTM-CRF fits completed 60 fixed epochs; all five logistic fits converged without warning.
- All 93 external feature, model, and full-score artifacts matched recorded sizes and SHA-256 values.
- The read-only validator reconstructed 198 threshold rows, selected thresholds, event outputs, participant tables, bootstrap intervals, and decisions; it passed 10/10 checks twice.
- A full immutable rerun changed no reviewed or external artifact.

## Knowledge Gained

The flattened bandpower representation was not the limit of the available train signal. A small structured temporal model suppressed many sequence-inconsistent alarms while retaining almost the same number of primary true detections. This is the first project experiment to pass the predefined meaningful F1 and alarm-burden point gates for a wearable-only temporal representation.

The experiment does not identify whether recurrence, the CRF transition terms, or their interaction produced the gain. It also does not show that F1 near 0.20 is adequate for the intended application.

## Decision and Boundary

Retain `LC-1` as a candidate for a separately committed evaluation on a new locked or external wearable cohort. Do not apply it to the already used validation or current test partitions, do not replace the frozen detector, and do not describe it as independently confirmed or deployable.

No post-result architecture, state-path, threshold, or loss-weight variation is authorized from this result. A future mechanism ablation would require a new protocol and confirmation boundary. Block 9 external-PSG compatibility remains the next scheduled project block; it is not an independent wearable confirmation cohort.
