# Stage-First Baseline Protocol v0.1

**Created:** 2026-08-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Project phase:** Catch-up completion of Block 5
**Label source:** PSG human consensus `stage_hum`
**Split:** `grouped_pid_split_v0.1`
**Quality membership:** `quality_analysis_membership_v0.1`
**Preprocessing:** `minimal_wearable_eeg_preprocessing_v0.2`
**Model results inspected before protocol:** No
**Test signal data opened before protocol:** No

## 1. Schedule Record

Block 5 was originally scheduled for 2026-07-27 to 2026-08-09. No project commit was made after 2026-07-19, and no model work is attributed to the inactive interval. Work resumed on 2026-08-15. This experiment is dated from the actual resumption date and is not backdated.

Aggregate sleep-stage class imbalance was already known from dataset-readiness work. Model predictions and event-performance results were not inspected before this protocol was frozen.

## 2. Technological Question

Can ordinary wearable sleep staging followed by deterministic REM-to-Wake transition derivation provide a credible event-level comparator under the frozen participant split, signal-quality tiers, and 30-second label uncertainty?

This experiment does not test the direct transition hypothesis. It establishes the comparator that the direct method must later exceed or complement.

## 3. Hypotheses

### H5.1 Fixed wearable staging comparator

The dataset-provided headband `stage_ai` sequence will detect some human-derived REM-to-Wake events but may have unknown training provenance and therefore can be interpreted only as a fixed descriptive comparator.

### H5.2 Transparent feature baseline

A five-stage multinomial logistic model using two-channel wearable EEG spectral features will produce measurable held-out stage and event performance without a deep architecture.

### H5.3 Temporal-context value

Concatenating two preceding, current, and two following epoch feature vectors will improve validation macro F1 or REM-to-Wake event F1 relative to the same epoch-only model. Both results will be retained regardless of outcome.

## 4. Comparators

| ID | Comparator | Role |
|---|---|---|
| `SF-A` | BOAS headband `stage_ai` | Fixed descriptive comparator; never ground truth |
| `SF-B` | Epoch-only multinomial logistic regression | Transparent non-temporal baseline |
| `SF-C` | Five-epoch-context multinomial logistic regression | Primary stage-first comparator |

The BOAS documentation identifies headband `stage_ai` as automatic scoring based on headband data but does not provide enough information here to establish its training set or independence from BOAS. `SF-A` must not be used for model selection or novelty claims.

## 5. Wearable Feature Configuration

- channels: `HB_1`, `HB_2`;
- continuous preprocessing: v0.2, 0.3-35 Hz fourth-order Butterworth SOS, zero-phase application, 256-to-128 Hz polyphase resampling, train-derived robust channel scaling;
- epoch duration: 30 seconds, 3,840 output samples per channel;
- PSD: Welch method, Hann windows, 512-sample segments, 256-sample overlap;
- bands per channel: delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, beta 16-30 Hz;
- feature: base-10 logarithm of mean band power with machine epsilon floor;
- `SF-B` input: 10 features for the current epoch;
- `SF-C` input: 50 features from epochs `t-2` through `t+2`;
- context never crosses a recording edge, PSG disconnection code 8, missing label, or non-30-second gap.

Feature arrays and fitted model files remain outside Git. Git retains configurations, compact summaries, predictions needed for reviewed metrics, and checksums for external artifacts.

## 6. Model Configuration

Both transparent models use scikit-learn 1.7.2:

- `StandardScaler` fitted on train rows only;
- `LogisticRegression` with `C=1.0`, `class_weight="balanced"`, `solver="lbfgs"`, `max_iter=500`, and `tol=1e-4`;
- five labels: Wake 0, N1 1, N2 2, N3 3, REM 4;
- no hyperparameter search;
- no validation-driven architecture change;
- no participant may cross partitions.

If convergence fails, the failed run is retained. Increasing `max_iter` without changing the objective is permitted only as a new version with the original failure preserved.

## 7. Event Derivation and Matching

A predicted REM-to-Wake event occurs at the boundary between adjacent predicted epochs when the previous prediction is REM 4 and the current prediction is Wake 0. Events never bridge an invalid epoch, missing prediction, disconnection, or recording boundary.

Matching is performed within recording using a one-to-one assignment that:

1. maximizes the number of matches within the tolerance;
2. among equal-cardinality solutions, minimizes total absolute timing error;
3. never allows one event to match twice.

Primary tolerance is +/-15 seconds, representing the frozen scoring-boundary uncertainty. A +/-45-second analysis tests sensitivity to a one-epoch displacement plus the 15-second uncertainty.

For primary quality evaluation, targeted-review and critical reference events are ignore zones after eligible references are matched. For expanded evaluation, only critical events are ignore zones. A prediction matched to an eligible reference takes precedence over an overlapping ignore zone.

## 8. Metrics

### Stage diagnostics

- macro F1;
- balanced accuracy;
- Cohen kappa;
- per-stage precision, recall, and F1;
- confusion matrix.

Overall accuracy is secondary because N1 and N3 are expected to be much less frequent than N2.

### Event endpoints

- event precision, recall, and F1;
- false alarms per supported hour;
- matched-event median and maximum absolute timing error;
- events and errors by participant;
- participant-cluster bootstrap 95% intervals with 2,000 resamples and seed `20260815`.

Primary conclusions use the conservative membership and +/-15-second tolerance. Expanded membership and +/-45-second tolerance are prespecified sensitivity analyses.

## 9. Test Lock

1. Build features and fit `SF-B` and `SF-C` using train participants only.
2. Inspect validation results only after the code and configuration above are frozen.
3. Record convergence, class coverage, stage metrics, event metrics, and failures.
4. Do not change features, model, tolerance, membership, or event rules after validation.
5. Open test signals once only after the validation decision is written.
6. Report the frozen test result whether positive, negative, or inconclusive.

`SF-A` is a fixed dataset output and requires no fitting. Its partition metrics are descriptive and cannot alter `SF-B` or `SF-C` configuration.

## 10. Decision Rule

The stage-first block is complete when all three comparator outputs, matching validation, stage diagnostics, event metrics, participant dispersion, and failure notes are preserved. Proceed to the simple direct feature baseline even if stage-first performance is poor, because the direct-versus-stage-first comparison is the next stated uncertainty. Do not add a CNN unless the simple direct comparison identifies a specific insufficiency that a CNN could test.

## 11. References

1. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*. 1967;15:70-73. https://doi.org/10.1109/TAU.1967.1161901
2. Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: end-to-end hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2019;27:400-410. https://doi.org/10.1109/TNSRE.2019.2896659
3. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
4. BOAS dataset, OpenNeuro `ds005555`, snapshot `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
