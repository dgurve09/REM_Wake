# Direct REM-to-Wake Event Baseline Protocol v0.1

**Created:** 2026-08-22  
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`  
**Project phase:** Block 6, simple direct transition baseline  
**Label source:** PSG human consensus `stage_hum`, transformed under `transition_label_spec_v0.1`  
**Split:** `grouped_pid_split_v0.1`  
**Quality membership:** `quality_analysis_membership_v0.1`  
**Wearable features:** frozen two-channel features from `stage_first_feature_baseline_v0.1`  
**Direct-model results inspected before protocol:** No  
**Direct-model test features or scores opened before protocol:** No

## 1. Schedule and Prior Evidence

Block 6 covers 2026-08-10 to 2026-08-23. The overdue stage-first work was completed on 2026-08-15, and this direct baseline is performed on its actual date, 2026-08-22.

The prior SF-C stage-first baseline obtained test event F1 0.0766, precision 0.0438, recall 0.3051, and 1.9692 false alarms per supported hour at the primary +/-15-second tolerance. Its predicted stage sequence had a 2.1784-fold excess transition rate and many short REM bouts. Those retained observations motivate a direct boundary comparator; they are not results of the present experiment.

The project test partition has therefore already been accessed for the planned stage-first comparator and its failure analysis. It is not a never-observed project-wide test set. The present protocol nevertheless blocks access to test recording features and direct-model scores until the direct models and validation-selected thresholds are frozen. The resulting test comparison is a fixed held-out comparison for this block, but future confirmatory claims require an external cohort or a newly locked dataset.

## 2. Technological Question

Can a simple classifier trained directly on PSG-hypnogram-derived REM-to-Wake boundary labels reduce the event-level failure of stage-first detection when its inputs are limited to frozen, simultaneously recorded wearable headband EEG spectral features?

The uncertainty is not whether sleep stages can be classified. It is whether direct discrimination of the target boundary avoids the error propagation and sequence fragmentation created by independently classified 30-second stages.

## 3. Hypotheses

### H6.1 Direct-boundary value

The primary direct model, DE-B, will produce higher event F1 and fewer false alarms per supported hour than transparent stage-first comparator SF-C on the fixed test partition at the primary +/-15-second tolerance.

### H6.2 Broader context

An eight-epoch context model, DE-B, will improve validation event F1 over a boundary-pair ablation, DE-A. The result is retained whether context helps, harms, or has no measurable effect.

### H6.3 Remaining representation limit

If DE-B improves on SF-C but still produces an impractical false-alarm burden or weak participant-consistent recall, the result will identify a limitation of fixed log-bandpower context. A CNN is not authorized merely because performance is low; its input representation and tested hypothesis must be stated in a separate protocol.

## 4. Comparators

| ID | Input at candidate boundary `t` | Role |
|---|---|---|
| `DE-A` | wearable features at epochs `t-30 s` and `t` | Direct boundary-pair ablation |
| `DE-B` | wearable features at `t-120 s` through `t+90 s` in 30-second steps | Primary direct context baseline |
| `SF-C` | five-epoch stage classifier followed by 4-to-0 transition derivation | Primary transparent stage-first comparator |
| `SF-A` | BOAS headband `stage_ai` followed by 4-to-0 derivation | Descriptive comparator only; training provenance is not established locally |

DE-A and DE-B use logistic regression. No tree ensemble, neural network, hyperparameter search, feature selection, or test-driven adjustment is included.

## 5. Candidate Labels and Training Set

The candidate time is an adjacent PSG epoch boundary. A positive training row is a primary-quality REM-to-Wake transition from `transition_analysis_membership_v0.1.tsv`. Its time is `nominal_boundary_sec` in signal-quality artifact v0.3.

A negative training row is any primary-quality row in the deterministic background review artifact. Both prespecified background tiers are used:

- `strict_same_stage_window`;
- `nontarget_window_no_remwake_nearby`.

The background construction already excludes a +/-135-second region around every REM/Wake boundary. All eligible reviewed backgrounds are used. There is no negative subsampling. Class imbalance is handled only by `class_weight="balanced"` during train fitting. Validation and test rows are never resampled or reweighted for metric calculation.

Pre-run primary counts are 180/37/59 REM-to-Wake positives and 2,563/620/810 reviewed backgrounds in train/validation/test. Participant assignment is frozen, and no `pid` may cross partitions.

## 6. Frozen Wearable Features

Each 30-second epoch has ten features produced previously without reference to the direct labels:

- channels `HB_1` and `HB_2`;
- delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-12 Hz, sigma 12-16 Hz, beta 16-30 Hz;
- base-10 logarithm of mean Welch power in each channel-band pair;
- minimal preprocessing v0.2: 0.3-35 Hz filtering, 256-to-128 Hz resampling, and train-derived robust channel scaling.

DE-A concatenates two epochs for 20 input features. DE-B concatenates eight epochs for 80 input features. Context never crosses a recording edge, invalid human-scored epoch, disconnection, missing epoch, or non-30-second gap. Human stage codes are used only to define labels, quality, and evaluable temporal support; they are not model inputs.

The precomputed feature arrays remain outside Git. Their SHA-256 values are recorded. The direct script must load only train and validation arrays during the first phase, even though prior stage-first test arrays exist on disk.

## 7. Model Configuration

Both direct models use:

- `StandardScaler` fitted on train rows only;
- `LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, tol=1e-4)`;
- binary target: REM-to-Wake boundary 1, reviewed background 0;
- no hyperparameter search;
- no validation-driven feature or architecture revision.

Fit warnings and missing-context rows are retained. A convergence failure cannot be silently removed by changing the frozen settings.

## 8. Continuous Alarm Rule

For event evaluation, each model scores every supported 30-second candidate boundary in a recording, not only the reviewed training windows. This is necessary to measure false alarms across the night.

For a fixed threshold:

1. mark every supported candidate with probability greater than or equal to the threshold;
2. divide marked candidates into contiguous runs separated by more than 30 seconds;
3. emit one alarm per run at its highest-probability candidate, using the earlier time for an exact score tie.

This deterministic consolidation avoids counting a continuous high-score run as repeated alarms. No other refractory period or postprocessing is used.

## 9. Validation Threshold Selection

Threshold candidates are fixed at 0.01 through 0.99 in increments of 0.01. Each model receives its own validation-selected threshold.

The threshold is selected on validation participants using primary-quality reference events and the primary +/-15-second tolerance. The ordered rule is:

1. highest event F1;
2. lower false alarms per supported hour;
3. higher recall;
4. higher threshold.

The complete validation threshold table is retained. DE-B remains the primary direct model regardless of whether DE-A has the higher validation score.

## 10. Event Matching and Metrics

The already validated one-to-one matcher is reused unchanged. It maximizes matched events inside the tolerance and then minimizes total absolute timing error. Eligible references are matched before quality-ignored references.

Primary reporting uses primary quality membership and +/-15 seconds. Prespecified sensitivities are expanded membership at +/-15 seconds and both memberships at +/-45 seconds.

Reported endpoints are:

- window-level validation average precision and ROC AUC as secondary discrimination diagnostics;
- event precision, recall, and F1;
- false alarms per supported hour;
- matched-event timing error;
- participant and recording dispersion;
- participant-cluster bootstrap 95% intervals using 2,000 resamples and seed `20260822`.

The primary DE-B versus SF-C comparison also uses a paired participant-cluster bootstrap. Each resample draws the same test `pid` values for both methods, recomputes aggregate event F1 and false alarms per hour separately, and reports the DE-B minus SF-C difference.

Direct value is directionally supported only if DE-B has both higher test event F1 and lower test false alarms per hour than SF-C under the primary configuration. SF-A remains descriptive and cannot establish independent superiority.

## 11. Phase Gate and Test Lock

1. Verify hashes and construct training rows from train participants only.
2. Fit DE-A and DE-B on train rows.
3. Score train and validation recordings only.
4. Select thresholds using the frozen validation rule and write a decision file containing the freeze marker.
5. Do not revise features, models, alarm consolidation, threshold rule, membership, tolerance, or comparator role after validation.
6. Only then load test feature arrays and score the two frozen models once.
7. Retain the test result regardless of outcome.

The script must fail closed if the validation decision or model hashes do not match before test access.

## 12. Block 6 Decision Rule

Block 6 closes when the direct data construction, fit record, validation threshold curves, frozen test metrics, participant uncertainty, comparator table, integrity checks, failure analysis, and next decision are preserved.

A CNN is deferred unless the direct baseline reveals a specific limitation that a separately declared representation experiment can test. Transfer work remains blocked until Block 7 begins on 2026-08-24.

## 13. References

1. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. *IEEE Transactions on Audio and Electroacoustics*. 1967;15:70-73. https://doi.org/10.1109/TAU.1967.1161901
2. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
3. BOAS dataset, OpenNeuro `ds005555`, snapshot `1.1.1`. https://doi.org/10.18112/openneuro.ds005555.v1.1.1
