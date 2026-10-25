# Multi-Dataset PSG External-Generalization Experiment v0.1

## Current stage

The frozen feature, model, threshold-selection, and held-out evaluation path is complete. The point advancement gate passed, but the participant-bootstrap interval for the F1 difference includes zero.

## Reviewed input

- 196 PSG nights from 130 participant groups.
- 128 BOAS, 58 gx36m, and 10 ISRUC group-3 recordings.
- Three common EEG channels: F4, C4, and O2.
- Native sampling rates retained at 256, 500, and 200 Hz.
- 378,054 past-only decision points beginning 240 seconds after recording start.

## Representations

`PSD3-BASE` contains 120 log band-power values from eight 30-second windows over the preceding four minutes.

`ADV3-CHANGE` adds 324 values from the final four 15-second windows: spectral entropy, a fixed aperiodic proxy, RMS, line length per second, Hjorth measures, three-channel-pair correlation and coherence, and signed/absolute immediate change. The combined representation contains 444 values.

The feature path uses forward chronological filtering at each native sampling rate. Synthetic controls at all three rates confirmed alpha-band localization and exact absence of future-sample influence on earlier filtered output.

## Controls

- Pre-model feature audit: 13/13 passed.
- Independent feature validation: 11/11 passed twice.
- Feature-cache hashes: 196/196 verified.
- Nonfinite values: none.
- Feature-bound violations: none.
- Future signal used at a decision: 0 seconds.
- Model-input audit: 11/11 passed.
- Supervised event targets: 521 total (365 BOAS, 124 gx36m, and 32 ISRUC).
- Clean negative targets after ambiguity exclusion: 285,095.
- Every participant-grouped inner fold contains both contributing training datasets and both target classes.

The first independent validation attempt stopped because its broad `*model*` file check matched the legitimate `pre_model_feature_audit_v0.1.tsv` filename. The corrected predicate checks only fit artifacts and an external model directory; it does not change features or audit results.

The first independent result-validation attempt also stopped because it expected all feature-grid rows to be scored. ISRUC consensus-unknown epochs are correctly excluded, so the reviewed total is 750,636 rows across two candidates rather than 756,108. Correcting that expected audit count did not change a model, probability, alarm, or metric.

## Held-out result

| Held-out evaluation | Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|
| gx36m | `PSD3-BASE` | 0.1645 | 0.2016 | 0.1812 | 0.2495 |
| gx36m | `ADV3-CHANGE` | 0.1898 | 0.2097 | 0.1992 | 0.2181 |
| ISRUC group 3 | `PSD3-BASE` | 0.5000 | 0.1875 | 0.2727 | 0.0925 |
| ISRUC group 3 | `ADV3-CHANGE` | 0.4800 | 0.3750 | 0.4211 | 0.2004 |
| Pooled primary external | `PSD3-BASE` | 0.1890 | 0.1987 | 0.1938 | 0.2318 |
| Pooled primary external | `ADV3-CHANGE` | 0.2346 | 0.2436 | 0.2390 | 0.2161 |

On the pooled primary external set, the advanced representation detected 38/156 events compared with 31/156 for the baseline. Its F1 difference was `+0.0452`, recall difference was `+0.0449`, and false-alarm difference was `-0.0157/hour`. Nineteen of 30 participant groups contributed at least one advanced-model true positive, compared with 18/30 for the baseline.

The stratified participant-bootstrap 95% interval for the F1 difference was `-0.0208` to `+0.1035`. The interval for recall difference was `-0.0390` to `+0.1222`, and the false-alarm interval was `-0.0672` to `+0.0358/hour`. The point gate passed, but the improvement is participant-inconclusive.

The descriptive BOAS holdout did not improve: F1 was 0.1971 advanced versus 0.1975 baseline, although false alarms decreased. Scorer sensitivity was also heterogeneous. The advanced direction was favorable for gx36m experts 1 and 3 and both ISRUC experts, but adverse for gx36m expert 2. These results prevent a universal cross-dataset claim.

## Verification

- 36/36 fixed logistic fits converged.
- 12/12 in-run checks passed.
- 13/13 independent result checks passed twice.
- 42/42 external model and score artifacts passed SHA-256 verification.
- The validator reconstructed 1,057 alarms, all event and participant metrics, six ranking diagnostics, ten scorer-sensitivity rows, the participant bootstrap, and the decision gate.

## Decision

Retain `ADV3-CHANGE` as a PSG mechanism candidate for a separately predeclared paired wearable-transfer study. Do not call it reliable: pooled external recall is 0.2436, 118/156 events were missed, and the F1-improvement interval crosses zero. Do not revise features, labels, thresholds, refractory interval, or event tolerance against these held-out results.

See the [frozen protocol](../../docs/evaluation/multidataset_psg_external_generalization_protocol_v0.1.md) and [pre-result amendment](../../docs/evaluation/multidataset_psg_protocol_amendment_2026-10-25.md).
