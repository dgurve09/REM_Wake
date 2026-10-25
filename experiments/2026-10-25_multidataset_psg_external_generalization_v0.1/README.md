# Multi-Dataset PSG External-Generalization Experiment v0.1

## Current stage

Feature generation and the pre-model audit are complete. No model has been fitted and no held-out performance result exists yet.

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

The first independent validation attempt stopped because its broad `*model*` file check matched the legitimate `pre_model_feature_audit_v0.1.tsv` filename. The corrected predicate checks only fit artifacts and an external model directory; it does not change features or audit results.

## Next execution boundary

The frozen protocol requires three leave-one-dataset-out comparisons with participant-grouped threshold selection. gx36m and ISRUC are the primary external tests; the BOAS holdout is descriptive because BOAS informed earlier feature development. No feature, label, model, threshold, refractory interval, or event tolerance may be changed after held-out results are available.

See the [frozen protocol](../../docs/evaluation/multidataset_psg_external_generalization_protocol_v0.1.md) and [pre-result amendment](../../docs/evaluation/multidataset_psg_protocol_amendment_2026-10-25.md).
