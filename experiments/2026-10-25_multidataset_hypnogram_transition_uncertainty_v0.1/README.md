# Multi-Dataset Hypnogram Transition-Uncertainty Analysis v0.1

## Current stage

The analysis protocol and pre-analysis input checkpoint are frozen. No transition, scorer, repeated-night, group, or decision result has been calculated in this experiment package yet.

## Intended input

- Ten harmonized public hypnogram sources represented by 2,381 stored scorer records.
- One primary consensus or provided human hypnogram per 1,935 nights.
- Fifty-eight additional gx36m nights with three scorers.
- Expected complete primary pool: 1,993 nights from eleven sources.

## Input checkpoint

- The primary audit passed 18/18 checks.
- The independent validator passed 11/11 checks twice.
- All 2,556 metadata and source-stage files passed size and SHA-256 verification.
- The primary pool contains 1,822 participant groups.
- Two Sleep-EDF Telemetry files contain timestamp gaps in late Wake padding. One missing epoch in `ST7121` and 65 missing epochs in `ST7221` were represented explicitly as `UNKNOWN`, preserving time and preventing transitions across missing support.

The first audit execution stopped on an invalid NumPy vocabulary expression. The corrected execution then exposed 28 nonzero Sleep-EDF Telemetry start offsets, which are valid constant grid origins. After allowing a constant origin, two real timestamp gaps remained and were regularized as described above. These corrections occurred before any transition result was calculated.

The first independent input-validation execution stopped because blank `night` fields in the reviewed TSV reload as missing values while the reconstructed table retains empty strings. A second attempt showed that numeric night identifiers such as `1` and `1.0` also serialize equivalently. Normalizing the display-only night field corrected the comparison without changing an input or audit result.

## Boundary

This is a label-only analysis. It reads no EEG signal, features, model, probability, or classifier output. The result will characterize the derived REM-to-Wake event prior and its annotation and repeated-night uncertainty; it cannot validate wearable detection.

See the [frozen protocol](../../docs/evaluation/multidataset_hypnogram_transition_uncertainty_protocol_v0.1.md).
