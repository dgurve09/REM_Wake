# Multi-Dataset Hypnogram Transition-Uncertainty Analysis v0.1

## Current stage

The analysis protocol is frozen. No transition, scorer, repeated-night, group, or decision result has been calculated in this experiment package yet.

## Intended input

- Ten harmonized public hypnogram sources represented by 2,381 stored scorer records.
- One primary consensus or provided human hypnogram per 1,935 nights.
- Fifty-eight additional gx36m nights with three scorers.
- Expected complete primary pool: 1,993 nights from eleven sources.

## Boundary

This is a label-only analysis. It reads no EEG signal, features, model, probability, or classifier output. The result will characterize the derived REM-to-Wake event prior and its annotation and repeated-night uncertainty; it cannot validate wearable detection.

See the [frozen protocol](../../docs/evaluation/multidataset_hypnogram_transition_uncertainty_protocol_v0.1.md).
