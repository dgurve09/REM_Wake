# Multi-Dataset PSG Scope Amendment

**Date:** 2026-10-25

## Reason for the amendment

Wearable-only experiments have repeatedly produced low absolute event F1 and recall. The latest causal multiscale features improved a matched wearable comparator, but detected only 18/180 train-cohort events. It remains unclear whether the dominant limitation is the transition target itself, the wearable montage, the cohort, or acquisition-specific overfitting.

The project will therefore test the transition mechanism across compatible PSG datasets before doing further wearable optimization.

## Revised sequence

1. Use every locally available hypnogram collection for transition-distribution and label-uncertainty analyses where its metadata support the question.
2. Use all locally eligible raw PSG sources, currently BOAS, gx36m, and ISRUC group 3, in a common-channel dataset-held-out experiment.
3. Keep entire datasets and participant groups out of fitting when they are the external test source.
4. Return to BOAS paired PSG-to-wearable transfer only if the PSG mechanism passes the frozen external-generalization gate.
5. Defer a wearable streaming prototype while the detector remains below its reliability gate.

## What does not change

- The final application direction remains wearable REM-to-Wake boundary measurement.
- The target remains event localization, not general sleep-stage classification.
- Thirty-second stage labels remain uncertain boundary annotations, not exact physiological onset times.
- Existing negative and inconclusive results remain part of the evidence record.
- PSG-only performance cannot be presented as wearable, diagnostic, or clinical performance.

## Scope boundary

This amendment increases sample diversity for mechanism testing; it does not make label-only datasets into EEG samples. Raw signals will not be pooled unless channel meaning, reference, sampling, participant linkage, and label support are explicitly audited. Dataset-held-out results will be reported separately so that a large source cannot conceal failure on a smaller source.
