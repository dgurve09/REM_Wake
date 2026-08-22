# Implementation Correction Record v0.1

**Recorded:** 2026-08-22  
**Phase:** Before direct-model test access  
**Scientific configuration changed:** No

## Initial Issue

The first train/validation execution completed successfully, but `train_validation_construction_summary_v0.1.tsv` used generic column names `level_0`, `level_1`, and `level_2`. The cause was concatenating two populated grouped series with an empty dropped-row series whose index had no names. No candidate row was actually dropped.

## Correction

The table-generation function was changed to merge explicitly named grouped data frames. No feature, label, split, model, alarm rule, threshold grid, selection rule, or metric was changed. The unchanged train/validation phase was rerun before any direct test-feature access.

## Verification

The rerun restored the intended columns `partition`, `label`, and `source_tier`. It reproduced:

- DE-A threshold 0.97, validation event F1 0.115183, and model SHA-256 `ebbd066d27da04f7ef683656d652f7e9887d2937da6b50c5191adbf2d340cdd4`;
- DE-B threshold 0.96, validation event F1 0.112676, and model SHA-256 `3667991d53536bafc06ad0de7163eb9f2489460c2f1b66b8cbb48b560084e92d`.

This was a reporting implementation correction, not an experimental retry or a response to model performance.
