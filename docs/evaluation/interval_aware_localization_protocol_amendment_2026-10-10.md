# Interval-Aware Localization Protocol Amendment: Shared Preprocessing

**Created:** 2026-10-10

**Applies to:** `interval_aware_localization_protocol_v0.1.md`

**Status:** Frozen before result-producing implementation or execution

## Ambiguity

The protocol fixes fold-local variance removal, correlation pruning, and standard scaling, but does not state whether the three positive rows per event in `INT15` may alter those preprocessing estimates. Candidate-specific preprocessing would mix the intended label-distribution change with a second change in retained features and scaling.

## Frozen clarification

For each inner or outer fit:

1. build the `HARD15` fitting table from eligible negatives and nominal positive rows belonging only to fitting participants;
2. fit variance removal, correlation pruning, and standard scaling once on that table;
3. reuse the exact selected feature indices and fitted scaler for both `HARD15` and `INT15`; and
4. change only the positive training rows and their weights when fitting the two classifiers.

The `INT15` positive rows retain weight `1/3` each, so both classifiers have the same total fitting weight. Negative rows, preprocessing, solver settings, folds, score grid, thresholds, outcomes, and gates remain identical.

This clarification prevents a preprocessing confound. It does not use or change any model result.
