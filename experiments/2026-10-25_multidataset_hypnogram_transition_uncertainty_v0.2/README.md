# Multi-Dataset Hypnogram Transition-Uncertainty Correction v0.2

## Current stage

The corrected transition, scorer, repeated-night, comparison, and decision analyses are complete.

## Correction

The source documents numeric codes `0..4`. The v0.1 loader converted the 818 observed `-1` labels to `REM` through negative array indexing. Version 0.2 maps every out-of-vocabulary value to `UNKNOWN`, then applies the same strict 2-of-3 known-stage majority.

All v0.1 artifacts remain preserved. Ten other sources and all analysis rules remain unchanged.

## Corrected result

- The input audit passed 12/12 checks and the independent input validator passed 10/10 checks.
- The analysis passed 15/15 controls and the independent result validator passed 13/13 checks.
- The version comparison passed 7/7 checks.
- Safe parsing changed 81 gx36m primary epochs from a false known stage to `UNKNOWN`, increasing primary unknown epochs from 1,398 to 1,479.
- The gx36m primary REM-to-Wake event count remained 85, and the eleven-source total remained 4,046.
- gx36m median stage disagreement changed from 0.1759 to 0.1684 and median tolerant event F1 changed from 0.3095 to 0.3333 after unresolved scorer epochs were excluded.
- All three frozen decisions remain material: cross-source heterogeneity, scorer dependence, and repeated-night dependence.

The stable event total does not make the parsing defect harmless. Scorer-specific results and REM support changed, and v0.2 replaces v0.1 for interpretation.

See the [frozen correction protocol](../../docs/evaluation/multidataset_hypnogram_gx_label_code_correction_protocol_v0.2.md).

See the [corrected decision](../../docs/evaluation/multidataset_hypnogram_transition_uncertainty_decision_v0.2_2026-10-25.md).
