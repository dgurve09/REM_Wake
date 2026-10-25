# Multi-Dataset Hypnogram Transition-Uncertainty Correction v0.2

## Current stage

The gx36m label-code correction protocol is frozen. No corrected transition result has been generated.

## Correction

The source documents numeric codes `0..4`. The v0.1 loader converted the 818 observed `-1` labels to `REM` through negative array indexing. Version 0.2 maps every out-of-vocabulary value to `UNKNOWN`, then applies the same strict 2-of-3 known-stage majority.

All v0.1 artifacts remain preserved. Ten other sources and all analysis rules remain unchanged.

See the [frozen correction protocol](../../docs/evaluation/multidataset_hypnogram_gx_label_code_correction_protocol_v0.2.md).
