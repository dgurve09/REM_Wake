# Multi-Dataset PSG External-Generalization Correction v0.2

## Current stage

The correction protocol is frozen. No corrected fit or performance result exists yet.

## Reason for correction

The v0.1 gx36m primary-label function converted three-way scorer ties to `W` through label-order `argmax`. The stated rule requires a strict 2-of-3 majority and therefore requires these epochs to be `UNKNOWN`.

## Fixed scope

- Reuse the exact 196 recordings, 130 participant groups, and verified signal-feature caches.
- Change only gx36m primary consensus.
- Repeat all 36 fits because gx36m participates in training or evaluation in every outer run.
- Preserve all v0.1 artifacts and write models, scores, and reviewed tables to v0.2 locations.
- Reapply the original fixed advancement gate and report a direct v0.2-minus-v0.1 comparison.

See the [frozen correction protocol](../../docs/evaluation/multidataset_psg_consensus_correction_protocol_v0.2.md).
