# Multi-Dataset PSG External-Generalization Correction v0.2

## Current stage

The correction protocol, amendment, and pre-fit checkpoint are frozen. No corrected fit or performance result exists yet.

## Reason for correction

The v0.1 gx36m primary-label function converted three-way scorer ties to `W` through label-order `argmax`. The stated rule requires a strict 2-of-3 majority and therefore requires these epochs to be `UNKNOWN`.

The first pre-fit audit then stopped on an expected-count mismatch before writing outputs. It found 818 raw scorer labels with undocumented code `-1`; an earlier label-only loader had converted these to `REM` through negative indexing. The frozen amendment requires out-of-vocabulary codes to remain `UNKNOWN` in both primary and scorer-specific labels.

## Fixed scope

- Reuse the exact 196 recordings, 130 participant groups, and verified signal-feature caches.
- Change only gx36m primary consensus.
- Repeat all 36 fits because gx36m participates in training or evaluation in every outer run.
- Preserve all v0.1 artifacts and write models, scores, and reviewed tables to v0.2 locations.
- Reapply the original fixed advancement gate and report a direct v0.2-minus-v0.1 comparison.

## Pre-fit checkpoint

- The primary audit passed 15/15 checks; independent validation passed 13/13 checks.
- All 196 feature-cache sizes and SHA-256 hashes matched the v0.1 records.
- Strict consensus changed 1,479 gx36m primary epochs to `UNKNOWN` and reduced primary gx36m events from 124 to 85; the complete target pool changed from 521 to 482 events.
- The 818 raw `-1` labels occurred only in experts 2 and 3. Safe parsing changed expert-3 events from 115 to 108; expert-1 and expert-2 event counts remained 156 and 56.
- All 260 reconstructed v0.1 participant-to-fold memberships were frozen. Every corrected inner fold still contains positive and negative examples from both training datasets.
- BOAS and ISRUC labels were unchanged, and no v0.2 model or score artifact existed at the checkpoint.

See the [frozen correction protocol](../../docs/evaluation/multidataset_psg_consensus_correction_protocol_v0.2.md).

See also the [pre-fit amendment](../../docs/evaluation/multidataset_psg_consensus_correction_amendment_2026-10-25.md).
