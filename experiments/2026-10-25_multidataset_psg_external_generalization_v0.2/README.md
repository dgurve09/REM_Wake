# Multi-Dataset PSG External-Generalization Correction v0.2

## Current stage

The correction protocol, amended pre-fit checkpoint, 36-fit rerun, independent reconstruction, and v0.1-to-v0.2 comparison are complete.

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

## Corrected result

| Held-out evaluation | Candidate | Precision | Recall | F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|
| BOAS, descriptive | `PSD3-BASE` | 0.2192 | 0.1562 | 0.1824 | 0.2051 |
| BOAS, descriptive | `ADV3-CHANGE` | 0.3290 | 0.2082 | 0.2550 | 0.1566 |
| gx36m | `PSD3-BASE` | 0.1151 | 0.1882 | 0.1429 | 0.2477 |
| gx36m | `ADV3-CHANGE` | 0.1532 | 0.2235 | 0.1818 | 0.2114 |
| ISRUC group 3 | `PSD3-BASE` | 0.5000 | 0.1562 | 0.2381 | 0.0771 |
| ISRUC group 3 | `ADV3-CHANGE` | 0.5238 | 0.3438 | 0.4151 | 0.1542 |
| Pooled primary external | `PSD3-BASE` | 0.1409 | 0.1795 | 0.1579 | 0.2280 |
| Pooled primary external | `ADV3-CHANGE` | 0.2069 | 0.2564 | 0.2290 | 0.2048 |

The corrected pooled F1 difference is `+0.0711`. Its participant-bootstrap 95% interval is `-0.0177` to `+0.1396`; the improvement therefore remains participant-inconclusive. The corrected point gate passes, but advanced recall is only 0.2564 and 87/117 external events were missed.

All three safely parsed gx36m expert variants and both ISRUC experts had nonnegative advanced-minus-baseline F1 point differences. This is more consistent than v0.1, but it is not evidence of reliable detection because event recall remains low and uncertainty spans no improvement.

## Verification

- 36/36 fixed logistic fits converged.
- 13/13 in-run controls passed.
- 14/14 independent result checks passed.
- 42/42 external model and score artifacts passed size and SHA-256 verification.
- 1,075 alarms, eight event summaries, 260 participant rows, ten scorer-sensitivity rows, the bootstrap, and the gate were independently reconstructed.
- The v0.1-to-v0.2 comparison passed 10/10 checks.

The first comparison execution stopped because boolean gate columns were subtracted as numbers. Casting numeric comparisons to float corrected the report without changing a fitted model or result.

See the [frozen correction protocol](../../docs/evaluation/multidataset_psg_consensus_correction_protocol_v0.2.md).

See also the [pre-fit amendment](../../docs/evaluation/multidataset_psg_consensus_correction_amendment_2026-10-25.md).

See the [corrected decision](../../docs/evaluation/multidataset_psg_external_generalization_decision_v0.2_2026-10-25.md).
