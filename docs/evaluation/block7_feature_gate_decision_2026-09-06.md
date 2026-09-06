# Block 7 Feature-Gate Decision

**Decision date:** 2026-09-06
**Protocol:** `block7_paired_transfer_protocol_v0.1.md`
**Validation plan:** `block7_feature_generation_validation_plan_v0.1.md`
**Partition processed:** Train only
**Model training performed:** No
**Validation or test data accessed:** No

## Decision

**Pass the feature-infrastructure gate and proceed to the fixed Block 7 train/validation transfer comparison.**

This decision authorizes model fitting under the existing Block 7 protocol. It does not establish a device-shift result, event-detection improvement, or Block 7 completion.

## Evidence

| Evidence | Result |
|---|---:|
| Train recordings | 82 |
| Train `pid` groups | 64 |
| Signal-path checks | 164/164 passed |
| Synthetic spectral checks | 5/5 passed |
| Per-recording feature/context checks | 82/82 passed |
| `PSG-6`/`PSG-2` overlap maximum absolute difference | 0 |
| `HB-2` reproduction maximum absolute difference | `4.76619863576e-7` |
| External feature artifacts | 246 hashed |
| Primary gate checks | 13/13 passed |
| Independent output-integrity checks | 15/15 passed |

The independent validator reconstructed train membership from the frozen split, confirmed that retained rows were train-only, verified exact ordered feature schemas, rehashed and reopened every external feature array, and recomputed cross-modality timing, PSG overlap, wearable reproduction, context, and manifest linkage.

## Technical Interpretation

The fixed preprocessing and feature implementation does not introduce a detectable difference between the overlapping `F3/F4` PSG paths. The wearable path also reproduces the earlier frozen train features within the declared float32 tolerance. Later differences among `P6-D`, `P2-D`, `H2-D`, and zero-shot transfer can therefore be interpreted as model/input-condition results rather than known feature-order or preprocessing-path inconsistencies.

This evidence does not show that the modalities are physiologically equivalent. `PSG_F3/F4` and `HB_1/HB_2` differ in electrode location and acquisition hardware, which is the unresolved Block 7 question.

## Next Decision Point

Run direct `PSG-6`, direct `PSG-2`, direct `HB-2`, and strict `PSG-2` to `HB-2` zero-shot comparisons on train/validation under the frozen protocol. Apply the one permitted robust alignment only if both predeclared validation gate conditions open. Keep the current test partition closed until all models, hashes, thresholds, comparator roles, and the alignment decision are frozen.
