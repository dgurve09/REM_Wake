# Block 7 Transfer Validation Execution Plan v0.1

**Created:** 2026-09-06
**Depends on:** `block7_paired_transfer_protocol_v0.1.md`
**Feature gate:** Passed 2026-09-06
**Scope:** Train and validation partitions only
**Test data access authorized:** No

## 1. Purpose

The paired-transfer protocol fixes the scientific comparison and model family. This execution plan fixes the remaining computational definitions needed to run its train/validation phase without using validation outcomes to revise the method.

The four required validation comparators are:

| Comparator | Fit data | Evaluation input | Validation threshold |
|---|---|---|---|
| `P6-D` | Direct `PSG-6` train candidates | `PSG-6` | Selected on `P6-D` validation |
| `P2-D` | Direct `PSG-2` train candidates | `PSG-2` | Selected on `P2-D` validation |
| `H2-D` | Direct `HB-2` train candidates | `HB-2` | Selected on `H2-D` validation |
| `P2-H2-Z` | The fitted `P2-D` pipeline | `HB_1/HB_2` normalized by mapped PSG train channel scalers | Inherited from `P2-D` |

`P2-H2-A` is generated once only if both zero-shot gate conditions in the frozen protocol open.

## 2. Feature Construction

- Reuse the passed train arrays for `PSG-6`, `PSG-2`, and direct `HB-2`.
- Generate validation arrays with the same committed filter, resampling, epoch, spectral, channel-order, and context definitions.
- Use only the robust channel scalers already fitted from the train partition.
- For zero-shot input, map the frozen `PSG_F3` center/scale to `HB_1` and `PSG_F4` center/scale to `HB_2` before spectral construction.
- Do not derive a zero-shot normalization value, threshold, coefficient, or schema choice from wearable validation data.
- Store full validation and zero-shot feature arrays outside Git and retain paths, sizes, and SHA-256 values.

## 3. Common Temporal Support

For every recording and comparator, a candidate center is retained only when all eight required 30-second feature epochs from `t-120` through `t+90` seconds exist. Paired source-target distribution calculations use the exact intersection of `PSG-2` and zero-shot wearable context centers within each train recording.

Labeled direct fits use the frozen primary-eligible REM-to-Wake events and reviewed background candidates. A row missing required context is retained in the construction record with its drop reason. Full-night validation scoring uses every supported context center and the same contiguous-alarm consolidation and event-matching implementation as DE-B.

## 4. Fixed Models and Thresholds

Each direct comparator uses the model and threshold rule frozen in the paired-transfer protocol. The input dimensions are 240 for `P6-D` and 80 for `P2-D` and `H2-D`. Fit warnings, iteration counts, candidate counts, window diagnostics, threshold curves, event metrics, and model hashes are retained.

The direct validation threshold is selected from `0.01` to `0.99` by maximum primary event F1 at +/-15 seconds, then minimum false alarms per supported hour, maximum recall, and highest threshold. `P2-H2-Z` and any `P2-H2-A` use the selected `P2-D` threshold without target-specific threshold selection.

## 5. Zero-Shot Distribution Gate

Gate condition 1 is evaluated exactly as written in the protocol: relative to `H2-D`, zero-shot validation event F1 must be at least `0.03` lower or false alarms must be at least `0.50` per supported hour higher.

For gate condition 2, each of the 80 paired context features is calculated over all common supported train boundaries:

1. compute the PSG train median and zero-shot wearable train median;
2. concatenate the paired PSG and wearable values for that dimension;
3. calculate pooled robust scale as `1.4826 * MAD` of the concatenated values;
4. use scale `1.0` when the pooled robust scale is zero and record that replacement; and
5. calculate absolute median difference divided by the pooled robust scale.

Condition 2 opens when at least 20% of dimensions exceed `0.50` pooled robust scale units. All 80 dimension-level results are retained whether the gate opens or closes.

## 6. Conditional Robust Alignment

If both gate conditions open, compute separate source and target median and `1.4826 * MAD` values for each dimension from the common supported train-boundary matrices. Replace a zero source or target scale with `1.0` and record it. Transform zero-shot wearable features as:

`aligned = ((wearable - wearable_train_median) / wearable_train_scale) * PSG_train_scale + PSG_train_median`

Apply the unchanged fitted `P2-D` pipeline and inherited `P2-D` threshold to aligned validation features once. Wearable labels and validation results do not influence the transformation. If either gate condition closes, record `P2-H2-A` as skipped rather than substituting another method.

## 7. Phase Boundary and Decision

This run may read train and validation signals, labels, and generated features. It must not enumerate, generate, load, score, or summarize the test partition.

After the run:

1. independently validate membership, construction counts, threshold selection, event-metric recomputation, model hashes, feature hashes, zero-shot threshold inheritance, gate arithmetic, and alignment execution or skip status;
2. freeze the complete validation result and comparator roles; and
3. create a separate test-opening record before the one permitted paired descriptive test evaluation.

No validation outcome can authorize a new model family, feature set, threshold rule, or second adaptation method in this protocol version.
