# Validation Decision and Test Freeze v0.1

**Created:** 2026-08-15
**Marker:** `FROZEN_FOR_SINGLE_TEST_EVALUATION`

The frozen train/validation run completed without changing features, model settings, split, quality membership, event derivation, or matching tolerance.

| Comparator | Validation macro F1 | Validation primary event F1 (+/-15 s) |
|---|---:|---:|
| SF-B epoch-only | 0.445999 | 0.020408 |
| SF-C five-epoch context | 0.496873 | 0.051020 |

Temporal context improved validation macro F1: **True**. Temporal context improved validation event F1: **True**. The prespecified H5.3 is supported only if the relevant metric improved; both outcomes are retained.

## Frozen Test Decision

Evaluate both SF-B and SF-C once on the untouched test recordings. SF-C remains the prespecified primary stage-first comparator and SF-B remains its ablation regardless of the validation ordering. No configuration or threshold will be revised after seeing test results.
