# Block 8 Train-Only Noise Augmentation v0.1

**Work date:** 2026-09-12
**Protocol commit:** `540cd87`
**Result-producing code commit:** `d8e9b62`
**H2-NA train-OOF threshold:** `0.99`
**H2-NA model SHA-256:** `cccb4632e9abb0c011ff40b44c1e1ccdef5de34abf3eb4b62364bed227cfce77`
**Validation role:** Reused development partition
**Test data accessed:** No

## Primary Validation Results

| Model | Condition | Threshold | Precision | Recall | F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|---:|
| H2-D | H2-CLEAN | 0.96 | 0.0645 | 0.4324 | 0.1123 | 1.4558 |
| H2-D | H2-BOTH-20DB | 0.96 | 0.0924 | 0.2973 | 0.1410 | 0.6777 |
| H2-D | H2-BOTH-10DB | 0.96 | 0.1250 | 0.1351 | 0.1299 | 0.2196 |
| H2-D | H2-BOTH-0DB | 0.96 | 0.0000 | 0.0000 | 0.0000 | 0.0126 |
| H2-D | H2-HB1-10DB | 0.96 | 0.0220 | 0.4324 | 0.0419 | 4.4616 |
| H2-D | H2-HB2-10DB | 0.96 | 0.0667 | 0.1081 | 0.0825 | 0.3514 |
| H2-NA | H2-CLEAN | 0.99 | 0.0599 | 0.3514 | 0.1024 | 1.2801 |
| H2-NA | H2-BOTH-20DB | 0.99 | 0.0789 | 0.3243 | 0.1270 | 0.8785 |
| H2-NA | H2-BOTH-10DB | 0.99 | 0.0877 | 0.1351 | 0.1064 | 0.3263 |
| H2-NA | H2-BOTH-0DB | 0.99 | 0.0000 | 0.0000 | 0.0000 | 0.0314 |
| H2-NA | H2-HB1-10DB | 0.99 | 0.0566 | 0.2432 | 0.0918 | 0.9413 |
| H2-NA | H2-HB2-10DB | 0.99 | 0.0685 | 0.2703 | 0.1093 | 0.8534 |

## Frozen Decisions

| Hypothesis | Value 1 | Value 2 | Decision |
|---|---:|---:|---|
| H8.6_clean_preservation | -0.0099 | -0.1757 | pass |
| H8.7_hb1_failure_mitigation | +0.0500 | -3.5204 | pass |
| H8.8_degradation_gap_reduction | +0.0599 | +3.3447 | pass |
| H8.9_meaningful_clean_advancement | -0.0099 | -0.1757 | fail |
| overall_robustness_method_advance | +1.0000 | +nan | advance |
| overall_core_detector_advance | +0.0000 | +nan | do_not_replace |

## Boundary

All 11/11 validation-phase checks passed. This experiment evaluates one fixed Gaussian-noise augmentation mechanism on a reused validation cohort. It does not establish clinical performance, natural artefact robustness, or independent generalization.
