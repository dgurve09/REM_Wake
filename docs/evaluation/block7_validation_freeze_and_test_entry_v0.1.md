# Block 7 Validation Freeze and Test Entry v0.1

**Created:** 2026-09-06
**Validation result commit:** `15dc9b4`
**Zero-shot hypothesis analysis commit:** `69391f2`
**Freeze marker:** `BLOCK7_VALIDATION_MODELS_THRESHOLDS_AND_ADAPTATION_DECISION_FROZEN`
**Block 7 test data accessed:** No

## 1. Validation Decision

The train/validation phase is complete. Reduced PSG produced the strongest validation event result, strict PSG-to-wearable transfer lost F1 relative to direct reduced PSG, and the predeclared robust-alignment branch remained closed.

| Comparator | Model SHA-256 | Threshold | Validation F1 | False alarms/hour | Test role |
|---|---|---:|---:|---:|---|
| `P6-D` | `243ab382909b30f8288bc24dc1e22b205fcdcea77c2011c2a3c2a452f7969082` | 0.99 | 0.1631 | 1.1107 | Frozen six-channel PSG reference |
| `P2-D` | `a3176329a142c4569a36813b08107b11a0acc4634ef47eb7e983b85fc51f7e51` | 0.99 | 0.1887 | 0.3702 | Frozen reduced-PSG reference |
| `H2-D` | `d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066` | 0.96 | 0.1123 | 1.4558 | Frozen direct wearable reference |
| `P2-H2-Z` | Uses the frozen `P2-D` model | 0.99 inherited | 0.0685 | 0.6526 | Frozen strict zero-shot transfer |

`P2-H2-A` is not a test comparator. The validation performance condition opened, but the distribution condition closed at 11/80 shifted dimensions versus the required 16/80. Running alignment despite that result would violate the frozen protocol.

## 2. Interpretation Frozen Before Test

- H7.1 is not supported on validation: `P6-D` did not improve event F1 over `P2-D` and produced more false alarms.
- H7.2 is supported for event-F1 loss relative to the direct reduced-PSG source comparator. The alarm-burden difference versus `P2-D` remains inconclusive.
- Relative to direct wearable fitting, zero-shot transfer has lower F1 but fewer false alarms, so uniform degradation is not supported.
- H7.3 was not evaluated because its two-part entry gate did not open.

These interpretations cannot be revised to accommodate the descriptive test result.

## 3. Test Entry Conditions

The one permitted Block 7 test opening must:

1. use the existing frozen 20-`pid`, 26-recording test assignment without moving a participant;
2. generate `PSG-6`, `PSG-2`, direct `HB-2`, and PSG-scaled `HB-2` features with the committed Block 7 feature code and train-fitted scalers;
3. verify exact onset, stage, context-center, feature-schema, and overlapping-PSG parity before scoring;
4. load the three models only after their SHA-256 values match this freeze;
5. apply the four fixed thresholds shown above without selection, calibration, or refitting;
6. score every comparator on the same supported test boundaries and evaluate them together;
7. retain primary and expanded membership results at +/-15 and +/-45 seconds;
8. label every result descriptive because the test partition was already used in Blocks 5 and 6;
9. preserve the result regardless of whether it confirms, reverses, or weakens the validation pattern; and
10. perform an independent output-integrity validation before closing Block 7.

No feature selection, threshold revision, model retraining, alignment, fine-tuning, error-driven exclusion, or replacement comparator is authorized after test access.

## 4. Validation Evidence

- 17/17 result-producing checks passed.
- 23/23 independent output-integrity checks passed twice.
- All 413 external input/output artifacts were rehashed.
- All 162 newly generated feature arrays and three fitted models were reopened.
- Model probabilities, threshold curves, event outputs, paired intervals, feature-shift statistics, and gate arithmetic were independently reproduced.

This freeze authorizes only the fixed descriptive test execution. A new locked partition or external cohort is required for a later confirmatory performance claim.
