# Block 7 Channel Compatibility Audit v0.1

**Work initiated:** 2026-08-25
**Audit executed:** 2026-08-26
**Protocol commit:** `1f6797f`
**Code commit:** `d73a2784395a`
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Frozen recording pairs | 128 |
| Complete pair-level checks passed | 128 |
| `PSG-6` eligible recordings | 128 |
| `PSG-2` eligible recordings | 128 |
| `HB-2` eligible recordings | 128 |
| Distinct PSG channel configurations | 7 |
| Distinct headband channel configurations | 3 |
| Gate decision | **pass** |

The common PSG-6, reduced PSG-2, and wearable HB-2 channel sets cover the complete frozen cohort. Optional sensor heterogeneity does not require exclusions.

The full common PSG comparator is six-channel EEG, not every clinical PSG sensor. Optional EOG, respiratory, pulse, oxygen-saturation, and wearable motion channels vary across recordings and remain excluded from the fixed Block 7 input sets. The reduced `F3/F4` to `HB_1/HB_2` mapping preserves laterality and feature dimension, but the electrode locations are not equivalent; subsequent transfer results combine device and location shift.

Raw EDF files remain outside Git. The retained tables contain public BOAS identifiers and technical channel metadata only.
