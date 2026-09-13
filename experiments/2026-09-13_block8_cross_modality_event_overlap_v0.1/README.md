# Block 8 Cross-Modality Event Overlap v0.1

**Work date:** 2026-09-13
**Protocol commit:** `a7ac70f`
**Partition:** Reused development validation only
**Model fitting or threshold selection:** None
**Test data accessed:** No

## Primary Result

Across 37 primary references at +/-15 seconds, the direct models produced 13 both, 7 PSG-only, 3 wearable-only, and 14 shared misses. Six-channel PSG recall was 0.5135, two-channel PSG recall was 0.2703, wearable recall was 0.4324, and their non-deployable union recall was 0.6216.

The union improved over the best individual direct-model recall by 0.1081. Expanding tolerance to +/-45 seconds changed union recall by +0.1081.

## Frozen Decisions

- `H8.10_shared_direct_failure`: fail (observed 0.3784; gate 0.50)
- `H8.11_wearable_specific_recovery_gap`: pass (observed 0.1892; gate 0.15)
- `H8.12_direct_model_complementarity`: pass (observed 0.1081; gate 0.10)
- `H8.13_boundary_tolerance_sensitivity`: pass (observed 0.1081; gate 0.10)

## Boundary

The union is a recoverability diagnostic, not an ensemble result, because its false-alarm burden was not evaluated. This analysis cannot improve or validate the detector. It identifies whether the next uncertainty concerns shared representation, boundary timing, or wearable-specific information loss.

All 9/9 in-run checks passed. The current test partition remained closed.
