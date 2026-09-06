# Block 7 Feature-Generation Validation v0.1

**Work date:** 2026-09-06
**Protocol commit:** `1f6797f`
**Feature-plan commit:** `71ddc92`
**Initial implementation commit:** `6c38a64`
**Execution code commit:** `dfbb3bf`
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Partition processed:** Train only
**Model training performed:** No
**Validation or test signals accessed:** No

## Result

| Check | Result |
|---|---:|
| Train `pid` groups | 64 |
| Train recordings | 82 |
| Train signal-path checks | 164/164 |
| Synthetic spectral checks | 5/5 |
| PSG-6/PSG-2 overlap maximum absolute difference | 0 |
| Wearable reproduction maximum absolute difference | 4.76619863576e-07 |
| Recording feature/context checks | 82/82 |
| Gate checks | 13/13 |
| Gate decision | **pass** |

The gate tests whether the three Block 7 feature paths are mechanically comparable. It does not test event performance or device transfer. Full feature arrays remain outside Git; their relative paths and SHA-256 values are retained in the external manifest.

No validation or test recording, feature, label row, score, model, threshold, alarm, or metric was accessed.
