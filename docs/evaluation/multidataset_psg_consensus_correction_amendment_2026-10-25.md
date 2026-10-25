# Multi-Dataset PSG Consensus-Correction Pre-Fit Amendment

**Date frozen:** 2026-10-25
**Parent protocol:** `multidataset_psg_consensus_correction_protocol_v0.2.md`
**Model status:** No v0.2 model, probability, alarm, or result table exists

## Reason

The first v0.2 pre-fit audit stopped because it found 1,479 unresolved gx36m primary epochs rather than the 1,398 expected from the label-only v0.1 result. Investigation showed that 818 raw scorer labels use code `-1`. The gx36m README documents only codes `0..4`. The label-only v0.1 loader used negative array indexing and converted `-1` to `REM`.

The PSG v0.1 primary consensus did not use that negative indexing, but it still converted every unresolved vote pattern to a known stage through unrestricted `argmax`. Its scorer-sensitivity path did use negative indexing. Both primary and scorer-specific PSG labels therefore require correction.

## Amended label rule

For every gx36m scorer, map only raw codes `0..4` to the five known stages and map all other values to `UNKNOWN`. The primary consensus requires at least two votes for the same known stage. A scorer-specific `UNKNOWN` epoch cannot form a transition.

This amendment does not change signal data, feature caches, candidate times, features, participant groups, model settings, threshold selection, alarm logic, metrics, bootstrap, or the advancement gate.

## Fold preservation

The original fold allocator balances event counts and supported hours. Corrected labels would therefore be capable of changing participant fold membership even with the same participants. To isolate the label correction, reconstruct and freeze the exact v0.1 participant-to-fold mapping before fitting. Apply that mapping to corrected labels and verify that every corrected inner fold retains both classes from both training datasets.

## Required audit additions

The repeated pre-fit audit must distinguish:

- all raw `-1` scorer labels;
- three-way or otherwise unresolved known-stage votes;
- primary consensus epochs changed from the v0.1 implementation;
- gx36m scorer-specific transitions after safe parsing; and
- exact preservation of v0.1 participant fold membership.

The stopped audit and its failed `1,398` expectation remain described in the experiment record. No result-driven setting change is permitted.
