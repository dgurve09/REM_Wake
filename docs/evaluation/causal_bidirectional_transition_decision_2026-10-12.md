# Causal Bidirectional REM/Wake Transition Decision

**Date:** 2026-10-12

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Partition:** Train only; validation and current test remained closed

**Decision:** Stop the bidirectional causal detection target at v0.1; retain the direction-separability finding for later confirmation

## Question tested

The experiment tested whether a past-only detector benefits from treating both REM-to-Wake and Wake-to-REM as positive transition events, rather than training only on REM-to-Wake. A separate classifier tested whether transition direction can be recovered when the true event checkpoint is already known.

## Causal formulation

Each decision used eight contiguous 30-second wearable feature windows ending at the decision time. The fixed positive decision was 15 seconds after the nominal PSG-derived boundary, when the complete 30-second annotation-uncertainty interval had elapsed. No future signal was used.

The support audit retained 180 REM-to-Wake and 61 Wake-to-REM events. Transition 406 was excluded before fitting because a feature-grid gap prevented the full causal context.

## Detection result

At the fixed 0.25 false-alarms/hour selection budget, the REM-to-Wake-only model reached F1 0.0517. The bidirectional model reached F1 0.0387 on the same REM-to-Wake evaluation. The F1 difference was -0.0130, below the required +0.03. The bidirectional model reduced false alarms/hour by 0.0556 but also detected fewer events.

Neither model matched a Wake-to-REM event. Combining the two transition directions therefore did not improve causal full-night event discovery under this formulation.

## Direction result

At true transition checkpoints, the direction classifier reached macro F1 0.8290 and balanced accuracy 0.8272. It correctly identified 165/180 REM-to-Wake and 45/61 Wake-to-REM events in participant-grouped outer out-of-fold evaluation.

This separates two technical issues. Direction-specific information is present in the preceding wearable signal when the event location is supplied, but the current full-night detector does not reliably localize the rare transition endpoint among background candidates.

The Stage A inner-selected thresholds varied by several orders of magnitude across participant folds, indicating unstable probability calibration even though the fits passed the numerical gate. The direction result may also use recent REM-versus-Wake occupancy rather than a distinct transition waveform; that mechanism remains unresolved.

## Numerical and validation controls

All 55 fits converged and passed the predeclared numerical controls. Independent validation passed 13/13 checks twice and verified 57 external artifacts. The result is therefore a stable negative detection comparison with a positive direction-separability mechanism result, not another solver-saturation failure.

## Next decision

Stop H-BI at v0.1 and do not select new timing windows, class weights, thresholds, or losses from these outer results. The direction classifier may be retained as a secondary component only after a future detector provides independently confirmed event candidates.
