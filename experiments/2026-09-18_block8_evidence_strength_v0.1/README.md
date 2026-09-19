# Block 8 Participant Evidence Strength v0.1

**Work date:** 2026-09-18
**Protocol commit:** `777e641`
**Claims reviewed:** 25
**New fitting, scoring, resampling, or threshold selection:** None
**Current test accessed:** No

## Evidence Grades

- 8 gates are supported by the full stored participant range.
- 8 gates pass only at the aggregate point estimate.
- 4 gate failures are supported by the full stored participant range.
- 4 failed point gates have ranges that overlap the gate.
- One joint degradation-gap claim has no direct stored difference-in-differences interval.

## Block Decisions

- `deployable_wearable_method_advances`: do_not_advance - clean F1 advancement gate is ruled out by the stored participant interval; PSG fusion is nondeployable
- `boundary_uncertainty_priority`: retain_priority - boundary effect is directionally positive, although the interval does not clear the 0.10 material gate
- `boundary_materiality_participant_supported`: not_supported - participant interval lower bound is above zero but below the predeclared 0.10 material gate
- `block8_closeout`: remain_closed - evidence-strength analysis changes interpretation strength but does not create a wearable advance

## Main Interpretation

The clean +0.05 F1 advancement gate for noise augmentation is ruled out by its participant interval, so wearable non-advancement is strengthened. The boundary-tolerance effect remains directionally positive, but its participant interval does not clear the original +0.10 materiality gate. Boundary uncertainty therefore remains a justified priority without being described as participant-supported material gain.

This analysis grades existing evidence only. It does not replace the original point decisions or reopen Block 8 development.
