# Block 8 Participant Evidence-Strength Decision

**Decision date:** 2026-09-18
**Protocol commit:** `777e641`
**Implementation commit:** `2842623`
**Result commit:** `687a858`
**New fitting, scoring, resampling, or threshold selection:** None
**Current test accessed:** No

## Question

Which Block 8 conclusions are supported by their full stored participant uncertainty ranges, rather than only by aggregate point estimates?

## Result

Twenty-five predeclared claims were reconstructed from ten compact source tables:

| Evidence grade | Claims |
|---|---:|
| Gate supported by uncertainty | 8 |
| Point gate only | 8 |
| Gate failure supported by uncertainty | 4 |
| Point failure with gate overlap | 4 |
| No direct interval for the exact contrast | 1 |

The primary analysis passed 11/11 checks. The independent reconstruction rehashed all ten sources, rebuilt the 25 claim values and classifications, and passed 10/10 checks. An immutable rerun changed no file.

## Participant-Supported Findings

The full stored participant range supported these passing gates:

- material F1 loss after `HB_1` feature neutralization;
- FAR preservation under mild both-channel noise;
- material F1 loss under severe both-channel noise;
- material FAR increase under isolated `HB_1` noise;
- clean FAR preservation under augmentation;
- material FAR reduction under augmentation for isolated `HB_1` noise;
- no FAR increase under laboratory consensus; and
- a material alarm penalty under all-direct OR fusion.

Four failures were also supported by their full ranges:

- isolated `HB_2` noise did not produce the hypothesized FAR increase;
- augmentation did not reach the clean `+0.05` F1 advancement gate;
- two-modality OR did not reach its F1 advancement gate; and
- two-modality OR exceeded its allowed FAR increase.

## Point-Only Findings

Eight passing aggregate gates did not have a full participant range beyond the gate. These include mild-noise F1 preservation, the material `HB_1` noise F1 loss, clean-F1 preservation under augmentation, augmented `HB_1` F1 improvement, PSG-only recovery, direct-model union gain, boundary-tolerance materiality, and consensus F1 noninferiority.

This does not reverse the original point decisions. It limits their interpretation to aggregate mechanism screens within the reused validation cohort.

All evidence-strength labels apply only within that reused cohort. They are not independent confirmation and do not compensate for the absence of a new locked wearable cohort.

The H8.8 joint degradation-gap gate has no direct stored participant interval for the exact difference-in-differences contrast. Separate within-model intervals cannot be substituted for that missing contrast.

## Boundary Interpretation

The boundary-tolerance gain remains directionally positive: its participant interval is +0.0238 to +0.2400. However, the lower bound does not clear the original `+0.10` materiality gate. The appropriate conclusion is therefore:

- boundary timing remains a justified prioritized uncertainty; but
- material boundary gain is not participant-supported at the predeclared threshold.

## Decision

Retain the Block 8 closeout and do not advance a deployable wearable method. The clean augmentation advancement failure is supported because its participant interval, -0.0438 to +0.0132, remains entirely below the required `+0.05` gain. PSG-dependent consensus remains nondeployable regardless of its FAR result.

Do not revise any original threshold or point decision. Carry the evidence grades forward so later summaries distinguish point-gate findings from participant-supported findings. Keep Block 9 unstarted before 2026-09-21 and keep the current test partition closed.
