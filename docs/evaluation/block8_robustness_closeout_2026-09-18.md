# Block 8 Robustness Closeout

**Decision date:** 2026-09-18
**Project block:** Block 8 signal and channel robustness
**Protocol commit:** `64e8ca9`
**Implementation commit:** `3139481`
**Result commit:** `08c0368`
**Current test accessed:** No

## Question

Do the completed Block 8 experiments justify advancing a wearable REM-to-Wake detector, and which unresolved uncertainty should govern the next planned work?

## Evidence Reviewed

The synthesis reconstructed ten fixed evidence statements from five frozen experiments:

1. single-channel feature neutralization;
2. raw-signal Gaussian-noise degradation;
3. train-only noise augmentation;
4. cross-modality event overlap; and
5. fixed alarm fusion.

All 132/132 applicable source integrity checks passed. The synthesis then passed 9/9 internal checks and 9/9 independent reconstruction checks. An immutable rerun changed no reviewed file. A warning-only regular-expression issue was removed in commit `463c49f`; the warning-free rerun reproduced the committed result.

## Findings

| Evidence | Result | Interpretation |
|---|---:|---|
| `HB_1` feature neutralization | F1 difference -0.1123 | Material dependence on one wearable channel |
| Mild both-channel noise | F1 difference +0.0287; FAR difference -0.7781/hour | Passed the controlled material bounds, but does not represent natural artefact |
| Severe both-channel noise | 0 true detections; F1 0 | Detection collapse under the fixed stress condition |
| Isolated `HB_1` noise | F1 difference -0.0704; FAR difference +3.0058/hour | Channel-specific high-alarm failure |
| Train-only augmentation on `HB_1` noise | F1 difference +0.0500; FAR difference -3.5204/hour | Mitigated the targeted degraded condition |
| Train-only augmentation on clean validation | F1 difference -0.0099; FAR difference -0.1757/hour | Failed the clean-advancement gate |
| PSG-only recovery | 7/37 primary events; fraction 0.1892 | Direct wearable input misses a material event subset |
| Boundary tolerance | Union-recall gain +0.1081; participant lower bound +0.0238 | Stage-derived boundary timing materially affects event matching |
| Fixed PSG/wearable OR | F1 difference -0.0530; FAR difference +1.1358/hour | Complementary detections accumulate too many alarms |
| Fixed 2-of-3 laboratory consensus | F1 interval -0.0192 to +0.1323; FAR interval upper bound -0.1346/hour | Supports false-alarm suppression, not conclusive F1 gain; requires PSG |

## Decisions

| Decision | Outcome | Reason |
|---|---|---|
| Wearable robustness sufficient | Fail | Material channel dependence, severe-noise collapse, and channel-specific alarm failure remain |
| Deployable wearable method advances | Do not advance | Augmentation failed clean advancement, and the only passing fusion mechanism requires PSG |
| Boundary uncertainty receives priority | Prioritize | The fixed tolerance effect exceeded its gate and its participant interval was above zero |
| Block 8 complete | Close | All five planned experiments, decisions, limitations, and 132 integrity checks are recorded |

Closing Block 8 is a completeness decision. It is not evidence that the wearable detector is reliable.

## Knowledge Gained

The negative result is more specific than a low aggregate F1 alone. The present wearable representation is sensitive to channel identity and degradation type. Generic noise augmentation can repair one induced failure without improving clean event separability. Complementary PSG and wearable detections cannot be combined by OR without a substantial alarm penalty. Cross-view agreement suppresses alarms, but that mechanism is not deployable from wearable inputs alone.

The strongest resolved next question concerns the target boundary rather than another threshold or augmentation variation. Moving from +/-15 to +/-45 seconds materially changes which reference events are counted as recovered. That result identifies label-timing uncertainty; it does not improve precision and cannot be interpreted as detector advancement.

## Unresolved Uncertainties

- compatibility of an external PSG dataset with the current target and channel definitions;
- temporal uncertainty in stage-derived REM-to-Wake boundaries;
- whether wearable-only temporal evidence can reproduce the specificity benefit of cross-view agreement;
- performance under natural wearable artefacts rather than Gaussian stress conditions; and
- independent confirmation on a new locked or external wearable cohort.

## Next Decision Boundary

Stop Block 8 threshold, Gaussian-augmentation, and fixed-fusion variations at v0.1. Begin the scheduled Block 9 external-PSG compatibility audit no earlier than 2026-09-21. Keep interval-aware boundary analysis in Block 10, with precision and false-alarm burden included. Keep the current test partition closed and require a new confirmation boundary before claiming a wearable-method advance.
