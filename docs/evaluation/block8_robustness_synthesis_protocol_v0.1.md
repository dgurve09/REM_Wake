# Block 8 Robustness Synthesis Protocol v0.1

**Protocol date:** 2026-09-18
**Project block:** Block 8 signal and channel robustness
**Analysis role:** Closeout synthesis of five frozen Block 8 experiments
**New model fitting or scoring:** None
**Threshold selection:** None
**Current test access:** Prohibited

## 1. Question

Do the completed Block 8 experiments support advancing any wearable REM-to-Wake detector, and which unresolved technical uncertainty should govern the next planned work?

This synthesis separates three concepts that cannot be treated as equivalent:

- a mechanism finding, where a controlled perturbation or comparison behaves as hypothesized;
- a robustness-method advance, where a method mitigates one identified failure; and
- a core wearable-detector advance, where clean event performance improves without unacceptable alarm burden and without requiring PSG at inference.

## 2. Frozen Evidence Sources

The synthesis will use compact reviewed outputs from these five completed experiments:

1. single-channel feature-contribution robustness;
2. raw-signal Gaussian-noise robustness;
3. train-only noise augmentation;
4. cross-modality event overlap; and
5. fixed alarm fusion.

For every experiment, the synthesis will hash the relevant decision or comparison table and all applicable in-run, phase, or independent integrity-check tables. Key event-metric and participant-bootstrap tables may be read only to verify the reported effect direction and uncertainty.

No raw EDF, feature array, model object, candidate-score table, full probability table, or current-test result may be opened.

## 3. Integrity Gate

The five experiments contain 132 applicable integrity checks:

| Experiment | Expected passing checks |
|---|---:|
| Single-channel robustness | 26 |
| Raw-signal noise | 29 |
| Noise-augmented training | 38 |
| Cross-modality overlap | 19 |
| Fixed alarm fusion | 20 |
| **Total** | **132** |

The synthesis cannot close Block 8 unless all 132 recorded checks are present and passing, every source file matches its recorded SHA-256, and no source path points to a current-test artifact.

## 4. Fixed Evidence Ledger

The analysis will reconstruct the following evidence statements directly from source tables:

| Evidence ID | Statement to test |
|---|---|
| B8-E1 | `HB_1` feature neutralization causes a material clean F1 loss |
| B8-E2 | Controlled mild both-channel noise passes the fixed material bounds |
| B8-E3 | Severe both-channel noise eliminates primary true detections |
| B8-E4 | Isolated `HB_1` noise creates a channel-specific high-alarm failure |
| B8-E5 | Train-only noise augmentation mitigates the `HB_1` failure |
| B8-E6 | Noise augmentation fails the meaningful clean-advancement gate |
| B8-E7 | Direct PSG recovers a material subset missed by direct wearable |
| B8-E8 | Boundary tolerance materially changes direct-model union recall |
| B8-E9 | Fixed OR fusion worsens both F1 and false-alarm burden |
| B8-E10 | Fixed 2-of-3 laboratory consensus suppresses false alarms but has participant-inconclusive F1 gain |

Each ledger row will record the source experiment, exact source values, fixed threshold or interval, result, relevance to a wearable detector, and remaining limitation.

## 5. Predeclared Block-Level Decisions

| Decision | Pass condition |
|---|---|
| C8.1 wearable robustness sufficient | No material single-channel failure, no severe-noise detection collapse, and no channel-specific FAR failure |
| C8.2 deployable wearable method advances | A method passes its clean-performance gate, does not materially increase FAR, and uses wearable inputs only at inference |
| C8.3 boundary uncertainty receives priority | A predeclared boundary-tolerance gate passes with participant direction supporting a positive effect |
| C8.4 Block 8 can close | All five experiments are present, all 132 integrity checks pass, stop/advance decisions are explicit, and unresolved limitations are carried forward |

C8.1-C8.3 are technical decisions. C8.4 is a completeness decision and must not be presented as evidence of detector success.

## 6. Interpretation Rules

- Passing a controlled mild-noise gate cannot override a severe-noise or channel-specific failure.
- Mitigating one degraded condition cannot advance the core detector if clean F1 fails its predeclared gate.
- PSG-dependent consensus cannot count as a deployable wearable method.
- A point F1 improvement with a participant interval crossing zero is inconclusive for event-performance advancement.
- A consistently lower false-alarm rate may support a mechanism even when F1 advancement is inconclusive.
- Sensitivity to a wider event tolerance identifies label-timing uncertainty but does not itself improve the detector.
- No Block 8 result may authorize current-test reopening or threshold revision.

## 7. Planned Outputs

The synthesis will produce:

- a source-artifact manifest with hashes;
- an integrity-check census by experiment;
- the ten-row evidence ledger;
- the four block-level decisions;
- an unresolved-uncertainty register;
- a Block 8 closeout report; and
- an independent reconstruction report.

The unresolved register will distinguish immediate scheduled work from later conditional model development. It will not create work dated after 2026-09-18 or claim that future experiments have occurred.

## 8. Closeout Boundary

Closing Block 8 means the planned robustness questions have been investigated and their limitations documented. It does not mean the wearable detector is reliable.

If C8.2 fails, the project must retain the current wearable result as inadequate and stop further Block 8 threshold, augmentation, or fusion variations. Block 9 may proceed with its scheduled external-PSG compatibility audit. Boundary-uncertainty and wearable-only temporal-representation work remain separate future protocols with a new locked or external confirmation requirement.
