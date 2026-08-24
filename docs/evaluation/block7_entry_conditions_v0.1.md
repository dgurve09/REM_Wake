# Block 7 Entry Conditions v0.1

**Created:** 2026-08-23
**Block:** Paired PSG-to-wearable transfer, 2026-08-24 to 2026-09-06
**Status:** Must be satisfied before Block 7 model fitting

## 1. Current Evidence Boundary

The frozen project test partition was first opened for the planned stage-first comparison and was later reused for a fixed direct-model comparison after direct models and thresholds were frozen. The direct result is retained, but the partition is no longer an independent confirmatory cohort. DE-D was designed after those results and remains validation-only.

Block 7 may use the existing split for a single paired, descriptive device-shift comparison because PSG and wearable recordings are simultaneous and participant matched. It must not describe that result as independent confirmation of DE-D or use current-test behavior to choose features, channels, thresholds, adaptation methods, or model architecture.

## 2. Question to Freeze

How much event-level performance changes when the same fixed transition method is trained or applied using full PSG, reduced PSG, and real wearable EEG, and is simple adaptation justified by a documented zero-shot transfer failure?

The block is about modality and device shift, not a search for the best detector.

## 3. Required Pre-Run Protocol

Before fitting, record:

- exact full-PSG, reduced-PSG, and wearable channel sets;
- one shared label, quality-membership, split, preprocessing, event-matching, and metric definition;
- the fixed method family and any modality-specific input dimensions;
- train-only fitting and validation-only threshold selection;
- zero-shot/no-adaptation evaluation before direct wearable fitting or fine-tuning;
- the allowed fine-tuning rule and the evidence that would justify it;
- missing-channel handling and recording exclusions;
- one paired participant-level comparison rule; and
- external artifact paths, hashes, seeds, software versions, and intended result files.

## 4. Test and Selection Rules

1. All feature, architecture, optimization, and threshold decisions use train and validation only.
2. The existing test partition may be opened once after every Block 7 configuration is frozen and applied to all fixed modality comparators together.
3. The paired test result is descriptive evidence about device shift, not a fresh confirmatory estimate.
4. No method may be revised because of the Block 7 test result.
5. A new locked dataset partition or external cohort is required for a later confirmatory performance claim.
6. Block 9 external data must not be chosen or filtered based on favorable performance.
7. Keep `sub-32`, `sub-39`, and `sub-50` in the primary analysis. Report any fixed sensitivity without these drift-review recordings separately; do not use it for model selection or automatic exclusion.

## 5. Gate Decision

- **Keep simple approach:** zero-shot or direct wearable performance does not show a defined device-shift limitation.
- **Test one adaptation method:** a predeclared paired comparison shows a material transfer deficit and identifies a specific mismatch that the method addresses.
- **No-go:** channel, label, or cohort incompatibility prevents a scientifically interpretable paired comparison.

Any outcome is retained with its configuration, result, limitation, and next decision.
