# Multi-Dataset PSG Stage-First Transition Experiment v0.1

## Current stage

The protocol and pre-fit checkpoint are frozen. No stage model or stage-first result exists.

## Purpose

Test whether the project has been formulating the task incorrectly as isolated rare-event classification. A five-stage model will produce causal stage predictions from the same corrected multi-dataset PSG features, and deterministic REM-to-Wake transitions will be compared with the frozen direct detector.

## Evidence boundary

This is an exploratory mechanism comparison on datasets whose direct-event results are already known. It cannot serve as independent confirmation and cannot establish wearable or clinical performance.

## Pre-fit checkpoint

- Primary audit: 13/13 checks passed.
- Independent input validation: 10/10 checks passed.
- Feature-cache hashes: 196/196 matched.
- Known causal stage rows: 367,697 across all five stages and all three datasets.
- Unknown decision rows: 672 BOAS, 2,958 gx36m, and 2,063 ISRUC group 3.
- Corrected REM-to-Wake events: 482/482 have the four consecutive decisions required by `STAGE-CONFIRM2`.
- Model and score artifacts present at checkpoint: none.

See the [frozen protocol](../../docs/evaluation/multidataset_psg_stage_first_protocol_v0.1.md).
