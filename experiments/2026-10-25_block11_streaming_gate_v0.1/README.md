# Block 11 Wearable Streaming Gate v0.1

## Current stage

The evidence synthesis and independent validation are complete. The decision is `no_go`; no streaming prototype was run.

## Boundary

This assessment uses reviewed offline evidence only. It cannot improve model performance or establish clinical utility. A no-go leaves paired wearable-transfer research open while stopping the conditional software-prototype branch.

## Result

Four of five required entry conditions failed:

- no wearable candidate has independently advanced;
- accepted direct-wearable F1 was 0.1123 on validation and 0.1493 on the already-used descriptive test;
- the Block 8 wearable robustness gate failed; and
- the causal peak candidate remains train-development only, with recall 0.1000.

The label-boundary condition can be handled only by reporting a coarse scored-boundary interval and explicit annotation uncertainty. Corrected external PSG F1 0.2290 cannot substitute for wearable validation.

The eight in-run checks and thirteen independent checks passed. Thirteen compact source files were hashed. No model, probability, alarm, prototype, latency test, or memory benchmark was created by this assessment.

See the [frozen protocol](../../docs/evaluation/block11_streaming_gate_protocol_v0.1.md).

See the [decision record](../../docs/evaluation/block11_streaming_go_no_go_decision_2026-10-25.md).
