# Stage-First Context Diagnostics v0.1

**Created:** 2026-08-15
**Plan:** `docs/evaluation/stage_first_context_diagnostic_plan_v0.1.md`
**Status:** Exploratory after primary results
**Raw signals, feature arrays, or models opened:** No
**Frozen input tables hashed:** 11

## Execution Record

The initial execution completed but emitted warnings because participant-level balanced accuracy used different class denominators when a participant lacked one or more human stages. Before interpreting the diagnostic results, that field was replaced with fixed-five-stage macro recall using labels 0-4 and zero contribution for absent stages. The original issue is preserved here; no event, timing, fragmentation, or bootstrap definition changed.

## Participant-Paired Context Effect

On test participants, SF-C improved stage macro F1 for 20 of 20, reduced it for 0, and left it unchanged for 0. Event F1 improved for 10, declined for 0, and was unchanged for 10. False alarms/hour decreased for 11 participants and increased for 9.

The paired participant bootstrap estimated an SF-C minus SF-B test event-F1 difference with median +0.0431 and exploratory 95% interval [+0.0199, +0.0676]. The false-alarms/hour difference had median -0.0576 and interval [-0.5028, +0.3445].

## One-Epoch Timing Direction

Under the already-frozen +/-45-second matching sensitivity, SF-C test eligible matches comprised 18 exact matches, 6 predictions one epoch early, and 1 one epoch late. This describes the additional timing-tolerance matches without rematching or changing the primary endpoint.

## Fragmentation Association

For SF-C test recordings, predicted REM bouts/hour had Spearman rho 0.5118 with false alarms/hour. Predicted all-stage transitions/hour had rho 0.3983. Mean predicted REM-bout duration had rho -0.0687. These correlations are descriptive, unadjusted, and not causal.

## REM-Bout Duration Distribution

In SF-C test sequences, 0.5920 of predicted REM bouts lasted only 30 or 60 seconds, compared with 0.1410 of human REM bouts over the same valid coverage. This distribution supports the previously observed fragmentation mechanism and shows that the median difference is not produced by a single extreme recording.

## Interpretation

Context reduces fragmentation relative to SF-B and improves aggregate event F1, but the benefit is not uniform across participants and does not remove the false-alarm problem. The additional +/-45-second matches include explicit early or late one-epoch errors, while the recording-level associations and bout-duration distribution support widespread short-run fragmentation.

No primary result, model, tolerance, threshold, or quality membership was changed. No later project phase was started.
