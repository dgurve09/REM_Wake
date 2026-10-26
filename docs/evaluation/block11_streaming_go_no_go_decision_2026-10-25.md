# Block 11 Wearable Streaming Go/No-Go Decision

**Decision date:** 2026-10-25
**Decision:** `no_go`
**Prototype executed:** No
**Latency or memory benchmark executed:** No

## Question

Does the accepted offline evidence justify entering the conditional wearable streaming-prototype block?

The project proposal requires a final justified offline model with interpretable and stable event performance. The frozen assessment used each source experiment's primary result and advancement status; it did not create a new post-result numeric threshold.

## Evidence

The direct two-channel wearable model reached validation F1 0.1123, precision 0.0645, recall 0.4324, and 1.4558 false alarms/hour. On the already-used descriptive test, it reached F1 0.1493, precision 0.0906, recall 0.4237, and 1.2621 false alarms/hour. Neither result advanced a deployable method.

Block 8 explicitly failed the wearable-robustness condition because channel dependence and controlled-noise failures remained material. Its deployability decision was `do_not_advance`.

Train-only candidates do not satisfy independent confirmation. LSTM-CRF reached F1 0.1965, enriched elastic-net features reached F1 0.1796, and the causal peak candidate reached F1 0.1125 with recall 0.1000. Their source decisions require a new confirmation cohort.

The corrected external PSG advanced representation reached F1 0.2290 and recall 0.2564, but it is not wearable evidence and its F1-difference interval includes zero. The eleven-source label analysis also found material source, scorer, and repeated-night dependence.

## Gate result

| Entry condition | Result |
|---|---|
| Wearable candidate independently advanced | Fail |
| Wearable event performance interpretable as ready | Fail |
| Wearable robustness supported | Fail |
| Causal wearable method independently confirmed | Fail |
| Boundary can be reported without exact-timing claim | Pass, using coarse scored-boundary intervals and explicit uncertainty |

Because four required conditions fail, the streaming decision is `no_go`.

## Consequence

Do not build or benchmark a wearable streaming prototype in Block 12. Latency and memory measurements would characterize software execution, not detector validity, and could be misread as readiness evidence when no final wearable model has advanced.

The no-go does not end the research project. The next permitted scientific model step is a separately frozen paired PSG-to-wearable transfer study using the corrected PSG candidate and participant-grouped evaluation. Until such evidence exists, retain offline reproducibility work and document the Block 12 no-go rather than implementing a demonstration.

## Verification

- Reviewed source artifacts hashed: 13/13.
- In-run evidence checks: 8/8 passed.
- Independent validation checks: 13/13 passed.
- Prototype, model fitting, and performance tuning performed in this assessment: none.
