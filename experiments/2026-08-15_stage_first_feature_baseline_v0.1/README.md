# Transparent Stage-First Feature Baseline v0.1

**Created:** 2026-08-15
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Models:** SF-B epoch-only logistic regression; SF-C five-epoch-context logistic regression

## Validation Result

| Comparator | Macro F1 | Balanced accuracy | Cohen kappa | Primary event F1 (+/-15 s) |
|---|---:|---:|---:|---:|
| SF-B | 0.4460 | 0.5384 | 0.3628 | 0.0204 |
| SF-C | 0.4969 | 0.6052 | 0.4163 | 0.0510 |

## Fit Record

| Comparator | Train epochs | Features | Iterations used | Convergence warnings | Decision |
|---|---:|---:|---:|---:|---|
| SF-B | 76,316 | 10 | 316 | 0 | pass |
| SF-C | 75,872 | 50 | 500 | 1 | convergence_warning_retained |

## Frozen Test Result

| Comparator | Macro F1 | Balanced accuracy | Cohen kappa | Primary event F1 (+/-15 s) |
|---|---:|---:|---:|---:|
| SF-B | 0.4382 | 0.5360 | 0.3874 | 0.0337 |
| SF-C | 0.4979 | 0.6080 | 0.4513 | 0.0766 |

## Interpretation Boundary

This is a transparent stage-first comparator, not the proposed direct boundary detector. Poor event performance despite useful stage metrics would identify a concrete limitation of deriving boundary alarms from independently classified 30-second stages.

SF-C reached the frozen 500-iteration ceiling and its convergence warning is retained. The model was not refitted or altered after validation; this limits claims about an optimum but does not erase the observed fixed-comparator result.

Feature arrays and fitted models are stored outside Git. Their paths and SHA-256 hashes are recorded in `external_artifact_manifest_v0.1.tsv`.
