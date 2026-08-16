# Fixed Stage-First Comparator v0.1

**Created:** 2026-08-15
**Comparator:** BOAS headband `stage_ai` (`SF-A`)
**Ground truth:** PSG human consensus `stage_hum`
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Model trained in this experiment:** No

## Readiness

- Event-table pairs checked: 128
- Event-table alignment passes: 128 of 128
- Valid stage-comparison epochs: 114,890
- Predicted REM-to-Wake events: 389

## Stage Diagnostics

| Partition | Epochs | Macro F1 | Balanced accuracy | Cohen kappa |
|---|---:|---:|---:|---:|
| Train | 74,012 | 0.7103 | 0.6961 | 0.7604 |
| Validation | 17,792 | 0.7469 | 0.7182 | 0.7810 |
| Test | 23,086 | 0.7248 | 0.7064 | 0.7863 |

## Primary Event Result (+/-15 seconds)

| Partition | Reference | Predicted | TP | FP | FN | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 180 | 257 | 44 | 201 | 136 | 0.1796 | 0.2444 | 0.2071 | 0.3272 |
| Validation | 37 | 55 | 10 | 43 | 27 | 0.1887 | 0.2703 | 0.2222 | 0.2912 |
| Test | 59 | 77 | 25 | 50 | 34 | 0.3333 | 0.4237 | 0.3731 | 0.2609 |

## Execution Record

The initial execution stopped before any metric file was written because the wrapper attempted to add a `tolerance_sec` column that was already returned by the evaluator. The output-assembly guard was corrected and the frozen scientific configuration was not changed. This was an implementation failure, not evidence about the research hypothesis.

## Interpretation Boundary

The fixed headband stage sequence is a useful stage-first comparator, but its training provenance and independence from BOAS are not established by the dataset files. It is not human ground truth and cannot replace the participant-independent transparent baselines.
