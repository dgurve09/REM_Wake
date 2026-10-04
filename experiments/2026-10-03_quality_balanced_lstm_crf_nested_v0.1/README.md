# Quality-Balanced LSTM-CRF Nested Experiment v0.1

This directory contains compact reviewed outputs for the frozen 2026-10-03 protocol.

| Pipeline | F1 | Recall | Precision | False alarms/hour | Clean recall | MAD-flagged recall |
|---|---:|---:|---:|---:|---:|---:|
| LC-1-NESTED | 0.1604 | 0.1778 | 0.1461 | 0.2971 | 0.0377 | 0.2362 |
| LC-QB1 | 0.1435 | 0.1889 | 0.1156 | 0.4130 | 0.0755 | 0.2362 |

Decision: `stop_v0.1`.

This is reused-cohort train-only model development, not independent confirmation. Full scores and model states remain outside Git under `REM_W_data`.
