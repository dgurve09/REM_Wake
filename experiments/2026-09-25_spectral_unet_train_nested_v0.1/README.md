# Spectral U-Net Train-Only Nested Experiment v0.1

This directory contains compact reviewed outputs for the frozen 2026-09-25 protocol.

| Pipeline | F1 | Recall | Precision | False alarms/hour |
|---|---:|---:|---:|---:|
| Nested BLSTM-CRF | 0.1604 | 0.1778 | 0.1461 | 0.2971 |
| Spectral U-Net | 0.1389 | 0.2667 | 0.0939 | 0.7355 |

Decision: `stop_v0.1`.

This is reused-cohort train-only model development, not independent confirmation. Raw EDFs, generated tensors, model states, and full-night scores remain under `REM_W_data` outside Git.
