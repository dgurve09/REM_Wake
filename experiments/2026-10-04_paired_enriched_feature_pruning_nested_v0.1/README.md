# Paired Enriched-Feature and Pruning Experiment v0.1

Train-only participant-grouped nested results using the frozen feature and pruning protocol.

| Pipeline | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| P2-F0-L2 | 0.1172 | 0.3444 | 0.1749 | 0.7419 |
| P2-F1-L2 | 0.1568 | 0.3833 | 0.2226 | 0.5894 |
| P2-F1-EN | 0.1631 | 0.3778 | 0.2278 | 0.5544 |
| H2-F0-L2 | 0.0575 | 0.3278 | 0.0978 | 1.5362 |
| H2-F1-L2 | 0.0701 | 0.2778 | 0.1120 | 1.0532 |
| H2-F1-EN | 0.1184 | 0.3722 | 0.1796 | 0.7927 |

Convergence check: 24 of 230 fits reached the predeclared 3,000-iteration limit; 0 were outer-final fits. Settings were not changed after inspection.

The advancement decisions are recorded in `hypothesis_decisions_v0.1.tsv`.
Validation and current-test data remained closed. Enriched arrays, fitted models, and full-night scores remain under `REM_W_data` outside Git.
