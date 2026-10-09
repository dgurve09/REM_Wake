# Full-Night Risk-Set Formulation Experiment v0.1

Train-only participant-grouped nested results under the predeclared alarm-budget protocol.

| Candidate | Precision | Recall | F1 | False alarms/hour | Average precision | Brier score |
|---|---:|---:|---:|---:|---:|---:|
| SAMP-BAL | 0.0526 | 0.0500 | 0.0513 | 0.2574 | 0.0242 | 0.061054 |
| RISK-BAL | nan | 0.0000 | 0.0000 | 0.0000 | 0.0492 | 0.035599 |
| RISK-NAT | 0.1480 | 0.1611 | 0.1543 | 0.2653 | 0.0719 | 0.002708 |

Full risk set: 73,656 rows, including 180 primary events.
Convergence: 44 of 50 new fits reached the iteration limit; failures were retained.

The frozen decisions are recorded in `hypothesis_decisions_v0.1.tsv`.
Validation and current-test data remained closed. This retrospective formulation test does not establish real-time performance or resolve 30-second label uncertainty.
