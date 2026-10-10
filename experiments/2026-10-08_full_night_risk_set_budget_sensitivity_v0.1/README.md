# Full-Night Risk-Set Alarm-Budget Sensitivity v0.1

Train-only participant-grouped nested out-of-fold sensitivity results. This report applies the four predeclared fold-specific thresholds to frozen outer scores; it performs no refitting or new threshold selection.

| Budget (/h) | Candidate | Precision | Recall | F1 | False alarms/hour |
|---:|---|---:|---:|---:|---:|
| 0.10 | RISK-BAL | undefined | 0.0000 | 0.0000 | 0.0000 |
| 0.10 | RISK-NAT | 0.1791 | 0.0667 | 0.0972 | 0.0874 |
| 0.10 | SAMP-BAL | 0.0000 | 0.0000 | 0.0000 | 0.1017 |
| 0.25 | RISK-BAL | undefined | 0.0000 | 0.0000 | 0.0000 |
| 0.25 | RISK-NAT | 0.1480 | 0.1611 | 0.1543 | 0.2653 |
| 0.25 | SAMP-BAL | 0.0526 | 0.0500 | 0.0513 | 0.2574 |
| 0.50 | RISK-BAL | 0.1188 | 0.0667 | 0.0854 | 0.1414 |
| 0.50 | RISK-NAT | 0.1343 | 0.2500 | 0.1748 | 0.4607 |
| 0.50 | SAMP-BAL | 0.1206 | 0.2111 | 0.1535 | 0.4400 |
| 1.00 | RISK-BAL | 0.0967 | 0.2778 | 0.1435 | 0.7419 |
| 1.00 | RISK-NAT | 0.1093 | 0.4000 | 0.1716 | 0.9325 |
| 1.00 | SAMP-BAL | 0.1154 | 0.4333 | 0.1822 | 0.9500 |

Metrics use the primary event membership and the +/-15 s tolerance. The alarm budget constrains inner-fold threshold selection; held-out outer-fold false-alarm rates can differ from the nominal budget.

These developmental train-partition results do not establish real-time or clinical performance. Validation and test partitions remain closed.
