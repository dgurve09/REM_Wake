# Quality-Balanced LSTM-CRF Execution Log v0.1

## 2026-10-03 first execution

All 50 declared nested fits completed and their model and score artifacts were written outside Git. Reviewed-output assembly then failed before any experimental result was accepted.

Failure:

- `evaluate_events` already returned `tolerance_sec` in recording-level and participant-level tables.
- The experiment wrapper attempted to insert the same column while adding configuration identifiers.
- Pandas raised `ValueError: cannot insert tolerance_sec, already exists`.

Resolution:

- Add configuration columns only when the evaluation table does not already contain that column.
- Do not change models, folds, seeds, thresholds, scoring, or the predeclared decision rule.
- Reuse the immutable cached models and scores, then run the complete independent validator.

This failure concerns result-table assembly only. It does not support or reject the scientific hypothesis.
