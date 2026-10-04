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

## 2026-10-03 second execution

The cached rerun completed event evaluation, then stopped during participant-cluster bootstrap assembly.

Failure:

- The quality-tier participant table initially contained only participants with at least one primary event.
- The declared bootstrap samples all 64 train participants, including participants with zero clean or MAD-flagged events.
- Indexing the incomplete tier table therefore raised a `KeyError` for sampled zero-event participants.

Resolution:

- Construct the tier table over the complete frozen train participant list crossed with both quality tiers.
- Fill absent participant-tier combinations with zero reference and detected counts.
- Keep the participant sampling, model outputs, thresholds, and decision rule unchanged.

This was a participant-accounting failure before confidence intervals were produced. It does not constitute an experimental result.

## 2026-10-03 third execution

The cached rerun reached the experiment integrity checks. The exact-control reproduction check failed with a maximum absolute full-night probability difference of `0.3652666658` from the frozen September nested BLSTM-CRF scores.

Cause:

- The original control fitted and applied `StandardScaler` directly to float32 sequence arrays.
- The first comparison implementation retained the fitted center and scale but applied them through a NumPy expression that promoted the arithmetic to float64 before casting back to float32.
- Repeated optimization amplified this numerical-path difference, so the result was not an exact control even though its mathematical formula was the same.

Resolution:

- Restore `StandardScaler.transform` for all control fitting and scoring transformations.
- Keep the robust median/IQR transformation unchanged for the experimental arm.
- Archive the invalid cached fits and rerun all 50 fits; do not treat the invalid probabilities as scientific results.
