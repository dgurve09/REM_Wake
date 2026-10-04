# Execution and Validation Record

**Primary execution:** 2026-10-03

**Isolated rerun completed:** 2026-10-04

**Result-producing code commit:** `858f7f9`

**Authorized data:** 82 train recordings from 64 participant groups

**Validation accessed:** No

**Current test accessed:** No

## Primary run

- Completed 40 inner fits and 10 outer-final fits.
- Produced 75,539 one-time outer scores per candidate.
- Reproduced all 75,539 prior nested BLSTM-CRF control probabilities exactly; maximum absolute difference was 0.
- The independent validator passed 9/9 checks twice.
- Event metrics and quality-tier recall were recomputed from predicted events and one-to-one matches.
- SHA-256 and byte sizes were checked for 101 external model and score artifacts.

## Isolated immutable rerun

The repository was checked out at detached commit `858f7f9`. The experiment used a separate result root with read-only links to the frozen Block 7 feature arrays and prior nested-control scores. All 50 fits and all result tables were regenerated.

- The isolated validator passed 9/9 checks.
- External files matched: 101/101, with zero SHA-256 differences.
- Reviewed output files matched: 19/19, with zero SHA-256 differences.

## Preserved execution failures

Three pre-result failures are documented in `docs/evaluation/quality_balanced_lstm_crf_execution_log_v0.1.md`:

1. duplicate `tolerance_sec` insertion during result-table assembly;
2. omission of zero-event participants from the tier bootstrap table; and
3. rejection of the first cached fits because their reconstructed control standardization did not reproduce the frozen control.

The invalid pre-scaler-fix artifacts remain outside Git in a separately named failed-run directory. No failed run was interpreted as an experimental result.
