# Signal-Quality Flag Specification v0.3

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Supersedes for current decisions:** `signal_quality_flag_spec_v0.2.md`
**Model training performed:** No

## 1. Trigger

Minimal preprocessing v0.1 failed for two retained train transitions because the extracted signals contained fewer than the required 61,440 input samples. Review found eight transition labels with incomplete 240-second coverage.

The v0.1 structural rule checked whether PSG and headband windows had equal lengths but did not check whether that equal length was the required 240 seconds. Six of the eight incomplete windows were already excluded by amplitude/flatline rules. Two were incorrectly retained.

## 2. New Critical Rule

For every transition and background window:

```text
window_stop_sample - window_start_sample = 61,440 samples
```

Any other length receives `coverage:incomplete_240s_signal_coverage` and `exclude_critical`.

No signal padding, duplication, extrapolation, or asymmetric-window substitution is allowed in the primary artifact. Those alternatives would change the defined input context and require a separate sensitivity specification.

## 3. Unchanged Rules

- Structural, alignment, amplitude, targeted-review, and 10-MAD sensitivity evidence from v0.2 is preserved.
- Existing critical exclusions remain in the table.
- Primary REM-to-Wake and secondary Wake-to-REM remain separate.
- Excluded rows are retained with their reasons.

## 4. Split Boundary

The participant assignment remains `splits/grouped_pid_split_v0.1/`. The two newly identified failures belong to the train partition. The split is not regenerated because train signals have now been inspected; changing assignments could move inspected participants into the locked validation or test partitions.

## 5. Required Checks

- 476 transition and 4,302 background rows retained in the artifact;
- eight transition rows identified with incomplete coverage;
- two newly excluded relative to v0.2;
- all background review windows retain full coverage;
- participant leakage remains zero;
- retained primary-event participant count is recalculated.
