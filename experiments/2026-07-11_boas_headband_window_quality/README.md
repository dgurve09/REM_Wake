# BOAS Headband Window Signal-Quality Assessment

**Work date:** 2026-07-11
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Protocol:** `docs/feasibility/headband_window_signal_quality_protocol_v0.1.md`
**Model training performed:** No

## 1. Question

Do the reviewed REM/Wake transition and background windows contain gross `HB_1` or `HB_2` amplitude failures that change the feasibility decision or require exclusion before grouped split design?

## 2. Result

| Window source | Include | Review | Exclude |
|---|---:|---:|---:|
| Transition | 99 | 362 | 15 |
| Background review | 2702 | 1580 | 20 |

- Channel-window measurements: 9,556
- Window decisions: 4,778
- Windows with review flags: 1,942
- Windows with critical exclusion flags: 35
- Unique `pid` values with a review or exclusion: 100
- Windows flagged only by the nonspecific 10-MAD rule: 1,561
- Windows with a targeted amplitude, jump, or endpoint review flag: 381
- Primary REM-to-Wake windows retained after critical exclusions: 350 of 365
- Primary REM-to-Wake windows critically excluded: 15
- Unique `pid` values retaining at least one primary REM-to-Wake window: 88

## 3. Decision

The critical failure rate requires revision of the preprocessing candidate set.

Review flags are not silently converted into exclusions. They remain available for sensitivity analysis and targeted visual inspection during the label/preprocessing gate.

The predeclared 10-MAD rule proved too nonspecific as a stand-alone review criterion because raw EEG is heavy-tailed and the rule reached all participant groups. Its result is preserved as `mad_only_review`, but it is not used as an exclusion. Targeted review is instead prioritized using amplitude range, peak-to-peak, abrupt-jump, and repeated-endpoint flags.

## 4. Outputs

| File | Purpose |
|---|---|
| `headband_channel_window_metrics_v0.1.tsv` | Per-channel amplitude and continuity measurements |
| `headband_window_signal_decisions_v0.1.tsv` | Combined two-channel decision for each reviewed window |
| `recording_window_signal_quality_summary_v0.1.tsv` | Counts by recording, source, and decision |
| `headband_window_signal_quality_summary_v0.1.tsv` | Aggregate decision counts |

## 5. Limitations

- Operational thresholds are conservative engineering screening rules, not validated clinical EEG-quality criteria.
- Endpoint repetition is only a clipping proxy because the physical device saturation limits are not independently validated here.
- A review flag does not prove that a window is unusable; later sensitivity analysis must test whether conclusions change when reviewed windows are excluded.
- This assessment does not evaluate model performance.
