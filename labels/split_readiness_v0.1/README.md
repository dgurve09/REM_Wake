# Split Readiness v0.1

**Created:** 2026-07-09
**Applies to:** `transition_labels_v0.1`, `background_windows_v0.1`, and `signal_quality_flags_v0.1`
**Status:** Readiness review only; no train/validation/test split assigned
**Model training performed:** No

## 1. Purpose

This artifact reviews whether participant-grouped split design is feasible after transition labels, background rules, and quality flags are available.

The uncertainty addressed here is leakage risk and group balance: BOAS has repeated recordings for some `pid` values, so future evaluation must group by `pid`, not by recording folder.

## 2. Result

| Item | Value |
|---|---:|
| `pid` values with any reviewed window | 100 |
| `pid` values with primary REM-to-Wake labels | 88 |
| `pid` values with background review windows | 100 |
| `pid` values with both primary positives and background windows | 88 |
| Background-only `pid` values | 12 |
| `pid` values with repeated recordings | 20 |
| Primary REM-to-Wake labels | 365 |
| Secondary Wake-to-REM labels | 111 |
| Background review windows | 4302 |
| Windows requiring quality review | 0 |

The repeated-recording count here is based on all reviewed windows after adding background candidates. The earlier transition-label draft counted repeated `pid` values only among recordings with transition labels.

## 3. Outputs

| File | Purpose |
|---|---|
| `pid_split_readiness_v0.1.tsv` | Participant-level counts for future grouped split design |
| `split_readiness_summary_v0.1.tsv` | Overall readiness count summary |

## 4. Decision

Do not assign final splits yet. The next split-design step should choose a deterministic policy that preserves all recordings from each `pid` in one partition and reports event/background balance before model work.
