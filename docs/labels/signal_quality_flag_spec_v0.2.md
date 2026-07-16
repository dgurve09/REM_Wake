# Signal-Quality Flag Specification v0.2

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Supersedes for current decisions:** `signal_quality_flag_spec_v0.1.md`
**Model training performed:** No

## 1. Purpose

Version 0.1 checked label structure, PSG-to-headband sample mapping, window geometry, and synchronization proxies. The July 11 full headband assessment then measured raw `HB_1` and `HB_2` amplitude and continuity, but those decisions remained in a separate experiment artifact.

Version 0.2 integrates both evidence sources so later split and preprocessing work uses one explicit decision per window.

## 2. Inputs

- structural recording/window flags from `labels/signal_quality_flags_v0.1/`;
- amplitude/continuity decisions from `experiments/2026-07-11_boas_headband_window_quality/`;
- 476 transition labels from `labels/transition_labels_v0.1/`;
- 4,302 deterministic background review windows from `labels/background_windows_v0.1/`.

No threshold is re-estimated in this integration step.

## 3. Combined Decision Rules

| Decision | Rule | Later use |
|---|---|---|
| `exclude_critical` | Structural critical flag or amplitude decision `exclude` | Exclude from preprocessing candidates |
| `review_targeted` | No critical flag and amplitude priority `targeted_review` | Retain for targeted review and sensitivity analysis |
| `include_mad_sensitivity` | Only the nonspecific 10-MAD rule is present | Include, but preserve for sensitivity analysis |
| `include` | No critical or review flag | Include |

The 10-MAD rule is not treated as an exclusion because the July 11 assessment showed that it flagged 2,663 of 9,556 channel-window measurements and reached every participant group. That negative result remains recorded rather than being deleted.

## 4. Required Integrity Checks

- exactly 476 transition rows and 4,302 background rows must join one-to-one;
- subject and `pid` must agree between structural and amplitude artifacts;
- every row must receive one combined decision;
- primary REM-to-Wake and secondary Wake-to-REM counts must remain separate;
- excluded windows must remain in the artifact with their reasons.

## 5. Boundary

This version revises quality decisions only. It does not select final background samples, assign train/validation/test partitions, preprocess EEG, train a model, or estimate performance.
