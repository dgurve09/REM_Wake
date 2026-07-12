# Signal Quality Flags v0.1

**Created:** 2026-07-09
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Transition labels:** `labels/transition_labels_v0.1/`
**Background windows:** `labels/background_windows_v0.1/`
**Model training performed:** No

## 1. Purpose

This artifact creates recording-level and window-level quality flags for the label/preprocessing gate.

The uncertainty addressed here is whether the transition and background windows can be carried forward with explicit critical flags and signal-proxy notes before any train/validation/test split or model work.

## 2. Method

- Critical flags use PSG label structure, PSG/headband sample mapping, and window geometry.
- Pulse and EEG-envelope lag summaries are retained as signal-proxy notes.
- Proxy notes do not automatically exclude windows because they are not ground-truth synchronization markers.
- Background review windows inherit recording-level critical flags and are checked for 240-second sample geometry and REM/Wake uncertainty separation.

## 3. Result

| Artifact | Rows | Include for preprocessing | Review before modeling |
|---|---:|---:|---:|
| Recordings | 128 | 128 | 0 |
| Transition windows | 476 | 476 | 0 |
| Background review windows | 4302 | 4302 | 0 |

## 4. Outputs

| File | Purpose |
|---|---|
| `recording_signal_quality_flags_v0.1.tsv` | Recording-level critical flags and signal-proxy status |
| `transition_window_quality_flags_v0.1.tsv` | Window-level flags for REM/Wake transition labels |
| `background_window_quality_flags_v0.1.tsv` | Window-level flags for deterministic background review candidates |
| `quality_flag_summary_v0.1.tsv` | Count summary by artifact |

## 5. Limitations

- These flags do not yet compute full amplitude artifact metrics from every headband channel.
- EEG-envelope and pulse lag statuses are review proxies, not exclusion rules by themselves.
- Final split creation remains blocked until the background sampling policy and these flags are reviewed together.

## 6. Decision

Use this artifact for the label/preprocessing gate. Model training remains blocked.
