# BOAS Full-Dataset Alignment Drift Audit v0.1

**Work date:** 2026-08-23
**Protocol:** `docs/evaluation/alignment_drift_audit_plan_v0.1.md`
**Input:** Saved July 4 `HB_PULSE` versus `PSG_PULSE` lag estimates
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Available pulse windows | 505 |
| Usable pulse windows | 383 |
| Recordings with at least three usable windows | 82 |
| Median slope | 0.0197 sec/hour |
| Median projected lag change | 0.1250 sec |
| 95th percentile absolute projected change | 1.4781 sec |
| Recordings above the 2-second review threshold | 3 |

The review recordings are: sub-32, sub-39, sub-50.

## Interpretation

Drift-like change was not widespread under this proxy: 79 of 82 analyzable recordings stayed within the 2-second projected-change screen. The flagged recordings are retained for review rather than excluded. Pulse waveform differences and cross-correlation instability can produce an apparent slope, so this result does not prove clock drift or exact sample synchronization.

The unchanged primary alignment evidence remains the matching EDF start time, sampling rate, sample count, duration, and extraction indices. This audit narrows the earlier limitation by explicitly quantifying change across the night.

## Post Hoc Frozen-Result Sensitivity

Two review recordings, `sub-32` and `sub-50`, are in the test partition. Removing only those recordings from the saved primary +/-15-second summaries reduced SF-C to F1 0.0732 and 1.9741 false alarms/hour and DE-B to F1 0.1384 and 1.3045 false alarms/hour. The directional DE-B versus SF-C comparison therefore persists.

This sensitivity was defined after the drift flags and does not replace the frozen result or justify exclusion. Its method boundary is recorded in `docs/audit/alignment_flagged_recording_sensitivity_note_v0.1.md`.
