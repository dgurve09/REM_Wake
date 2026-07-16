# Signal Quality Flags v0.2

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/signal_quality_flag_spec_v0.2.md`
**Model training performed:** No

## 1. Purpose

This artifact integrates structural quality flags v0.1 with the predeclared full headband amplitude/continuity assessment. It provides one current preprocessing decision per reviewed transition and background window without deleting excluded or inconclusive cases.

## 2. Result

| Artifact | Include | Include with 10-MAD sensitivity flag | Targeted review | Critical exclusion |
|---|---:|---:|---:|---:|
| Primary REM-to-Wake | 76 | 201 | 73 | 15 |
| Secondary Wake-to-REM | 23 | 69 | 19 | 0 |
| Background review | 2702 | 1291 | 289 | 20 |

- Primary REM-to-Wake windows retained before targeted-review sensitivity decisions: 350
- `pid` values retaining at least one primary REM-to-Wake window: 88
- Recordings retained with window-level critical exclusions: 12

## 3. Interpretation

The v0.1 statement that every reviewed window could be included is superseded for current preprocessing decisions. Critical amplitude failures now exclude 15 primary transition windows and 20 background review windows. The feasibility conclusion remains unchanged because 350 primary windows remain across all 88 contributing `pid` groups.

The 10-MAD-only outcome remains visible but does not trigger exclusion. Targeted amplitude, jump, and endpoint flags remain separate so later sensitivity analysis can test their effect rather than treating uncertain windows as automatically clean or unusable.

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_window_quality_flags_v0.2.tsv` | Integrated transition-window decisions |
| `background_window_quality_flags_v0.2.tsv` | Integrated background-window decisions |
| `recording_signal_quality_summary_v0.2.tsv` | Recording-level counts and retained status |
| `quality_flag_summary_v0.2.tsv` | Counts by artifact and combined decision |

## 5. Decision

Use v0.2, not v0.1 alone, for grouped split design and the label/preprocessing gate. No final split is assigned in this artifact, and model training remains blocked.
