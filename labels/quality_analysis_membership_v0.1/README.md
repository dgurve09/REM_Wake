# Quality Analysis Membership v0.1

**Created:** 2026-07-18
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/quality_analysis_membership_spec_v0.1.md`
**Input quality artifact:** `signal_quality_flags_v0.3`
**Frozen split:** `grouped_pid_split_v0.1`
**Model training performed:** No
**Raw signal data read:** No

## Result

Targeted-review windows are excluded from the primary analysis but retained in an expanded quality-sensitivity analysis. Critical windows are excluded from both. The nonspecific 10-MAD-only tier remains primary eligible and separately identifiable.

| Partition | Primary REM-to-Wake events | Primary positive `pid` | Expanded events | Expanded positive `pid` |
|---|---:|---:|---:|---:|
| Train | 180 | 47 | 227 | 56 |
| Validation | 37 | 10 | 51 | 14 |
| Test | 59 | 15 | 70 | 18 |
| Total | 276 | 72 | 348 | 88 |

The primary set contains 276 events across 72 participant groups. The expanded quality-sensitivity set contains 348 events across 88 participant groups. The 72 targeted-review primary events therefore account for 16 participant groups with no clean or 10-MAD-only primary event.

The smaller primary validation set, 37 events across 10 positive participant groups, must be reported as a precision limitation. The expanded analysis tests whether conclusions depend on the conservative targeted-review exclusion.

## Outputs

| File | Purpose |
|---|---|
| `transition_analysis_membership_v0.1.tsv` | Row-level membership for all transition labels |
| `background_analysis_membership_v0.1.tsv` | Row-level membership for the background review pool |
| `membership_summary_v0.1.tsv` | Counts by artifact, split, quality decision, and tier |
| `primary_event_balance_v0.1.tsv` | Primary REM-to-Wake counts for each analysis set |
| `primary_pid_coverage_v0.1.tsv` | Per-participant primary, expanded, and targeted-only event coverage |
| `background_balance_v0.1.tsv` | Background review-pool counts for each analysis set |

## Decision

Use the primary analysis membership for the first baseline. Run the expanded quality-sensitivity membership as a prespecified robustness comparison. Do not use model performance to alter these tiers or the frozen participant split.
