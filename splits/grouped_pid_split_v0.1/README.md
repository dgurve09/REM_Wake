# Grouped Participant Split v0.1

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/splits/grouped_pid_split_spec_v0.1.md`
**Quality artifact:** `labels/signal_quality_flags_v0.2/`
**Search seed:** `20260715`
**Candidate assignments searched:** 50,000
**Selected balance score:** 0.79274736
**Model training performed:** No

## 1. Selection-Time Result

The assignment below records the v0.2 quality counts used when selecting the split. It is retained as the decision-time evidence.

| Partition | `pid` | Recordings | Positive `pid` | Primary retained | Secondary retained | Background retained | Repeated `pid` | Critical-history `pid` | F/M | Median age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 64 | 82 | 56 | 229 | 75 | 2761 | 13 | 8 | 38/26 | 31 |
| Validation | 16 | 20 | 14 | 51 | 16 | 651 | 3 | 2 | 11/5 | 31 |
| Test | 20 | 26 | 18 | 70 | 20 | 870 | 4 | 2 | 11/9 | 28 |

No `pid` occurs in more than one partition. All recordings belonging to a repeated `pid` inherit the same assignment.

## 2. Interpretation

The assignment is balanced using pre-model participant, label, and quality counts only. It does not use raw signal values, learned features, predictions, or performance results. The test partition is now locked and must not guide preprocessing, threshold selection, model selection, or error-driven revisions.

Critical-exclusion history remains represented in every partition. Windows marked `exclude_critical` are not counted as retained events or backgrounds, but their rows remain in the quality artifact.

## 3. Outputs

| File | Purpose |
|---|---|
| `pid_split_assignments_v0.1.tsv` | Frozen partition for each `pid` and its recordings |
| `split_balance_summary_v0.1.tsv` | Actual versus proportional target for every balance metric |
| `split_search_diagnostics_v0.1.tsv` | Twenty lowest valid candidate scores |

## 4. Decision

Use this split for the first stage-first and direct-transition comparisons after the label/preprocessing gate passes. Any later change requires a new version and a reason independent of test performance.

## 5. Quality v0.3 Update

Minimal preprocessing identified two additional incomplete primary train windows. Quality v0.3 excludes them, reducing current retained primary counts to 227/51/70 in train/validation/test and 348 overall. The participant assignment is unchanged: the failures were found after train signals were inspected, so regenerating the split could move inspected participants into validation or test.

Current counts are stored in `current_quality_balance_v0.3.tsv`. Validation and test signals remain untouched.
