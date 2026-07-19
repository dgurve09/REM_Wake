# Quality Analysis-Membership Specification v0.1

**Created:** 2026-07-18
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Input quality artifact:** `signal_quality_flags_v0.3`
**Frozen split:** `grouped_pid_split_v0.1`
**Model training performed:** No
**Raw signal data read:** No

## 1. Remaining Uncertainty

Signal-quality v0.3 leaves four deterministic outcomes: `include`, `include_mad_sensitivity`, `review_targeted`, and `exclude_critical`. The targeted-review outcome identifies possible high amplitude, abrupt jumps, or repeated endpoint values, but it is not a validated unusable-signal label. A final rule is required before the label/preprocessing gate can close.

Aggregate v0.3 counts were already known before this specification was written. This work therefore evaluates analysis-set consequences of existing flags; it is not a blinded test of new signal-quality thresholds.

## 2. Candidate Policies

### A. Include every noncritical window in the primary analysis

This maximizes event retention but treats unresolved targeted indicators as equivalent to clean signal. It is insufficient because later performance could depend on windows already identified as possible artifact cases.

### B. Remove every targeted-review window from the project

This creates a conservative primary set but discards unresolved windows as though their unusability were established. It is insufficient because the targeted thresholds are engineering screening rules, not calibrated physiological or device-failure criteria.

### C. Separate primary and quality-sensitivity membership

Use clean and 10-MAD-only windows in the primary analysis. Reserve targeted-review windows for a prespecified quality-sensitivity analysis. Exclude critical windows from both. This preserves a conservative primary comparison while retaining the unresolved cases for a direct robustness test.

Policy C is selected for v0.1.

## 3. Deterministic Mapping

| v0.3 preprocessing decision | Primary eligible | Quality-sensitivity eligible | Membership tier |
|---|---:|---:|---|
| `include` | Yes | Yes | `primary_clean` |
| `include_mad_sensitivity` | Yes | Yes | `primary_mad_flagged` |
| `review_targeted` | No | Yes | `quality_sensitivity_only` |
| `exclude_critical` | No | No | `excluded_critical` |

The 10-MAD-only outcome remains primary eligible because the July 11 assessment found that rule to be nonspecific across all participant groups. The flag remains available for stratified reporting.

## 4. Split Boundary

- Membership is derived from existing tables only; no EDF is opened.
- The frozen `pid` assignment is not changed.
- Validation and test signals are not visually adjudicated.
- No label, quality, preprocessing, or split decision may be revised using later model performance.
- Any future manual signal review must use a new version and cannot silently replace this artifact.

## 5. Required Outputs

1. One membership row for each of the 476 transition windows.
2. One membership row for each of the 4,302 background review windows.
3. Counts by partition, label class, quality decision, and membership tier.
4. Primary REM-to-Wake event and positive-`pid` counts by partition.
5. Checks for one-to-one row preservation, complete split assignment, participant leakage, and deterministic mapping.

The background table remains a review pool. Membership does not convert it into the final sampled negative training set.
