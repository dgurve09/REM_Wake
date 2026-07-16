# Grouped Participant Split Specification v0.1

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Quality artifact:** `labels/signal_quality_flags_v0.2/`
**Model training performed:** No

## 1. Purpose

This specification freezes a participant-independent development split before model preprocessing or training. BOAS contains repeated recordings for some `pid` values, so every recording from one `pid` must remain in one partition.

## 2. Partition Sizes

| Partition | `pid` count | Purpose |
|---|---:|---|
| Train | 64 | Parameter fitting |
| Validation | 16 | Threshold and model-selection decisions |
| Test | 20 | Final locked evaluation |

The test partition must not be used for preprocessing-parameter fitting, threshold selection, model selection, or iterative error-driven revision.

## 3. Eligible Counts

- Retain `include`, `include_mad_sensitivity`, and `review_targeted` windows for split-balance counting.
- Exclude `exclude_critical` windows from eligible event/background counts.
- Keep primary REM-to-Wake and secondary Wake-to-REM counts separate.
- Preserve targeted-review, 10-MAD sensitivity, and critical-exclusion histories in the assignment table.

This split balances candidate availability. It does not decide whether targeted-review windows enter the primary analysis; that remains a label/preprocessing-gate decision.

## 4. Deterministic Search

- Random generator: NumPy `PCG64`.
- Seed: `20260715`.
- Candidate assignments: `50,000`.
- Exact partition sizes: 64 train, 16 validation, and 20 test `pid` values.
- Select the valid candidate with the lowest predeclared balance score.

The score compares each partition with its proportional target using only:

- recording count;
- retained primary and secondary transition counts;
- retained background review count;
- positive-participant and background-only-participant counts;
- repeated-recording participant count;
- female participant count;
- age-band counts: under 30, 30-49, 50 or older, and missing;
- primary/background targeted-review counts;
- participant count with at least one critical exclusion.

Primary event count and positive-participant balance receive the highest weights. No raw signal value, spectral feature, model prediction, or performance result enters the search.

## 5. Hard Constraints

- All recordings from a `pid` remain in one partition.
- Train, validation, and test contain both recorded sex categories.
- Each partition contains at least one background-only `pid`, one repeated-recording `pid`, and one `pid` with a critical window exclusion.
- Retained-primary positive `pid` minimums are 55 train, 13 validation, and 17 test.
- Subject, `pid`, and quality joins must be complete and one-to-one at their declared level.

## 6. Demographic Handling

Sex is required to be consistent across repeated recordings for a `pid`. Age and BMI can differ between repeated recording rows, so the assignment table records their median and observed range. Missing age remains an explicit balance category.

Demographics are used only to avoid a grossly unbalanced partition; no subgroup outcome is inspected.

## 7. Revision Rule

Once generated, v0.1 remains frozen through baseline comparison. A revision requires a new split version, a stated technical reason independent of test performance, and retention of the superseded assignment.
