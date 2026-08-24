# End-to-End Project Audit

**Audit date:** 2026-08-23
**Scope:** Project start through Block 6 closeout
**Repository base reviewed:** `d936ea4`
**Purpose:** Check scientific consistency, data integrity, result traceability, implementation behavior, and readiness for Block 7

## 1. Audit Conclusion

The project remains scientifically coherent and worth continuing. The central question is consistently event-specific REM-to-Wake detection under coarse-label and device uncertainty, not general sleep staging or a clinical claim. Label, split, quality, and model-result totals reconcile across the reviewed artifacts, and all rerun output validators passed.

No frozen model metric or scientific decision required correction. The audit identified several evidence-boundary and repository-policy problems that required explicit correction before Block 7.

## 2. Confirmed Findings and Actions

| Priority | Finding | Action on 2026-08-23 | Residual boundary |
|---|---|---|---|
| High | The project test partition has already been inspected for stage-first and direct comparisons, so it cannot provide fresh confirmation for Block 7 or DE-D. | Added `block7_entry_conditions_v0.1.md` and updated the proposal/timeline boundary. | Existing-test Block 7 results are paired descriptive evidence; confirmation requires a new lock or external cohort. |
| Medium | The July full-alignment record sampled five points per night but did not explicitly calculate across-night lag change, despite mentioning possible drift. | Added a retrospective drift audit using the saved pulse windows. | Pulse lag is a proxy, not a direct clock measurement. Three recordings exceed the review screen. |
| Medium | Full EDF acquisition was verified only by remote byte size; full-set content hashes were absent. | Verified all 256 EDFs against official `SHA256E` git-annex keys. | Future raw-data replacement must rerun the identity audit. |
| Medium | Several records said continuous scores were outside Git, while compact labeled-row probability tables are tracked for recomputation. The privacy rule also used the overly broad phrase "participant information" despite public BOAS split metadata being tracked. | Corrected the artifact descriptions and clarified both retention and public-metadata boundaries. | Full-night score artifacts, models, features, and all direct/non-public participant information remain external; compact reviewed tables and required public pseudonymous metadata are permitted. |
| Medium | Most protocols and their results first appear in the same Git commit. | Recorded this chronology limitation and added a future protocol-first commit rule. | The documents state their execution order, but Git alone cannot independently prove within-session order for those past runs. |
| Low | The repository had no root navigation document. | Added `README.md` with status, boundaries, and primary entry points. | Keep it current at each block closeout. |
| Low | A local empty experiment directory has no tracked files. | Left it untouched because it is not part of Git history. | Remove or reuse it only through an intentional local cleanup. |

## 3. Quantitative Reconciliation

- Split assignment contains 100 unique `pid` groups and 128 recordings with no participant or recording overlap: 64/16/20 groups and 82/20/26 recordings in train/validation/test.
- Primary membership contains 180/37/59 REM-to-Wake events, totaling 276 across 47/10/15 positive groups.
- Expanded membership contains 227/51/70 events, totaling 348 across 56/14/18 positive groups.
- The earlier 229/51/70 split-balance count predates two train-window coverage exclusions; the frozen split was not changed after those exclusions.
- Full timeline and sample mapping passed for 128/128 EDF pairs and 476/476 transition windows.
- Stage-first, direct-event, factorization, participant, threshold-robustness, endpoint-contribution, event-matching, and Block 4 artifact validators all passed on rerun.
- All 33 unique DOI references in project Markdown resolved to structured metadata with matching identifiers; titles include the expected publication or dataset records.
- No raw EDF, fitted model, binary feature array, or credential is tracked in Git.

## 4. New Integrity Results

### Full EDF identity

All 256 local EDFs, totaling 35,913,652,480 bytes, matched both the official annex size and SHA-256 for OpenNeuro `ds005555` tag `1.1.1`.

### Across-night alignment proxy

The saved full-dataset table contains 505 available pulse windows and 383 usable windows. Eighty-two recordings had at least three usable windows. Median fitted lag slope was 0.0197 seconds/hour, median projected change was 0.1250 seconds, and the 95th percentile absolute projected change was 1.4781 seconds. Three recordings exceeded the retrospective 2-second review screen: `sub-32`, `sub-39`, and `sub-50`.

This does not overturn the alignment decision. All three still have matching EDF timelines and sample-index mapping. They should remain visible as proxy-review cases rather than being silently excluded.

Two review recordings, `sub-32` and `sub-50`, are in the test partition. In a post hoc audit using saved per-recording summaries only, removing them left SF-C at F1 0.0732 and 1.9741 false alarms/hour and DE-B at F1 0.1384 and 1.3045 false alarms/hour. The descriptive direction therefore persists, but the calculation does not authorize exclusion or replace the frozen primary metrics.

## 5. Git Chronology Boundary

The stage-first protocol and result package first appear together in `176fdd8`. The direct baseline and factorization protocol/result packages first appear together in `ab76720`. The endpoint-contribution package records repository base `ab76720` and first appears in `d936ea4`, but its plan and results are also introduced together.

The folder-level mapping is retained in `experiment_commit_index_2026-08-23.md`. It is explicitly labeled as a first-repository-commit index rather than a code-used-for-run record.

These facts do not show that the stated within-session sequence is false. They mean the repository history cannot independently establish that sequence. Starting with Block 7, freeze and commit the protocol before executing the result-producing run. If an implementation correction is required, retain the failed attempt and commit the correction record before reopening any locked evaluation partition.

## 6. Scientific Boundaries Retained

- The derived labels are valuable because their definition, uncertainty, quality sensitivity, and learnability are tested systematically; mechanical stage-pair conversion alone is not a technological advance.
- BOAS does not validate sleep-paralysis detection, narcolepsy diagnosis, intervention, or clinical utility.
- SF-A remains descriptive because its training provenance is unresolved locally.
- DE-B improved directionally over SF-C but remains a low-precision detector.
- DE-D is promising validation-only mechanism evidence and has no independent performance estimate.
- Future complexity must address a documented failure mechanism rather than routine model tuning.

## 7. Block 7 Readiness

Block 7 can begin on schedule only after its complete experiment protocol is committed. The protocol must treat the existing test partition as a fixed paired descriptive comparison, run zero-shot transfer before adaptation, isolate all selection to train/validation, and specify what new lock or external evidence will support later confirmation.
