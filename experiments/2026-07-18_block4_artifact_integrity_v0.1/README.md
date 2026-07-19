# Block 4 Artifact Integrity v0.1

**Created:** 2026-07-18
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Protocol:** `docs/feasibility/block4_artifact_integrity_protocol_v0.1.md`
**Model training performed:** No
**Raw signal data read:** No

## Result

Integrity decision: **pass**.

- Checks passed: 71 of 71
- Frozen files hashed: 19
- Transition rows linked label-to-quality-to-membership: 476
- Background rows linked source-to-quality-to-membership: 4,302
- Noncritical train windows linked quality-to-preprocessing: 3,063
- Train recordings linked split-to-preprocessing: 82
- Validation/test windows found in preprocessing output: 0
- LF-normalized split assignment SHA-256: `52450EDA07795D198E2722D4D804E71D0E17A8A4B62BA5AF93AE811B211D83A7`

## Method

The audit compared row identities and invariant fields rather than relying on aggregate totals. It independently rebuilt the expected noncritical train-window set from quality v0.3 and the frozen split, then required exact equality with preprocessing v0.2.

## Outputs

| File | Purpose |
|---|---|
| `artifact_integrity_checks_v0.1.tsv` | Pass/fail result and observed value for each linkage check |
| `frozen_artifact_manifest_v0.1.tsv` | Relative path, size, row count, and LF-normalized SHA-256 for each frozen file |

## Decision

Retain the July 18 label/preprocessing gate pass. The frozen artifacts are internally traceable and no split contamination was found.

This result establishes internal consistency only. It does not estimate detector performance or establish clinical validity.
