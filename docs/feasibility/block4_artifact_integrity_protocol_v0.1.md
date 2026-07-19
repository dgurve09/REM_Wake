# Block 4 Artifact-Integrity Protocol v0.1

**Created:** 2026-07-18
**Project phase:** Block 4 label/preprocessing gate
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Model training performed:** No
**Raw signal data read:** No

## 1. Technical Uncertainty

The transition labels, background review pool, signal-quality decisions, participant split, analysis membership, and preprocessing checks were generated in separate steps. Matching aggregate counts do not prove that the same row identities, sample windows, participants, or decisions propagated through every step. Silent row loss, duplication, stale quality versions, or split mismatch would invalidate later comparisons even if each artifact looked plausible in isolation.

## 2. Hypothesis

All frozen Block 4 artifacts will preserve one-to-one row identity and invariant fields. The preprocessing v0.2 train-window set will equal the complete set of noncritical v0.3 train windows, with no validation/test participant or recording present.

## 3. Frozen Inputs

- transition labels `v0.1`;
- background review windows `v0.1`;
- signal-quality flags `v0.3`;
- grouped participant split `v0.1`;
- quality analysis membership `v0.1`;
- minimal preprocessing validation `v0.2`.

## 4. Required Cross-Artifact Checks

### Transition chain

- 476 unique transition IDs in label, quality, and membership tables;
- exact agreement on subject, participant identifier, `pid`, transition direction, and primary-label status;
- exact agreement between label headband start/stop samples and quality window start/stop samples;
- exact agreement between quality decision and membership decision.

### Background chain

- 4,302 unique background review IDs in source, quality, and membership tables;
- exact agreement on subject, participant identifier, `pid`, and background tier;
- exact agreement between source headband start/stop samples and quality window start/stop samples;
- exact agreement between quality decision and membership decision.

### Split and membership

- every row maps to exactly one frozen `pid` partition;
- no `pid` occurs in more than one partition;
- membership tier and primary/expanded eligibility exactly follow the v0.1 mapping;
- the split-assignment file LF-normalized SHA-256 remains `52450EDA07795D198E2722D4D804E71D0E17A8A4B62BA5AF93AE811B211D83A7`.

### Preprocessing linkage

- preprocessing checks contain exactly every noncritical v0.3 train transition and background row;
- no excluded, validation, or test row is present;
- all window IDs, subjects, `pid` values, label classes, and quality decisions agree;
- every input/output window contains 61,440/30,720 samples and passes;
- the 82 recording checks exactly match the 82 frozen train recordings and all pass.

## 5. Outputs

1. Pass/fail table with observed and expected values for every check.
2. LF-normalized SHA-256 manifest for the frozen source and result files so verification is independent of operating-system line endings.
3. Dated interpretation and gate decision.

## 6. Decision Rule

The Block 4 integrity audit passes only if every required check passes. A failure must remain recorded and be resolved with a new artifact version; aggregate-count agreement alone is not an acceptable substitute. Passing establishes internal traceability, not model performance or clinical validity.
