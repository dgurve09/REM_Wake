# Direct Event Failure Analysis v0.1

**Created:** 2026-08-22
**Status:** Post-result exploratory analysis
**Plan:** `docs/evaluation/direct_event_failure_analysis_plan_v0.1.md`

## Retained Primary Result

DE-B test event F1 was 0.1497 with precision 0.0909, recall 0.4237, and 1.2571 false alarms per supported hour. This analysis does not change the frozen model, threshold, or result.

## Residual False Alarms

The largest false-positive stage-pair category was `human_other_to_Wake`, contributing 108 of 250 false alarms (43.20%). Its enrichment relative to all supported candidate boundaries was 2.31-fold.

Only 6.80% of false positives occurred within 135 seconds of a human-derived REM/Wake boundary, the region excluded from reviewed background construction. The excluded boundary zone is therefore not the dominant observed source of false alarms.

False positives occurred in 18 of 20 test participants. The four highest-burden participants contributed 58.80%. The failure is broad across participants with moderate concentration in the highest-burden group, rather than attributable to a single outlier.

## Context Tradeoff

Relative to DE-A, DE-B added 12 true positives and +55 false positives. Event F1 changed by +0.0523, while false alarms per hour changed by +0.2868. Eight-epoch context improved test recall and F1 despite failing the prespecified validation F1 comparison, but it also increased the alarm burden.

## Timing Sensitivity

At +/-45 seconds, 8 matches were outside the primary +/-15-second tolerance: 2 were one epoch early and 6 were one epoch late. This quantifies how much one-epoch boundary displacement contributes to the remaining error.

## Decision Boundary

Direct modeling produced measurable value over transparent stage-first SF-C, but fixed log-bandpower context did not resolve event precision. The next method must address a stated source of residual error and must be validated without modifying this frozen test result. A CNN is not automatically justified by the remaining error; its proposed representation must be tied to a specific hypothesis in a new protocol.

No raw EDF or fitted model was opened. All diagnostic inputs and frozen feature-stage lookup arrays are listed with SHA-256 hashes in `input_artifact_manifest_v0.1.tsv`.
