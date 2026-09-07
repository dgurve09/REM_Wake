# Block 7 Zero-Shot Hypothesis Completion Plan v0.1

**Created:** 2026-09-06
**Analysis status:** Post-result completion analysis
**Input result commit:** `15dc9b4`
**Input partition:** Validation only
**Raw signal, feature, or model access:** No
**Test data access:** No

## 1. Reason for the Analysis

Block 7 hypothesis H7.2 states that strict `PSG-2` to wearable transfer should be compared with both direct `PSG-2` and direct wearable fitting. The primary validation result includes all three point estimates, but its paired-bootstrap table contains `P2-H2-Z` versus `H2-D` and does not contain the direct source comparison `P2-D` versus `P2-H2-Z`.

This analysis completes that planned hypothesis comparison from the committed participant-level validation counts. It does not revise a model, feature, threshold, gate, or primary result.

## 2. Frozen Inputs

- `validation_event_participants_v0.1.tsv` from result commit `15dc9b4`;
- `validation_event_metrics_v0.1.tsv` from result commit `15dc9b4`;
- `paired_participant_bootstrap_v0.1.tsv` as a reproduction control; and
- primary membership at the fixed +/-15-second event tolerance only.

The comparator thresholds remain those already frozen: direct models use their selected validation thresholds, while `P2-H2-Z` inherits the `P2-D` threshold.

## 3. Comparisons

1. `P2-D - P2-H2-Z`: cost of applying the unchanged reduced-PSG model and threshold to mapped wearable features.
2. `H2-D - P2-H2-Z`: direct wearable fitting versus strict zero-shot transfer.
3. `P2-D - H2-D`: direct reduced-PSG versus direct wearable acquisition under the same model family.

For each comparison, calculate the aggregate event-F1 difference and false-alarms-per-hour difference. Use paired participant-cluster bootstrap resampling with 2,000 resamples and seed `20260906`.

## 4. Interpretation Boundary

H7.2 will not be reduced to a new post-result binary success rule. Event F1 and false alarms per hour will be interpreted jointly:

- lower zero-shot F1 indicates loss of event detection under transfer;
- higher zero-shot false alarms indicates added alarm burden;
- lower zero-shot false alarms together with lower F1 is a sensitivity-specificity tradeoff, not uniform improvement or uniform degradation; and
- intervals crossing zero are inconclusive for the corresponding paired difference.

The analysis cannot establish independent generalization because it uses the same validation participants as the main result.

## 5. Required Records

Retain the input hashes, participant-pair counts, point differences, bootstrap intervals, per-participant descriptive differences, reproduction checks, and interpretation. Preserve mixed or inconclusive findings.
