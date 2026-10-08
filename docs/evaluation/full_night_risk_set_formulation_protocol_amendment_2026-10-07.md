# Full-Night Risk-Set Protocol Amendment: No-Alarm Sentinel

**Created:** 2026-10-07

**Applies to:** `full_night_risk_set_formulation_protocol_v0.1.md`

**Status:** Frozen before resuming threshold selection or evaluating any outer result

## Observed implementation failure

The v0.1 protocol specified threshold `1.0` to guarantee a budget-feasible no-alarm option. After outer-fold-1 score generation, but before outer evaluation, the threshold-selection code stopped because no threshold met the `0.10` false-alarms/hour budget.

Inspection found that the nonconverged balanced-loss logistic models produced some floating-point probabilities exactly equal to `1.0`: 227, 108, 76, and 65 values in the four inner score sets for outer fold 1. The rule `probability >= threshold` therefore emitted alarms at threshold `1.0`. The protocol statement that `1.0` guarantees no alarms was technically incorrect.

No event metric, hypothesis decision, or outer result had been written when the failure occurred. The generated models and scores are retained unchanged.

## Frozen correction

Add `nextafter(1.0, +infinity)` to the threshold grid as an explicit no-alarm sentinel. In IEEE-754 double precision this value is approximately `1.0000000000000002`, strictly greater than every valid probability.

The sentinel is eligible only through the same false-alarm-budget selection rule as every other threshold. If it is selected, it represents a valid zero-alarm, zero-recall operating point. Keep threshold `1.0` in the grid because it remains a distinct observed operating point.

No model, feature, participant, label, score, alarm budget, metric, hypothesis gate, ranking criterion, or stop rule changes. This amendment corrects only the failed assumption that a valid probability cannot equal one.
