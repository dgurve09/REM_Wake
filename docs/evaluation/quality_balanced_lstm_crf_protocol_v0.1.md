# Nested Quality-Balanced LSTM-CRF Protocol v0.1

**Created:** 2026-10-03

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Technological uncertainty

The earlier nested temporal comparison produced primary +/-15-second event F1 of 0.1604 for the matched BLSTM-CRF. A frozen error diagnostic then found clean-event recall of 9/53 for the earlier non-nested `LC-1` and 1/53 for the nested selected pipeline, versus 47/127 and 31/127 for MAD-flagged events. The 10-MAD tier is nonspecific and is not an artefact ground truth, but the repeated difference shows that aggregate performance depends strongly on amplitude-defined event groups.

The unresolved uncertainty is whether conventional mean/standard-deviation scaling and unequal representation of the two positive tiers bias the learned boundary score. The experiment isolates that mechanism while holding architecture, features, context, folds, optimizer, event evaluator, and total positive-versus-negative loss contribution fixed.

## 2. Fixed candidate pair

### C1: `LC-1-NESTED`

Exact nested BLSTM-CRF control:

- the existing eight consecutive 30-second epochs and ten stored `HB_1`/`HB_2` log-bandpower features per epoch;
- one bidirectional LSTM with hidden size 16 per direction;
- linear projection from 32 values to three CRF emissions;
- endpoint path `[0, 0, 0, 1, 2, 0, 0, 0]` versus the all-background path;
- mean and population standard deviation fitted per feature only on the current fitting subset; and
- positive sequence weight `N_negative / N_positive`, with negative weight 1.

### C2: `LC-QB1`

The architecture, optimizer, epochs, context, features, CRF paths, seed, and inference procedure are identical to C1. Only two training mechanisms change:

1. Per-feature center is the fitting-subset median and scale is the fitting-subset interquartile range. Use `max(IQR, 1e-6)` as the fixed denominator guard.
2. Negative sequences retain weight 1. If the fitting subset contains `N_negative` negatives, `N_clean` clean positives, and `N_flagged` MAD-flagged positives, assign:

   - clean-positive weight `N_negative / (2 * N_clean)`; and
   - flagged-positive weight `N_negative / (2 * N_flagged)`.

The clean tier therefore contributes half of total positive loss, the flagged tier contributes half, and combined positive weight remains exactly equal to total negative weight. Quality tier is never supplied as an inference feature.

No normalization alternative, weight ratio, hidden size, depth, optimizer, epoch count, threshold grid, context, feature, or post-processing change is permitted after results are available.

## 3. Frozen quality tiers

Use `membership_tier` from `labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv` for primary-analysis-eligible train REM-to-Wake transitions:

- `primary_clean`: 53 events across 22 `pid` groups; and
- `primary_mad_flagged`: 127 events across 42 `pid` groups.

Every positive candidate must map one-to-one to one of these tiers. Background candidates receive `not_applicable`. All 20 three-fold inner fitting subsets must contain both positive tiers. The MAD tier remains a sensitivity grouping, not a causal artefact label.

## 4. Fixed optimization

Both candidates use:

- Adam with learning rate 0.003 and weight decay 0.0001;
- batch size 128;
- gradient-norm clipping at 5;
- 60 fixed epochs;
- no early stopping, scheduler, augmentation, pretrained weights, or hyperparameter search;
- deterministic CPU algorithms and one PyTorch thread; and
- the prior nested seeds `20260922 + 100 * outer_fold + 10 * inner_fold`, with inner fold 9 for the outer-final fit.

Using the same seed within each fit pair makes the normalization and weighting mechanism the only candidate-level training difference.

## 5. Nested participant-grouped design

Reuse the exact five frozen train `pid` folds. For each outer fold:

1. keep the outer participants closed;
2. rotate each of the other four fold identifiers once as the inner held-out fold;
3. fit both candidates on the remaining three folds;
4. score every supported full-night context for the inner-held-out participants;
5. concatenate four inner out-of-fold score sets separately for each candidate;
6. select one candidate-specific threshold using only those inner scores;
7. refit both unchanged candidates on all four outer-training folds; and
8. apply each frozen candidate-specific threshold once to the closed outer fold.

After all five outer folds, concatenate each candidate's one-time outer predictions. Outer labels do not influence scaling, weights, thresholds, epochs, or any model choice.

## 6. Threshold and event evaluation

Use the exact nested temporal threshold grid:

- 0.01 through 0.95 in increments of 0.05; and
- `sigmoid(x)` for logits 3.0 through 14.0 in increments of 0.25.

Select maximum primary +/-15-second event F1, then minimum false alarms/hour, maximum recall, and maximum threshold. Contiguous above-threshold 30-second candidates form one alarm; the highest probability represents the run and exact ties use the earliest time.

The primary endpoint is primary-label event F1 at +/-15 seconds. Report fixed sensitivity analyses for primary +/-45 seconds, expanded +/-15 seconds, and expanded +/-45 seconds.

## 7. Tier-specific endpoints

At primary +/-15 seconds, separately report:

- clean-event recall among 53 `primary_clean` references; and
- flagged-event recall among 127 `primary_mad_flagged` references.

Use the same one-to-one event matches as the aggregate primary evaluation. A reference is detected only when it participates in an eligible match. Report event counts as well as recall.

## 8. Hypothesis and decision rule

**H-QB1:** `LC-QB1` will improve overall primary +/-15-second event F1 by at least 0.05 and clean-event recall by at least 0.10 relative to `LC-1-NESTED`, without increasing false alarms per supported hour.

Use paired `pid`-cluster bootstrap resampling with 2,000 resamples and seed 20261003. Report point differences and 95% percentile intervals for:

- overall F1;
- false alarms/hour;
- clean-event recall; and
- flagged-event recall.

The point advancement gate passes only if all three primary conditions pass. Directional participant support additionally requires the F1 and clean-recall intervals entirely above zero and the false-alarm interval entirely at or below zero. Full material support requires the F1 lower bound at least +0.05, clean-recall lower bound at least +0.10, and false-alarm upper bound at most zero.

If the point gate fails, stop robust-scaler and quality-tier weight variations on the reused train cohort. Do not change weight ratios or add normalization variants after observing the result. If it passes, the mechanism may be frozen only for separately authorized new-cohort confirmation.

## 9. Required controls

1. Confirm exact train membership of 82 recordings and 64 participants.
2. Confirm no validation or current-test file, participant, feature, score, model, or result is accessed.
3. Reuse the exact five frozen participant folds and verify disjointness at every fit boundary.
4. Confirm 2,743 retained candidates, including 180 positives, 53 clean positives, and 127 flagged positives.
5. Confirm both positive tiers occur in every fitting subset.
6. Verify control scaling against `StandardScaler` and robust scaling against direct median/IQR calculations.
7. Verify the exact loss-weight totals for negatives, all positives, clean positives, and flagged positives in every fit.
8. Complete exactly 60 epochs with finite losses, gradients, parameters, and probabilities.
9. Verify C1 reproduces the prior nested BLSTM-CRF outer scores, thresholds, events, and metrics.
10. Select thresholds only from complete inner out-of-fold full-night scores.
11. Produce one outer score per supported boundary per candidate and one outer assignment per participant.
12. Run an independent read-only validator twice and complete one isolated immutable rerun.
13. Keep model states and full-night score artifacts outside Git; record sizes and SHA-256 values.

## 10. Interpretation boundary

This is bounded model development on reused train participants. Nested outer evaluation limits direct threshold fitting but does not provide independent confirmation because the hypothesis was motivated by earlier results from the same cohort. A positive result cannot establish clinical performance, prospective alarm utility, causality of the MAD tier, or deployment readiness. Validation and current test remain closed.
