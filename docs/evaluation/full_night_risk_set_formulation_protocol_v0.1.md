# Full-Night Risk-Set Formulation Protocol v0.1

**Created:** 2026-10-07

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized partition:** Train only, 82 recordings from 64 `pid` groups

**Validation partition authorized:** No

**Current test partition authorized:** No

**Protocol status:** Frozen before result-producing implementation or execution

## 1. Formulation error being tested

The current event detector was trained with 180 primary REM-to-Wake examples and 2,563 negative examples from `background_review_windows_v0.1.tsv`. That negative table was created for manageable preprocessing review: within each subject, background tier, and centre-stage pair, eligible rows were sorted by time and the first two were retained. It was not designed as a representative training sample.

A pre-experiment audit regenerated 73,476 eligible train-partition background centres before signal-context intersection. The review subset therefore covered only 3.488% of this pool. Its median centre time was 4,110 seconds, compared with 13,830 seconds in the full pool. The N2-to-N2 share was 12.41% in the review sample and 59.93% in the full pool. The Jensen-Shannon divergence between the sampled and full centre-stage-pair distributions was 0.2743 bits.

This creates two separable uncertainties:

1. whether the small, early-night, stage-stratified review sample omitted the diversity needed to distinguish REM-to-Wake events from ordinary full-night boundaries; and
2. whether balanced class weights distorted probability calibration and threshold transfer relative to the natural event prevalence.

The raw positive prevalence was 6.562% in the old labelled training table but is approximately 0.238% when the same 180 events are compared with the full eligible background pool. With balanced class weights, the effective training loss assigns half of its mass to each class, approximately 210 times the natural positive prior. These observations do not prove that the old detector is invalid, but they require a controlled formulation test before adding another architecture.

## 2. Scope and interpretation boundary

This experiment changes only the negative risk set and class weighting for the existing wearable enriched elastic-net detector. It does not change:

- train participants or frozen participant folds;
- primary REM-to-Wake labels;
- signal preprocessing or feature definitions;
- the eight-epoch `[-120, -90, -60, -30, 0, +30, +60, +90]` second context;
- correlation pruning, elastic-net penalty, or regularization strength;
- event matching; or
- validation/test authorization.

Because the context includes the boundary epoch and 90 seconds after it, this remains a retrospective boundary-localization experiment. It is not a prospective alarm model and cannot establish real-time detection latency. The separate Block 10 interval-aware experiment remains necessary to test 30-second label uncertainty.

## 3. Frozen modality, representation, and model

Use wearable `H2` only: channels `HB_1` and `HB_2`.

Reuse the reviewed `F1-EN` representation from the paired enriched-feature experiment:

- 45 features per 30-second epoch;
- eight contiguous epochs and 360 input values;
- fitting-subset variance removal at or below `1e-12`;
- fitting-subset greedy absolute-correlation pruning at `0.95`;
- standard scaling fitted only on the fitting subset; and
- elastic-net logistic regression with `saga`, `l1_ratio=0.5`, `C=0.1`, seed `20261007`, tolerance `1e-4`, and at most 3,000 iterations.

`C=0.1` is fixed because all five outer folds selected it in the prior nested wearable experiment. It is not re-tuned here.

## 4. Frozen candidate formulations

| Candidate | Negative fitting rows | Class weight | Purpose |
|---|---|---|---|
| `SAMP-BAL` | Prior review-sampled negatives | Balanced | Exact prior wearable control |
| `RISK-BAL` | Every eligible full-risk-set negative with complete H2 context | Balanced | Isolate negative-set coverage |
| `RISK-NAT` | Same full risk set | None | Isolate training-prior/calibration effect |

The `SAMP-BAL` control must reuse the exact prior inner and outer full-night score artifacts for `H2-F1-EN`, `C=0.1`; it must not be refitted. Each new candidate uses the same positive rows as the control and every eligible negative row in the relevant fitting participants. No negative downsampling, tier weighting, resampling, or hard-negative mining is allowed.

## 5. Full risk-set construction

For the 82 authorized train recordings:

1. regenerate background eligibility using the frozen `background_window_spec_v0.1` rules;
2. exclude REM-to-Wake and Wake-to-REM centres;
3. exclude incomplete 240-second windows, disconnections, missing/non-30-second epochs, onset gaps, and centres within 135 seconds of a REM-to-Wake boundary;
4. intersect eligible centres with complete wearable enriched-feature context;
5. retain all remaining negative rows, including their background tier and centre-stage pair; and
6. add the same 180 primary eligible REM-to-Wake rows used previously.

Record the eligible, retained, and excluded counts by subject, `pid`, tier, and centre-stage pair. The exact retained count is a result of the frozen intersection and is not forced to equal the 73,476 pre-audit count.

## 6. Participant-grouped nested execution

Reuse the exact five frozen train `pid` folds. For each outer fold:

1. hold out the outer-fold participants;
2. rotate each remaining fold once as the inner-held-out fold;
3. fit `RISK-BAL` and `RISK-NAT` on the other three folds;
4. score every supported full-night boundary in the inner-held-out recordings;
5. select one threshold per candidate from concatenated inner out-of-fold scores;
6. refit each candidate on all four outer-training folds; and
7. score the outer-held-out recordings once.

The prior `SAMP-BAL` inner and outer scores pass through the same new threshold-selection and evaluation code. Outer labels cannot influence pruning, scaling, fitting, thresholds, or model choice.

## 7. Alarm-budget threshold selection

Evaluate thresholds formed by sigmoid-transforming logits from -16 through +16 in increments of 0.25, plus threshold `1.0`. The primary alarm budget is `0.25` false alarms per supported hour, approximately two false alarms in an eight-hour recording.

For each inner pooled score set, select among thresholds meeting the budget by:

1. maximum primary-label recall within +/-15 seconds;
2. maximum precision;
3. maximum F1; and
4. maximum threshold.

Threshold `1.0` ensures that a budget-feasible no-alarm option exists. Also report sensitivity at fixed budgets of `0.10`, `0.50`, and `1.00` false alarms/hour. Consolidate contiguous above-threshold candidates into one alarm at the highest-scoring boundary, with the earliest boundary breaking exact ties.

## 8. Outcomes and hypotheses

### Primary event outcomes

On concatenated outer held-out scores, report primary-label precision, recall, F1, and false alarms/hour at +/-15 seconds under the inner-selected `0.25/hour` threshold. Report the same event outcomes at +/-45 seconds and for expanded-quality labels as sensitivity analyses.

### Risk-set outcomes

At every held-out labelled risk-set row, report average precision, Brier score, and log loss. These metrics are descriptive for `SAMP-BAL` and `RISK-BAL`, whose balanced loss does not target natural prevalence, but they permit a direct calibration comparison with `RISK-NAT`.

### H-RISK: negative-set coverage

`RISK-BAL` passes its point gate only if outer primary +/-15-second recall exceeds `SAMP-BAL` by at least `0.05` while both use inner-selected thresholds at or below `0.25` false alarms/hour.

### H-PRIOR: natural-prior fitting

`RISK-NAT` passes its point gate only if its outer risk-set Brier score is at least 10% lower than `RISK-BAL` and its primary +/-15-second recall is no more than `0.05` lower at the same alarm budget.

Use paired participant-cluster bootstrap resampling with 2,000 resamples and seed `20261007` for differences in event recall, F1, false alarms/hour, Brier score, and average precision. Bootstrap intervals describe participant-level uncertainty; the frozen point gates determine the v0.1 decision.

## 9. Required controls

1. Confirm exactly 82 train recordings and 64 train `pid` groups.
2. Confirm that no validation or current-test signal, feature, label, score, model, or result is read.
3. Verify full-risk-set eligibility against the frozen background rules.
4. Confirm identical positive rows across all candidates.
5. Reuse the exact five participant folds and verify participant disjointness at every fit.
6. Verify that prior `SAMP-BAL` score hashes match the reviewed external-artifact manifest.
7. Fit pruning, scaling, and the classifier using fitting participants only.
8. Confirm one full-night score per supported boundary and one outer assignment per participant for each candidate.
9. Record convergence failures without changing the frozen settings.
10. Keep feature arrays, fitted models, full-night scores, and the full risk-set table outside Git; retain compact reviewed summaries and artifact hashes in Git.
11. Run an independent read-only validator twice.

## 10. Stop rule

Do not revise the risk set, class weight, alarm budget, thresholds, features, or model after outer results are available. If neither hypothesis passes, stop negative-set and loss-weight variations on the reused train cohort. Preserve the failure and continue Block 10 as the distinct interval-aware label-uncertainty experiment.

A passing result identifies a better train-cohort formulation only. It does not establish clinical utility, prospective detection, or independent generalization and does not authorize reopening the current validation/test cohorts.

## 11. Method references

1. King G, Zeng L. Logistic regression in rare events data. *Political Analysis*. 2001;9(2):137-163. https://doi.org/10.1093/oxfordjournals.pan.a004868
2. Fithian W, Hastie T. Local case-control sampling: efficient subsampling in imbalanced data sets. *The Annals of Statistics*. 2014;42(5):1693-1724. https://doi.org/10.1214/14-AOS1220
3. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
