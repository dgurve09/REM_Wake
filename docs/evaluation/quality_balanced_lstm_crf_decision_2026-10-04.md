# Quality-Balanced LSTM-CRF Decision

**Decision date:** 2026-10-04

**Protocol commits:** `6422a23`, `b09c857`

**Result-producing code commit:** `858f7f9`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Stop robust-scaler and quality-tier weighting variants on the reused train cohort

## Question

Does replacing mean/standard-deviation scaling with median/interquartile-range scaling, while equalizing clean and MAD-flagged positive loss contribution, materially improve the nested BLSTM-CRF's overall event F1 and clean-event recall without increasing false alarms?

## Fixed comparison

Both candidates used the same ten wearable bandpower features, eight-epoch context, BLSTM-CRF, optimizer, 60 epochs, participant folds, seeds, full-night scoring, nested threshold selection, and event evaluator. `LC-QB1` changed only the fitting-subset scaler and positive-tier weights. Quality tier was not an inference feature.

The experiment contained 2,743 reviewed candidates: 2,563 negatives, 53 clean positives, and 127 MAD-flagged positives. Each candidate completed 20 inner fits and five outer-final fits.

## Primary result

| Pipeline | True positive | False positive | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
| LC-1-NESTED | 32 | 187 | 0.1461 | 0.1778 | 0.1604 | 0.2971 |
| LC-QB1 | 34 | 260 | 0.1156 | 0.1889 | 0.1435 | 0.4130 |

`LC-QB1` detected two additional primary events but produced 73 additional false positives. Its F1 difference was `-0.0169`, against the required `+0.05`, and its false-alarm difference was `+0.1160`/hour, while the gate required no increase.

Participant-cluster bootstrap intervals for `LC-QB1 - LC-1-NESTED` were:

- F1: `-0.0503` to `+0.0204`;
- false alarms/hour: `+0.0555` to `+0.1825`;
- clean-event recall: `0.0000` to `+0.1176`; and
- MAD-flagged recall: `-0.0455` to `+0.0455`.

The entire false-alarm interval is adverse. The F1 interval excludes the required material gain.

## Quality-tier result

| Pipeline | Clean detected | Clean recall | MAD-flagged detected | MAD-flagged recall |
|---|---:|---:|---:|---:|
| LC-1-NESTED | 2/53 | 0.0377 | 30/127 | 0.2362 |
| LC-QB1 | 4/53 | 0.0755 | 30/127 | 0.2362 |

Clean recall increased by `0.0377`, below the required `+0.10`; MAD-flagged recall did not change. The result does not support unequal tier loss as the main explanation for the earlier clean-versus-flagged gap.

## Sensitivity results

At primary +/-45 seconds, `LC-QB1` reached F1 0.2000 and 0.3860 false alarms/hour, versus 0.2317 and 0.2716/hour for the control. Under expanded membership, `LC-QB1` also remained worse at both +/-15 seconds and +/-45 seconds. The negative conclusion is not specific to the primary tolerance.

## What was learned

Robust scaling and equal tier contribution modestly shifted detections toward clean events, but the shift was too small and came with substantially more alarms. The experiment rejects this fixed combined mechanism as an advancement path on the present representation and cohort.

It does not establish that amplitude or recording quality is irrelevant. `membership_tier` is based on a nonspecific 10-MAD screen, and the experiment combined scaling and weighting into one predeclared treatment. The result is therefore evidence against this treatment, not a causal decomposition of signal quality.

## Verification

- The exact control reproduced 75,539/75,539 prior outer probabilities with maximum absolute difference 0.
- The independent validator passed 9/9 checks twice.
- A detached isolated rerun regenerated all 50 fits.
- All 101 external artifacts and all 19 reviewed outputs reproduced byte-for-byte.
- Three pre-result implementation failures and their resolutions are retained in the execution log.

## Next decision

Stop robust-scaler, tier-weight-ratio, and closely related normalization variants on these reused train participants. Do not tune the ratio or scaler after observing this result.

Return to the scheduled scientific sequence. Complete the overdue Block 9 external-PSG compatibility audit before beginning Block 10. Block 9 cannot confirm wearable performance, but it can determine whether external PSG supports a valid generalization comparison. Keep validation and current test closed, and do not claim a reliable detector from the current F1.
