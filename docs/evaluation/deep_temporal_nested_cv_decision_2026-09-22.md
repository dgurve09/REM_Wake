# Nested Deep Temporal Architecture Decision

**Decision date:** 2026-09-22

**Protocol commit:** `533cad3`

**Pre-execution receptive-field correction:** `2eb0b7a`

**Result-producing code commit:** `b12f4f3`

**Accepted result commit:** `cf1207a`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Stop this four-candidate exploration; retain `LC-1` as the existing train-only temporal candidate

## Question

Can nested participant-grouped selection among four small deep temporal architectures produce a material REM-to-Wake event improvement over the prior bidirectional LSTM-CRF mechanism without increasing false alarms?

## Frozen Comparison

The experiment compared `BLSTM-CRF`, `BGRU-CRF`, `TCN-CRF`, and `BLSTM-2H`. Every candidate used the same 2,743 reviewed train candidates, 180 positive sequences, ten wearable bandpower features per epoch, eight-epoch contexts, five outer participant folds, 60 epochs, optimizer settings, full-night event evaluator, and 64-threshold grid.

Within each closed outer fold, the other four participant folds rotated through inner held-out evaluation. Candidate thresholds and architecture choice used only the pooled inner predictions. The chosen model and a matched `BLSTM-CRF` control were then fit on all outer-training participants and evaluated once on the outer-held-out participants.

## Architecture Stability

No architecture dominated the five inner selections:

| Outer fold | Selected architecture | Inner F1 | Inner FAR/hour |
|---:|---|---:|---:|
| 1 | BLSTM-2H | 0.2180 | 0.2151 |
| 2 | BLSTM-CRF | 0.2167 | 0.4135 |
| 3 | BGRU-CRF | 0.2021 | 0.3971 |
| 4 | BGRU-CRF | 0.1983 | 0.1466 |
| 5 | TCN-CRF | 0.2367 | 0.5786 |

`BLSTM-2H` had the best prespecified mean inner rank, but it was selected in only one outer fold. The variation across recurrent, convolutional, structured, and endpoint-head models is evidence against a stable architecture advantage on the available participants.

## Primary Outer Result

| Pipeline | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
| Nested BLSTM-CRF | 0.1461 | 0.1778 | 0.1604 | 0.2971 |
| Nested selected | 0.1531 | 0.1778 | 0.1645 | 0.2812 |

Both pipelines detected 32 of 180 reference events. Selection reduced false positives from 187 to 177 but did not improve true positives. The selected-minus-control differences were:

- F1: `+0.0041`, versus the required `+0.05`;
- false alarms/hour: `-0.0159`;
- F1 95% participant interval: `-0.0532` to `+0.0608`; and
- false-alarm difference interval: `-0.0621` to `+0.0296`/hour.

Neither favorable direction was participant-supported because both intervals crossed zero. The advancement hypothesis failed.

## Relation to the Earlier LC-1 Result

The earlier `LC-1` screen reported train out-of-fold F1 0.1965 at a pooled threshold. The present nested BLSTM-CRF control produced F1 0.1604 because its threshold was selected independently inside each outer fold and then applied once to that outer-held-out group. This stricter estimate should not be treated as evidence that retraining damaged the architecture; it quantifies the added uncertainty of fold-local threshold selection.

Thresholds selected for the five chosen outer models remained extreme, from 0.9820 to 0.9991. Together with unchanged recall, this indicates that swapping small temporal encoders did not resolve score separation at true boundaries. The limiting factor is more likely the information and label uncertainty in the eight-epoch bandpower representation than insufficient depth within this candidate family.

## Verification

- All 14/14 in-run controls passed.
- All four synthetic model checks and the full-sequence TCN check passed.
- All 89 fits completed the fixed 60 epochs.
- Four duplicate fits reproduced state hashes, scalers, and probabilities exactly.
- The read-only validator passed 12/12 reconstruction checks twice after the first run and once after the full rerun.
- The final manifest verified 202 external artifacts by size and SHA-256.
- The full rerun reproduced all reported selections, event counts, metrics, and decisions.

## Reporting Correction

The generated architecture table uses `recommended_for_new_confirmation=True` to identify the best mean-inner-rank candidate. Because the experiment-level gate failed, that label does not authorize confirmation. The original generated files remain unchanged, and the append-only `INTERPRETATION_CORRECTION.md` records the governing interpretation.

## Knowledge Gained and Next Decision

The earlier LSTM-CRF gain over flattened logistic regression was not followed by a further material gain from GRU, TCN, or two-head variants. Deepening or swapping the sequence encoder within the same bandpower representation is therefore stopped, rather than continuing architecture search against reused participants.

Any later deep-learning experiment must test a different documented mechanism, such as multiscale raw-waveform information or explicit boundary-label uncertainty, and must use a new prespecified evaluation boundary. It must not be selected retrospectively from these outer results. Block 9 external-PSG compatibility remains the scheduled next project block and is not a wearable confirmation cohort.
