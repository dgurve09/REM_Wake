# Paired Enriched-Feature and Pruning Decision

**Decision date:** 2026-10-04

**Protocol commits:** `0f3c694`, `c431783`, `7cc6253`

**Initial result code commit:** `086c43c`

**Result-finalization commit:** `e71a042`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

**Decision:** Freeze both enriched elastic-net candidates for future confirmation; do not claim a reliable detector

## Question

Can a bounded set of spectral shape, amplitude, time-domain, interchannel, and asymmetry features improve participant-held-out REM-to-Wake event detection over the existing ten-bandpower representation? Does fold-local correlation pruning with elastic-net regularization improve the enriched representation?

## Fixed comparison

The experiment held the reviewed REM-to-Wake labels, 82 train recordings, 64 participant groups, five outer folds, four inner folds, eight-epoch context, logistic model class, threshold grid, full-night scoring, event matching, and advancement gate fixed.

`F0-L2` used the existing 80 context values. `F1-L2` used 360 enriched context values without pruning. `F1-EN` used the same enriched values with fitting-fold-only variance and correlation pruning followed by elastic-net logistic regression. No feature, regularization value, threshold, or gate was added after the results were observed.

The feature definitions followed established methods: Welch periodograms (Welch, 1967, DOI: `10.1109/TAU.1967.1161901`), Hjorth parameters (Hjorth, 1970, DOI: `10.1016/0013-4694(70)90143-4`), power-spectrum entropy (Inouye et al., 1991, DOI: `10.1016/0013-4694(91)90138-T`), and elastic-net regularization (Zou and Hastie, 2005, DOI: `10.1111/j.1467-9868.2005.00503.x`).

## Primary result

| Modality | Pipeline | True positive | False positive | Precision | Recall | F1 | False alarms/hour |
|---|---|---:|---:|---:|---:|---:|---:|
| Reduced PSG | F0-L2 | 62 | 467 | 0.1172 | 0.3444 | 0.1749 | 0.7419 |
| Reduced PSG | F1-L2 | 69 | 371 | 0.1568 | 0.3833 | 0.2226 | 0.5894 |
| Reduced PSG | F1-EN | 68 | 349 | 0.1631 | 0.3778 | 0.2278 | 0.5544 |
| Wearable | F0-L2 | 59 | 967 | 0.0575 | 0.3278 | 0.0978 | 1.5362 |
| Wearable | F1-L2 | 50 | 663 | 0.0701 | 0.2778 | 0.1120 | 1.0532 |
| Wearable | F1-EN | 67 | 499 | 0.1184 | 0.3722 | 0.1796 | 0.7927 |

Relative to `F0-L2`, `F1-EN` improved reduced-PSG F1 by `+0.0529` and reduced false alarms by `0.1875`/hour. It improved wearable F1 by `+0.0818` and reduced false alarms by `0.7435`/hour. Both comparisons passed the predeclared point gate of at least `+0.05` F1 with no false-alarm increase.

## Participant uncertainty

Participant-cluster bootstrap intervals for `F1-EN - F0-L2` were:

- reduced-PSG F1: `+0.0186` to `+0.0882`;
- reduced-PSG false alarms/hour: `-0.2851` to `-0.0951`;
- wearable F1: `+0.0439` to `+0.1169`; and
- wearable false alarms/hour: `-1.1938` to `-0.3371`.

Both modalities have directional participant support, but neither F1 lower bound reaches the predeclared `+0.05` material-support threshold. The point-gate decisions are therefore frozen for new-cohort confirmation rather than treated as established effects.

For wearable EEG, `F1-EN` also improved over unpruned `F1-L2`: F1 difference `+0.0676`, interval `+0.0380` to `+0.0946`; false-alarm difference `-0.2605`/hour, interval `-0.4630` to `-0.0867`. This supports pruning and regularization as an important part of the wearable result. The corresponding reduced-PSG elastic-net versus unpruned differences were inconclusive.

## Feature and convergence findings

All five wearable outer folds selected `C=0.1`; their outer-final fits converged and retained 320-335 of 360 inputs after correlation pruning, with 115-130 nonzero coefficients. Stable nonzero wearable terms included present-epoch interchannel correlation and coherence, channel-specific spectral entropy, log bandpower, line length, relative theta power, and zero-crossing rate.

Twenty-four of 230 fits reached the predeclared 3,000-iteration limit. Every failure was an inner-fold `F1-EN` fit at `C=1.0`; all 30 outer-final fits converged. Four of five reduced-PSG folds selected `C=0.1`, while fold 1 selected `C=1.0` using pooled inner results that included three nonconverged fits. The reduced-PSG result therefore carries an additional selection-stage limitation. No solver, tolerance, iteration limit, or candidate grid was changed after inspection.

## What was learned

The negative results from earlier architecture changes did not imply that the signal contained no useful transition information. On the reused train cohort, the tested enriched representation improved both modalities, and fold-local elastic-net regularization was particularly important for the wearable channels. The wearable gain came from 17 more true detections and 468 fewer false positives than the original feature baseline.

This does not establish clinical or operational reliability. Wearable precision remained `0.1184`, F1 remained `0.1796`, and approximately four of five emitted wearable alarms were false positives. The result is participant-held-out train development on one dataset, not independent confirmation.

## Verification

- All 82 train recordings and 64 participant groups were retained; 2,743 candidates included 180 positives.
- Existing F0 values were reproduced within maximum absolute difference `9.54e-7`.
- All 624 external feature, model, and score artifacts were hashed.
- The independent validator passed 10/10 checks twice.
- Validation and current-test data remained closed.
- The 24 convergence failures and the pre-result coherence failure are retained in the execution record.

## Next decision

Freeze the `F1-EN` definitions, pruning rules, `C` grid, and event evaluation. Do not tune this feature family further on the same outer results. Continue the scheduled Block 10 interval-aware label-uncertainty work separately.

The frozen enriched candidates may be evaluated only on a genuinely new locked or scientifically matched external cohort. The wearable candidate is the stronger mechanistic result, but it is not a deployable detector at the observed precision and F1.
