# Multi-Dataset PSG Consensus-Correction Protocol v0.2

**Date frozen:** 2026-10-25
**Precondition:** Hypnogram uncertainty result commit `368c910`
**Result status at freeze:** No corrected model has been fitted and no corrected probability, alarm, or performance metric has been calculated

The pre-fit source-code audit subsequently found an additional gx36m raw-code issue before any v0.2 model was fitted. The frozen [pre-fit amendment](multidataset_psg_consensus_correction_amendment_2026-10-25.md) governs that correction without changing the model or evaluation design below.

## 1. Correction question

Does the v0.1 cross-dataset PSG conclusion remain unchanged when gx36m primary labels implement the written strict 2-of-3 scorer-majority rule?

The v0.1 gx36m function selected `argmax` over the five stage counts. At epochs where the three experts chose three different stages, all leading counts equalled one and `argmax` selected the first stage in label order, `W`. The written protocol instead requires a majority. Such epochs must be `UNKNOWN` because no stage has two votes.

This is a validity correction, not a performance-improvement experiment. The v0.1 artifacts remain immutable and are superseded for scientific interpretation.

## 2. Fixed cohort and inputs

Retain the exact 196 recordings and 130 participant groups used in v0.1:

| Dataset | Nights | Participant groups | Corrected primary-label rule |
|---|---:|---:|---|
| BOAS | 128 | 100 | Published human consensus; unchanged |
| gx36m | 58 | 20 | Stage requires at least 2 of 3 expert votes; otherwise `UNKNOWN` |
| ISRUC group 3 | 10 | 10 | Both experts must agree; otherwise `UNKNOWN`; unchanged |

Reuse the verified v0.1 feature caches. Signal channels, causal filtering, decision times, feature values, and participant identifiers must not be regenerated or changed. Verify all 196 cache hashes before fitting.

## 3. Only permitted change

For gx36m, count the three expert labels independently at every epoch. Assign the corresponding stage only when its count is at least two. Assign `UNKNOWN` otherwise.

No other label, timing, support, feature, split, model, threshold, alarm, metric, bootstrap, or decision rule may change. In particular, retain:

- `PSD3-BASE` with 120 features and `ADV3-CHANGE` with 444 features;
- the same five participant-grouped inner folds;
- natural-prevalence L2 logistic regression with `C=0.1`, `lbfgs`, no class weighting, `max_iter=1000`, and `tol=1e-6`;
- the 0.25 false-alarm/hour inner threshold budget;
- peak-first selection, greater-than-60-second separation, and the `nominal +15` through `nominal +30` second match interval;
- the same 2,000 stratified participant-bootstrap resamples and seed `20261025`; and
- all v0.1 advancement-gate thresholds.

## 4. Pre-fit correction audit

Before fitting, independently record and verify:

1. the number of gx36m epochs changed from the v0.1 implementation;
2. the number of affected gx36m recordings;
3. the number of strict-consensus `UNKNOWN` epochs;
4. per-dataset and pooled REM-to-Wake target counts under both implementations;
5. unchanged recording, participant, feature-cache, and fold membership; and
6. absence of any v0.2 model or score artifact.

Commit this audit before model fitting. Stop if a BOAS or ISRUC label changes, if a feature-cache hash fails, or if a split differs from v0.1.

## 5. Corrected rerun and comparison

Repeat all 36 fits because gx36m is an evaluation source in one outer run and a training source in the other two. Write models and outer scores to a separate v0.2 derived directory. Reconstruct all alarms and result tables independently from the saved scores.

Report the complete v0.2 metrics without selecting favorable datasets or scorers. Also provide a locked v0.2-minus-v0.1 comparison for target counts, selected thresholds, TP, FP, FN, precision, recall, F1, false alarms/hour, bootstrap intervals, and gate decision.

## 6. Interpretation rules

- If the point gate changes, replace the v0.1 advancement decision with the corrected v0.2 decision.
- If the point gate is unchanged, still use v0.2 values because v0.1 did not implement its stated label rule.
- A passing point gate does not establish reliability when recall is low or the participant-bootstrap interval includes zero.
- A PSG-only result cannot establish wearable performance, clinical detection, diagnosis, or utility for sleep paralysis or narcolepsy.
- Preserve the defect, correction, and any adverse result as part of the experimental record.

## 7. References

1. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9(1):81-87. https://doi.org/10.5664/jcsm.2350
2. Guillot A, Sauvet F, During EH, Thorey V. Dreem Open Datasets: Multi-Scored Sleep Datasets to Compare Human and Automated Sleep Staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2020;28(9):1955-1965. https://doi.org/10.1109/TNSRE.2020.3011181
3. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
4. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
