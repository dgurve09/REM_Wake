# Multi-Dataset PSG External-Generalization Decision

**Decision date:** 2026-10-25  
**Frozen protocol:** `multidataset_psg_external_generalization_protocol_v0.1.md`  
**Primary external datasets:** gx36m and ISRUC group 3  
**Descriptive reused-source holdout:** BOAS

## Question

Does the causal `ADV3-CHANGE` representation carry REM-to-Wake boundary information across independent PSG datasets, or was its earlier positive direction specific to BOAS development data?

## Controlled test

The experiment used all 196 eligible raw-PSG nights from 130 participant groups and all 521 reviewed REM-to-Wake events. Each dataset was held out in turn. Model fitting and threshold selection used only the other two datasets, with participant-grouped inner folds. The fixed natural-prevalence logistic model compared 120 baseline features with 444 baseline-plus-advanced features.

gx36m and ISRUC are the primary external evaluations because they did not inform the earlier BOAS model development. BOAS is descriptive only. This distinction was frozen before fitting.

## Result

Across gx36m and ISRUC, `ADV3-CHANGE` reached precision 0.2346, recall 0.2436, F1 0.2390, and 0.2161 false alarms/hour. `PSD3-BASE` reached precision 0.1890, recall 0.1987, F1 0.1938, and 0.2318 false alarms/hour.

The advanced-minus-baseline differences were `+0.0452` F1, `+0.0449` recall, and `-0.0157` false alarms/hour. The advanced model detected 38/156 external events versus 31/156 for the baseline. The predefined point gate passed all seven conditions.

The stratified participant-bootstrap intervals remained inconclusive:

- F1 difference: `-0.0208` to `+0.1035`;
- recall difference: `-0.0390` to `+0.1222`; and
- false-alarm difference: `-0.0672` to `+0.0358/hour`.

The result was not uniform. gx36m F1 improved from 0.1812 to 0.1992, while ISRUC F1 improved from 0.2727 to 0.4211 on only ten participants and 32 events. The descriptive BOAS result was effectively unchanged at 0.1975 baseline versus 0.1971 advanced. The advanced direction was also adverse under gx36m expert 2 labels, despite being favorable under gx36m experts 1 and 3 and both ISRUC experts.

## Interpretation

The experiment provides point evidence against the narrower concern that the advanced feature direction exists only in BOAS: a positive point difference appeared in both independently held-out PSG datasets. It does not establish a stable universal improvement because the participant interval includes zero, the ISRUC contribution is small, and scorer-specific gx36m results disagree.

Absolute performance remains insufficient for a reliable detector. The advanced model missed 118/156 primary external events. This PSG result also does not validate wearable EEG, sleep-paralysis detection, narcolepsy diagnosis, or clinical utility.

## Decision

Retain `ADV3-CHANGE` for a separately predeclared paired PSG-to-wearable transfer experiment. Do not tune this v0.1 representation or alarm rule against the held-out results. Any all-196-night fit is an unevaluated transfer-source model, not another performance estimate.

The next technical question is whether the cross-dataset PSG feature direction survives projection to simultaneously recorded wearable channels under fixed feature mappings and a new participant-grouped evaluation. The label-only datasets remain assigned to transition-prevalence and annotation-uncertainty analyses; they are not counted as raw-signal training samples.

## Verification

All 36 fits converged. The run passed 12/12 internal controls. An independent reconstruction passed 13/13 checks twice, verified 42 external artifact hashes, and reproduced thresholds, 1,057 alarms, event metrics, participant summaries, ranking diagnostics, scorer sensitivity, bootstrap intervals, and the decision gate.

The first result-validator execution retained an expected-count failure: it counted all feature-grid rows rather than excluding ISRUC consensus-unknown epochs. Correcting the audit expectation from 756,108 to 750,636 score rows changed no scientific output.

## References

1. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
2. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
3. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*. 2015;10(3):e0118432. https://doi.org/10.1371/journal.pone.0118432
4. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
