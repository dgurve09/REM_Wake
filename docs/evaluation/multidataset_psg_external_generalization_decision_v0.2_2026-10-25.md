# Corrected Multi-Dataset PSG External-Generalization Decision v0.2

**Decision date:** 2026-10-25
**Eligible PSG nights:** 196
**Participant groups:** 130
**Corrected stage-derived targets:** 482
**Primary external datasets:** gx36m and ISRUC group 3

## Question

Does the fixed advanced PSG representation retain cross-dataset REM-to-Wake boundary information after correcting gx36m primary consensus and out-of-vocabulary scorer labels, with every other experimental choice held fixed?

## Correction integrity

The pre-fit checkpoint verified all 196 feature-cache hashes, froze all 260 original participant-to-fold assignments, and confirmed that every corrected inner fold retained both classes from both training datasets. The only data interpretation change was gx36m labeling: codes outside `0..4` became `UNKNOWN`, and a primary stage required two known votes.

The corrected target pool contains 482 events: 365 BOAS, 85 gx36m, and 32 ISRUC. The v0.1 pool contained 521 because unresolved gx36m epochs were assigned through unrestricted `argmax`.

## Held-out result

| Held-out evaluation | Baseline F1 | Advanced F1 | Advanced recall | Advanced false alarms/hour |
|---|---:|---:|---:|---:|
| gx36m | 0.1429 | 0.1818 | 0.2235 | 0.2114 |
| ISRUC group 3 | 0.2381 | 0.4151 | 0.3438 | 0.1542 |
| Pooled primary external | 0.1579 | 0.2290 | 0.2564 | 0.2048 |
| BOAS, descriptive | 0.1824 | 0.2550 | 0.2082 | 0.1566 |

On the corrected pooled primary external set, `ADV3-CHANGE` detected 30/117 events and `PSD3-BASE` detected 21/117. The advanced F1 difference is `+0.0711`; recall difference is `+0.0769`; and false-alarm difference is `-0.0232/hour`.

## Uncertainty

The stratified participant-bootstrap 95% interval for the advanced-minus-baseline F1 difference is `-0.0177` to `+0.1396`. The recall-difference interval is `-0.0451` to `+0.1731`, and the false-alarm-rate interval is `-0.0734` to `+0.0263/hour`.

All three corrected gx36m expert variants and both ISRUC expert variants had nonnegative advanced-minus-baseline F1 point differences. This removes the adverse expert-2 direction reported by v0.1, but does not resolve the participant uncertainty or low event recall.

## v0.1 comparison

The corrected primary external reference count decreased from 156 to 117. Absolute advanced F1 changed from 0.2390 to 0.2290, while baseline F1 changed from 0.1938 to 0.1579. The point F1 difference therefore changed from `+0.0452` to `+0.0711`. Both bootstrap intervals include zero, and both point gates pass.

Version 0.2 replaces v0.1 for scientific interpretation because it implements the stated label rule. The v0.1 files remain preserved as the failed implementation record.

## Decision

The corrected point gate passes, so retain `ADV3-CHANGE` as a PSG mechanism candidate for the separately predeclared paired wearable-transfer experiment. Do not describe it as reliable: recall is 0.2564, 87/117 external events were missed, and the participant interval includes no improvement.

Do not optimize this external result after inspection. The next valid model question is whether the frozen PSG representation transfers to simultaneously recorded wearable EEG under a new paired protocol. A PSG-only result cannot support a wearable, diagnostic, sleep-paralysis, or narcolepsy performance claim.

## Verification

- Pre-fit correction audit: 15/15 checks passed.
- Independent input validation: 13/13 checks passed.
- Fixed fits converged: 36/36.
- In-run controls: 13/13 passed.
- Independent result reconstruction: 14/14 passed.
- External artifact hashes: 42/42 passed.
- Version comparison: 10/10 passed.

## References

1. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
2. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
3. Welch PD. The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*. 1967;15(2):70-73. https://doi.org/10.1109/TAU.1967.1161901
4. Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics*. 2006;7:91. https://doi.org/10.1186/1471-2105-7-91
