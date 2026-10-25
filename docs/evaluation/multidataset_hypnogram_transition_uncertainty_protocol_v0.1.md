# Multi-Dataset Hypnogram Transition-Uncertainty Protocol v0.1

**Date frozen:** 2026-10-25  
**Analysis type:** Label-only descriptive mechanism study  
**Signal use:** None  
**Model fitting:** None

## 1. Question

How strongly do REM-to-Wake event frequency, within-night timing, human-scorer choice, and repeated-night sampling vary across the locally harmonized public hypnograms?

This analysis addresses label and sampling uncertainty. It does not estimate EEG classifier performance and does not add hypnogram-only records to a signal-training sample.

## 2. Input scope

The primary corpus contains ten harmonized sources under `Project_hypno`:

| Source | Stored scorer records | Primary night rule |
|---|---:|---|
| Challenge 2018 | 994 | Provided human hypnogram |
| DOD-H/DOD-O | 400 | Five scorers collapsed to one 3-of-5 epoch majority per 80 nights |
| ISRUC | 252 | Two experts collapsed to agreement-only labels per 126 recordings |
| Sleep-EDF Expanded | 197 | Provided human hypnogram |
| HMC sleep staging | 151 | Provided human hypnogram |
| BOAS | 128 | Published three-human consensus (`stage_hum`) |
| CAP Sleep | 108 | Provided human hypnogram |
| iSLEEPS | 100 | Provided human hypnogram |
| Sleep-Accel | 31 | Provided PSG-scored hypnogram |
| DREAMS | 20 | Provided AASM hypnogram, harmonized from 5-second to 30-second epochs |

These 2,381 stored scorer records represent 1,935 primary nights after DOD and ISRUC scorer collapse. The separately stored gx36m source adds 58 nights from 20 participants, each with three scorers collapsed by 2-of-3 epoch majority. The expected complete analysis pool is therefore 1,993 primary nights from eleven sources.

No `stage_ai` label is used. BOAS human consensus is the only BOAS label source.

## 3. Stage and consensus rules

The known common stages are `W`, `N1`, `N2`, `N3`, and `REM`. `UNKNOWN`, missing, movement, and unscored epochs break a run and cannot form an event or a confirmed REM-exit opportunity.

- DOD: retain an epoch when at least three of all five scorers assign the same known stage; otherwise mark it unknown.
- ISRUC: retain an epoch only when both experts assign the same known stage; otherwise mark it unknown.
- gx36m: retain an epoch when at least two of three experts assign the same known stage; otherwise mark it unknown.
- Other sources: use the stored harmonized human stage.
- Restrict multi-scorer records to their common epoch support. Trailing epochs missing from any scorer do not enter the primary consensus.

Scorer-specific hypnograms are retained only for the scorer-uncertainty analysis. They cannot replace the primary consensus after results are observed.

## 4. Sleep-period support

Long pre-sleep and post-sleep Wake padding differs across datasets, particularly Sleep-EDF. The primary support for each night is therefore:

1. the first known non-Wake sleep epoch;
2. through the last known non-Wake sleep epoch; and
3. one immediately following known Wake epoch, when present, so a terminal REM-to-Wake event can be represented.

A night without a known sleep epoch is excluded and reported. Unknown epochs inside support remain unknown rather than being removed or bridged.

Full-record counts will be reported as a sensitivity check, but cross-source rates and positions use the fixed sleep-period support.

## 5. Event and opportunity definitions

A primary REM-to-Wake event is an immediately consecutive known `REM` epoch followed by a known `W` epoch. Its time is the shared 30-second boundary.

For each event report:

- preceding contiguous REM-run duration;
- following contiguous Wake-run duration observed in the source;
- whether the Wake run is right-censored by the source end or an unknown epoch;
- normalized position within sleep-period support;
- spacing from the preceding REM-to-Wake event in the same night; and
- the source, recording, participant group, and descriptive cohort group.

A known REM-exit opportunity is a contiguous REM run whose next epoch is a known non-REM stage. The REM-to-Wake fraction is events divided by known REM exits. REM runs ending at an unknown epoch or source end are not opportunities.

## 6. Primary descriptive endpoints

For each source report:

- primary nights and participant groups;
- sleep-period support hours and REM hours;
- REM-to-Wake event count;
- nights with at least one event;
- events per sleep-period hour;
- median and interquartile night-level event count and rate;
- REM-to-Wake fraction among known REM exits;
- event-position quartiles;
- preceding REM-duration quartiles;
- uncensored following-Wake-duration quartiles; and
- right-censored event count.

Do not pool source rows into a single clinical prevalence estimate. Dataset summaries describe different cohorts and annotation systems.

## 7. Cross-source heterogeneity decision

Restrict the fixed heterogeneity screen to sources with at least 20 primary nights and at least 10 events. Mark cross-source event structure as materially heterogeneous if both conditions hold:

1. the maximum-to-minimum aggregate event-rate ratio is at least 2.0; and
2. the range in event-positive-night fraction is at least 0.20.

Also report the same quantities without applying this threshold. This decision describes transportability of the label-derived event prior; it is not a test of biological population differences.

## 8. Human-scorer uncertainty

Evaluate DOD, ISRUC, and gx36m separately. For every scorer pair on the same night report:

- known-epoch stage disagreement fraction;
- absolute REM-to-Wake event-count difference;
- exact-boundary event F1;
- event F1 with one-to-one matching within plus or minus 30 seconds; and
- the number of informative pairs with at least one event under either scorer.

Pairs with no event under either scorer are retained for event-presence agreement but do not receive an event F1 of 1. For each source report medians across informative pairs.

Mark scorer dependence as material if any source has median exact-boundary F1 below 0.80 or median plus-or-minus-30-second F1 below 0.90. These thresholds are decision rules for this project, not clinical standards.

## 9. Repeated-night variability

Use only sources with verified participant linkage and at least two primary nights for a participant:

- BOAS participant-table `pid`;
- Sleep-EDF participant code;
- ISRUC subgroup 2 participant code; and
- gx36m participant code.

For each participant report the number of nights, event-count range, event-rate range, and whether event presence is discordant across nights. For each source report:

- participants with repeated nights;
- fraction with discordant event presence;
- median within-participant event-count range;
- median within-participant event-rate range; and
- interquartile range of participant-mean event rates.

Mark repeated-night dependence as material if any eligible source has at least 25% discordant event presence or a median within-participant event-rate range of at least 0.10 events/hour. This cannot distinguish physiology from annotation or acquisition variability.

## 10. Cohort-group summaries

Report source-specific descriptive groups only when a cell contains at least ten primary nights. Do not merge similarly named groups across sources. Do not perform causal, diagnostic, treatment, medication, or disease-effect inference.

## 11. Required controls

Before calculating result summaries, verify:

- all 2,381 expected harmonized scorer files are present and match metadata lengths;
- DOD has 80 complete five-scorer nights;
- ISRUC has 126 complete two-expert recordings;
- gx36m has 58 complete three-scorer nights;
- all epoch grids are consecutive 30-second grids;
- stage values belong to the allowed vocabulary;
- the primary pool contains 1,993 nights;
- every night maps to exactly one source and participant group;
- consensus construction is deterministic; and
- no model, signal feature, probability, or classifier output is read.

Hash every source hypnogram and the harmonized metadata file. Store only paths relative to the external data root, byte sizes, and SHA-256 values in Git.

## 12. Required outputs

The dated experiment package must include:

- source manifest and pre-analysis input audit;
- primary recording-level transition measures;
- event-level transition table;
- source and source-group summaries;
- scorer-pair record and source summaries;
- repeated-participant and source summaries;
- full-record support sensitivity;
- fixed decision table;
- in-run checks; and
- an independent reconstruction report.

Any failed first execution or changed audit assumption must be retained in the experiment record. After results exist, do not change consensus, support, event, matching, eligibility, or decision rules in v0.1.

## 13. Interpretation boundary

This analysis can establish that the derived event prior, scorer dependence, or repeated-night sampling is heterogeneous. It cannot establish exact physiological transition time, EEG detectability, wearable performance, narcolepsy or sleep-paralysis detection, diagnosis, treatment response, or clinical utility.

## 14. References

1. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9(1):81-87. https://doi.org/10.5664/jcsm.2350
2. Guillot A, Sauvet F, During EH, Thorey V. Dreem Open Datasets: Multi-Scored Sleep Datasets to Compare Human and Automated Sleep Staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2020;28(9):1955-1965. https://doi.org/10.1109/TNSRE.2020.3011181
3. Khalighi S, Sousa T, Santos JM, Nunes U. ISRUC-Sleep: A comprehensive public dataset for sleep researchers. *Computer Methods and Programs in Biomedicine*. 2016;124:180-192. https://doi.org/10.1016/j.cmpb.2015.10.013
4. Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering*. 2000;47(9):1185-1194. https://doi.org/10.1109/10.867928
5. Ying S, Fu J, Wang L, et al. A continuous multi-night sleep database for exploring sleep structure similarity. *Scientific Data*. 2025;12:946. https://doi.org/10.1038/s41597-025-05202-6
6. Walch O, Huang Y, Forger D, Goldstein C. Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device. *Sleep*. 2019;42(12):zsz180. https://doi.org/10.1093/sleep/zsz180
