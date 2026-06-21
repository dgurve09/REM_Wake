# Preliminary Literature Review

## Wearable EEG REM-to-Wake Transition Detection

**Project period covered:** 2026-06-01 to 2026-06-21  
**Record consolidated:** 2026-06-21  
**Review type:** Targeted scoping review  
**Purpose:** Establish the pre-project technological knowledge base, identify defensible gaps, and guide the experimental plan

## 1. Record Status

This document consolidates preliminary literature work associated with the June 1-21 planning period. It is not a claim that a formal systematic review or meta-analysis was completed. Search methods and limitations are recorded so the review can be repeated and expanded.

The evidence supports a narrower project than the original concept. Wearable EEG sleep staging, temporal sleep-stage modeling, and sleep-staging domain adaptation are established research areas. The potentially useful research gap is direct REM-to-Wake event detection from simultaneously recorded wearable EEG, evaluated against a strong stage-first baseline while explicitly handling event rarity and 30-second label uncertainty.

## 2. Questions Addressed

1. Are REM/Wake transitions scientifically meaningful and already studied?
2. How reliable are 30-second human sleep-stage labels near boundaries?
3. Can reduced-channel wearable EEG support sleep staging?
4. What performance can modern automatic sleep-staging methods already achieve?
5. Are cross-dataset and PSG-to-wearable transfer established techniques?
6. Has direct wearable EEG REM-to-Wake event detection already been demonstrated?
7. What experimental comparison would establish new knowledge rather than repeat routine staging work?

## 3. Search Method

### 3.1 Sources

- PubMed for biomedical literature and reproducible query counts.
- OpenAlex for broad scholarly discovery and DOI verification.
- Crossref for current publication metadata and final journal versions.
- Semantic Scholar for paper abstracts and open-access links.
- Publisher, PubMed Central, PhysioNet, OpenNeuro, NSRR, and official dataset pages for primary documentation.

### 3.2 PubMed searches performed on 2026-06-21

The following targeted searches were run. Counts are database results at the search date, not the number of papers finally included.

| Topic | Search concept | Results |
|---|---|---:|
| REM/Wake transition physiology | `"REM sleep" AND wake AND transition* AND (EEG OR electroencephalography)` in title/abstract | 90 |
| Wearable EEG staging | `(wearable OR headband OR ear-EEG) AND sleep AND (staging OR scoring)` in title/abstract | 220 |
| Automated stage-transition methods | `"sleep stage transition" AND ("machine learning" OR "deep learning" OR automated)` in title/abstract | 7 |
| Domain adaptation | `"sleep staging" AND "domain adaptation"` in title/abstract | 17 |
| Human scoring reliability | `"sleep stage scoring" AND (interscorer OR interrater)` in title/abstract | 23 |

The seven results from the narrow automated stage-transition search were checked by title. They primarily concerned conventional sleep staging, temporal context, or boundary refinement. None of those seven titles described direct REM-to-Wake event detection from wearable EEG. This is evidence of a possible gap, not proof that no such study exists.

### 3.3 Inclusion priorities

- Human sleep studies.
- EEG or PSG measurements.
- Wearable validation against simultaneous PSG.
- Explicit transition, boundary, temporal-context, uncertainty, or domain-transfer analysis.
- Public datasets or methods relevant to reproducible experiments.
- Peer-reviewed papers where available; a preprint was used only when directly connected to BOAS and was replaced by its final 2026 journal article when verified.

### 3.4 Exclusions from the core synthesis

- Consumer-device claims without PSG comparison.
- Drowsiness detection while awake.
- Animal-only sleep-transition physiology.
- REM behavior disorder detection when it did not inform stage boundaries.
- Studies using only PPG or actigraphy unless needed as a contrast.
- Papers without a verifiable title, authors, journal or repository, and DOI or stable official link.

## 4. Evidence Synthesis

## 4.1 REM/Wake transitions are meaningful, but not a new concept

REM/Wake transition structure has been studied in clinical sleep research. Weinhold et al. specifically compared Wake-REM transitions among narcolepsy, idiopathic hypersomnia, and healthy controls. Sorensen et al. examined sleep-transition patterns in hypocretin-deficient narcolepsy. These studies show that transition organization can contain disease-relevant information, so the general concept of transition analysis is prior art, not the proposed advancement [1,2].

Bianchi et al. analyzed stage-transition dynamics in Sleep Heart Health Study hypnograms and found that obstructive sleep apnea was associated with shorter REM and NREM bouts and more transitions. Importantly, they reported that one night can contain too few transitions to characterize individual dynamics reliably. That result directly motivates an event-count feasibility gate and cautions against strong per-person stability claims from a small number of nights [3].

Stephansen et al. showed that probabilistic sleep-stage outputs, or hypnodensity representations, can reveal stage ambiguity relevant to narcolepsy. Their work used approximately 3,000 recordings and demonstrated that conventional hard hypnograms discard information. It supports retaining probabilities and uncertainty rather than reducing every epoch immediately to a hard class [4].

**Implication for this project:** Transition analytics are scientifically plausible, but the novelty cannot be stated as “using sleep transitions.” The narrower question is whether a direct wearable EEG detector provides event-level value beyond transitions derived from ordinary wearable sleep staging.

## 4.2 REM detection is not the same as REM-to-Wake event detection

Single-channel EEG REM detection has already been demonstrated. Imtiaz and Rodriguez-Villegas presented a low-computational-cost algorithm for REM stage detection from a single EEG channel [5]. Modern staging models also classify REM as one of several stages.

However, classifying an isolated epoch as REM does not establish that a REM-to-Wake boundary was detected correctly. Event detection requires at least:

- correct ordering of REM followed by Wake;
- temporal matching between predicted and reference events;
- control of duplicate alarms around one boundary;
- false alarms per hour or per night;
- handling of short awakenings and uncertain labels;
- participant-independent evaluation.

**Implication:** A model that reports high REM sensitivity may still produce poor REM-to-Wake event precision. Stage-level accuracy cannot substitute for event-level evaluation.

## 4.3 Thirty-second human labels are useful but imperfect

Manual PSG scoring remains the reference standard, but it is not error-free. Lee et al. synthesized 11 interrater studies and reported an overall Cohen kappa of 0.76. Stage-specific kappa was 0.70 for Wake, 0.69 for REM, and only 0.24 for N1. Thus, even human scorers disagree, especially at ambiguous or transitional epochs [6]. Earlier multi-scorer studies also documented substantial but incomplete agreement [7,8].

Standard hypnograms label consecutive epochs, commonly 30 seconds long. A change from REM to Wake between two scored epochs defines a scoring boundary. It does not reveal the exact physiological second at which REM-related EEG, eye movement, muscle tone, or arousal characteristics changed.

U-Sleep demonstrated that a model trained using standard labels can emit shorter-interval predictions, and the authors argued that high-frequency segmentation may add information [9]. Stephansen et al. similarly produced five-second hypnodensity outputs [4]. These are important methodological precedents, but shorter model outputs do not create independent one-second ground truth.

BTCRSleep explicitly models boundary temporal context because discriminative waveforms can cross epoch boundaries. Its reported improvement across Sleep-EDF, SHHS, and CAP supports the idea that boundaries require temporal context [10]. The target remained sleep-stage classification rather than direct REM-to-Wake event detection.

**Implications:**

- The project should represent each transition as a scored boundary with an uncertainty interval.
- One-second windows may be analyzed for localization evidence, but not scored as exact ground truth.
- Sensitivity analyses should test label tolerance and boundary definitions.
- Model confidence or entropy should be retained near ambiguous boundaries.

## 4.4 Wearable EEG sleep staging is technically feasible and established

A 2024 systematic review included 60 papers covering 34 unique EEG wearables and 42 validation studies. It concluded that many devices show good feasibility and generally high staging accuracy, while device properties and validated populations vary [11]. A 2025 meta-analysis of 43 validation studies found moderate-to-substantial agreement with PSG, variation by sleep stage, persistent N1 difficulty, and comparatively reliable N3 detection [12].

Several direct validation studies are especially relevant:

- Arnal et al. studied 25 simultaneous Dreem headband and PSG recordings. Automatic headband staging achieved mean accuracy of 83.5% and F1 of 83.8%, compared with average expert values of 86.4% and 86.3% [13].
- Mikkelsen et al. collected 80 full-night simultaneous PSG and dry-contact ear-EEG recordings from 20 healthy participants. Ear-EEG automatic scoring achieved average Cohen kappa of 0.73 [14].
- Chen et al. evaluated a single-channel forehead EEG recorder in 197 participants, most with obstructive sleep apnea. Reported sensitivities were 79.7% for Wake and 82.7% for REM, with substantial kappa agreement [15].
- The BOAS dataset contains 128 nights from 108 unique participants with simultaneous clinical PSG and a two-channel forehead EEG headband. Its consensus stage labels are derived from multiple scorers [16].

The final 2026 BOAS-related modeling paper reported that a single frontal channel could support reliable sleep scoring, additional sensors yielded limited improvement in that study, real-time scoring could approach offline scoring, and performance decreased in sleep-disordered participants [17].

**Implication:** It would not be defensible to claim that detecting REM or staging sleep from wearable EEG is itself new. BOAS is valuable because its simultaneous paired recordings allow a controlled transition-specific comparison across devices.

## 4.5 Modern staging baselines are strong and use temporal context

SeqSleepNet formulated sleep staging as sequence-to-sequence prediction rather than independent epoch classification. On a public dataset with 200 subjects, it reported 87.1% accuracy, 83.3% macro F1, and kappa of 0.815 [18].

U-Sleep trained on 15,660 participants from 16 studies and was designed to operate across cohorts and typical EEG/EOG channel combinations. Its scale and channel flexibility show that large, general staging models already exist [9].

SleepTransformer added sequence context, attention-based interpretation, and entropy-based uncertainty. It illustrates that an appropriate baseline can expose uncertain epochs rather than returning only hard labels [19].

Reviews of automatic EEG sleep staging conclude that performance on many healthy cohorts can approach scorer agreement, while clinical adoption, generalization, scoring ambiguity, data variation, and interpretability remain unresolved [20].

**Implications:**

- A weak independent-epoch classifier is not a sufficient comparator.
- The primary stage-first baseline should use temporal context and output probabilities.
- The direct transition detector must be compared with transitions derived from this strong baseline.
- Complex architectures should be added only after simple baselines and error analysis.

## 4.6 Cross-dataset generalization is a known problem

Alvarez-Estevez and Rijsman evaluated an automatic staging architecture across six public databases. Average kappa was 0.80 on independent local test data but fell to 0.54 when local models predicted external datasets; an ensemble improved external kappa to 0.62 [21]. This demonstrates that internal cross-validation can substantially overestimate external performance.

Heremans et al. evaluated adversarial domain adaptation on wearable sleep data. Relative accuracy gains of 7%-27% were reported over unadapted target application, with smaller personalization improvements. The paper also found that source-target compatibility and available target labels affected results [22]. Domain adaptation is therefore established prior art, not automatically an advancement.

**Implications:**

- First quantify BOAS PSG-to-wearable degradation.
- Compare no adaptation, direct wearable training, and simple fine-tuning before adversarial methods.
- Use paired recordings to separate channel/device effects from participant effects.
- Treat reduced-channel PSG as simulated wearable input, not real wearable validation.
- External PSG datasets can test generalization but cannot replace a second wearable dataset.

## 4.7 Evidence for the specific proposed gap

The targeted searches identified extensive work on:

- wearable EEG validation;
- REM stage detection;
- sleep-transition statistics;
- narcolepsy transition patterns;
- boundary-aware staging;
- temporal sequence models;
- high-resolution probabilistic staging;
- domain adaptation and cross-dataset staging.

No directly matching paper was identified in this scoping search that simultaneously satisfied all of the following:

1. direct REM-to-Wake event detection as the primary target;
2. real wearable EEG input;
3. event-level precision, recall, timing tolerance, and false-alarm evaluation;
4. participant-independent validation;
5. comparison with a stage-first transition baseline;
6. explicit treatment of 30-second boundary uncertainty.

This is a provisional gap statement. A publication discovered later could narrow or remove it. The project should therefore say “not identified in the targeted review” rather than “no prior work exists.”

## 5. Consequences for Experimental Design

### 5.1 Primary endpoint

REM-to-Wake event detection from BOAS wearable EEG.

### 5.2 Exploratory endpoint

Wake-to-REM events only if the feasibility audit finds adequate independent examples.

### 5.3 Required baselines

1. no-event and prevalence checks;
2. temporal stage-first wearable model followed by transition derivation;
3. simple direct feature model;
4. compact direct CNN;
5. temporal direct model only if justified by errors.

### 5.4 Required metrics

- event precision, recall, and F1;
- precision-recall area under the curve;
- false alarms per hour and per night;
- event matching at predeclared tolerances such as 30 and 60 seconds;
- participant-level distributions and confidence intervals;
- calibration and uncertainty near boundaries.

### 5.5 Required split unit

BOAS `pid`, because 128 nights come from 108 unique participants. Recording-level random splitting could place different nights from one person in training and testing.

## 6. Clinical Interpretation Limits

The literature supports clinical relevance of transition organization in narcolepsy and sleep instability, but BOAS is a general-population dataset without event labels for sleep paralysis or a dedicated narcolepsy cohort. Therefore this project cannot validate:

- sleep-paralysis detection or prevention;
- narcolepsy screening;
- a clinical REM-stability biomarker;
- treatment response;
- a closed-loop intervention.

These remain future applications requiring suitable cohorts, outcomes, ethics review, and prospective validation.

## 7. Review Limitations

- This was a targeted scoping review, not a PRISMA systematic review.
- Searches emphasized English-language indexed literature.
- Citation databases differ in indexing and publication dates.
- Conference papers, patents, dissertations, and proprietary commercial studies were not exhaustively searched.
- Full-text review was concentrated on the most relevant open or abstract-indexed papers.
- Search terms may miss papers that use “arousal,” “awakening,” “boundary,” or “state change” without the phrase REM-to-Wake.
- Absence of a matching paper in these searches is not proof of worldwide novelty.

## 8. Literature Review Conclusion

The project is scientifically reasonable only if it is framed as a transition-specific event-detection and uncertainty problem. The technological knowledge base already supports wearable EEG acquisition, REM staging, temporal sleep models, high-frequency predictions, and domain adaptation. New knowledge may come from determining whether direct REM-to-Wake detection adds event-level value over stage-first inference on paired wearable data, and from documenting why it succeeds or fails under event scarcity, participant variation, device shift, and coarse labels.

## 9. References

1. Weinhold SL, Seeck-Hirschner M, Nowak A, Goder R, Baier PC. Wake-REM sleep transitions for measuring REM sleep disturbance: comparison between narcolepsy, idiopathic hypersomnia and healthy controls. 2011. https://doi.org/10.1111/j.1479-8425.2011.00503.x
2. Sorensen GL, Knudsen S, Jennum P. Sleep transitions in hypocretin-deficient narcolepsy. *Sleep*. 2013;36:1173-1177. https://doi.org/10.5665/sleep.2880
3. Bianchi MT, Cash SS, Mietus J, Peng CK, Thomas R. Obstructive sleep apnea alters sleep stage transition dynamics. *PLoS ONE*. 2010;5:e11356. https://doi.org/10.1371/journal.pone.0011356
4. Stephansen JB, Olesen AN, Olsen M, et al. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. *Nature Communications*. 2018;9:5229. https://doi.org/10.1038/s41467-018-07229-3
5. Imtiaz SA, Rodriguez-Villegas E. A low computational cost algorithm for REM sleep detection using single channel EEG. *Annals of Biomedical Engineering*. 2014;42:2344-2359. https://doi.org/10.1007/s10439-014-1085-6
6. Lee YJ, Lee JY, Cho JH, Choi JH. Interrater reliability of sleep stage scoring: a meta-analysis. *Journal of Clinical Sleep Medicine*. 2022;18:193-202. https://doi.org/10.5664/jcsm.9538
7. Danker-Hopfe H, Anderer P, Zeitlhofer J, et al. Interrater reliability for sleep scoring according to the Rechtschaffen and Kales and the new AASM standard. *Journal of Sleep Research*. 2009;18:74-84. https://doi.org/10.1111/j.1365-2869.2008.00700.x
8. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9:81-87. https://doi.org/10.5664/jcsm.2350
9. Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum P, Igel C. U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*. 2021;4:72. https://doi.org/10.1038/s41746-021-00440-5
10. Zhao C, Li J, Guo Y. BTCRSleep: a boundary temporal context refinement-based fully convolutional network for sleep staging with single-channel EEG. *Physiological Measurement*. 2023;44. https://doi.org/10.1088/1361-6579/acdb46
11. de Gans CJ, Burger PC, van den Ende ES, et al. Sleep assessment using EEG-based wearables: a systematic review. *Sleep Medicine Reviews*. 2024;76:101951. https://doi.org/10.1016/j.smrv.2024.101951
12. Markov K, Elgendi M, Menon C. Evaluating the performance of wearable EEG sleep monitoring devices: a meta-analysis approach. *npj Biomedical Innovations*. 2025. https://doi.org/10.1038/s44385-025-00034-w
13. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
14. Mikkelsen KB, Tabar YR, Kappel SL, et al. Accurate whole-night sleep monitoring with dry-contact ear-EEG. *Scientific Reports*. 2019;9:16824. https://doi.org/10.1038/s41598-019-53115-3
15. Chen X, Jin X, Zhang J, et al. Validation of a wearable forehead sleep recorder against polysomnography in sleep staging and desaturation events in a clinical sample. *Journal of Clinical Sleep Medicine*. 2023;19:711-718. https://doi.org/10.5664/jcsm.10416
16. Lopez-Larraz E, Sierra-Torralba M, Clemente S, et al. Bitbrain Open Access Sleep Dataset. OpenNeuro, version 1.2.1. https://doi.org/10.18112/openneuro.ds005555.v1.2.1
17. Esparza-Iaizzo M, Sierra-Torralba M, Klinzing JG, Minguez J, Montesano L, Lopez-Larraz E. Automatic sleep scoring for real-time monitoring and stimulation in individuals with and without sleep apnea. *Computers in Biology and Medicine*. 2026;205:111560. https://doi.org/10.1016/j.compbiomed.2026.111560
18. Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet: end-to-end hierarchical recurrent neural network for sequence-to-sequence automatic sleep staging. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2019;27:400-410. https://doi.org/10.1109/TNSRE.2019.2896659
19. Phan H, Mikkelsen KB, Chen OY, Koch P, Mertins A, De Vos M. SleepTransformer: automatic sleep staging with interpretability and uncertainty quantification. *IEEE Transactions on Biomedical Engineering*. 2022;69:2456-2467. https://doi.org/10.1109/TBME.2022.3147187
20. Phan H, Mikkelsen KB. Automatic sleep staging of EEG signals: recent development, challenges, and future directions. *Physiological Measurement*. 2022;43. https://doi.org/10.1088/1361-6579/ac6049
21. Alvarez-Estevez D, Rijsman RM. Inter-database validation of a deep learning approach for automatic sleep scoring. *PLoS ONE*. 2021;16:e0256111. https://doi.org/10.1371/journal.pone.0256111
22. Heremans ERM, Phan H, Borzee P, Buyse B, Testelmans D, De Vos M. From unsupervised to semi-supervised adversarial domain adaptation in electroencephalography-based sleep staging. *Journal of Neural Engineering*. 2022;19. https://doi.org/10.1088/1741-2552/ac6ca8

