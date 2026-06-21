# Technological Knowledge Assessment

## Wearable EEG REM-to-Wake Transition Detection

**Project period covered:** 2026-06-01 to 2026-06-21  
**Record consolidated:** 2026-06-21  
**Assessment date for starting knowledge base:** 2026-06-01  
**Purpose:** Separate established technical capability from unresolved technological uncertainty and define the advancement sought

## 1. Assessment Basis

This assessment uses the literature and dataset evidence recorded in:

- `01_literature_review.md`
- `02_dataset_search.md`
- `Proposal.md`

The assessment applies the current Canada Revenue Agency description of SR&ED eligibility as a planning framework, not as a guarantee of tax eligibility or professional tax advice. CRA states that eligible work must both:

1. seek scientific or technological advancement; and
2. use systematic investigation or search through experiment or analysis.

Routine implementation, quality control, ordinary data collection, training, and acquisition of existing know-how do not become eligible merely because they are difficult or documented [1].

## 2. System Boundary

### Input system

- Two-channel forehead wearable EEG from BOAS, approximately AF7 and AF8.
- Optional IMU and PPG used only as separately reported ablations.
- Human consensus PSG sleep stages used as the reference sequence.

### Primary output

A validated derived event-label specification, followed by a probability and event decision for REM-to-Wake boundaries. The project output is not a general sleep-stage classifier.

### Primary comparator

A temporal wearable sleep-stage model whose predicted stages are converted to REM-to-Wake events.

### Evaluation boundary

- Unseen participants grouped by BOAS `pid`.
- Event-level matching and false-alarm burden.
- Explicit 30-second label uncertainty.
- No diagnosis, intervention, or medical-device claim.

## 3. Technological Knowledge Already Available

## 3.1 Wearable EEG can support sleep staging

This is established enough that it cannot be the central advancement. Simultaneous PSG studies have shown useful staging agreement for forehead headbands and ear-EEG. Systematic and meta-analytic evidence now covers dozens of wearable systems and validation studies [2-6].

The BOAS-related 2026 paper further reports reliable single-frontal-channel staging and comparable real-time versus offline scoring under the studied conditions [7].

**Available know-how:** A practitioner can select standard preprocessing, train a classifier, and calculate conventional epoch-level stage metrics.

## 3.2 Temporal context improves sleep staging

Sequence-to-sequence recurrent, convolutional, and transformer models are established. SeqSleepNet, U-Sleep, SleepTransformer, and boundary-context models all exploit neighboring epochs or longer temporal structure [8-11].

**Available know-how:** Using temporal context is standard method selection, not new knowledge by itself.

## 3.3 REM can be detected from reduced EEG

Single-channel REM classification has been published, and modern staging systems already treat REM as a normal output class [12].

**Available know-how:** A REM classifier does not itself constitute a new transition detector.

## 3.4 Sleep-stage transition statistics have prior art

Researchers have analyzed Wake-REM transitions in hypersomnia/narcolepsy, transition patterns in hypocretin-deficient narcolepsy, and stage-transition dynamics in obstructive sleep apnea [13-15].

**Available know-how:** Counting or statistically summarizing transitions from a hypnogram is not new.

## 3.5 Domain adaptation and external generalization are established methods

Cross-database performance degradation is documented, as are transfer learning and adversarial domain-adaptation methods for wearable sleep staging [16,17].

**Available know-how:** Applying an existing domain-adaptation package or published algorithm without a new technical question is routine method use.

## 3.6 Shorter-than-30-second model outputs are possible

Published systems can emit high-frequency or probabilistic stage outputs even when trained on conventional labels [9,18].

**Available know-how:** Generating one-second predictions is computationally possible.

**Critical limitation:** Prediction resolution is not ground-truth resolution. Existing 30-second labels do not reveal the exact physiological transition second.

## 4. Knowledge Not Established for This System

The targeted review did not identify established answers to the following combined questions.

### K0. Derived event-target validity

BOAS provides 30-second sleep stages, not REM-to-Wake event labels. It is unknown how adjacency rules, uncertainty intervals, artifacts, disconnections, device coverage, and event-matching definitions affect the resulting event population and conclusions. A reproducible label table can be generated mechanically, but its validity as a first-class wearable prediction target must be established empirically.

### K1. BOAS event sufficiency

The number and participant distribution of usable REM-to-Wake events after wearable signal-quality filtering are unknown.

### K2. Wearable transition information

It is unknown whether two frontal wearable channels preserve transition-specific information beyond ordinary stage classification and temporal smoothing.

### K3. Direct versus stage-first value

It is unknown whether direct event training improves event recall, precision, timing tolerance, or false alarms compared with deriving transitions from a strong stage model.

### K4. Paired PSG-to-wearable transfer

General staging transfer is studied, but the magnitude and structure of paired device shift for REM-to-Wake event detection in BOAS are unknown.

### K5. Boundary-label uncertainty

It is unknown whether uncertainty-aware targets improve event detection without creating unjustified pseudo-labels from 30-second stages.

### K6. Individual-level stability

It is unknown whether the limited repeated nights in BOAS support reliable participant-level transition measures.

## 5. Why Routine Engineering Is Insufficient

A competent machine-learning practitioner can implement an EEG classifier, but cannot determine the following from standard practice or published parameters alone:

- whether the target event occurs often enough in BOAS;
- whether the headband signal contains discriminative boundary information;
- whether apparent performance survives participant grouping;
- whether a direct detector outperforms a stage-derived detector;
- whether improvement comes from EEG, PPG, IMU, or temporal priors;
- whether device alignment or scorer uncertainty dominates errors;
- whether one-second outputs represent evidence or only interpolation of coarse labels.
- whether conclusions remain stable under alternative defensible transition-label derivations.

These outcomes require experiments on the paired data. Merely choosing a different neural-network architecture or tuning hyperparameters is not the uncertainty.

## 6. Technological Uncertainties and Experiments

| ID | Technological uncertainty | Testable hypothesis | Required experiment | Resolution evidence |
|---|---|---|---|---|
| U0 | Do derived hypnogram transitions define a stable target? | A deterministic specification with explicit uncertainty produces reproducible labels and interpretable conclusions across sensitivity rules | Generate versioned event tables under primary and alternative rules; inspect events and compare counts, exclusions, and downstream metrics | Label specification, validated event table, sensitivity analysis, and documented limitations |
| U1 | Are enough independent events available? | Usable REM-to-Wake events are sufficiently distributed for grouped evaluation | Count events after quality control by `pid`, night, and direction | Feasibility report with counts, distributions, exclusions, and expected interval width |
| U2 | Does wearable EEG contain transition information? | EEG windows around true boundaries differ sufficiently from matched non-boundary windows | Simple feature model and compact direct model on held-out participants | Event precision-recall, false alarms, participant distribution, error analysis |
| U3 | Does direct detection add value? | Direct event training improves at least one predeclared event metric at comparable false-alarm burden | Compare stage-first and direct models using identical grouped splits | Paired participant-level metric differences and confidence intervals |
| U4 | Can knowledge transfer across devices? | Paired PSG pretraining or fine-tuning improves wearable event detection over unadapted transfer | Compare PSG, reduced PSG, wearable-only, zero-shot, and fine-tuned conditions | Device-specific degradation and improvement with uncertainty intervals |
| U5 | Can label uncertainty be handled usefully? | Interval-aware targets improve calibration or event matching without unstable pseudo-label dependence | Compare hard boundaries, tolerance windows, and uncertainty-aware loss or targets | Sensitivity analysis across label definitions and tolerances |
| U6 | Are transition measures repeatable? | Repeated nights show measurable within-person consistency beyond threshold artifacts | Analyze eligible repeated-night participants only | Reliability estimates with sample size and uncertainty; no claim if underpowered |

## 7. Advancement Sought

The proposed advancement is new technological knowledge about the feasibility and limits of wearable REM-to-Wake event detection, specifically:

1. how to define and validate a derived REM-to-Wake event target from coarse hypnograms without hiding uncertainty;
2. whether the resulting event population is sufficiently stable and represented for participant-independent analysis;
3. whether a paired two-channel wearable signal supports participant-independent transition detection;
4. whether direct event modeling provides value beyond established sleep staging;
5. how real paired device shift affects transition events;
6. how coarse human boundary labels constrain temporal claims;
7. which failure modes prevent reliable transition analytics.

The advancement is knowledge, not merely a trained model. A negative result can provide advancement if systematic experiments show why the available signals, labels, or event prevalence are insufficient.

## 8. What Would Not Constitute the Advancement

- Downloading BOAS.
- Converting EDF files.
- Training a standard sleep-stage CNN.
- Reproducing published staging accuracy.
- Trying many architectures without a prior hypothesis.
- Hyperparameter search performed only to maximize a score.
- Producing a GUI or command-line interface.
- Mechanically calculating `REM -> Wake` rows from human hypnograms without testing the target definition, uncertainty, or downstream validity.
- Writing documentation or a manuscript.
- Calling one-second predictions one-second truth.

These tasks may support the investigation, but they are not the technological advancement.

The derived label table is still an important research artifact. Its knowledge contribution comes from the systematic choices, validation, sensitivity analysis, event characterization, and learnability experiments built around it, not from relabeling alone.

## 9. Working Hypotheses and Falsification

### H0. Derived label validity

**Hypothesis:** A deterministic transition specification with uncertainty and quality flags produces reproducible event labels, and the main feasibility conclusion is not an artifact of one arbitrary derivation rule.

**Falsified or weakened if:** reasonable label definitions produce materially conflicting event populations or reverse the primary conclusion.

### H1. Feasibility

**Hypothesis:** BOAS has enough artifact-free REM-to-Wake events across independent participants for grouped evaluation.

**Falsified or weakened if:** events are concentrated in too few participants, removed by alignment/quality checks, or produce unusably wide performance intervals.

### H2. Direct event value

**Hypothesis:** A direct detector improves event sensitivity at a controlled false-alarm rate compared with the stage-first baseline.

**Falsified if:** differences are absent, inconsistent across participants, or explained by threshold selection.

### H3. Paired transfer

**Hypothesis:** Paired PSG information improves wearable transition detection through pretraining or simple fine-tuning.

**Falsified if:** wearable-only training performs equivalently, transfer is unstable, or apparent gains disappear on held-out participants.

### H4. Uncertainty-aware boundaries

**Hypothesis:** Representing a transition as an uncertain interval improves calibration or tolerance-based event detection.

**Falsified if:** results are insensitive to the representation or depend strongly on arbitrary pseudo-label choices.

## 10. Measurement Requirements

### Event metrics

- precision, recall, and F1 after one-to-one event matching;
- precision-recall area under the curve;
- false alarms per hour and per night;
- detection at predeclared 30-second and 60-second tolerances;
- probability calibration;
- participant-level results and confidence intervals.

### Controls

- no-event and prevalence baselines;
- fixed participant groups across compared models;
- untouched final test group;
- repeated analysis under alternative boundary tolerances;
- EEG-only result before PPG/IMU ablations;
- device timestamp and alignment checks.

### Evidence quality

- dataset version and manifest;
- versioned transition-label specification and deterministic generation code;
- derived event table with source, quality, uncertainty, and derivation-rule fields;
- manual inspection and sensitivity checks for label construction;
- configuration for each run;
- linked Git commit;
- raw metric tables, not only selected figures;
- failed and inconclusive experiments retained.

## 11. Decision Gates

### Gate A: Existing-knowledge boundary

**Question:** Is the planned experiment distinguishable from established wearable staging and transition statistics?

**Current decision:** Yes, only under the narrowed direct-event comparison. Generic wearable staging would not pass.

### Gate B: Dataset feasibility

**Question:** Are event count, alignment, and quality adequate?

**Decision options:** Proceed, narrow to REM-to-Wake only, redesign target, or stop.

### Gate C: Baseline comparison

**Question:** Does direct detection add evidence beyond stage-first inference?

**Decision options:** Continue direct modeling, retain stage-first as sufficient, or investigate one documented failure mode.

### Gate D: Device shift

**Question:** Is there a measurable PSG-to-wearable gap that simple transfer does not resolve?

**Decision options:** Use simple fine-tuning, test one justified adaptation method, or conclude adaptation is unnecessary.

### Gate E: Temporal analysis

**Question:** Do short-window outputs remain interpretable under 30-second label uncertainty?

**Decision options:** Report interval localization, restrict to epoch-level events, or stop fine-resolution claims.

## 12. SR&ED Work Classification for Planning

### Candidate experimental work

- defining alternative transition-label rules and testing whether they yield a stable, valid event target;
- event feasibility experiments tied to U1;
- direct versus stage-first comparison tied to U2/U3;
- paired device-transfer experiments tied to U4;
- boundary-uncertainty experiments tied to U5;
- documented failure analysis required to resolve those uncertainties.

### Potential support work when directly required

- programming the label and evaluation pipeline;
- mathematical/event analysis;
- data extraction needed for a specific experiment;
- testing required to verify experimental correctness.

### Routine or separately tracked work

- general literature review and learning existing methods;
- environment setup beyond what is directly required;
- routine data conversion and storage management;
- ordinary model implementation;
- routine quality assurance;
- interface development;
- manuscript and claim preparation;
- administrative and financial work.

The final treatment of work and expenditures requires professional review. Project documentation should not label every hour as eligible by default.

## 13. Record Integrity

For the June 1-21 period, this assessment documents literature review, dataset search, definition of the knowledge base, uncertainty refinement, and planning. It does not claim that preprocessing or model experiments occurred before June 22. Actual hours must come from the researcher's contemporaneous records and must not be invented from this document.

Each later experiment record should include:

- uncertainty ID;
- hypothesis;
- date and actual time;
- dataset and version;
- participant split version;
- code commit;
- configuration;
- result and uncertainty;
- failure or limitation;
- conclusion and next decision.

## 14. Assessment Conclusion

The project has a potentially defensible experimental core, but only after removing generic wearable staging and unsupported clinical claims from the claimed advancement. This is a transition-boundary project: sleep stages are source annotations and a benchmark, not the research output. The strongest research question is whether a reproducibly derived REM-to-Wake target can be detected from paired wearable EEG and yield reliable event-level information beyond stage-first inference under real device shift and coarse labels. The feasibility gate is mandatory because the knowledge gap may be resolved by discovering that the derived event target is too sparse, unstable, or uncertain for the available dataset.

## 15. References

1. Canada Revenue Agency. What work is eligible: Scientific Research and Experimental Development tax incentives. Updated 2026-04-01. https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program/sred-eligibility.html
2. de Gans CJ, Burger PC, van den Ende ES, et al. Sleep assessment using EEG-based wearables: a systematic review. *Sleep Medicine Reviews*. 2024;76:101951. https://doi.org/10.1016/j.smrv.2024.101951
3. Markov K, Elgendi M, Menon C. Evaluating the performance of wearable EEG sleep monitoring devices: a meta-analysis approach. *npj Biomedical Innovations*. 2025. https://doi.org/10.1038/s44385-025-00034-w
4. Arnal PJ, Thorey V, Debellemaniere E, et al. The Dreem Headband compared to polysomnography for electroencephalographic signal acquisition and sleep staging. *Sleep*. 2020;43:zsaa097. https://doi.org/10.1093/sleep/zsaa097
5. Mikkelsen KB, Tabar YR, Kappel SL, et al. Accurate whole-night sleep monitoring with dry-contact ear-EEG. *Scientific Reports*. 2019;9:16824. https://doi.org/10.1038/s41598-019-53115-3
6. Chen X, Jin X, Zhang J, et al. Validation of a wearable forehead sleep recorder against polysomnography in sleep staging and desaturation events in a clinical sample. *Journal of Clinical Sleep Medicine*. 2023;19:711-718. https://doi.org/10.5664/jcsm.10416
7. Esparza-Iaizzo M, Sierra-Torralba M, Klinzing JG, Minguez J, Montesano L, Lopez-Larraz E. Automatic sleep scoring for real-time monitoring and stimulation in individuals with and without sleep apnea. *Computers in Biology and Medicine*. 2026;205:111560. https://doi.org/10.1016/j.compbiomed.2026.111560
8. Phan H, Andreotti F, Cooray N, Chen OY, De Vos M. SeqSleepNet. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2019;27:400-410. https://doi.org/10.1109/TNSRE.2019.2896659
9. Perslev M, Darkner S, Kempfner L, Nikolic M, Jennum P, Igel C. U-Sleep: resilient high-frequency sleep staging. *npj Digital Medicine*. 2021;4:72. https://doi.org/10.1038/s41746-021-00440-5
10. Phan H, Mikkelsen KB, Chen OY, et al. SleepTransformer. *IEEE Transactions on Biomedical Engineering*. 2022;69:2456-2467. https://doi.org/10.1109/TBME.2022.3147187
11. Zhao C, Li J, Guo Y. BTCRSleep. *Physiological Measurement*. 2023;44. https://doi.org/10.1088/1361-6579/acdb46
12. Imtiaz SA, Rodriguez-Villegas E. A low computational cost algorithm for REM sleep detection using single channel EEG. *Annals of Biomedical Engineering*. 2014;42:2344-2359. https://doi.org/10.1007/s10439-014-1085-6
13. Weinhold SL, Seeck-Hirschner M, Nowak A, Goder R, Baier PC. Wake-REM sleep transitions for measuring REM sleep disturbance. 2011. https://doi.org/10.1111/j.1479-8425.2011.00503.x
14. Sorensen GL, Knudsen S, Jennum P. Sleep transitions in hypocretin-deficient narcolepsy. *Sleep*. 2013;36:1173-1177. https://doi.org/10.5665/sleep.2880
15. Bianchi MT, Cash SS, Mietus J, Peng CK, Thomas R. Obstructive sleep apnea alters sleep stage transition dynamics. *PLoS ONE*. 2010;5:e11356. https://doi.org/10.1371/journal.pone.0011356
16. Alvarez-Estevez D, Rijsman RM. Inter-database validation of a deep learning approach for automatic sleep scoring. *PLoS ONE*. 2021;16:e0256111. https://doi.org/10.1371/journal.pone.0256111
17. Heremans ERM, Phan H, Borzee P, et al. From unsupervised to semi-supervised adversarial domain adaptation in electroencephalography-based sleep staging. *Journal of Neural Engineering*. 2022;19. https://doi.org/10.1088/1741-2552/ac6ca8
18. Stephansen JB, Olesen AN, Olsen M, et al. Neural network analysis of sleep stages enables efficient diagnosis of narcolepsy. *Nature Communications*. 2018;9:5229. https://doi.org/10.1038/s41467-018-07229-3
