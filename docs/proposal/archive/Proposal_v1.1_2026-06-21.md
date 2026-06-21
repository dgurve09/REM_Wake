# Project Proposal

## Wearable EEG REM-to-Wake Transition Detection Under Label and Device Uncertainty

**Version:** 1.1  
**Planning date:** 2026-06-21  
**Project window:** 2026-06-01 to 2026-11-29  
**Status:** Initial research plan; expected to change as evidence is generated

## 1. Technology Area

Wearable EEG, sleep science, event detection, machine learning, temporal modeling, real-time inference, and digital health research.

## 2. Executive Summary

This project will investigate whether REM-to-Wake transitions can be detected reliably from reduced-channel wearable EEG. The primary dataset will be the Bitbrain Open Access Sleep (BOAS) dataset, which contains 128 nights of simultaneous clinical PSG and wearable headband recordings from 108 unique participants. Human consensus sleep stages are available at 30-second resolution.

The project began on June 1, 2026 with preliminary scientific work: literature review, examination of the existing technological knowledge base, research-question refinement, project planning, and public-dataset identification. Experimental implementation is planned to begin on June 22, 2026. No preprocessing, model training, or experimental result is attributed to the preliminary period unless supported by a dated record.

The project will not assume that transition-specific modeling is better than conventional sleep staging. It will compare a stage-first approach, in which predicted sleep stages are converted into transitions, with a model trained directly to detect transition events. The main technological questions concern event rarity, label uncertainty, participant-level generalization, reduced-channel signal quality, and PSG-to-wearable device shift.

REM-to-Wake will be the primary event. Wake-to-REM will remain exploratory until the initial data audit establishes that enough independent events exist. Sub-epoch analysis will be treated as localization within an uncertain 30-second interval, not as one-second ground truth.

The six-month outcome will be a reproducible feasibility assessment, transition-detection pipeline, comparative evaluation, documented experimental record, and, only if performance supports it, a lightweight streaming inference demonstration. The project will not claim diagnosis, sleep-paralysis prevention, narcolepsy screening, or clinical utility from a general-population dataset without relevant clinical labels.

## 3. Scientific and Technological Context

Wearable forehead EEG can support automated sleep staging, and the BOAS dataset already provides paired PSG and wearable recordings. Therefore, ordinary sleep staging alone is not the technological advancement sought in this project.

Prior work has also examined Wake-REM transition patterns in narcolepsy and idiopathic hypersomnia. The question that remains for this project is narrower: whether a transition-specific detector operating on wearable EEG can provide useful event-level performance beyond a stage-first baseline under realistic signal, label, and participant variability.

## 4. Primary Research Question

Can REM-to-Wake events be detected from two-channel forehead wearable EEG with participant-independent event-level performance that is meaningfully better than, or complementary to, transitions derived from conventional wearable sleep-stage predictions?

## 5. Secondary Research Questions

1. How many usable REM-to-Wake and Wake-to-REM events exist after signal-quality filtering, and how are they distributed across unique participants?
2. How much performance is lost when moving from PSG signals to reduced-channel wearable EEG?
3. Does paired PSG-to-wearable transfer or adaptation improve performance over direct wearable training and simple fine-tuning?
4. Can transition likelihood be localized within the 30-second scoring interval without claiming an unsupported exact physiological onset time?
5. Are transition-derived technical measures repeatable across nights for participants with repeated recordings?
6. Can the final justified model run with sufficiently low latency for streaming use on the available hardware?

## 6. Technological Uncertainties

### U1. Event feasibility

It is unknown whether BOAS contains enough independent, artifact-free transition events for participant-level training and evaluation. Wake-to-REM events may be especially rare in a general-population cohort.

### U2. Wearable signal sufficiency

It is unknown whether two approximately AF7/AF8 wearable EEG channels preserve enough information to distinguish a true REM-to-Wake boundary from short awakenings, artifacts, and neighboring stage transitions.

### U3. Transition-specific value

It is unknown whether direct event detection adds measurable value over a conventional sleep-stage model followed by transition derivation.

### U4. Device and participant generalization

It is unknown whether knowledge learned from PSG or one subset of participants will transfer to simultaneous wearable recordings and unseen participants without unacceptable degradation.

### U5. Temporal localization under coarse labels

Human consensus labels identify 30-second scored epochs, not the exact second of physiological transition. It is unknown whether signal evidence can narrow the likely transition interval without introducing misleading pseudo-ground truth.

## 7. Hypotheses

### H1. Feasibility

After quality control, REM-to-Wake events will be sufficiently numerous and sufficiently distributed across unique participants to support grouped cross-validation and uncertainty estimates. The initial feasibility report will determine whether this hypothesis is retained.

### H2. Direct detection

A transition-specific wearable EEG model will improve at least one clinically relevant event metric, such as event recall at a controlled false-alarm rate, compared with the stage-first baseline.

### H3. Paired transfer

Training strategies that use the simultaneously recorded PSG and wearable signals will improve unseen-participant wearable performance compared with an unadapted PSG-trained model.

### H4. Temporal evidence

Short-window signal features around a scored boundary will provide informative transition likelihood within the label-uncertainty interval, even though exact one-second transition accuracy cannot be established.

Negative or inconclusive results will still answer the technological uncertainties and will be retained in the experimental record.

## 8. Scope

### Included

- BOAS dataset verification, inventory, and quality assessment.
- Reproducible derivation of REM-to-Wake labels from human consensus hypnograms.
- Exploratory Wake-to-REM analysis if event counts support it.
- Two-channel wearable EEG as the primary input.
- Separate IMU and PPG ablations only after the EEG-only baseline is established.
- Stage-first, simple feature-based, direct transition, and justified temporal baselines.
- Participant-independent evaluation grouped by BOAS `pid`.
- Paired PSG-to-wearable transfer experiments.
- Event-level metrics, uncertainty intervals, error analysis, and reproducibility checks.
- Conditional lightweight streaming inference demonstration.

### Excluded from the current six-month study

- Diagnosis of narcolepsy or other sleep disorders.
- Detection or prevention of sleep paralysis episodes.
- Closed-loop auditory or vibration intervention.
- Claims of clinical utility without clinical outcomes and an appropriate validation cohort.
- Exact one-second ground truth derived from 30-second sleep-stage labels.
- A production application or polished GUI.
- New participant data collection.

## 9. Dataset and Ground Truth

### Primary dataset

The BOAS dataset contains:

- 128 nights of simultaneous PSG and Bitbrain wearable headband recordings;
- 108 unique participants identified by `pid`;
- PSG EEG, EOG, EMG, respiratory, and other physiological channels when available;
- two wearable forehead EEG channels approximately located at AF7 and AF8;
- wearable IMU and PPG signals when available;
- 30-second human consensus sleep-stage labels derived from multiple expert scorers.

All participant-level splits will use `pid`, not the recording identifier, because some participants contributed multiple nights.

### Transition labels

- **Primary positive event:** a human-consensus REM epoch followed by a Wake epoch.
- **Exploratory event:** a Wake epoch followed by a REM epoch.
- **Boundary representation:** the scored epoch boundary plus an explicit uncertainty interval.
- **Excluded or flagged regions:** disconnections, missing data, severe artifacts, and boundaries without the required signal coverage.

The label-generation code will be deterministic and tested on manually checked examples. No short window will be assigned an exact physiological transition second unless an independent annotation supports it.

## 10. Feasibility Gate

Before model development, the project will produce a dated feasibility report containing:

- event counts by direction;
- number of unique participants contributing events;
- events per participant and per night;
- usable signal duration and missing-channel rates;
- artifact and disconnection rates around event boundaries;
- class prevalence and candidate negative-window definitions;
- expected uncertainty of event-level estimates under grouped validation.

The decision will be one of the following:

1. **Proceed:** adequate REM-to-Wake coverage for participant-level evaluation.
2. **Narrow:** retain REM-to-Wake but drop Wake-to-REM or reduce secondary analyses.
3. **Redesign:** treat the work as transition-risk estimation or representation analysis if event counts cannot support supervised detection.
4. **Stop:** document that the dataset cannot answer the proposed question and identify the evidence required for a future study.

This gate prevents months of modeling on an unsupported target.

## 11. Experimental Design

### 11.1 Data splitting

- Group all recordings by unique participant `pid`.
- Keep every recording from one participant in only one train, validation, or test group within a split.
- Fix and version the split definitions before comparing models.
- Use grouped cross-validation or repeated grouped holdout, depending on the number and distribution of positive events.
- Keep the final test partition untouched until the analysis plan and model-selection procedure are fixed.

### 11.2 Baselines

1. **Prevalence and no-event checks:** sanity baselines for rare-event performance.
2. **Stage-first baseline:** predict conventional sleep stages from wearable EEG, then derive transitions from the predicted stage sequence.
3. **Simple direct baseline:** use a small, interpretable feature model to classify candidate transition windows.
4. **Small CNN baseline:** use a compact EEG model only after the simple baseline is working.
5. **Temporal model:** add a lightweight temporal model only if error analysis shows that temporal context is needed.

### 11.3 PSG-to-wearable experiments

- Compare full PSG, reduced PSG channel configurations, and real wearable EEG.
- Measure zero-shot PSG-to-wearable transfer.
- Compare direct wearable training and simple fine-tuning.
- Add paired feature alignment or another adaptation method only if simpler approaches leave a documented device-shift problem.
- Treat channel-reduced PSG as simulated wearable input, not as validation on a real wearable device.

### 11.4 Modality ablations

The primary result will use wearable EEG only. IMU and PPG will be added separately to determine whether any improvement comes from EEG or from multimodal sensing.

### 11.5 Label-uncertainty experiments

- Compare hard boundary labels with interval or uncertainty-aware targets.
- Test short windows only within the known 30-second scoring limitation.
- Report timing relative to the scored boundary, not an unobserved physiological onset.
- Document any heuristic pseudo-label and analyze its sensitivity.

## 12. Evaluation

### Primary metrics

- event precision, recall, and F1;
- precision-recall area under the curve;
- false positive events per hour and per night;
- event sensitivity within predeclared boundary tolerances, such as 30 and 60 seconds;
- participant-level performance distribution;
- confidence intervals obtained with participant-level resampling when appropriate.

### Secondary metrics

- calibration of transition probabilities;
- timing relative to the scored boundary and its uncertainty interval;
- subgroup error summaries by age, sex, signal quality, and repeated-night status when sample sizes permit;
- computational latency and memory for the final justified model.

Accuracy and ROC-AUC may be reported for completeness but will not be used alone because transition events are rare.

### Success interpretation

The project will not use an arbitrary fixed F1 threshold before the event distribution is known. Evidence will be judged by comparison with the stage-first baseline, uncertainty intervals, false-alarm burden, participant-level consistency, and reproducibility. Failure to outperform the baseline is a valid experimental result.

## 13. Cross-Dataset Work

An external PSG dataset such as Sleep-EDF may be used to test sleep-stage or reduced-channel PSG generalization. It will not be described as real wearable validation because it does not contain the paired BOAS headband device.

Cross-dataset work will proceed only after the BOAS feasibility gate and primary baseline. Dataset label definitions, channels, sampling rates, and cohort differences will be documented before comparison.

## 14. Transition-Derived Measures

The project may calculate technical measures such as nightly REM-to-Wake event count, event probability burden, and REM stability. In the absence of validated clinical outcomes, these measures will be evaluated for computational definition, robustness, and repeated-night reliability only. They will not be presented as validated measures of sleep quality, disease, or treatment response.

## 15. Real-Time Demonstration

A simple streaming command-line demonstration will be attempted only if the final offline model provides interpretable event performance. The demonstration will:

- process windows in chronological order;
- avoid future-signal leakage;
- report measured latency and memory on specified hardware;
- use the same preprocessing as offline evaluation;
- be clearly labeled as a research prototype, not a medical device.

A GUI is outside the current research scope.

## 16. Six-Month Work Plan

The planning allocation is up to 20 hours per week. Over 26 weeks this is up to 520 planned hours, not 1,040 hours. Only actual contemporaneously recorded time will be used for any SR&ED accounting.

### Block 1: June 1 to June 14 - Literature review and initial planning

- Review published work on wearable EEG sleep staging, REM/Wake transitions, label reliability, and PSG-to-wearable transfer.
- Identify what is established in the technological knowledge base and what remains uncertain.
- Search for public datasets with suitable signals, labels, and access conditions.
- Draft the initial research scope, hypotheses, work plan, and SR&ED documentation approach.

**Deliverable:** Initial proposal, literature evidence, and public-dataset candidate list.

### Block 2: June 15 to June 28 - Scope refinement, dataset selection, and setup

- Continue literature review and dataset comparison through June 21.
- Select BOAS as the primary candidate and document why other datasets are secondary.
- Refine the transition target, feasibility gate, evaluation design, and clinical limitations.
- From June 22, initialize version control and the organized repository structure.
- Record the environment and check existing dependencies before installing anything.
- Verify BOAS access, version, files, license, channels, labels, and metadata.
- Begin contemporaneous weekly technical and SR&ED records.

**Deliverable:** Revised proposal, dataset-selection rationale, repository, and initial dataset manifest.

### Block 3: June 29 to July 12 - Event inventory and feasibility gate

- Count transitions by direction, participant, and night.
- Quantify missing data and artifacts around boundaries.
- Assess participant-level evaluation feasibility.
- Produce the proceed, narrow, redesign, or stop decision.

**Deliverable:** Feasibility report and documented decision.

### Block 4: July 13 to July 26 - Minimal preprocessing and labels

- Implement only the preprocessing required by the feasibility decision.
- Generate deterministic transition labels and uncertainty intervals.
- Validate alignment between PSG labels and wearable recordings.
- Test label generation on manually inspected examples.

**Deliverable:** Versioned label table, quality flags, and preprocessing notebook or script.

### Block 5: July 27 to August 9 - Stage-first baseline

- Build or reuse a simple wearable sleep-stage baseline.
- Derive transition events from its predicted sequence.
- Establish grouped evaluation and event-matching code.

**Deliverable:** Stage-first baseline and event-level metrics.

### Block 6: August 10 to August 23 - Direct transition baselines

- Train the simple feature baseline.
- Add a small CNN only after the simple baseline passes basic checks.
- Compare direct detection with the stage-first approach.

**Deliverable:** Comparative baseline report with failure analysis.

### Block 7: August 24 to September 6 - Paired PSG-to-wearable transfer

- Measure full PSG, reduced PSG, and wearable performance.
- Test zero-shot transfer, direct wearable training, and simple fine-tuning.
- Quantify the device-shift problem before adding adaptation complexity.

**Deliverable:** Paired transfer results and decision log.

### Block 8: September 7 to September 20 - Robustness and justified adaptation

- Test missing channels, signal degradation, and participant variability.
- Add one justified adaptation method only if required by Block 7 evidence.
- Perform modality ablations where data coverage permits.

**Deliverable:** Robustness and ablation report.

### Block 9: September 21 to October 4 - External PSG generalization

- Audit compatibility with one external PSG dataset.
- Test reduced-channel PSG generalization if scientifically valid.
- Clearly separate simulated channel reduction from real wearable validation.

**Deliverable:** External generalization report or documented no-go decision.

### Block 10: October 5 to October 18 - Temporal localization under uncertainty

- Analyze short-window evidence around scored boundaries.
- Compare hard labels with interval-aware targets.
- Report localization relative to the 30-second scoring boundary.

**Deliverable:** Label-uncertainty and temporal-localization report.

### Block 11: October 19 to November 1 - Transition-derived measures and streaming decision

- Define event burden and REM stability measures.
- Test sensitivity to detector threshold and label uncertainty.
- Evaluate repeated-night reliability when sample size permits.
- Decide whether offline evidence justifies a streaming demonstration.

**Deliverable:** Technical measure definitions, reliability results, and streaming go/no-go decision.

### Block 12: November 2 to November 15 - Conditional streaming prototype and QA

- If justified, build chronological windowed inference without future leakage.
- Benchmark latency and memory on documented hardware.
- Re-run the selected experiment from a clean environment.
- Verify participant splits, seeds, configuration, and result linkage.
- Stress-test the most important failure conditions.

**Deliverable:** Command-line research prototype or documented no-go decision, plus the reproducibility and QA report.

### Block 13: November 16 to November 29 - Final analysis and documentation

- Freeze reviewed results without deleting failed runs.
- Write the technical report and manuscript outline.
- Complete the SR&ED evidence index using actual logs and hours.
- Separate eligible experimental work from routine or excluded work.

**Deliverable:** Final technical report, reproducible artifacts, paper outline, and SR&ED evidence index.

## 17. Milestones

- **M0, July 12:** Feasibility decision completed.
- **M1, July 26:** Reproducible labels and preprocessing completed.
- **M2, August 23:** Stage-first and direct transition baselines compared.
- **M3, September 20:** PSG-to-wearable transfer and robustness assessed.
- **M4, November 15:** Temporal analysis completed and streaming decision implemented or documented as no-go.
- **M5, November 29:** Final results, reproducibility package, and evidence index completed.

## 18. Risks and Decision Rules

### Insufficient event count

Narrow the target to REM-to-Wake, redesign the question, or stop. Do not compensate by leaking participants across splits or reporting unstable metrics without uncertainty.

### Direct model does not beat stage-first baseline

Retain the negative result and investigate errors. Do not add complex models without a documented hypothesis.

### Thirty-second labels cannot support exact timing

Report interval-based localization only. Do not present pseudo-labels as ground truth.

### External dataset incompatibility

Document the mismatch and omit the comparison rather than forcing incompatible channels, cohorts, or label systems.

### Clinical outcomes are absent

Restrict conclusions to technical feasibility and repeatability.

### Real-time performance is not justified

Document the no-go decision and omit the prototype.

## 19. SR&ED Work Framing

Potentially eligible experimental work will be organized around the stated technological uncertainties, hypotheses, experiments, observations, and conclusions. Eligibility is not guaranteed merely because work is difficult or documented.

Candidate experimental work includes:

- determining whether the rare-event target is technically learnable from wearable signals;
- comparing stage-first and transition-specific methods;
- investigating PSG-to-wearable device shift;
- testing label-uncertainty strategies;
- analyzing failure modes and robustness where existing methods are insufficient.

Programming, testing, mathematical analysis, and data collection will be treated as support work only when they directly support and are commensurate with the experimental investigation.

Routine setup, ordinary data conversion, general training, routine quality control, production engineering, interface polishing, manuscript preparation, and administrative claim preparation will be recorded separately and will not automatically be treated as eligible SR&ED work.

The project will maintain contemporaneous records of:

- the uncertainty and hypothesis addressed;
- the experiment plan and configuration;
- data and code versions;
- results, failures, and limitations;
- the conclusion and next decision;
- actual time spent;
- the relevant experiment identifier and Git commit.

No expenditure or tax-credit amount is estimated in this technical proposal. Financial eligibility and calculation require separate professional review.

## 20. Repository and Evidence Practices

- Keep the repository organized according to `PROJECT_RULES.md`.
- Use simple section-by-section notebooks for exploration.
- Move stable reusable logic into small scripts only when duplication justifies it.
- Keep each experiment run immutable and identifiable.
- Preserve failed and inconclusive runs.
- Keep large data and model artifacts outside Git while recording their source, version, path, and identifier or checksum.
- Maintain dated weekly technical and SR&ED notes.
- Review staged changes before every push and use concise professional language.

## 21. Expected Outcomes

The project may produce one of several valid outcomes:

1. evidence that wearable EEG supports useful REM-to-Wake event detection beyond stage-first inference;
2. evidence that the stage-first approach is sufficient and a direct detector adds no value;
3. evidence that device shift, event scarcity, or label uncertainty prevents reliable detection with the available data;
4. a documented requirement for different labels, sensors, or a clinical cohort.

Each outcome advances the project's technical understanding if obtained through the planned systematic investigation.

## 22. Future Work Outside This Project

After technical feasibility is established, a separate study could investigate clinically labeled populations, sleep-paralysis events, narcolepsy-related REM intrusion, prospective validation, or closed-loop intervention. These applications require appropriate clinical labels, ethics review, safety analysis, and independent validation and are not outcomes of the current BOAS study.

## 23. References

1. Lopez-Larraz E, Sierra-Torralba M, Clemente S, et al. *Bitbrain Open Access Sleep Dataset*. OpenNeuro, version 1.2.1. https://doi.org/10.18112/openneuro.ds005555.v1.2.1
2. Esparza-Iaizzo M, Sierra-Torralba M, Klinzing JG, Minguez J, Montesano L, Lopez-Larraz E. Automatic sleep scoring for real-time monitoring and stimulation in individuals with and without sleep apnea. *Computers in Biology and Medicine*. 2026;205:111560. https://doi.org/10.1016/j.compbiomed.2026.111560
3. Weinhold SL, Seeck-Hirschner M, Nowak A, Goder R, Baier PC. Wake-REM sleep transitions for measuring REM sleep disturbance: comparison between narcolepsy, idiopathic hypersomnia and healthy controls. 2011. https://doi.org/10.1111/j.1479-8425.2011.00503.x
4. Chen X, Jin X, Zhang J, et al. Validation of a wearable forehead sleep recorder against polysomnography in sleep staging and desaturation events in a clinical sample. *Journal of Clinical Sleep Medicine*. 2023;19. https://doi.org/10.5664/jcsm.10416
5. Danker-Hopfe H, Anderer P, Zeitlhofer J, et al. Interrater reliability for sleep scoring according to the Rechtschaffen & Kales and the new AASM standard. *Journal of Sleep Research*. 2009;18:74-84. https://doi.org/10.1111/j.1365-2869.2008.00700.x
6. Rosenberg RS, Van Hout S. The American Academy of Sleep Medicine inter-scorer reliability program: sleep stage scoring. *Journal of Clinical Sleep Medicine*. 2013;9:81-87. https://doi.org/10.5664/jcsm.2350
7. Canada Revenue Agency. What work is eligible: Scientific Research and Experimental Development tax incentives. Updated 2026-04-01. https://www.canada.ca/en/revenue-agency/services/scientific-research-experimental-development-tax-incentive-program/sred-eligibility.html
