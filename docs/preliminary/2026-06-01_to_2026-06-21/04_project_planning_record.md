# Project Planning Record

## Wearable EEG REM-to-Wake Transition Detection

**Project period covered:** 2026-06-01 to 2026-06-21  
**Record consolidated:** 2026-06-21  
**Experimental implementation start:** 2026-06-22  
**Planned project end:** 2026-11-29  
**Planning status:** Preliminary phase completed; experimental work not yet started

**Metadata correction recorded 2026-06-22:** The OpenNeuro API identifies BOAS version 1.1.1 as the latest snapshot. Its README reports 108 individuals, while `participants.tsv` contains 100 unique `pid` values. See `docs/data/boas_dataset_manifest.md`.

## 1. Purpose of This Record

This document records the planning decisions produced by the preliminary literature review, dataset search, and technological knowledge assessment. It translates the current proposal into an executable sequence of experiments and decision gates.

It does not assign invented daily hours or claim that preprocessing, data download, or model training occurred before June 22. Git commits will use their real creation and push dates. When these records are committed, the commit should say that they document preliminary work covering June 1-21 rather than pretending the commit itself was created earlier.

## 2. Preliminary Phase Deliverables

| Deliverable | File | Status on 2026-06-21 |
|---|---|---|
| Revised scientific proposal | `Proposal.md` | Completed, version 1.2 |
| Literature review | `01_literature_review.md` | Completed |
| Dataset search and selection | `02_dataset_search.md` | Completed |
| Technological knowledge assessment | `03_technological_knowledge_assessment.md` | Completed |
| Detailed execution plan | `04_project_planning_record.md` | Completed with this record |
| Project operating rules | `PROJECT_RULES.md` | Updated |

## 3. Work Covered During June 1-21

The preliminary period covers the following activity categories:

- review of wearable EEG sleep-staging evidence;
- review of sleep transition, narcolepsy transition, scorer-reliability, temporal-context, and domain-adaptation literature;
- identification and comparison of BOAS, DOD, Sleep-EDF, MASS, ISRUC-Sleep, SHHS, and CAP datasets;
- selection of BOAS as the conditional primary dataset;
- definition of the event-count feasibility gate;
- correction of the one-second ground-truth claim;
- narrowing of clinical claims;
- separation of established staging capability from the transition-specific knowledge gap;
- definition of hypotheses, metrics, participant splits, stop rules, and documentation practices;
- revision of the six-month work plan.

Actual dates and hours for individual activities must come from the researcher's real calendar, notes, or time records. This summary must not be used to invent a daily chronology.

## 4. Project Objective

Determine whether REM-to-Wake events can be detected from two-channel forehead wearable EEG with participant-independent event-level performance that adds useful information beyond transitions derived from a temporal sleep-stage model.

This is not a sleep-stage classification project. Existing hypnogram stages are source annotations for constructing transition events, and an established stage-first method is retained only as the necessary comparator.

The longer-term application direction includes REM-to-Wake phenomena such as awakening from REM and sleep paralysis, and Wake-to-REM or abnormal transition structure relevant to narcolepsy research. The present project establishes technical measurement feasibility and does not validate diagnosis, prevention, or clinical utility.

## 5. Project Boundaries

### Included

- BOAS paired PSG and wearable EEG.
- Human consensus PSG stages.
- REM-to-Wake as the primary event.
- Wake-to-REM only after event feasibility is established.
- Participant-independent evaluation grouped by `pid`.
- Stage-first versus direct-event comparison.
- PSG-to-wearable transfer experiments.
- Boundary-label uncertainty analysis.
- EEG-only primary results and conditional sensor ablations.
- Conditional streaming command-line demonstration.

### Excluded

- New participant data collection.
- Narcolepsy diagnosis.
- Sleep-paralysis detection or prevention.
- Closed-loop intervention.
- A medical-device claim.
- Exact one-second physiological ground truth from 30-second labels.
- Production application or polished GUI.
- Unnecessary framework or infrastructure development.

## 6. Planning Assumptions

1. BOAS version 1.1.1 remains accessible and contains the files described in its official README.
2. Human PSG consensus labels can be aligned with the simultaneous headband signal.
3. The project is performed by one researcher at up to 20 planned hours per week.
4. Actual hours, not the 520-hour planning ceiling, control any later accounting.
5. Existing installed packages will be checked before any installation.
6. Exploration will use simple, section-by-section notebooks.
7. Stable reusable logic will move to small scripts only when justified.
8. Large datasets and model artifacts will remain outside Git.
9. The work plan will change when evidence invalidates an assumption.

## 7. Research Questions and Hypotheses

| ID | Research question | Working hypothesis |
|---|---|---|
| H0 | Do the derived transition labels define a stable target? | A deterministic specification with uncertainty and exclusion flags yields reproducible labels and interpretable conclusions across sensitivity rules |
| H1 | Are enough usable events available? | BOAS contains sufficient REM-to-Wake events across independent participants for grouped evaluation |
| H2 | Does wearable EEG contain useful event evidence? | Direct wearable features discriminate true boundaries from matched non-boundaries |
| H3 | Does direct detection add value? | A direct model improves event sensitivity at comparable false-alarm burden versus stage-first inference |
| H4 | Does paired transfer help? | PSG pretraining or fine-tuning improves wearable event detection over unadapted transfer |
| H5 | Does interval-aware labeling help? | Explicit boundary uncertainty improves calibration or tolerance-based event matching |
| H6 | Are technical measures repeatable? | Eligible repeated nights show measurable within-person consistency, if sample size permits |

## 8. Experiment Sequence

## P0. Repository and environment setup

**Purpose:** Create the minimum reproducible workspace.

**Tasks:**

- initialize Git in `REM_W`;
- configure a remote and upstream when supplied;
- create only required directories;
- inspect installed Python and packages before installing anything;
- record platform, Python version, relevant package versions, and hardware;
- add `.gitignore` before downloading data;
- establish the weekly record template.

**Output:** Environment record, repository baseline, and first verified push.

**SR&ED note:** Mostly setup and routine support, not the advancement.

## D0. BOAS metadata and pilot access

**Purpose:** Verify that the assumed dataset structure is real and usable.

**Tasks:**

- record DOI, version, access date, and license;
- inspect BIDS structure and participant metadata;
- inspect one or a few paired PSG/headband records;
- verify EDF reading, channel names, sampling rates, events, and timestamps;
- determine storage requirement before full download;
- record source manifest or checksums.

**Output:** Dataset manifest and pilot inspection notebook.

**Stop rule:** Do not download the full dataset if paired files or human labels cannot be read or aligned.

## E0. Transition feasibility audit

**Uncertainty:** U0/H0 and U1/H1.

**Purpose:** Determine whether supervised event detection is statistically supportable.

**Tasks:**

- parse `stage_hum` labels;
- define and version the primary transition-label rule;
- define predeclared sensitivity rules for adjacency, uncertainty, disconnections, and artifacts;
- map recordings to unique `pid`;
- count REM-to-Wake and Wake-to-REM boundaries;
- count event-contributing participants and nights;
- quantify events per participant;
- inspect signal and label coverage around boundaries;
- define candidate negative windows;
- compare event populations across the primary and sensitivity rules;
- estimate the uncertainty expected under grouped validation.

**Output:** Immutable event inventory and feasibility report.

**Decision:**

- proceed with both directions;
- narrow to REM-to-Wake;
- redesign as transition-risk or boundary analysis;
- stop and document dataset insufficiency.

No model development begins before this decision.

## E1. Deterministic labels and minimal preprocessing

**Purpose:** Create a tested input and label pipeline required by the feasibility decision.

**Tasks:**

- define the scored boundary and uncertainty interval;
- exclude or flag disconnections and missing coverage;
- validate PSG/headband alignment;
- resample and filter only as required;
- record every preprocessing parameter;
- manually inspect a small set of positive and negative examples;
- test deterministic regeneration of label tables.

The event table will record participant, recording, direction, source stages, boundary time, uncertainty interval, quality flags, derivation-rule version, and source-dataset version.

**Output:** Versioned label table, QC table, and preprocessing code.

**Stop rule:** Resolve alignment errors before any training. Do not hide failed records.

## E2. Stage-first baseline

**Uncertainty:** Comparator for U2/U3.

**Purpose:** Establish the strongest reasonable standard approach before direct-event claims.

**Tasks:**

- select a simple temporal staging baseline;
- train on grouped participant splits;
- retain stage probabilities;
- convert predicted stage sequences to REM-to-Wake events;
- perform one-to-one event matching;
- report stage metrics and event metrics separately.

**Output:** Stage-first event baseline with raw per-participant metrics.

**Rule:** Do not compare a direct temporal model only against a weak independent-epoch baseline.

## E3. Simple direct-event baseline

**Uncertainty:** U2/U3.

**Purpose:** Test whether direct event learning is possible without architecture complexity.

**Tasks:**

- define matched candidate windows without participant leakage;
- extract a small, justified feature set or use a simple linear/tree model;
- use training-only class balancing;
- tune thresholds only on validation participants;
- evaluate on the fixed test participants.

**Output:** Direct baseline and comparison with E2.

**Decision:** Add a CNN only if the simple model shows signal or a specific representation limitation.

## E4. Compact direct EEG model

**Uncertainty:** U2/U3.

**Purpose:** Test whether learned EEG representations improve the direct-event result.

**Tasks:**

- use a compact model consistent with available data size;
- keep split, windows, metrics, and event matching identical to E3;
- control parameter count and overfitting;
- compare participant-level paired results with E2 and E3.

**Output:** Compact-model result and error analysis.

**Stop rule:** Do not add deeper models if gains are unstable or driven by a few participants.

## E5. Paired PSG-to-wearable transfer

**Uncertainty:** U4/H4.

**Purpose:** Quantify real device shift before selecting adaptation methods.

**Conditions:**

1. full PSG input;
2. reduced PSG channel input;
3. unadapted PSG model on wearable input where technically compatible;
4. wearable-only training;
5. simple PSG pretraining plus wearable fine-tuning.

**Output:** Device-transfer matrix and failure analysis.

**Decision:** Test one more complex adaptation method only if a measurable gap remains and simple fine-tuning does not resolve it.

## E6. Conditional domain adaptation

**Uncertainty:** U4.

**Entry criterion:** Written evidence from E5 that there is an unresolved device-domain problem.

**Tasks:**

- select one published method matching the observed mismatch;
- state the expected mechanism before implementation;
- compare with simple fine-tuning using identical splits;
- run an ablation of the adaptation component.

**Output:** Adaptation result or documented no-go decision.

## E7. Boundary uncertainty and temporal localization

**Uncertainty:** U5/H5.

**Purpose:** Determine whether short-window evidence can be interpreted under coarse labels.

**Conditions:**

- hard boundary target;
- tolerance-based event matching;
- interval-aware or uncertainty-weighted target;
- sensitivity to alternative interval widths.

**Output:** Calibration, event metrics, and timing relative to the scored boundary.

**Rule:** Never report exact physiological onset error without independent fine-resolution labels.

## E8. Modality and robustness ablations

**Purpose:** Identify what information drives performance.

**Ablations:**

- EEG only;
- EEG plus IMU;
- EEG plus PPG;
- channel 1 only;
- channel 2 only;
- simulated dropout and noise where justified.

**Output:** Controlled ablation table.

**Rule:** EEG-only remains the primary result. Multimodal improvement must not be described as EEG improvement.

## E9. External PSG-only validation

**Purpose:** Test preprocessing or stage-generalization limits after BOAS results exist.

**First candidate:** Sleep-EDF Expanded.

**Tasks:**

- document label and cohort differences;
- harmonize stages explicitly;
- test channel-reduced PSG only if compatible;
- report it as PSG-only external validation.

**Output:** External result or incompatibility/no-go record.

## E10. Repeated-night technical measures

**Uncertainty:** U6/H6.

**Purpose:** Explore repeatability without claiming clinical validity.

**Tasks:**

- identify repeated `pid` records;
- define REM-to-Wake event burden and probability burden;
- test threshold sensitivity;
- estimate repeatability only if sample size supports it.

**Output:** Technical reliability result or underpowered/no-go conclusion.

## P1. Conditional streaming demonstration

**Entry criterion:** Offline event performance is stable enough to make a streaming demonstration meaningful.

**Tasks:**

- use chronological windows only;
- remove future-signal leakage;
- reuse the verified preprocessing pipeline;
- benchmark latency and memory on named hardware;
- implement a command-line demonstration, not a GUI.

**Output:** Research prototype or documented no-go decision.

## 9. Data Splitting Plan

### Unit

Unique BOAS participant `pid`.

### Rules

- Every night from one `pid` stays in one split within an experiment.
- Split definitions are versioned before model comparison.
- Event-count balance is assessed across groups without moving test participants after seeing performance.
- Validation data choose thresholds and hyperparameters.
- Final test participants remain untouched until the analysis procedure is fixed.
- If event scarcity makes a fixed test set unstable, use grouped repeated validation and report that limitation rather than leaking participants.

## 10. Event Evaluation Plan

### Event matching

- Sort reference and predicted events chronologically per night.
- Match at most one prediction to one reference event within a predeclared tolerance.
- Count duplicate predictions as false positives.
- Report results at 30-second and 60-second tolerances unless feasibility evidence requires another documented choice.

### Primary metrics

- event precision;
- event recall;
- event F1;
- precision-recall area under the curve;
- false alarms per hour and per night;
- participant-level distributions;
- participant-level confidence intervals or resampling intervals.

### Secondary metrics

- calibration;
- stage confusion matrix for stage-first baseline;
- boundary-relative timing;
- signal-quality stratification;
- compute time, memory, and latency for the final model.

Accuracy and ROC-AUC may be reported but cannot be the only results for a rare event.

## 11. Repository Plan

Directories will be created only when first needed.

```text
REM_W/
|-- Proposal.md
|-- PROJECT_RULES.md
|-- docs/
|   |-- preliminary/
|   |-- decisions/
|   |-- reports/
|   `-- sred/
|       `-- weekly/
|-- notebooks/
|-- src/
|-- experiments/
|-- results/
|   |-- current/
|   `-- archive/
`-- tests/
```

### Folder rules

- `notebooks/`: numbered exploratory notebooks, section by section.
- `src/`: only stable reused logic.
- `experiments/`: immutable run records, including failures.
- `results/current/`: reviewed current tables and figures.
- `results/archive/`: superseded reviewed outputs, not raw run duplication.
- `docs/decisions/`: method decisions and rejected alternatives.
- `docs/sred/weekly/`: dated contemporaneous technical narratives.
- raw data and large model files: outside Git, referenced by manifest.

## 12. Experiment Record Template

Every run should record:

```text
Experiment ID:
Date:
Researcher:
Uncertainty ID:
Hypothesis:
Purpose:
Dataset and version:
Transition-label specification version:
Participant split version:
Input channels:
Preprocessing configuration:
Model or method:
Training configuration:
Code commit:
Expected result:
Observed result:
Primary metrics:
Failures or deviations:
Interpretation:
Decision and next step:
Actual time:
Artifact paths:
```

## 13. Weekly Workflow

1. State the uncertainty and planned test before running it.
2. Record actual work time contemporaneously.
3. Run the smallest experiment that can resolve the next question.
4. Preserve raw metrics and failed attempts.
5. Update the weekly record with what changed, why, result, limitation, and next decision.
6. Review staged Git changes for secrets, private working material, and unprofessional text.
7. Commit meaningful work with a concise human-readable message.
8. Push at least once per calendar week and verify that the upstream contains the commit.

## 14. Evidence and Change Control

### Before changing a major method

- record the observed problem;
- identify the hypothesis for the change;
- state why the current method is insufficient;
- define what result would support or reject the change.

### After the change

- compare against the unchanged baseline;
- retain both successful and failed results;
- record whether the change resolved the uncertainty;
- update `Proposal.md` only when scope, milestones, or major assumptions change;
- archive superseded standalone planning documents when Git history is not yet available.

## 15. Risk Register

| Risk | Probability before audit | Impact | Mitigation or decision |
|---|---|---|---|
| Too few independent events | Medium to high | Critical | E0 feasibility gate; narrow or stop |
| Event concentration in few participants | Medium | High | Report per-`pid` distribution; grouped validation |
| PSG/headband misalignment | Unknown | Critical | D0/E1 timestamp and signal checks |
| Wearable channel dropout | Unknown | High | Coverage table and exclusion sensitivity |
| Thirty-second label ambiguity | High | High | Interval targets and tolerance analysis |
| Direct model adds no value | Medium | High | Accept stage-first sufficiency as valid result |
| Domain adaptation unnecessary | Medium | Low | Conditional E6; do not implement by default |
| External datasets incompatible | Medium | Medium | Document no-go rather than force harmonization |
| Clinical claims exceed data | High without controls | High | Restrict to technical feasibility |
| Overengineering | Medium | Medium | Simple baselines first; no framework building |
| Data or artifacts pushed to Git | Low with controls | High | `.gitignore`, manifest, staged-diff review |
| SR&ED records reconstructed late | Medium | High | Weekly contemporaneous records from June 22 onward |

## 16. Milestones

| Milestone | Target | Evidence |
|---|---|---|
| Preliminary planning complete | June 21 | Four preliminary records and Proposal v1.2 |
| Dataset and repository setup | June 28 | P0/D0 records and verified repository |
| Feasibility decision | July 12 | E0 report |
| Labels and preprocessing | July 26 | E1 artifacts and tests |
| Baseline comparison | August 23 | E2-E4 comparison |
| Transfer and robustness decision | September 20 | E5/E6/E8 reports |
| External PSG assessment | October 4 | E9 result or no-go |
| Boundary analysis | October 18 | E7 report |
| Technical measures and streaming decision | November 1 | E10 and go/no-go |
| Prototype and QA | November 15 | P1 or no-go plus reproducibility report |
| Final project package | November 29 | Final report, evidence index, and reviewed artifacts |

## 17. Immediate Actions for June 22-28

1. Initialize the local Git repository.
2. Configure the remote once its URL and repository are confirmed.
3. Add `.gitignore` before any data acquisition.
4. Inspect Python and installed scientific packages; do not install preemptively.
5. Create the dataset manifest.
6. Verify BOAS metadata and a small pilot record.
7. Confirm participant mapping and event-file structure.
8. Create the first contemporaneous weekly record.
9. Push the preliminary documents with their real commit date and a message stating that they document work covering June 1-21.

## 18. Planning Completion Criteria

The preliminary planning phase is complete when:

- the knowledge gap is stated without claiming worldwide novelty;
- BOAS is selected conditionally rather than assumed feasible;
- sleep stages are explicitly limited to source annotations and a comparator, not the project objective;
- the derived transition-label artifact and its validation are explicitly planned;
- the 30-second label limitation is explicit;
- stage-first and direct-event methods are both planned;
- participant grouping and event metrics are fixed in principle;
- clinical claims are outside the current validation scope;
- decision gates and stop rules are documented;
- repository and SR&ED record practices are defined;
- no experiments or hours are falsely backdated.

These criteria are satisfied by the current preliminary records, subject to final consistency verification.
