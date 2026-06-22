# Preliminary Dataset Search and Selection

## Wearable EEG REM-to-Wake Transition Project

**Project period covered:** 2026-06-01 to 2026-06-21  
**Record consolidated:** 2026-06-21  
**Purpose:** Identify datasets capable of answering the primary research question and document why each dataset is selected, deferred, or rejected

**Metadata correction recorded 2026-06-22:** The OpenNeuro API identifies BOAS version 1.1.1 as the latest snapshot. Its README reports 108 individuals, while `participants.tsv` contains 100 unique `pid` values. See `docs/data/boas_dataset_manifest.md`.

## 1. Dataset Question

The primary question is not ordinary sleep-stage classification. The required dataset must support event-specific comparison of REM-to-Wake detection between clinical PSG and a real wearable EEG device.

The ideal dataset therefore has:

1. simultaneous PSG and wearable EEG from the same night;
2. synchronized signals or reliable alignment information;
3. human sleep-stage labels;
4. identifiable unique participants for leakage-free splitting;
5. enough REM-to-Wake events across independent participants;
6. wearable signal channels available as raw data;
7. clear access, license, and citation requirements;
8. sufficient metadata to characterize cohort and missingness.

No dataset should be called “wearable validation” if it contains only channel-reduced PSG.

## 2. Search Sources

- OpenNeuro and DataCite for BOAS.
- PhysioNet for Sleep-EDF Expanded and CAP Sleep Database.
- National Sleep Research Resource for SHHS.
- CÉAMS and Borealis for MASS.
- ISRUC-Sleep official project site and its dataset paper.
- Zenodo and the official DreemLearning repository for DOD-H and DOD-O.
- Crossref, OpenAlex, PubMed, and Semantic Scholar for dataset publications.

## 3. Selection Criteria

### 3.1 Mandatory for the primary experiment

- Raw wearable EEG.
- Human-scored reference stages from PSG.
- Same-night PSG and wearable recordings.
- Participant identifiers suitable for grouped splitting.
- Research access that is feasible within the six-month schedule.

### 3.2 Desirable

- Multiple human scorers or consensus labels.
- Repeated nights for some participants.
- Signal-quality or missing-data annotations.
- Forehead channels resembling future wearable use.
- IMU or PPG for controlled modality ablations.
- A versioned repository and persistent DOI.

### 3.3 Useful only for secondary work

- PSG-only external cohorts.
- Clinical cohorts without wearable recordings.
- Multi-scored staging datasets for label uncertainty.
- Large hypnogram collections for transition-rate analysis.

## 4. Candidate Summary

| Dataset | Nights/records and cohort | Real wearable EEG | Simultaneous PSG pair | Human labels | Access | Best role |
|---|---|---|---|---|---|---|
| BOAS | 128 nights; README reports 108 individuals, participant table contains 100 unique `pid` values | Yes, two forehead EEG channels plus IMU/PPG | Yes | Multi-scorer PSG consensus | OpenNeuro | Primary dataset |
| Dreem Open Datasets | DOD-H: 25 healthy; DOD-O: 55 OSA | Released sleep signals, but not a paired PSG-versus-wearable benchmark equivalent to BOAS | No suitable paired-device comparison identified | Five scorers | Open Zenodo release | Multi-scorer staging and uncertainty benchmark |
| Sleep-EDF Expanded | 197 PSG recordings from two older studies | No | No | Technician-scored R&K hypnograms | Open PhysioNet | Simple external PSG staging check |
| MASS | 200 complete PSG nights, five subsets, ages 18-76 | No | No | Subset-dependent expert annotations | PSG access requires project and ethics documentation | External PSG benchmark if access justified |
| ISRUC-Sleep | 100 single-session subjects, 8 repeated-session subjects, 10 healthy subjects | No | No | Two experts | Public research dataset | Clinical heterogeneity and scorer comparison |
| SHHS | 5,804 adults in the current NSRR summary; thousands of PSGs and follow-up data | No | No | Central PSG scoring | NSRR data-access request | Large-scale transition distribution and external PSG testing |
| CAP Sleep Database | 108 PSG recordings: 16 controls and 92 pathological | No | No | Sleep stages plus CAP annotations | Open PhysioNet | Exploratory clinical transition context |

## 5. Detailed Candidate Assessment

## 5.1 Bitbrain Open Access Sleep Dataset (BOAS)

### Contents

The official BOAS documentation describes 128 nights of simultaneous recordings from:

- a Micromed Brain Quick Plus Evolution PSG system; and
- a Bitbrain wearable EEG headband.

The official README reports 108 unique individuals. However, `participants.tsv` in snapshot 1.1.1 contains 100 unique `pid` values. The `pid` field remains the explicit grouping variable because some individuals contributed more than one night; the count discrepancy must be resolved or reported before evaluation.

The wearable headband includes:

- two EEG channels approximately located at AF7 and AF8;
- three-axis accelerometer and three-axis gyroscope signals;
- a PPG pulse signal.

The PSG may include six EEG channels, EOG, chin EMG, respiratory channels, and other physiological signals depending on the recording. Individual channel availability is described in channel metadata.

### Labels

Three expert scorers independently annotated the PSG according to the AASM framework. A majority consensus was used, with a fourth scorer resolving three-way disagreements. The human consensus stage appears as `stage_hum` in PSG event files. AI-generated labels are also present but must not replace the human reference.

The human PSG labels need to be aligned to the simultaneously recorded headband signal. This alignment must be verified rather than assumed.

### Cohort limitations

- Participants came from the general population rather than a dedicated narcolepsy or sleep-paralysis cohort.
- Severe neurological or psychiatric conditions and relevant medication use were exclusion criteria.
- The dataset cannot validate sleep-paralysis prevention or narcolepsy diagnosis.
- Wake-to-REM events may be rare.

### Strengths for this project

- Only candidate in this search with substantial paired PSG and real headband EEG.
- Direct device comparison within the same person and night.
- Consensus human labels.
- Forehead channels relevant to wearable use.
- Repeated nights enable limited within-person reliability analysis.
- BIDS organization and persistent OpenNeuro DOI.

### Unresolved checks before full analysis

- Exact event counts by direction and participant.
- Clock and sample alignment between devices.
- Recording-specific channel coverage and sampling rates.
- Headband failure, dropout, and artifact rates.
- Duration and meaning of disconnection labels.
- Whether enough events remain after quality filtering.
- Current dataset version, file manifest, and checksums at download.

### Decision

**Selected as the primary dataset, subject to the event-count and alignment feasibility gate.**

Official record: https://doi.org/10.18112/openneuro.ds005555.v1.1.1

## 5.2 Dreem Open Datasets: DOD-H and DOD-O

### Contents

The dataset paper reports:

- DOD-H: 25 healthy volunteers;
- DOD-O: 55 patients with obstructive sleep apnea;
- five independent sleep technologist scorings per record.

The current Zenodo record contains approximately 21.9 GB for DOD-H and 36.2 GB for DOD-O and provides checksums. The associated repository contains preprocessing and model code.

### Strengths

- Five scorers permit consensus and scorer-disagreement analysis.
- Healthy and OSA cohorts.
- Established staging benchmark.
- Open, versioned release with code.

### Limitations

- It does not provide the same paired clinical-PSG versus independent wearable-headband comparison required for the primary BOAS device-shift experiment.
- Small sample size relative to modern staging studies.
- Large download for a secondary purpose.
- It primarily addresses sleep staging, not direct REM-to-Wake event validation.

### Decision

**Deferred.** Use only if multi-scorer boundary disagreement or an external staging benchmark becomes necessary. Do not download approximately 58 GB before that need is established.

Dataset: https://doi.org/10.5281/zenodo.15900394  
Paper: https://doi.org/10.1109/TNSRE.2020.3011181

## 5.3 Sleep-EDF Database Expanded

### Contents

PhysioNet provides 197 whole-night PSG recordings containing EEG, EOG, chin EMG, event markers, and in some cases respiration and body temperature. The main EEG channels are Fpz-Cz and Pz-Oz at 100 Hz. Hypnograms were manually scored under the older Rechtschaffen and Kales system.

The collection combines:

- 153 home sleep-cassette records from an age-effects study; and
- 44 telemetry records from 22 participants with mild sleep-onset difficulty under temazepam/placebo conditions.

### Strengths

- Open access and easy to download.
- Widely used benchmark.
- Small channel set resembles a reduced-channel technical test.
- Multiple nights for many subjects.

### Limitations

- No real wearable headband pair.
- Old recording protocols and R&K labels.
- Cohort and acquisition differ substantially from BOAS.
- Recording-level splitting can leak repeated participants unless subject IDs are handled correctly.

### Decision

**Selected as the first external PSG-only comparison if external validation proceeds.** It can test preprocessing and channel-reduction generalization, not real wearable transfer.

Official page: https://physionet.org/content/sleep-edfx/1.0.0/  
Related paper: https://doi.org/10.1109/10.867928

## 5.4 Montreal Archive of Sleep Studies (MASS)

### Contents

MASS cohort 1 contains 200 complete laboratory PSG nights from 97 men and 103 women aged 18-76. It is divided into five subsets with different acquisition and annotation properties.

### Access conditions

The current CÉAMS page states that PSG biosignal access requires:

- a project description;
- proof of local research ethics approval;
- acceptance of the dataset license.

Some annotations and descriptors are more openly available, but PSG signals are not an immediate unrestricted download.

### Strengths

- Well-known, heterogeneous PSG benchmark.
- Multiple subsets support cross-protocol analysis.
- Useful for conventional sleep-stage benchmarking.

### Limitations

- No real wearable pair.
- Access requirements may not fit the immediate schedule.
- Subset heterogeneity requires careful reporting.
- Ethics approval cannot be assumed.

### Decision

**Deferred unless ethics and access requirements are satisfied and a specific external hypothesis justifies it.** Sleep-EDF is operationally simpler for the first external check.

Official page: https://ceams-carsm.ca/en/mass/  
Paper: https://doi.org/10.1111/jsr.12169

## 5.5 ISRUC-Sleep

### Contents

The official site describes three groups:

- 100 subjects with one session each;
- 8 subjects with two sessions each;
- 10 healthy subjects with one session each.

The cohort includes adults with sleep disorders and medication effects. PSG recordings were scored independently by two experts and include electrophysiological, respiratory, and contextual information.

### Strengths

- Clinical heterogeneity.
- Two scorers.
- Repeated sessions in a small subgroup.
- Published dataset description.

### Limitations

- No paired wearable data.
- Medication and disorder heterogeneity may confound transition rates.
- Group definitions and file organization need careful harmonization.
- AASM/R&K label differences must be verified per group.

### Decision

**Secondary candidate for clinical PSG robustness after BOAS.** It should not be introduced before the primary feasibility and baseline work.

Official page: https://sleeptight.isr.uc.pt/  
Paper: https://doi.org/10.1016/j.cmpb.2015.10.013

## 5.6 Sleep Heart Health Study (SHHS)

### Contents

The current NSRR summary describes a large multi-center cohort focused on sleep-disordered breathing and cardiovascular outcomes. It lists 5,804 adults aged 40 or older in its summary, while the detailed description notes 6,441 people enrolled at Visit 1 and 3,295 second PSGs. This difference reflects dataset and cohort accounting and must be resolved for any specific analysis subset.

### Strengths

- Thousands of PSGs.
- Longitudinal and clinical outcomes.
- Established use for sleep-transition dynamics.
- Suitable for population transition-rate estimates.

### Limitations

- No wearable EEG.
- NSRR access request and data-use requirements.
- Large storage and processing burden.
- Older adult cohort differs from BOAS.
- Scope would expand substantially beyond a six-month wearable feasibility study.

### Decision

**Deferred.** Consider only if the project later requires large-scale reference distributions or outcome associations. It is unnecessary for the first transition detector.

Official dataset DOI: https://doi.org/10.25822/ghy8-ks59  
NSRR page: https://sleepdata.org/datasets/shhs

## 5.7 CAP Sleep Database

### Contents

The open PhysioNet dataset includes 108 PSG recordings:

- 16 healthy controls;
- 40 nocturnal frontal lobe epilepsy;
- 22 REM behavior disorder;
- 10 periodic limb movements;
- 9 insomnia;
- 5 narcolepsy;
- 4 sleep-disordered breathing;
- 2 bruxism.

Signals include at least three EEG channels, EOG, chin and tibial EMG, respiratory signals, SaO2, and ECG. Sleep-stage and cyclic alternating pattern annotations are supplied.

### Strengths

- Open clinical PSG collection.
- Includes sleep-instability annotations and several disorders.
- Contains a small narcolepsy subgroup.

### Limitations

- Only five narcolepsy recordings, insufficient for strong disease conclusions.
- No wearable EEG.
- Older R&K stage framework.
- Pathology groups are highly imbalanced.
- CAP is an NREM microstructure construct and is not equivalent to REM-to-Wake detection.

### Decision

**Exploratory-only clinical context.** It may support later hypothesis generation but cannot validate the wearable model or narcolepsy screening.

Dataset: https://doi.org/10.13026/C2VC79  
CAP scoring paper: https://doi.org/10.1016/S1389-9457(01)00149-6

## 6. Selection Decision

### Primary

**BOAS version 1.1.1**, because it uniquely provides the required simultaneous PSG and real wearable headband signals with human consensus stages.

### First external comparison

**Sleep-EDF Expanded**, because it is open, relatively simple, and widely used. Its role must be labeled PSG-only generalization.

### Conditional secondary datasets

- **DOD-H/DOD-O:** multi-scorer uncertainty and staging benchmark.
- **ISRUC-Sleep:** clinical and medication heterogeneity.
- **CAP:** exploratory pathological transition context.
- **MASS:** only after access and ethics requirements are met.
- **SHHS:** only for large-scale transition distributions or outcomes.

## 7. Download and Storage Plan

No large dataset should be downloaded merely because it is available.

### Stage 1: Metadata verification

- Record dataset version and DOI.
- Save the official README and manifest reference.
- Confirm license and citation.
- Estimate storage requirement.
- Confirm file structure and subject identifiers.

### Stage 2: BOAS pilot

- Download metadata and one or a few representative paired records.
- Verify EDF readability, channel names, sampling rates, event files, and time alignment.
- Do not build a custom downloader if OpenNeuro tooling or standard HTTP access is sufficient.

### Stage 3: BOAS full acquisition

- Download only after pilot validation.
- Store outside Git.
- Record exact version, acquisition date, file count, total bytes, and checksums or official manifest.
- Keep raw data immutable.

### Stage 4: Secondary acquisition

- Download Sleep-EDF only after the BOAS baseline exists.
- Download DOD, ISRUC, CAP, MASS, or SHHS only when tied to a written hypothesis and decision.

## 8. Required Dataset Manifest Fields

For every acquired dataset, record:

- dataset name and abbreviation;
- persistent DOI and official URL;
- version and access date;
- license and data-use restrictions;
- cohort size and unique participant unit;
- repeated-recording structure;
- signal channels and sampling rates;
- label standard, scorers, and epoch duration;
- path outside Git;
- total file count and size;
- checksum or source manifest;
- local preprocessing status;
- known missingness and exclusions.

## 9. Dataset Risks

| Risk | Consequence | Required response |
|---|---|---|
| Too few REM-to-Wake events | Unstable model estimates | Narrow, redesign, or stop after feasibility report |
| Same participant in multiple splits | Inflated performance | Group all records by BOAS `pid` |
| Device clocks misaligned | Incorrect transition windows | Verify timestamps and cross-signal alignment |
| Missing wearable channels | Biased usable cohort | Report coverage and sensitivity analysis |
| Human labels only in PSG events | Misuse of AI labels | Use PSG human consensus and align to headband |
| Old R&K labels in external datasets | Incompatible stage definitions | Harmonize explicitly and document mapping |
| Restricted access | Schedule and compliance risk | Do not assume access; obtain approval first |
| Excessive downloads | Storage and organization burden | Acquire only hypothesis-driven datasets |

## 10. Dataset Search Conclusion

BOAS is the correct primary dataset, but its suitability is conditional rather than proven. The decisive next activity is not model training; it is a metadata, alignment, signal-quality, and event-count audit grouped by unique participant. Public PSG-only datasets can support secondary comparisons, but none identified here can substitute for BOAS in the paired wearable experiment.
