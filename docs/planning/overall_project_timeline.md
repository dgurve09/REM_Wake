# Overall Project Timeline

**Project:** REM-to-Wake transition-boundary detection from wearable EEG
**Project window:** 2026-06-01 to 2026-11-29
**Prepared during:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Current status:** Blocks 3 and 4 are complete; Block 5 was completed as catch-up work on 2026-08-15 after an inactive 2026-07-20 to 2026-08-14 interval; the stage-first results show that useful epoch staging does not translate into adequate REM-to-Wake event precision; Block 6 direct transition work is next

## 1. Project Boundary

This project is not a general sleep-stage classification project. Sleep stages are used as source annotations and as a later comparator. The direct project target is event-specific REM-to-Wake boundary detection from wearable EEG, with explicit handling of 30-second label uncertainty.

The project should not train models until the feasibility gate confirms that BOAS has enough usable events and participant-level spread to justify modeling.

## 2. Full Timeline

```mermaid
gantt
    title REM-to-Wake Wearable EEG Project Timeline
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Planning and Setup
    Literature review and initial planning           :done, b1, 2026-06-01, 2026-06-14
    Scope refinement, dataset selection, setup       :done, b2, 2026-06-15, 2026-06-28

    section Feasibility and Labels
    Event inventory and feasibility gate             :b3, 2026-06-29, 2026-07-12
    Minimal preprocessing and deterministic labels   :b4, 2026-07-13, 2026-07-26

    section Baselines
    Stage-first baseline                             :b5, 2026-07-27, 2026-08-09
    Direct transition baselines                      :b6, 2026-08-10, 2026-08-23

    section Transfer and Robustness
    Paired PSG-to-wearable transfer                  :b7, 2026-08-24, 2026-09-06
    Robustness and justified adaptation              :b8, 2026-09-07, 2026-09-20
    External PSG generalization check                :b9, 2026-09-21, 2026-10-04

    section Analysis and Measures
    Temporal localization under uncertainty          :b10, 2026-10-05, 2026-10-18
    Transition-derived measures and streaming gate   :b11, 2026-10-19, 2026-11-01

    section Prototype and Final Package
    Conditional streaming prototype and QA           :b12, 2026-11-02, 2026-11-15
    Final analysis and documentation                 :b13, 2026-11-16, 2026-11-29
```

## 3. End-to-End Flow

```mermaid
flowchart TD
    A["June 1-21: Literature review, dataset search, technology assessment, planning"] --> B["June 22-28: Repository setup, BOAS metadata audit, pilot acquisition, E0 readiness"]
    B --> C{"E0 Feasibility Gate<br/>June 29-July 12"}
    C -->|Proceed| D["Deterministic transition labels and minimal preprocessing"]
    C -->|Narrow| D2["Restrict target, e.g. REM-to-Wake only or stricter quality subset"]
    C -->|Redesign| D3["Boundary/risk analysis instead of full detector"]
    C -->|Stop| D4["Document dataset insufficiency and close BOAS modeling path"]
    D --> E["Stage-first baseline: predict sleep stages, derive transitions"]
    E --> F["Direct transition baseline: simple features, then small CNN only if justified"]
    F --> G["PSG-to-wearable transfer and device-shift analysis"]
    G --> H["Robustness, ablations, and justified adaptation"]
    H --> I["External PSG compatibility or no-go"]
    I --> J["Temporal localization under 30-second label uncertainty"]
    J --> K["Transition-derived measures and streaming go/no-go"]
    K --> L["Conditional prototype and reproducibility QA"]
    L --> M["Final report, reproducible artifacts, project artifact index, manuscript outline"]
```

## 4. Phase Table

| Block | Dates | Main Question | Main Work | Deliverable | Status as of 2026-08-15 |
|---:|---|---|---|---|---|
| 1 | Jun 1-Jun 14 | What is known and what is uncertain? | Literature review, initial planning, public-dataset search | Initial proposal, literature evidence, dataset candidate list | Complete |
| 2 | Jun 15-Jun 28 | Can the project be set up cleanly with a defensible dataset and target? | Scope refinement, BOAS selection, repository setup, environment audit, pilot checks, E0 readiness | Revised proposal, manifest, setup records, pilot reports, E0 readiness package | Complete |
| 3 | Jun 29-Jul 12 | Are there enough usable REM/Wake events to justify modeling? | Full event inventory, participant grouping, label-quality audit, feasibility decision | E0 feasibility report and proceed/narrow/redesign/stop decision | Count-level inventory, proceed decision, full EDF acquisition, full signal-alignment validation, label artifact v0.1, background-window rules, quality flags, and split-readiness review complete |
| 4 | Jul 13-Jul 26 | Can labels and minimal preprocessing be made reproducible? | Transition-event table, uncertainty intervals, alignment validation | Versioned label table and preprocessing artifacts | Complete; membership v0.1 separates conservative primary and expanded quality-sensitivity tiers; gate passed 2026-07-18 |
| 5 | Jul 27-Aug 9 | What does a stage-first comparator achieve? | Wearable sleep-stage baseline, transition derivation from predicted stages | Stage-first event metrics | Completed as catch-up work on Aug 15; fixed `stage_ai`, epoch-only logistic, and five-epoch-context logistic comparators evaluated under frozen event matching |
| 6 | Aug 10-Aug 23 | Does direct transition detection add value? | Simple direct baseline, small CNN only if justified, comparison to stage-first | Comparative baseline report | Scheduled interval active; direct baseline protocol and experiment not started |
| 7 | Aug 24-Sep 6 | How large is the PSG-to-wearable device-shift problem? | Full PSG, reduced PSG, wearable, zero-shot and fine-tuning tests | Paired transfer results and decision log | Not started |
| 8 | Sep 7-Sep 20 | Is the approach robust to signal/channel variability? | Missing-channel tests, degradation tests, ablations, justified adaptation if needed | Robustness and ablation report | Not started |
| 9 | Sep 21-Oct 4 | Is external PSG comparison scientifically valid? | Audit one external PSG dataset and test reduced-channel generalization if appropriate | External generalization report or no-go | Not started |
| 10 | Oct 5-Oct 18 | How should 30-second label uncertainty be handled? | Hard-label versus interval-aware temporal analysis | Label-uncertainty and localization report | Not started |
| 11 | Oct 19-Nov 1 | Are transition-derived measures stable enough to report? | Event burden, REM stability, repeated-night reliability, streaming decision | Technical measures and streaming go/no-go | Not started |
| 12 | Nov 2-Nov 15 | Is a streaming demonstration justified and reproducible? | Conditional command-line prototype, latency/memory check, clean rerun, QA | Prototype or no-go plus reproducibility report | Not started |
| 13 | Nov 16-Nov 29 | What is the final defensible package? | Freeze reviewed results, final report, artifact index, manuscript outline | Final technical report and reproducible project package | Not started |

## 5. Gate Logic

| Gate | Target Date | Decision | Required Evidence |
|---|---|---|---|
| E0 feasibility gate | 2026-07-12 | Proceed, narrow, redesign, or stop | Event counts by recording and `pid`, label-quality flags, unlabeled-tail summary, participant spread |
| Label/preprocessing gate | 2026-07-26 | Passed 2026-07-18; freeze v0.1/v0.3 inputs | Tested label generation, alignment checks, quality flags, participant split, preprocessing checks, analysis membership |
| Baseline gate | 2026-08-23 | Continue direct detector, narrow, or revise | Stage-first versus direct event-level comparison |
| Transfer/robustness gate | 2026-09-20 | Add adaptation or keep simpler model | PSG-to-wearable transfer result and robustness evidence |
| External-data gate | 2026-10-04 | External test or documented no-go | Compatibility audit and clear label/channel mapping |
| Streaming gate | 2026-11-01 | Prototype or no-go | Offline evidence, threshold sensitivity, repeated-night reliability if possible |
| Final QA gate | 2026-11-15 | Freeze prototype/results or no-go | Clean rerun, split/seed/config verification, artifact linkage |

## 6. Current Completed Evidence

Completed before or by 2026-06-28:

- preliminary literature review, dataset search, technology assessment, and planning records for June 1-21;
- revised project proposal and working rules;
- BOAS metadata manifest and version correction to OpenNeuro snapshot `1.1.1`;
- repository setup and GitHub remote configuration;
- environment audit confirming existing `SMRI` environment can read EDF files with `mne` and `edfio`;
- limited `sub-53` pilot acquisition outside Git;
- EDF header verification for paired PSG/headband files;
- pilot REM/Wake transition candidates from PSG `stage_hum`;
- transition-window quality check around six `sub-53` candidates;
- transition-label specification `v0.1`;
- E0 feasibility audit protocol and metadata data dictionary;
- all-recording BOAS metadata/event readiness check without EDF download.

Additional evidence completed on 2026-07-01:

- all-recording E0 transition inventory across 128 PSG `stage_hum` event files;
- 365 direct REM-to-Wake candidates across 112 recordings and 88 unique `pid` values;
- 111 Wake-to-REM secondary candidates across 57 recordings and 46 unique `pid` values;
- label-quality audit confirming no missing `stage_hum` values, no non-30-second epochs, and no sidecar duration or sampling-frequency mismatch;
- E0 proceed decision for deterministic label-table generation and minimal preprocessing, with model training still blocked.

Additional evidence completed on 2026-07-04:

- `sub-53` PSG-to-headband signal-level alignment pilot using the already downloaded paired EDF files;
- sample-index validation showing all six `sub-53` REM/Wake transition windows use matching PSG/headband extraction indices;
- pulse-based drift proxy using `HB_PULSE` versus `PSG_PULSE`, with four usable windows and all usable lags within +/-2 seconds;
- EEG-envelope artifact/physiology proxy showing `HB_1` versus `PSG_F3` peaked within +/-1 second for all six transition windows;
- decision to treat `sub-53` as sample-aligned for pilot label-table and minimal preprocessing work, without generalizing this result to all BOAS recordings;
- full BOAS EDF acquisition outside Git: 256 EDF files, 35,913,652,480 bytes, with no partial files and no size mismatches against remote object sizes;
- full-dataset signal-alignment validation across 128 paired PSG/headband EDF recordings;
- timeline validation showing all 128 EDF pairs matched on start time, sampling rate, sample count, and duration;
- transition-window sample-index validation showing all 476 E0 REM/Wake candidate windows used matching PSG/headband sample indices;
- pulse and EEG-envelope proxy analyses recorded as supporting evidence, not as ground-truth synchronization markers;
- decision to proceed to versioned deterministic label-table generation and minimal preprocessing, with model training still blocked.
- transition-label artifact `v0.1` with 476 REM/Wake rows: 365 primary REM-to-Wake labels and 111 secondary Wake-to-REM labels;
- each label preserves nominal boundary time, +/-15 second uncertainty interval, PSG/headband sample indices, label source, and quality flags;
- participant-level label distribution and grouped split-policy draft added for later leakage-safe splitting by `pid`, without assigning final splits yet.

Additional evidence completed on 2026-07-09:

- background-window specification `v0.1` added to prevent negative-window contamination around REM/Wake boundary uncertainty intervals;
- 119,967 possible adjacent-epoch background centers checked;
- 115,275 eligible background centers found after edge, disconnection, REM/Wake-boundary, and uncertainty-overlap exclusions;
- deterministic background review artifact written with 4,302 rows, without treating it as a final training set;
- signal-quality flags `v0.1` added for 128 recordings, 476 transition windows, and 4,302 background review windows;
- current critical quality checks include all reviewed recordings/windows for preprocessing while preserving pulse and EEG-envelope proxy notes;
- split-readiness review `v0.1` added by `pid`, showing 88 `pid` values with both primary REM-to-Wake labels and background review windows;
- final train/validation/test split still not assigned.

Additional evidence completed on 2026-07-11:

- predeclared full headband window signal-quality protocol applied to 9,556 `HB_1`/`HB_2` channel-window measurements;
- 15 transition windows and 20 background review windows met critical constant, near-flat, or flatline exclusion rules;
- 350 of 365 primary REM-to-Wake events remain after critical screening, with all 88 contributing `pid` groups retained;
- critical failures identified as participant-concentrated, including 16 windows in `pid 89`, five in `pid 77`, and five in `pid 91`;
- the 10-MAD review rule found to be nonspecific for raw EEG after flagging 2,663 channel-window measurements across all participant groups;
- nonspecific 10-MAD flags preserved for sensitivity analysis, while 381 windows with targeted amplitude, jump, or endpoint flags receive review priority;
- final E0 decision remains `proceed` to Block 4, with model training still blocked.

Additional evidence completed on 2026-07-15:

- integrated structural and full headband amplitude evidence into signal-quality artifact v0.2;
- retained four explicit outcomes: clean include, 10-MAD sensitivity include, targeted review, and critical exclusion;
- confirmed 350 primary REM-to-Wake windows remain across 88 `pid` groups after 15 critical exclusions;
- predeclared deterministic grouped split rules using participant metadata and pre-model label/quality counts only;
- evaluated 50,000 assignments using seed `20260715` and froze a 64/16/20 `pid` train/validation/test split;
- obtained 229/51/70 retained primary events and 56/14/18 primary-positive `pid` values in train/validation/test;
- verified that no repeated participant crosses partitions and locked the test partition before preprocessing or model work.
- preprocessing v0.1 passed 82/82 recording and 3/3 synthetic checks but failed two of 3,065 retained train windows because 240-second coverage was incomplete;
- identified the insufficient earlier rule: equal PSG/headband lengths did not guarantee the required absolute window length;
- added quality v0.3 with an exact 61,440-sample rule, identifying eight incomplete primary windows, including two new critical exclusions;
- retained 348 primary REM-to-Wake events across all 88 contributing `pid` groups and kept the frozen participant assignment unchanged;
- preprocessing v0.2 passed 82/82 train recordings, 3,063/3,063 retained train windows, and 3/3 synthetic checks without reading validation/test signals.

Additional evidence completed on 2026-07-18:

- predeclared deterministic treatment of targeted-review flags without opening raw signal files;
- rejected unrestricted primary inclusion because unresolved artifact indicators would be ignored, and rejected total removal because targeted thresholds are not validated failure labels;
- froze quality analysis membership v0.1 with clean and 10-MAD-only rows in the primary tier, targeted rows in an expanded sensitivity tier, and critical rows excluded from both;
- preserved all 476 transition and 4,302 background review rows with complete partition assignment and zero participant leakage;
- obtained 276 primary-tier REM-to-Wake events across 72 `pid` groups and 348 expanded-tier events across 88 groups;
- identified 16 `pid` groups represented only by targeted-review primary events, distributed 9/4/3 across train/validation/test;
- passed the label/preprocessing gate with transition labels v0.1, background rules v0.1, quality v0.3, membership v0.1, split v0.1, and preprocessing v0.2 frozen;
- retained the primary validation limitation of 37 events across 10 positive groups for later uncertainty reporting;
- completed a cross-artifact integrity audit with 71/71 checks passed, exact linkage of 3,063 noncritical train windows and 82 train recordings, zero validation/test preprocessing rows, and an LF-normalized SHA-256 manifest covering 19 frozen files.

Additional evidence completed on 2026-08-15:

- recorded that no project work is attributed to the inactive 2026-07-20 to 2026-08-14 interval and completed the overdue Block 5 work on its actual date;
- froze a stage-first protocol before inspecting model results, including three comparators, participant-grouped partitions, primary and expanded quality membership, +/-15-second and +/-45-second tolerances, and participant-cluster bootstrap intervals;
- validated the one-to-one event matcher on 10 synthetic cases covering tolerance edges, duplicate predictions, ignore zones, tie-breaking, empty inputs, and recording isolation;
- evaluated fixed BOAS headband `stage_ai` as a provenance-limited descriptive comparator, obtaining test stage macro F1 0.7248 and primary event F1 0.3731 at +/-15 seconds;
- fitted transparent epoch-only and five-epoch-context logistic baselines using two-channel Welch log-bandpower features and train-only scaling;
- preserved the SF-C convergence warning at the frozen 500-iteration limit instead of silently changing the configuration;
- wrote the validation decision before opening test signals, confirmed zero cached test features, and then performed one frozen test evaluation;
- obtained SF-C test stage macro F1 0.4979 but primary event F1 0.0766, precision 0.0438, recall 0.3051, and 1.9692 false alarms/hour at +/-15 seconds;
- confirmed that context improved both validation and test metrics over the epoch-only ablation, while neither transparent stage-derived detector approached a usable event alarm rate;
- passed output-integrity checks for participant isolation, phase separation, exact metric recomputation, event accounting, and SHA-256 verification of 130 external model/feature artifacts;
- completed an exploratory failure analysis using frozen tables only, with 18 input files recorded by SHA-256 and no raw-signal, external-feature, or model access;
- found that SF-C correctly retained the preceding REM endpoint at only 22 of 59 primary test references, compared with 46 of 59 correct following Wake endpoints;
- found that all 20 test participants produced false positives and that 306 of 393 false positives occurred at human REM-to-other or other-to-Wake pairs;
- measured a 2.1784-fold excess in SF-C test all-stage transition rate, 799 predicted versus 156 human REM bouts, and median REM-bout durations of 60 versus 600 seconds, supporting sequence fragmentation as the stage-first failure mechanism;
- completed a participant-paired context diagnostic using 11 hashed frozen input tables and no raw-signal, external-feature, or model access;
- preserved and corrected an initial participant balanced-accuracy warning caused by participant-dependent missing-stage denominators before interpreting the diagnostic;
- found SF-C stage macro F1 improved for all 20 test participants, while event F1 improved for 10 and remained unchanged for 10;
- obtained an exploratory paired-bootstrap SF-C minus SF-B test event-F1 difference of +0.0431 with a 95% interval of +0.0199 to +0.0676, while the false-alarm-rate difference interval crossed zero;
- found that six of seven additional SF-C test matches admitted at +/-45 seconds were one epoch early and that 59.20% of predicted REM bouts lasted only 30 or 60 seconds, compared with 14.10% of human bouts;
- observed a descriptive test-recording Spearman association of 0.5118 between predicted REM bouts/hour and false alarms/hour, recorded as exploratory and noncausal.

## 7. Next Work

After the completed stage-first baseline:

1. predeclare the Block 6 direct REM-to-Wake detection hypothesis, inputs, negative-window sampling, validation threshold, and event metrics before fitting;
2. implement the simplest direct feature baseline using the frozen split, preprocessing, labels, and quality memberships;
3. compare its validation and frozen test event metrics directly with SF-A and the primary SF-C comparator;
4. add a small CNN only if the direct feature baseline identifies a specific representational insufficiency that the CNN can test;
5. keep the SF-C convergence warning and poor event precision visible when interpreting later comparisons.

## 8. Rules for the Rest of the Project

- Keep raw data, model weights, and generated arrays outside Git.
- Preserve failed, negative, and inconclusive results.
- Keep weekly records contemporaneous.
- Push meaningful work at least weekly during active work periods.
- Do not train or evaluate models before the label/preprocessing gate.
- Do not use headband `stage_ai` as human ground truth.
- Do not claim clinical utility from BOAS alone.
- Before any model experiment, record the known-method baseline, hypothesis, configuration, metric, result, limitation, and next decision.
- Keep routine data setup separate from experiments that test a stated technical uncertainty.
