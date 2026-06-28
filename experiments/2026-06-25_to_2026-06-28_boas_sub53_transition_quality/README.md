# BOAS Sub-53 Transition-Window Quality Check

**Work period:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Project phase:** Block 2 closeout, setup and dataset verification
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Recording:** `sub-53`
**Raw data in Git:** No
**Model training performed:** No

## 1. Purpose

This check inspects signal quality around the six pilot REM/Wake transition candidates derived from PSG `stage_hum` on 2026-06-24. It is a limited pilot check before the scheduled E0 full event inventory.

The goal is not to judge model performance. The goal is to decide whether the first pilot windows are usable enough to justify the next feasibility step.

## 2. Method

The inspection script is:

```text
scripts/inspect_sub53_transition_quality.py
```

For each of the six transition candidates, the script reads a 240-second window centered on the nominal boundary:

```text
boundary - 120 seconds to boundary + 120 seconds
```

Channels checked:

- headband: `HB_1`, `HB_2`, `HB_PULSE`;
- PSG reference channels: `PSG_F3`, `PSG_F4`, `PSG_C3`, `PSG_C4`, `PSG_O1`, `PSG_O2`, `PSG_EOG`, `PSG_EMG`.

Basic checks:

- expected sample count for the window;
- finite-sample fraction;
- missing-sample count;
- peak-to-peak range;
- standard deviation;
- robust extreme-point fraction using a 10-MAD rule;
- PSG `stage_hum` sequence and PSG disconnection count inside the window.

## 3. Main Findings

All six windows were readable from both the headband and PSG EDF files.

All `HB_1` and `HB_2` windows passed the basic quality checks:

| Channel | Windows checked | Pass windows | Non-pass windows |
|---|---:|---:|---:|
| `HB_1` | 6 | 6 | 0 |
| `HB_2` | 6 | 6 | 0 |
| `HB_PULSE` | 6 | 6 | 0 |

The PSG reference channels mostly passed, but four PSG-channel windows were flagged for many extreme points:

| Transition ID | Transition | Boundary (s) | Channel | Flag |
|---:|---|---:|---|---|
| 4 | REM-to-Wake | 18,840 | `PSG_EOG` | many extreme points |
| 5 | Wake-to-REM | 18,870 | `PSG_EOG` | many extreme points |
| 5 | Wake-to-REM | 18,870 | `PSG_EMG` | many extreme points |
| 6 | REM-to-Wake | 19,080 | `PSG_EMG` | many extreme points |

No PSG disconnection epochs occurred inside any of the six inspected windows.

## 4. Window Decisions

| Transition ID | Transition | Boundary (s) | Headband EEG pass | PSG reference pass | Window decision |
|---:|---|---:|---:|---:|---|
| 1 | REM-to-Wake | 1,620 | 2/2 | 8/8 | pass basic |
| 2 | Wake-to-REM | 1,650 | 2/2 | 8/8 | pass basic |
| 3 | REM-to-Wake | 18,540 | 2/2 | 8/8 | pass basic |
| 4 | REM-to-Wake | 18,840 | 2/2 | 7/8 | PSG reference issue |
| 5 | Wake-to-REM | 18,870 | 2/2 | 6/8 | PSG reference issue |
| 6 | REM-to-Wake | 19,080 | 2/2 | 7/8 | PSG reference issue |

Interpretation: the wearable EEG pilot windows are usable under basic checks, but PSG EOG/EMG artifacts appear near the later transition cluster. These flagged PSG windows should be visually inspected before using them as clean reference examples.

## 5. Scientific Interpretation

This result supports proceeding to the scheduled E0 feasibility audit because the pilot wearable EEG windows are present, readable, and not obviously missing or flatlined around the candidate boundaries.

The result does not prove that REM-to-Wake transitions are detectable from wearable EEG. It only confirms that the pilot data are technically usable enough for the next planned feasibility work.

## 6. Files Produced

| File | Purpose |
|---|---|
| `input_transition_candidates.tsv` | Six PSG `stage_hum` transition candidates used as input |
| `transition_channel_quality.tsv` | Row-level quality metrics for each transition, source, and channel |
| `transition_window_summary.tsv` | Per-transition pass/fail summary |
| `channel_quality_overview.tsv` | Per-channel aggregate pass/fail summary |

## 7. Next Step

Begin the E0 feasibility audit on or after 2026-06-29:

- run the transition inventory across BOAS using the versioned label specification;
- quantify event counts by participant and recording;
- record missing labels, PSG disconnection epochs, and unlabeled tails;
- decide whether to proceed, narrow, redesign, or stop before model training.
