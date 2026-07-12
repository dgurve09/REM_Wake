# E0 Feasibility Closeout

**Closeout date:** 2026-07-11
**Audit period:** 2026-06-29 to 2026-07-11
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Model training performed:** No

## 1. Final Decision

**Proceed to the label/preprocessing gate with revised signal-quality exclusions.**

The full event inventory, participant distribution, PSG-to-headband sample mapping, background-window construction, and raw headband amplitude assessment support continued work. The E0 decision does not authorize model training. The deterministic grouped split policy and final preprocessing specification remain required first.

## 2. Evidence at Closeout

| Evidence item | Result |
|---|---:|
| Paired PSG/headband recordings checked | 128 |
| Unique `pid` values | 100 |
| Primary REM-to-Wake labels before amplitude screening | 365 |
| Secondary Wake-to-REM labels | 111 |
| Primary REM-to-Wake labels retained after critical screening | 350 |
| Primary REM-to-Wake labels critically excluded | 15 |
| `pid` values retaining at least one primary label | 88 |
| Eligible background centers before review sampling | 115,275 |
| Background review windows assessed | 4,302 |
| Background review windows critically excluded | 20 |
| PSG/headband pairs with matching timeline fields | 128 of 128 |
| Transition windows with matching sample indices | 476 of 476 |
| Headband channel-window measurements | 9,556 |

The 35 critical signal exclusions occur in 12 `pid` groups. They are not evenly distributed: 16 occur in `pid 89`, five in `pid 77`, and five in `pid 91`. This concentration must be preserved in later participant-grouped reporting rather than diluted through recording-level random splitting.

## 3. Signal-Quality Test

### Hypothesis

Most reviewed transition and background windows would contain finite, non-flat signal in both `HB_1` and `HB_2`, with a smaller subset requiring review or exclusion.

### Method

The predeclared protocol in `headband_window_signal_quality_protocol_v0.1.md` measured finite fraction, robust amplitude range, peak-to-peak amplitude, median absolute deviation, flatline duration, abrupt jumps, and repeated signal endpoints. Critical failures and review-only indicators were separated before the full result was inspected.

### Result

- 15 of 476 transition windows met a critical exclusion rule; all 15 were primary REM-to-Wake windows.
- 20 of 4,302 background review windows met a critical exclusion rule.
- All 88 `pid` groups that originally contributed primary events still retain at least one primary event.
- 381 windows received a targeted review flag based on amplitude, abrupt jumps, or repeated endpoints.
- 1,561 additional windows were flagged only by the 10-MAD rule.

### Failed or insufficient rule

The predeclared rule marking a window for review when more than 1% of samples exceeded 10 median absolute deviations was too nonspecific for raw EEG. At channel level it flagged 2,663 of 9,556 measurements and reached all participant groups. Heavy-tailed physiological signals and artifacts both contribute to this measure, so it does not discriminate usable from unusable windows well enough by itself.

### Resolution

The 10-MAD result is preserved as a sensitivity flag rather than discarded or converted into an exclusion. Manual and sensitivity review will prioritize the more specific amplitude-range, peak-to-peak, abrupt-jump, and repeated-endpoint indicators. Critical flatline, constant-signal, near-flat, missing, and nonfinite rules remain exclusions.

## 4. Remaining Uncertainty

- Review flags do not by themselves prove a window is unusable.
- Device saturation limits have not been independently calibrated, so repeated endpoints remain a clipping proxy.
- Label boundaries remain uncertain within the 30-second scoring interval.
- The participant-count discrepancy remains: the dataset README reports 108 individuals, while the versioned participant table contains 100 unique `pid` values.
- Feasibility does not establish wearable event-detection performance or clinical relevance.

## 5. Next Decision

Block 4 must:

1. incorporate the critical amplitude exclusions into a revised quality artifact;
2. preserve targeted and 10-MAD-only review flags separately;
3. define a deterministic split grouped by `pid` and report event/background balance;
4. freeze minimal preprocessing and record the label/preprocessing gate decision.

No model training should begin before these items are complete.
