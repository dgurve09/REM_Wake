# Headband Window Signal-Quality Protocol v0.1

**Prepared:** 2026-07-11
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Project phase:** Block 3 feasibility closeout
**Model training performed:** No

## 1. Purpose

The existing quality artifact verifies label structure, PSG-to-headband sample mapping, window geometry, and alignment proxies. It does not measure the actual `HB_1` and `HB_2` signal amplitude inside every reviewed transition and background window.

This protocol tests whether gross headband signal failures are common enough to change the feasibility decision or require exclusions before participant-grouped split design.

## 2. Hypothesis

Most reviewed 240-second windows will contain finite, non-flat signal in both headband EEG channels. A smaller subset may require review for unusually large amplitude, abrupt jumps, or repeated endpoint values. If critical failures are concentrated by participant or recording, the later split policy must preserve those groups and report the imbalance.

## 3. Inputs

- 476 transition windows from `labels/signal_quality_flags_v0.1/transition_window_quality_flags_v0.1.tsv`;
- 4,302 deterministic background review windows from `labels/signal_quality_flags_v0.1/background_window_quality_flags_v0.1.tsv`;
- raw BOAS headband EDF files stored outside Git;
- headband EEG channels `HB_1` and `HB_2` at 256 Hz.

## 4. Predeclared Measurements

Measurements are calculated separately for each channel and window after converting MNE values from volts to microvolts:

- actual and expected sample count;
- finite-sample fraction;
- minimum, maximum, median, standard deviation, and median absolute deviation;
- 1st-to-99th percentile robust amplitude range;
- peak-to-peak amplitude;
- fraction beyond 10 median absolute deviations;
- longest run of unchanged samples, using a `0.01 uV` difference tolerance;
- fraction of adjacent differences greater than `500 uV`;
- fraction of samples equal to the observed minimum or maximum as a clipping proxy.

## 5. Predeclared Decisions

### Critical channel flags

A channel is excluded from the current preprocessing candidate set if any of the following occur:

- sample-count mismatch or empty window;
- any nonfinite sample;
- zero peak-to-peak amplitude;
- 1st-to-99th percentile range below `1 uV`;
- an unchanged run lasting at least 5 seconds.

### Review channel flags

The channel is retained for review, not automatically excluded, if any of the following occur:

- 1st-to-99th percentile range above `1,000 uV`;
- peak-to-peak amplitude above `5,000 uV`;
- more than 1% of finite samples lie beyond 10 median absolute deviations;
- more than 0.1% of adjacent differences exceed `500 uV`;
- more than 1% of samples equal the observed minimum or maximum.

### Window decision

- `include`: both channels have no critical or review flag;
- `review`: neither channel has a critical flag, but at least one review flag is present;
- `exclude`: at least one channel has a critical flag.

These are conservative engineering screening rules, not validated clinical EEG-quality criteria. Review flags preserve uncertain cases for later inspection rather than silently removing them.

## 6. Required Outputs

1. channel-window metric table;
2. window-level decision table;
3. recording-level summary;
4. aggregate summary by window source and decision;
5. a decision on whether the existing feasibility conclusion changes.
