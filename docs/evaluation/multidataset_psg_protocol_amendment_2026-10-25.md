# Multi-Dataset PSG Protocol Amendment

**Date:** 2026-10-25
**Timing:** Before feature extraction, model fitting, or performance results

## Issue

The original protocol specified resampling all inputs to 128 Hz. A standard symmetric polyphase implementation can use samples on both sides of an output time. That is avoidable and conflicts with the intended past-only decision boundary.

## Frozen correction

- Retain each dataset's native sampling rate.
- Apply the 0.3-35 Hz Butterworth filter forward in chronological order with `sosfilt`.
- Use four-second Welch segments and two-second overlap expressed at each native rate.
- Express line length and zero crossing per second and Hjorth mobility in Hz.
- Keep the first decision at 240 seconds, outside the filter startup region.

No channel, feature family, window duration, label, model, split, threshold, alarm rule, metric, or advancement gate changes. The correction is an implementation-validity decision made before any result exists.
