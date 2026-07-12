# Signal Quality Flag Specification v0.1

**Created:** 2026-07-09
**Applies to:** `labels/signal_quality_flags_v0.1/`
**Model training performed:** No

## 1. Purpose

This specification defines v0.1 quality flags for recordings, REM/Wake transition windows, and background review windows.

The technical issue is whether derived labels and background windows can be carried into preprocessing with explicit review flags instead of silently assuming that every window is equally reliable.

## 2. Critical Flags

Critical flags can block a window from later model work until reviewed. They include:

- PSG/headband timeline mismatch;
- transition-window sample-index mismatch;
- missing PSG `stage_hum`;
- non-30-second PSG epochs;
- PSG onset gaps;
- PSG/headband sampling or duration mismatch;
- unlabeled PSG tail of at least one full epoch;
- PSG disconnection inside a candidate window;
- background-window geometry mismatch;
- background-window overlap with REM/Wake uncertainty.

## 3. Context and Proxy Flags

Context and proxy flags are retained for later interpretation but do not automatically exclude a window in v0.1. These include:

- recording-level PSG disconnection outside the candidate window;
- missing pulse proxy channel;
- unusable or mixed pulse-lag proxy;
- unusable or mixed EEG-envelope proxy.

Pulse and EEG-envelope lag checks are useful timing proxies, but they are not treated as ground-truth synchronization markers.

## 4. Current Result

The v0.1 build found:

- 128 of 128 recordings include for preprocessing under critical checks;
- 476 of 476 transition windows include for preprocessing under critical checks;
- 4,302 of 4,302 background review windows include for preprocessing under critical checks;
- 0 windows requiring critical quality review.

The artifact still preserves proxy notes such as missing pulse channels, unusable EEG-envelope comparisons, or mixed lag evidence.

## 5. Decision

Use these flags for the label/preprocessing gate. Do not treat them as a final amplitude-artifact detector; full signal artifact metrics can be added later if needed before model training.
