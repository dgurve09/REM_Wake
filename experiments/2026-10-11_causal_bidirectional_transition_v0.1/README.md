# Causal Bidirectional REM/Wake Transition v0.1

**Date:** 2026-10-11

**Status:** Protocol frozen; result-producing implementation not yet executed

**Partition:** Train only; validation and current test remain closed

## Purpose

Test whether a past-only wearable EEG detector benefits from treating both REM-to-Wake and Wake-to-REM boundaries as transition positives, while keeping transition direction as a separate diagnostic outcome.

## Frozen design

- decision at `nominal + 15 s`;
- eight 30-second feature windows ending at the decision time;
- `CAUSAL-RW` versus `CAUSAL-BI` with shared full-night negatives;
- participant-grouped nested threshold selection at 0.25 false alarms/hour; and
- separate participant-grouped REM-to-Wake versus Wake-to-REM direction classifier.

The complete pre-result specification is in `docs/evaluation/causal_bidirectional_transition_protocol_v0.1.md`.
