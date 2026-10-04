# Execution and Validation Record

**Execution date:** 2026-10-04

**Protocol commit:** `a3c81fe`

**Result-producing code commit:** `28dfd1a`

**Validator correction commit:** `c11f133`

## Data boundary

The run accessed only the official metadata files and two preselected pilot PSG/hypnogram pairs permitted by the protocol. It did not download the complete Sleep-EDF dataset, fit a model, or access BOAS validation/current-test artifacts.

## Integrity checks

- Seven local files matched the official Sleep-EDF SHA-256 annex.
- The official manifest reconstructed 197 unique PSG/hypnogram pairs.
- Filename grouping reconstructed 100 participants without recording-level splitting.
- Both pilot PSGs contained `Fpz-Cz` and `Pz-Oz` at 100 Hz.
- Both pilot hypnograms contained Wake and REM labels.
- Eight R-to-W transitions were reconstructed from contiguous annotations.

## Independent validation

The read-only validator passed 8/8 checks twice. It independently reopened both PSGs and hypnograms, rehashed all local files, reconstructed manifest pairing and participant counts, reconstructed the channel and transition claims, and confirmed the complete no-go decision.

## Preserved failures

The first shell hash command had an inverted final Boolean condition after displaying seven successful comparisons. The first validator then rejected a narrative that contradicted the stored annotation-support failure. Both errors and their corrections are documented in `docs/evaluation/block9_sleep_edf_execution_log_v0.1.md`; neither changed the frozen gate.
