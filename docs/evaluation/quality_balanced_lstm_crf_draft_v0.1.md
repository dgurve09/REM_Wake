# Quality-Balanced LSTM-CRF Experiment Draft v0.1

**Created:** 2026-09-23

**Status:** Draft only; not authorized for execution

## Technical Uncertainty

All evaluated train-only models detect MAD-flagged primary events more often than clean primary events. It is unknown whether this difference is caused by training-tier imbalance, amplitude normalization, participant composition, or physiologically meaningful signal differences.

## Proposed Hypothesis

A participant-fit robust scaler combined with equal clean/MAD-flagged positive loss contribution will improve clean-event recall and overall event F1 relative to an otherwise identical nested `LC-1` control, without increasing false alarms/hour.

## Fixed Candidate Pair

### Control: `LC-1-NESTED`

- Existing bidirectional LSTM and three-state CRF.
- StandardScaler fit only on each fitting subset.
- Existing positive-versus-negative class weighting.

### Test: `LC-QB1`

- Exact same architecture, optimizer, epochs, contexts, features, and CRF path.
- Median and interquartile range fitted only on each fitting subset, with a fixed small denominator guard.
- Positive sequence weights adjusted so clean and MAD-flagged positives contribute equal total loss within each fitting subset.
- Combined positive contribution remains equal to the total negative contribution, preventing an unintended change in overall class weighting.
- Quality tier is used only for training weights and reporting, never as an inference input.

No hidden-size, depth, optimizer, epoch, threshold-grid, context-length, feature, or post-processing variation is included.

## Proposed Evaluation Boundary

This would be continued development on reused train participants, not independent confirmation. Reuse nested participant-grouped selection with every outer participant held out from scaler fitting, loss-weight estimation, and threshold selection. Validation and current test remain closed.

Because the existing outer results motivated this hypothesis, any repeated-train result must be labeled exploratory. Independent advancement still requires a new locked or external wearable cohort.

## Proposed Endpoints

Primary comparison at primary membership and +/-15 seconds:

- overall event F1;
- false alarms per supported hour;
- clean-event recall; and
- MAD-flagged-event recall.

Proposed point gate for `LC-QB1 - LC-1-NESTED`:

- overall F1 difference at least +0.05;
- clean-event recall difference at least +0.10; and
- false-alarm difference no greater than zero.

Use paired participant bootstrap intervals for all four contrasts. Report the aggregate gate separately from interval support.

## Stop Rule

If the point gate fails, stop normalization and tier-weight variations on the reused train cohort. Do not add weight ratios after observing the result. Move to a separately specified raw-waveform feasibility study only if its data volume, participant boundary, storage, and compute controls are established first.

## Evidence Boundary

The observed quality-tier association is not a causal artefact label. This experiment tests whether reducing amplitude- and tier-dependent learning changes performance; it cannot prove what physiological or technical process produced the original association.
