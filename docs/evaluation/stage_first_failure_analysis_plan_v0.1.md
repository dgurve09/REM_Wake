# Stage-First Failure Analysis Plan v0.1

**Created:** 2026-08-15
**Project phase:** Block 5 diagnostic closeout
**Primary stage-first results already known:** Yes
**Analysis status:** Exploratory and explanatory, not confirmatory
**Raw signal access required:** No
**Model refitting or threshold selection permitted:** No

## 1. Purpose

The primary stage-first comparison showed that the transparent five-epoch-context model improved over the epoch-only ablation but still produced low REM-to-Wake event precision and a high false-alarm rate. This analysis identifies where that failure arises without changing the frozen models or primary endpoint.

Because the primary results were known before this plan was written, the diagnostic outputs cannot be presented as independently prespecified evidence. They are intended to refine the technical question for the direct-event experiment.

## 2. Questions

### D5.1 Boundary miss mechanism

At eligible human REM-to-Wake references, is a missed event caused primarily by failure to predict the preceding REM epoch, failure to predict the following Wake epoch, failure at both endpoints, or missing prediction coverage?

### D5.2 False-positive mechanism

When the transparent models predict a REM-to-Wake boundary, what human stage pair is present at that boundary? Predicted events will be grouped as:

- true REM-to-Wake;
- human REM to another stage;
- another human stage to Wake;
- no human stage change;
- another human stage transition.

The frozen +/-15-second event matcher determines whether each prediction is an eligible match, an ignored-quality match, or a false positive.

### D5.3 Participant concentration

Are false positives broadly distributed, or concentrated in a small number of participants? For each comparator and partition, report participants with references, participants with at least one true positive, participants with references but no true positive, median and maximum participant false alarms/hour, and the fraction of false positives contributed by the highest-FP 20% of participants.

### D5.4 Stage-to-event discordance

Which stage classes are weakest, and how large is the difference between stage macro F1 and event F1? This is descriptive because event performance is not a linear function of stage macro F1.

### D5.5 Quality and timing sensitivity

For each comparator, quantify the change from primary to expanded quality membership at +/-15 seconds and the change from +/-15 to +/-45 seconds under primary membership. These are the already-prespecified sensitivity conditions from the baseline protocol.

### D5.6 Sequence fragmentation

Do the transparent models create more stage transitions and shorter REM bouts than the human hypnogram over the same supported epochs? Report human and predicted all-stage transition rates, REM-to-Wake transition rates, REM-bout counts, and pooled REM-bout durations. This is a mechanistic diagnostic, not an additional primary endpoint.

## 3. Frozen Inputs

- fixed comparator stage and event result tables;
- transparent train/validation and test stage predictions;
- transparent stage and event result tables;
- frozen event-match outputs;
- quality analysis membership v0.1;
- signal quality flags v0.3;
- grouped participant split v0.1.

The analysis must not load EDF signals, external feature arrays, or fitted model files.

## 4. Outputs

- stage-class comparison;
- reference-boundary mechanism rows and summaries;
- predicted-stage distributions at the preceding REM and following Wake endpoints;
- predicted-event failure-mode summaries;
- participant concentration summary;
- stage/event discordance table;
- quality and tolerance sensitivity contrasts;
- recording-level and aggregate sequence-fragmentation summaries;
- concise interpretation and next decision.

## 5. Interpretation Limits

- Do not redefine the primary result using a favorable subgroup.
- Do not select a new model, threshold, tolerance, or quality tier from these diagnostics.
- Do not treat SF-A as independently trained because its provenance is unresolved.
- Do not infer clinical performance from BOAS.
- Any later intervention motivated by these results must be a separately frozen experiment.
