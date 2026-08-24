# Alignment-Flagged Recording Sensitivity Note v0.1

**Created:** 2026-08-23
**Status:** Post hoc audit after the drift-review recordings were known
**Model fitting or threshold selection:** None

The across-night proxy audit flagged `sub-32`, `sub-39`, and `sub-50` for review. `sub-32` and `sub-50` are test recordings; `sub-39` is a train recording. This sensitivity check asks whether the frozen SF-C versus DE-B descriptive test comparison depends on the two flagged test recordings.

The check uses only the saved primary, +/-15-second, per-recording event summaries. It removes `sub-32` and `sub-50`, sums true positives, false positives, false negatives, and supported hours over the remaining recordings, and recomputes precision, recall, F1, and false alarms/hour. No alarm, match, threshold, feature, model, or label is regenerated.

Because the review set and this analysis were defined after the primary results, the sensitivity is explanatory only. It cannot replace the frozen result or justify excluding a recording.
