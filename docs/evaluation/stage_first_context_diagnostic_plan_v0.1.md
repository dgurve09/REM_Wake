# Stage-First Context Diagnostic Plan v0.1

**Created:** 2026-08-15
**Project phase:** Block 5 diagnostic closeout
**Primary and failure-analysis results already known:** Yes
**Status:** Exploratory post-result analysis
**Raw signal, feature-array, or model access:** Prohibited

## 1. Purpose

Five-epoch context improved aggregate SF-C stage and event metrics over epoch-only SF-B, but both models remained inadequate event detectors. This diagnostic tests whether the context improvement is participant-consistent, characterizes one-epoch timing errors, and examines the association between sequence fragmentation and false-alarm burden.

These analyses were defined after the primary results were inspected. They may explain the observed failure and refine later hypotheses, but they cannot replace the frozen primary endpoint or justify post-test model changes.

## 2. Diagnostic Questions

### D5.7 Participant consistency of context effects

- Recompute fixed-five-stage macro F1, fixed-five-stage macro recall, and kappa by `pid` from saved predictions.
- Pair SF-B and SF-C participants within each partition.
- Count participants with improved, unchanged, or reduced stage macro F1.
- Pair frozen primary +/-15-second participant event metrics.
- Count participants with improved event F1 and with lower false alarms/hour.
- Use a paired participant bootstrap with 2,000 resamples and seed `20260815` to estimate SF-C minus SF-B differences in aggregate event precision, recall, F1, and false alarms/hour.

The bootstrap intervals are exploratory because the comparison was defined after aggregate results were known.

**Execution amendment before interpretation:** The first script execution showed that some participants do not contain all five human stages. Scikit-learn balanced accuracy therefore used participant-dependent class denominators and emitted warnings. Before any diagnostic result was interpreted, participant balanced accuracy was replaced with fixed-five-stage macro recall using labels 0-4 and zero contribution for absent stages. The execution issue is retained in the experiment record; event, timing, fragmentation, and bootstrap definitions were unchanged.

### D5.8 Direction of one-epoch timing errors

Use the existing primary +/-45-second eligible matches for SF-A, SF-B, and SF-C. Classify signed prediction offsets as exact, 30 seconds early, 30 seconds late, or another allowed offset. Do not rematch events or change the tolerance.

### D5.9 Fragmentation and false-alarm burden

Join the frozen per-recording event results to the saved sequence-fragmentation output. Calculate descriptive Spearman correlations between false alarms/hour and:

- predicted all-stage transitions/hour;
- predicted REM bouts/hour;
- predicted-to-human all-transition count ratio;
- mean predicted REM-bout duration.

Report coefficients and unadjusted descriptive p-values. Do not treat them as confirmatory tests or causal effects.

### D5.10 REM-bout duration distribution

Within the same valid prediction coverage, compare human and predicted REM bouts in five bins: 30 seconds, 60 seconds, 90-150 seconds, 180-300 seconds, and longer than 300 seconds. Report SF-B and SF-C separately because their supported epochs differ.

## 3. Frozen Inputs

- saved SF-B and SF-C train/validation/test stage predictions;
- saved participant and recording event metrics;
- saved +/-45-second event matches for SF-A, SF-B, and SF-C;
- stage-first failure-analysis sequence-fragmentation tables.

All input files will be recorded by relative path, size, and SHA-256 hash.

## 4. Interpretation Limits

- No model selection, refitting, threshold change, or new subgroup endpoint.
- No claim that a bootstrap interval or correlation is independently confirmatory.
- No causal interpretation of fragmentation correlations.
- No direct-event, transfer, robustness, clinical, or later-phase work.
