# Execution and Validation Record

**Execution date:** 2026-09-25

**Protocol commit:** `03d0e0c`

**Result-producing code commit:** `90477d8`

**Validator path correction:** `62744b1`

**Authorized data:** 82 train recordings from 64 `pid` groups

**Validation accessed:** No

**Current test accessed:** No

## Accepted execution

The accepted run generated two-second wearable spectral features from all 82 train recordings, completed 20 inner fits and five outer-final fits, selected one threshold inside each outer fold, and scored every supported full-night context once in the closed outer folds.

All eight synthetic checks passed. All 25 fits completed the fixed 30 epochs with finite losses, gradients, parameters, and probabilities. Gradient norms were recorded before the fixed clip at 5; no configuration was changed after execution.

## Independent validation

The read-only validator passed all 15 checks twice after one path-only correction. It reconstructed:

- exact 82-recording and 64-participant train membership;
- all 75,539 outer score rows and participant-fold assignments;
- fold-specific threshold event consolidation;
- all reported event metrics;
- every feature-cache and external-artifact hash; and
- one cached epoch directly from the original EDF through filtering, resampling, robust scaling, and two-second spectral calculation.

## Full isolated rerun

The original result-producing commit was executed again in a detached worktree with a separate derived-data directory. It regenerated all 82 feature caches and all 25 model fits from the raw train recordings.

Comparison with the accepted run found:

- maximum absolute feature-array difference: `0.0`;
- 25/25 model artifacts byte-identical;
- 26/26 score artifacts byte-identical; and
- 18/18 reviewed output files byte-identical.

The temporary worktree and isolated derived-data directory were removed after comparison.

## Rejected checks retained in the record

The first validator invocation failed because its read-only cache path omitted the `derived` directory. The corrected path did not alter preprocessing, models, thresholds, scores, or metrics.

A cache-reuse rerun then stopped when it reconstructed the consolidated score TSV through pandas and produced shorter decimal strings than the stored file. Parsed keys and all 75,539 probabilities were exactly equal, with maximum absolute difference `0.0`. The stronger isolated rerun regenerated the model scores directly and reproduced the original score artifact byte-for-byte.
