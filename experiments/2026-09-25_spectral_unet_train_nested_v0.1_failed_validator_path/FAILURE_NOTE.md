# Failed Validator Path Check

**Date:** 2026-09-25

The first read-only validator invocation stopped before completing its checks because the feature-cache path omitted the `derived` directory between `REM_W_data` and the experiment folder.

The correction changed only two validator path constructions. It did not modify the result-producing script, external artifacts, thresholds, predictions, or metrics. The corrected validator subsequently passed all 15 checks twice.
