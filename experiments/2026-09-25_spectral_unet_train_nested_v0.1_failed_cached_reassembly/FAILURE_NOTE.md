# Rejected Cached Score Reassembly Check

**Date:** 2026-09-25

A cache-reuse rerun stopped while verifying the consolidated outer-score gzip. Re-reading the per-fold TSV files through pandas shortened trailing decimal text when the consolidated table was serialized again.

The parsed tables had identical columns, keys, row order, and all 75,539 probability values; the maximum absolute probability difference was `0.0`. No stored artifact was overwritten.

This cached reassembly was not accepted as the immutable rerun. The original result-producing commit was instead executed in an isolated data root, regenerating every feature cache and all 25 model fits. That full rerun reproduced all model, score, and reviewed output files byte-for-byte.
