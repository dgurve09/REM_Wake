# Post-Run Numerical Review

This review was added after the frozen v0.1 outputs were generated. It does not alter the predeclared H-INT gate or rewrite the reviewed result tables.

All 50 SGD fits stopped before the iteration limit and contained finite parameters. Those controls were necessary but not sufficient. The fitted intercepts were extreme, and almost all outer-fold probabilities rounded to an exact endpoint of zero or one. The selected inner thresholds were therefore `1.0` or its no-alarm successor in most folds, leaving only three outer alarms per candidate at every tested alarm budget.

The v0.1 event table records the literal outcome of the frozen pipeline: INT15 did not improve over HARD15. However, the endpoint saturation means this run is not a stable test of whether interval-distributed labels improve REM-to-Wake localization. It establishes a numerical failure mode of the chosen `SGDClassifier` configuration.

The predeclared stop rule is retained. No solver substitution, smoothing-width change, or threshold revision will be evaluated against these same outer results. Any future formulation test must be a separately justified version with a solver whose probability scale is demonstrated to remain finite and non-degenerate before outcome evaluation, and it requires a new confirmation boundary rather than a retroactive replacement of v0.1.

The descriptive evidence is stored in `post_run_fit_numerical_audit_v0.1.tsv` and `post_run_numerical_summary_v0.1.tsv`. No post-run threshold has been used to replace the frozen decision.
