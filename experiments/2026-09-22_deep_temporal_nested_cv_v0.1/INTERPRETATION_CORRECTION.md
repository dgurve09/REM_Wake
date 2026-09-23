# Interpretation Correction

**Recorded:** 2026-09-22

**Original result commit:** `cf1207a`

The generated `architecture_recommendation_v0.1.tsv` marks `BLSTM-2H` as `recommended_for_new_confirmation=True`, and the generated README calls it the inner-ranked candidate for future confirmation. Those fields identify only the candidate with the best prespecified mean inner rank. They do not override the experiment-level advancement gate.

The primary gate failed: nested selection improved F1 by only 0.0041 rather than the required 0.05, and its participant interval crossed zero. Under Section 8 of the committed protocol, the correct decision is to stop this four-candidate exploration at v0.1. No candidate, including `BLSTM-2H`, is authorized for new-cohort confirmation from this result.

The original generated files are retained unchanged. This append-only note corrects their interpretation without changing any score, threshold, ranking, event output, uncertainty interval, or decision row.
