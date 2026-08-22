# Access Check Wording Correction v0.1

**Recorded:** 2026-08-22  
**Affected record:** `output_integrity_checks_v0.1.tsv`  
**Numerical result affected:** No

## Original Wording

The in-run check was named `no_test_model_train_or_raw_input`. That name is broader than the condition actually tested. The check searched the input-manifest role names for prohibited test, model, train, or raw inputs.

## Clarification

The pre-analysis plan explicitly permitted the frozen project-wide transition-membership table to be read only to select validation references. That table contains partition metadata, after which non-validation rows were discarded before event evaluation. The analysis did not load or evaluate:

- current-test candidate scores, features, predictions, or metrics;
- fitted endpoint models;
- train labeled-score rows;
- raw EDF signals;
- Block 7 transfer inputs.

The more precise guarantee is therefore `no_test_score_model_train_score_or_raw_input`. The independent validator uses this corrected wording. The original in-run output is retained rather than rewritten, and all thresholds, alarms, metrics, category counts, hashes, and decisions are unchanged.
