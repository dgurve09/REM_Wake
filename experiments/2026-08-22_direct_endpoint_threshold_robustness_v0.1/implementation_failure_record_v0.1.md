# Implementation Failure Record v0.1

**Recorded:** 2026-08-22  
**Affected run:** First execution of `validate_direct_endpoint_threshold_robustness_v0_1.py`  
**Scientific result affected:** No

## Failure

The validator stopped at the threshold-interval table check because it used exact pandas DataFrame equality after TSV serialization. The saved and recomputed interval values and data types appeared identical, but the floating-point representation of the calculated threshold span was not bitwise identical after writing and reading the table.

## Correction

Numeric fields are now compared with a fixed absolute tolerance, while Boolean and other nonnumeric fields remain exact. The same correction was applied proactively to the final decision-table check.

No threshold, fold selection, alarm, metric, bootstrap value, scientific configuration, or input artifact changed.
