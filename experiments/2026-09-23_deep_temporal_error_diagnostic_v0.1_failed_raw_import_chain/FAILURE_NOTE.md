# Rejected Raw-Stack Import Record

**Date:** 2026-09-23

The corrected diagnostic rerun was stopped before output creation after an unnecessary historical import chain spent several minutes loading the raw-signal MNE stack. The diagnostic does not read EDF files, so the implementation was narrowed to lightweight local membership, threshold, and event helpers.

The accepted method and numerical calculations are unchanged. The corrected diagnostic imports without the raw-signal dependency and retains the same source tables, threshold grid, matching logic, and integrity checks.
