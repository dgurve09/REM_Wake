# Execution and Validation Record

**Execution date:** 2026-09-23

**Analysis type:** Retrospective train-only diagnostic; no model fitting

**Validation accessed:** No

**Test accessed:** No

## Failed and Rejected Executions

Early executions exposed a redundant membership merge and reversed immutable-output helper arguments before reviewed output creation. A first complete output lacked the historical LR-OOF and LC-1 controls. Later complete outputs were rejected for absolute machine paths and trailing empty TSV fields. The full rejected outputs are retained outside Git where needed, with compact failure notes in the repository.

An additional rerun was stopped before output creation because an unnecessary historical import chain loaded the raw-signal MNE stack. The accepted implementation uses lightweight local membership, threshold, and event helpers; no scientific calculation changed.

## Accepted Execution

The accepted run used only frozen train outer-fold scores and reviewed train labels. All 9/9 in-run checks passed, including exact reconstruction of F1 0.164524421594 and all 177 primary false alarms.

The complete analysis was rerun without changing any reviewed table or README. The independent read-only validator then reconstructed source hashes, frozen metrics, nested and historical quality-tier recall, false-alarm context counts, and threshold-bound values. It passed 8/8 checks twice.
