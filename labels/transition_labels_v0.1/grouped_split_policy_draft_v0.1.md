# Grouped Split Policy Draft v0.1

**Created:** 2026-07-04
**Applies to:** `transition_labels_v0.1.tsv`
**Status:** Historical draft; superseded for current evaluation by `splits/grouped_pid_split_v0.1/` on 2026-07-15
**Model training performed:** No

## 1. Purpose

This draft defines split constraints for later model evaluation without creating a split prematurely.

The uncertainty addressed here is leakage risk: BOAS contains repeated recordings for some `pid` values, so splitting by recording alone could place the same participant in more than one evaluation partition.

## 2. Hard Rules

- Split by `pid`, not by recording folder.
- Never place recordings from the same `pid` in more than one partition.
- Keep primary REM-to-Wake labels separate from secondary Wake-to-REM labels during stratification summaries.
- Do not create final train/validation/test assignments until background-window rules and signal-quality flags are complete.
- Record the random seed and exact label-table version when a split is eventually created.

## 3. Current Label Distribution

| Item | Value |
|---|---:|
| `pid` values with transition labels | 88 |
| `pid` values with primary REM-to-Wake labels | 88 |
| `pid` values with repeated labeled recordings | 18 |
| Total transition labels | 476 |
| Primary REM-to-Wake labels | 365 |
| Secondary Wake-to-REM labels | 111 |

## 4. Later Split Design Requirements

Before creating a split, summarize candidate partitions by:

- `pid` count;
- recording count;
- primary REM-to-Wake count;
- secondary Wake-to-REM count;
- label-quality flags;
- signal-quality flags;
- recording duration distribution if needed.

## 5. Decision

Do not assign final splits yet. Use this draft and `pid_transition_distribution_v0.1.tsv` to design a leakage-safe split after background-window and signal-quality rules are defined.
