# Causal Bidirectional Transition Protocol Amendment

**Date:** 2026-10-12

**Status:** Frozen after support audit and before any model fit

## Reason

The protocol expected 242 quality-eligible train transitions before checking whether the exact causal decision point had eight contiguous past feature windows. The audit stopped before model fitting because one event did not pass that existing support rule.

Transition 406 is a Wake-to-REM event in `sub-108` with a nominal boundary at 29,220 seconds and a causal decision time at 29,235 seconds. The +15-second feature phase has a gap from 28,905 to 29,085 seconds. Eight contiguous 30-second windows ending at 29,235 seconds therefore cannot be formed.

## Frozen resolution

Exclude transition 406 from fitting and primary evaluation. Do not interpolate the feature gap, shorten the context, move the decision time, or substitute a different phase.

The supported primary cohort is fixed at:

- 180 REM-to-Wake events;
- 61 Wake-to-REM events; and
- 241 bidirectional events in total.

The event remains in the support audit with `causal_context_supported=False`. No classifier was fitted and no model score or outcome was inspected before this amendment.
