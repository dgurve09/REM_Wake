# Block 8 Participant Evidence-Strength Protocol v0.1

**Protocol date:** 2026-09-18
**Project block:** Block 8 robustness closeout extension
**Analysis type:** Frozen-output participant-uncertainty synthesis
**New fitting, scoring, threshold selection, or resampling:** None
**Current test access:** Prohibited

## 1. Question

Which Block 8 conclusions are supported by the stored participant-level uncertainty ranges, which pass only at the aggregate point estimate, and which failed gates are ruled out by the existing uncertainty bounds?

This analysis does not change any original decision. The original experiments used predeclared point-estimate gates. The present analysis adds an evidence-strength layer so point-gate results are not described as participant-stable when their stored uncertainty range overlaps the gate.

## 2. Frozen Sources

The analysis will hash and read ten compact tables from the five completed Block 8 experiments:

- single-channel primary comparisons and leave-one-`pid`-out summaries;
- raw-noise hypothesis decisions and paired participant bootstrap intervals;
- noise-augmentation hypothesis decisions and paired participant bootstrap intervals;
- event-overlap hypothesis decisions and participant bootstrap intervals; and
- fixed-fusion hypothesis decisions and paired participant bootstrap intervals.

No raw signal, feature array, model, candidate score, full probability table, event rematching, new bootstrap resampling, validation threshold revision, or current-test artifact may be used.

## 3. Fixed Classification Rules

Each claim has one original gate with operator `>=` or `<=`.

For a `>= gate` claim:

- the point passes when `point >= gate`;
- the uncertainty range supports the gate when `lower >= gate`; and
- the uncertainty range supports gate failure when `upper < gate`.

For a `<= gate` claim:

- the point passes when `point <= gate`;
- the uncertainty range supports the gate when `upper <= gate`; and
- the uncertainty range supports gate failure when `lower > gate`.

The fixed evidence grades are:

| Grade | Definition |
|---|---|
| `gate_supported_by_uncertainty` | Point passes and the full stored range remains on the passing side of the gate |
| `point_gate_only` | Point passes but the stored range overlaps the gate |
| `gate_failure_supported_by_uncertainty` | Point fails and the full stored range remains on the failing side of the gate |
| `point_failure_with_gate_overlap` | Point fails but the stored range overlaps the gate |
| `no_direct_uncertainty_interval` | The original point gate has no stored interval for that exact derived contrast |

For bootstrap rows, `lower_95` and `upper_95` define the stored range. For the single-channel analysis, the minimum and maximum across the 16 leave-one-`pid`-out folds define a sensitivity range, not a confidence interval.

Each ranged claim will also be classified relative to zero as `positive`, `negative`, `spans_zero`, or `exact_zero`. This direction label is descriptive and does not replace the original material gate.

## 4. Predeclared Claims

### Single-channel contribution

| ID | Claim | Gate |
|---|---|---:|
| ES01 | `HB_1` neutralization has a material F1 loss | F1 difference `<= -0.03` |
| ES02 | `HB_2` neutralization has a material F1 loss | F1 difference `<= -0.03` |

### Raw-signal noise

| ID | Claim | Gate |
|---|---|---:|
| ES03 | Mild both-channel noise preserves F1 | F1 difference `>= -0.03` |
| ES04 | Mild both-channel noise preserves FAR | FAR difference `<= +0.50/hour` |
| ES05 | Severe both-channel noise causes material F1 loss | F1 difference `<= -0.03` |
| ES06 | Isolated `HB_1` noise causes material F1 loss | F1 difference `<= -0.03` |
| ES07 | Isolated `HB_1` noise causes material FAR increase | FAR difference `>= +0.50/hour` |
| ES08 | Isolated `HB_2` noise causes material F1 loss | F1 difference `<= -0.03` |
| ES09 | Isolated `HB_2` noise causes material FAR increase | FAR difference `>= +0.50/hour` |

### Train-only augmentation

| ID | Claim | Gate |
|---|---|---:|
| ES10 | Augmentation preserves clean F1 | F1 difference `>= -0.03` |
| ES11 | Augmentation preserves clean FAR | FAR difference `<= +0.50/hour` |
| ES12 | Augmentation materially improves `HB_1`-noise F1 | F1 difference `>= +0.03` |
| ES13 | Augmentation materially reduces `HB_1`-noise FAR | FAR difference `<= -0.50/hour` |
| ES14 | Augmentation meaningfully advances clean F1 | F1 difference `>= +0.05` |
| ES15 | Augmentation reduces the joint `HB_1` degradation gap | Original H8.8 point gate; no direct stored difference-in-differences interval |

### Cross-modality overlap

| ID | Claim | Gate |
|---|---|---:|
| ES16 | Shared direct-model misses are material | Fraction `>= 0.50` |
| ES17 | PSG-only recovery is material | Fraction `>= 0.15` |
| ES18 | Direct-model union gain is material | Recall gain `>= 0.10` |
| ES19 | Boundary-tolerance gain is material | Recall gain `>= 0.10` |

### Fixed fusion

| ID | Claim | Gate |
|---|---|---:|
| ES20 | Two-modality OR meaningfully improves F1 | F1 difference `>= +0.05` |
| ES21 | Two-modality OR keeps FAR increase within its bound | FAR difference `<= +0.25/hour` |
| ES22 | Laboratory consensus preserves or improves F1 | F1 difference `>= 0` |
| ES23 | Laboratory consensus does not increase FAR | FAR difference `<= 0/hour` |
| ES24 | Wider tolerance materially improves consensus F1 | F1 difference `>= +0.05` |
| ES25 | All-direct OR causes a material alarm penalty | FAR difference `>= +0.50/hour` |

## 5. Fixed Summary Questions

The analysis will report:

1. the count of claims in each evidence grade;
2. the count of ranged claims whose interval or sensitivity range is entirely positive, entirely negative, spans zero, or is exactly zero;
3. which passing point gates are only `point_gate_only`;
4. which failed gates are `gate_failure_supported_by_uncertainty`; and
5. whether the Block 8 non-advancement decision changes.

The non-advancement decision can change only if ES14 is not a supported failure and a deployable wearable method has a passing clean-advancement gate. No PSG-dependent fusion claim can satisfy that deployability condition.

## 6. Interpretation Limits

- Stored percentile intervals are reused descriptively; no multiplicity correction or new inferential claim is introduced.
- Leave-one-participant ranges are sensitivity summaries, not confidence intervals.
- An interval entirely beyond a gate strengthens that claim within the reused validation cohort; it is not independent confirmation.
- An interval overlapping a gate does not prove no effect.
- A directionally stable result can still fail a materiality gate.
- The analysis cannot reopen Block 8 model development, the current test partition, or any threshold search.

## 7. Planned Outputs

- source manifest with file hashes;
- 25-row evidence-strength table;
- grade and direction summary;
- list of point-only conclusions and uncertainty-supported failures;
- unchanged block-level decision statement;
- internal checks; and
- independent reconstruction report.
