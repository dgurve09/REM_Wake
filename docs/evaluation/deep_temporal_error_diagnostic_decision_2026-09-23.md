# Deep Temporal Error Diagnostic Decision

**Decision date:** 2026-09-23

**Source experiment:** `2026-09-22_deep_temporal_nested_cv_v0.1`

**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`

**Authorized data:** frozen train outer-fold outputs only

**Model fitting:** No

**Validation accessed:** No

**Current test accessed:** No

## Question

Why does wearable REM-to-Wake event F1 remain low after temporal deep-learning models, and which mechanism is most defensible to test next?

## Timing Contribution

The frozen selected pipeline produced:

| Tolerance | True positives | Recall | F1 | False alarms/hour |
|---:|---:|---:|---:|---:|
| +/-15 s | 32 | 0.1778 | 0.1645 | 0.2812 |
| +/-45 s | 41 | 0.2278 | 0.2113 | 0.2653 |
| +/-75 s | 41 | 0.2278 | 0.2119 | 0.2637 |
| +/-105 s | 42 | 0.2333 | 0.2171 | 0.2621 |

One adjacent 30-second bin recovered nine events, but widening to +/-105 seconds recovered only one additional event. Boundary timing contributes to error but does not explain most of the 148 primary false negatives.

## Threshold Upper Bound

The frozen fold-local thresholds gave F1 0.1645. A post-result pooled outer-label oracle threshold reached F1 0.2177, and separate label-informed thresholds for every outer fold reached pooled F1 0.2324.

These are deliberately optimistic and cannot replace the frozen thresholds because they use outer labels. They show that threshold transfer contributes approximately 0.05-0.07 possible F1 in this reused sample. They also show that even highly optimistic threshold choice leaves F1 below 0.25; calibration alone cannot make the detector reliable.

## Quality-Tier Dependence

At +/-15 seconds:

| Model | Clean detected | MAD-flagged detected | Clean recall | MAD-flagged recall |
|---|---:|---:|---:|---:|
| LR-OOF | 3/53 | 54/127 | 0.0566 | 0.4252 |
| LC-1 | 9/53 | 47/127 | 0.1698 | 0.3701 |
| Nested selected | 1/53 | 31/127 | 0.0189 | 0.2441 |

The same direction appears in all three model procedures. For the nested scores, median exact-boundary probability was 0.2623 for clean events and 0.9495 for MAD-flagged events, while fold thresholds ranged from approximately 0.98 to 0.999.

The 10-MAD flag is nonspecific and was never validated as an unusable-signal label. Therefore this result does not prove that models detect artefacts rather than physiology. It does establish a large quality-tier dependence that aggregate F1 hides. The dependence could reflect amplitude structure, participant composition, or other correlated recording characteristics and must be isolated experimentally.

## False-Alarm Context

Of 177 primary false alarms:

- 167 were not within 45 seconds of any labeled transition;
- nine were 16-45 seconds from a primary REM-to-Wake boundary;
- one was within 45 seconds of a quality-sensitivity-only REM-to-Wake boundary; and
- none was assigned to the Wake-to-REM proximity category.

A simple veto around labeled reverse transitions therefore cannot explain or remove most false alarms. Most alarms require signal-level characterization rather than label-context relabeling.

## Decision

Do not perform another unrestricted encoder or threshold search. The next defensible development mechanism is a small LSTM-CRF experiment that isolates quality-tier imbalance and amplitude normalization while preserving the event-specific target:

1. exact `LC-1` control;
2. participant-fit robust normalization using median and interquartile range instead of mean and standard deviation;
3. equal total positive loss contribution from clean and MAD-flagged events; and
4. unchanged negative sampling, event consolidation, thresholds, participant grouping, and evaluation.

The experiment must report overall event F1, clean-event recall, MAD-flagged recall, and false alarms/hour. Advancement should require at least +0.05 overall F1, at least +0.10 clean-event recall, and no false-alarm increase relative to its nested `LC-1` control. A result that only improves flagged-event recall does not address the observed uncertainty.

If this controlled mechanism fails, the next representation question is whether a compact raw-waveform epoch encoder recovers morphology discarded by the ten bandpower features. Raw EEG convolutional representation learning is established in sleep analysis, and subject-level robust normalization has improved a different sleep-EEG event detector; neither prior result establishes effectiveness for REM-to-Wake boundaries here [1-3]. Because only 180 positive train events are available, raw-waveform work should remain small and should not begin as a broad architecture search.

## Verification

- All 9/9 in-run integrity checks passed.
- The accepted analysis completed an immutable rerun without changing any reviewed output.
- The independent read-only validator passed 8/8 checks twice.
- Nine source files matched their recorded sizes and SHA-256 values.
- Validation and test artifacts remained closed.

## References

1. Supratak A, Dong H, Wu C, Guo Y. DeepSleepNet: a model for automatic sleep stage scoring based on raw single-channel EEG. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2017;25(11):1998-2008. https://doi.org/10.1109/TNSRE.2017.2721116
2. Chambon S, Galtier MN, Arnal PJ, Wainrib G, Gramfort A. A deep learning architecture for temporal sleep stage classification using multivariate and multimodal time series. *IEEE Transactions on Neural Systems and Rehabilitation Engineering*. 2018;26(4):758-769. https://doi.org/10.1109/TNSRE.2018.2813138
3. Hartmann S, Baumert M. Subject-level normalization to improve A-phase detection of cyclic alternating pattern in sleep EEG. *2023 45th Annual International Conference of the IEEE Engineering in Medicine & Biology Society*. 2023:1-4. https://doi.org/10.1109/EMBC40787.2023.10340124
