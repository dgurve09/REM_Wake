# Block 9 Sleep-EDF Compatibility Audit v0.1

Sleep-EDF Expanded version 1.0.0 passed identity, participant-grouping, REM/Wake-label, 30-second-grid, nonoverlap, and pilot signal-readability checks. The manifest contains 197 PSG/hypnogram pairs from 100 participant groups. One excluded `Sleep stage ?` interval begins at the PSG end and extends beyond signal support, so the protocol's complete annotation-support criterion failed.

The direct external model gate did not pass. Sleep-EDF provides bipolar `Fpz-Cz` and `Pz-Oz`, whereas the frozen BOAS reduced-PSG model uses ordered left/right `F3-M1` and `F4-M1` inputs. Substituting the Sleep-EDF derivations would change both electrode location and feature meaning. It would be technically executable but scientifically uninterpretable as direct model generalization.

**Decision:** `complete_no_go`.

No model was fitted, no full-dataset download was authorized, and no BOAS validation or current-test artifact was accessed. The pilot demonstrates that R-to-W boundaries can be derived, but the complete Block 9 gate does not authorize further Sleep-EDF analysis. It is not a wearable confirmation cohort.

Official dataset: https://physionet.org/content/sleep-edfx/1.0.0/

Dataset DOI: https://doi.org/10.13026/C2X676

Original paper: https://doi.org/10.1109/10.867928
