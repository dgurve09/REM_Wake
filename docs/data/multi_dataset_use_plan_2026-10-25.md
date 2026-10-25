# Multi-Dataset Use Assessment

**Date:** 2026-10-25
**Purpose:** Define valid roles for the other locally available sleep datasets without conflating PSG, wearable EEG, and hypnogram-only evidence.

## Local inventory

The adjacent `Project_hypno` collection contains 2,381 harmonized scoring rows from ten public datasets: Challenge 2018 (994), DOD (400 scorer-record rows), ISRUC (252 expert-record rows), Sleep-EDF Expanded (197), HMC (151), BOAS (128), CAP Sleep (108), iSLEEPS (100), Sleep-Accel (31), and DREAMS (20).

Most local entries contain hypnograms and metadata, not raw EEG. Three collections provide local raw EEG with a common F4/C4/O2 PSG channel intersection relevant to the mechanism study:

| Dataset | Local signal evidence | Valid role here | Invalid inference |
|---|---|---|---|
| BOAS `ds005555` | 128 simultaneous PSG and two-channel forehead-headband recordings | Wearable development; paired PSG/wearable comparisons; participant-grouped evaluation | Clinical detection of sleep paralysis or narcolepsy |
| gx36m | 58 PSG recordings from 20 adults over three nights, with five EEG channels and three expert scorings | External PSG mechanism portability; scorer and repeated-night boundary variability | Wearable performance or device transfer |
| ISRUC group 3 | 10 PSG recordings from 10 adults, with six EEG channels and two expert scorings | External PSG mechanism portability and scorer sensitivity | Wearable performance or device transfer |

The BOAS dataset version used here is the public OpenNeuro snapshot `1.1.1` ([dataset DOI](https://doi.org/10.18112/openneuro.ds005555.v1.1.1)). Sleep-EDF remains useful for label-level analyses, but its `Fpz-Cz` and `Pz-Oz` montage is not a channel-semantic match to the BOAS left/right frontal input; the completed Block 9 audit therefore correctly stopped direct application of the frozen BOAS model.

## Why the sources have different roles

1. The final endpoint is wearable REM-to-Wake localization, and BOAS is the only local dataset with simultaneous PSG, human hypnograms, and the actual two-channel wearable signal.
2. BOAS, gx36m, and ISRUC group 3 were pooled only as training sources in a leave-one-dataset-out PSG mechanism study. This tests acquisition-domain portability but cannot estimate wearable generalization.
3. The hypnogram-only collections can test transition prevalence, scorer dependence, and label uncertainty. They cannot supply signal features.
4. Pooling EEG channels with different references and scalp locations as if they were exchangeable would make device, montage, and cohort effects inseparable.
5. Sleep-Accel contains a wearable modality but not wearable EEG, so it cannot validate the present signal model.

## Block 11 use

- Derive REM-to-Wake counts, event spacing, preceding REM duration, subsequent Wake duration, and night-position distributions from the ten-dataset hypnogram corpus.
- Quantify same-night transition disagreement in DOD, ISRUC, and gx36m rather than treating one scorer as exact physiological time.
- Quantify repeated-night variability where participant linkage is available.
- Use pathology labels only for descriptive transition stratification when sample definitions support it; do not infer diagnosis or clinical utility.
- The separately predeclared gx36m/ISRUC raw-EEG mechanism experiment used participant-level grouping across nights and preserved its PSG-only interpretation. Its point gate passed, but the improvement interval included zero.

## Confirmation boundary

No currently local non-BOAS dataset can confirm the wearable result. A valid wearable confirmation requires another simultaneously recorded wearable-PSG cohort with compatible signal access, participant identifiers, and independent scoring. Until then, external PSG and multi-dataset hypnograms provide mechanism and label-uncertainty evidence only.
