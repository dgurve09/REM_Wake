# Block 9 Sleep-EDF Compatibility Protocol v0.1

**Created:** 2026-10-04

**Candidate dataset:** Sleep-EDF Database Expanded, PhysioNet version 1.0.0

**Dataset DOI:** https://doi.org/10.13026/C2X676

**Block:** External PSG generalization compatibility

**Protocol status:** Frozen before local metadata or signal inspection

**Model fitting authorized:** No

## 1. Question

Can Sleep-EDF Expanded support a scientifically interpretable external evaluation of the existing reduced-channel BOAS PSG method for REM-to-Wake boundary detection?

This is a compatibility gate, not a performance experiment. It cannot confirm the BOAS wearable detector because Sleep-EDF has no simultaneous BOAS headband recording.

## 2. Prior selection and official description

The June dataset review selected Sleep-EDF Expanded as the first external PSG candidate because it is openly accessible, relatively small, widely used, and includes expert hypnograms. The current official PhysioNet record describes 197 whole-night PSG recordings from two studies, stored as PSG EDF and hypnogram EDF+ files under the Open Data Commons Attribution License.

PhysioNet states that the EEG derivations are `Fpz-Cz` and `Pz-Oz`, sampled at 100 Hz, and that hypnograms use the 1968 Rechtschaffen and Kales labels `W`, `R`, `1`, `2`, `3`, `4`, `M`, and `?`.

Official source: https://physionet.org/content/sleep-edfx/1.0.0/

Original dataset paper:

Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberyé JJL. Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. *IEEE Transactions on Biomedical Engineering*. 2000;47(9):1185-1194. https://doi.org/10.1109/10.867928

## 3. Frozen BOAS comparator

The relevant BOAS comparator is the reduced PSG input `PSG_F3` and `PSG_F4`, referenced to M1, with five log-bandpower features per channel. The fixed direct feature schema therefore has ten ordered values representing a left/right frontal pair.

Sleep-EDF channels must not be renamed to imply that `Fpz-Cz` and `Pz-Oz` are equivalent to `F3-M1` and `F4-M1`. A technically executable ten-feature input is not sufficient for scientific compatibility.

## 4. Authorized inspection

Before the gate decision, access only:

1. the official `RECORDS` list;
2. the official `SHA256SUMS.txt` file;
3. the official SC and ST subject spreadsheets;
4. the lexicographically first complete PSG/hypnogram pair in the SC study; and
5. the lexicographically first complete PSG/hypnogram pair in the ST study.

Pilot selection is determined from the official manifest before EDF contents are inspected. Raw pilot files remain outside Git. Do not download the complete 8.1 GB dataset unless the direct-generalization gate passes.

## 5. Compatibility checks

### Identity and access

- dataset version is exactly 1.0.0;
- all manifest paths have official SHA-256 entries;
- the license permits project use with attribution; and
- each PSG has one identifiable hypnogram partner.

### Participant grouping

- study, subject, and night can be derived without using recording-level random splits;
- repeated nights map to one participant group; and
- treatment/placebo telemetry nights remain identifiable and are not mixed without a declared cohort rule.

### Labels and time base

- explicit Wake and REM annotations exist;
- annotation onsets and durations are finite, non-overlapping, and within PSG support;
- scoring intervals are compatible with 30-second boundary uncertainty;
- `R -> W` can be derived directly;
- `M` and `?` are excluded rather than remapped; and
- R&K stages 3 and 4 would be combined only if a broader N3 label is needed.

### Signal compatibility

- EEG sampling rate, units, channel names, and duration are readable from EDF headers;
- preprocessing can operate at 100 Hz without using unavailable BOAS sensors;
- no channel is silently treated as a BOAS electrode equivalent; and
- the external input preserves the physiological role and ordered feature meaning required by the frozen BOAS model.

## 6. Decision rules

### Direct-generalization pass

Proceed to a separately predeclared reduced-channel external evaluation only if:

1. participant grouping is reliable;
2. REM-to-Wake labels and timing are compatible;
3. the required EEG derivations preserve the physiological meaning of the frozen BOAS reduced-PSG feature positions; and
4. no model, threshold, or dataset subset is selected using external performance.

### Limited compatibility

If labels and timing pass but channel meaning does not, Sleep-EDF may support transition-count, label-timing, or within-dataset preprocessing studies. It must not receive the frozen BOAS model as a claimed external generalization test.

### No-go

Stop Block 9 model evaluation if participant identity, labels, timing, or channel physiology prevent an interpretable comparison. Record the mismatch rather than forcing harmonization.

## 7. Required outputs

- official manifest inventory and pairing table;
- subject/night grouping summary;
- pilot EDF channel and timing table;
- pilot annotation-label and duration table;
- BOAS-to-Sleep-EDF compatibility matrix;
- explicit pass, limited-compatibility, or no-go decision;
- source URLs, file hashes, script version, and software versions; and
- independent read-only validation of every retained claim.

## 8. Interpretation boundary

A passing external PSG result would address PSG-domain generalization only. It would not validate `HB_1/HB_2`, establish wearable deployment performance, confirm clinical utility, or authorize access to the closed BOAS validation/current-test partitions.
