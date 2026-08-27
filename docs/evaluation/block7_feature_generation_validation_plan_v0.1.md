# Block 7 Feature-Generation Validation Plan v0.1

**Created:** 2026-08-26
**Depends on:** `block7_paired_transfer_protocol_v0.1.md`
**Channel gate:** Passed 128/128
**Initial scope:** Train partition only
**Model training performed:** No
**Validation or test signals accessed:** No

## 1. Purpose

The channel audit established that every recording supports `PSG-6`, `PSG-2`, and `HB-2`. The next risk is implementation-induced modality difference: separate code paths could apply different filters, scalers, epoch boundaries, spectral definitions, or feature ordering and create an apparent device shift.

This validation will establish that all three modality paths implement the committed feature definition consistently before any Block 7 model is fitted.

## 2. Fixed Feature Schemas

| Input | Channels | Bands per channel | Epochs of context | Expected features |
|---|---:|---:|---:|---:|
| `PSG-6` | 6 | 5 | 8 | 240 |
| `PSG-2` | 2 | 5 | 8 | 80 |
| `HB-2` | 2 | 5 | 8 | 80 |

Band order is delta, theta, alpha, sigma, beta. Context order is `t-120`, `t-90`, `t-60`, `t-30`, `t`, `t+30`, `t+60`, and `t+90` seconds. Channel order is the order frozen in the transfer protocol.

## 3. Train-Only Scalers

1. Fit robust channel centers and `1.4826 * MAD` scales from a deterministic 1 Hz sample of filtered/resampled train recordings only.
2. Fit one six-channel PSG scaler and one two-channel wearable scaler.
3. Reuse the `PSG_F3/F4` parameters from the six-channel PSG scaler for `PSG-2`; do not refit a second PSG scaler.
4. Record sample counts, centers, scales, source recordings, split version, and hashes.
5. Treat a nonfinite or nonpositive scale as a failed run; do not replace it silently.

Validation and test recordings cannot contribute to a scaler.

## 4. Required Checks

### 4.1 Signal-path checks

- all train EDFs have 256 Hz input and finite required samples;
- the 0.3-35 Hz filter and 256-to-128 Hz resampling are identical across modalities;
- output length follows the same deterministic resampling rule;
- no rereferencing or separate notch filter is introduced; and
- incomplete 30-second epochs or eight-epoch contexts are dropped by the same coverage rule.

### 4.2 Synthetic spectral checks

Run fixed synthetic sinusoids through the shared filter/resample/Welch path. For a frequency inside each band, its intended band must have greater power than every nonadjacent band. Retain the complete check table, including any failure.

### 4.3 Overlapping-channel parity

For every train recording and retained epoch, the `PSG_F3/F4` base features produced through the `PSG-6` path must match those produced through the `PSG-2` path within absolute tolerance `1e-10` before float32 storage. Onsets, retained-row masks, feature names, and values must agree.

This is the primary implementation control: failure means the later `PSG-6` versus `PSG-2` comparison would not isolate channel count.

### 4.4 Wearable reproduction

The new `HB-2` base epoch features for all train recordings must reproduce the existing frozen stage-first feature arrays:

- identical retained onsets and feature names;
- maximum absolute feature difference no greater than `1e-6`, allowing the existing float32 storage boundary; and
- identical SHA-256 only when the complete serialized artifact is expected to be byte-for-byte unchanged.

If values differ, record whether the cause is scaler, filtering, epoch coverage, Welch configuration, feature order, numerical precision, or an unresolved implementation difference. Do not proceed to fitting until it is resolved or a new protocol version explicitly authorizes the change.

### 4.5 Context construction

- every context has exactly eight adjacent 30-second epochs;
- context never crosses a recording edge, invalid scored epoch, disconnection, missing epoch, or gap;
- all three modality paths retain the same context onset for a given recording when their signal coverage is complete; and
- all feature values are finite.

## 5. Success Rule

The feature gate passes only if:

1. all train recordings complete the required signal-path checks;
2. all synthetic spectral checks pass;
3. `PSG-6` and `PSG-2` overlapping features satisfy exact schema/onset parity and the `1e-10` value tolerance;
4. `HB-2` reproduces the frozen train features within `1e-6`;
5. context onsets agree across modalities;
6. no validation or test artifact is accessed; and
7. all scalers and generated feature artifacts are hashed and recorded.

Any failed check is retained. A scientific method change requires a new protocol version; a coding correction requires an implementation-correction record and a deterministic rerun.

## 6. Artifact Boundary

Full recording feature arrays and scaler sampling arrays remain outside Git. Git may retain:

- scaler summaries and hashes;
- per-recording construction counts and feature hashes;
- synthetic checks;
- cross-path parity summaries;
- the wearable reproduction summary;
- software versions; and
- the feature-gate decision.

No model fit, candidate probability, alarm, event metric, or test result is authorized by this plan.
