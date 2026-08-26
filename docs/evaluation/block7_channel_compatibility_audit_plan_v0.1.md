# Block 7 Channel Compatibility Audit Plan v0.1

**Created:** 2026-08-25
**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`
**Scope:** 128 simultaneous PSG/headband recording pairs
**Model training performed:** No

## 1. Purpose

Block 7 requires comparable full-PSG, reduced-PSG, and wearable inputs. Preliminary inspection shows that optional PSG and headband sensor channels are not uniform across recordings. Before feature extraction, this audit will determine whether a fixed EEG-only comparison can include the complete frozen cohort without channel-dependent exclusions.

This is a compatibility gate, not a performance experiment. It does not access model scores or select a model from validation or test behavior.

## 2. Proposed Input Sets

| Identifier | Proposed channels | Role |
|---|---|---|
| `PSG-6` | `PSG_F3`, `PSG_F4`, `PSG_C3`, `PSG_C4`, `PSG_O1`, `PSG_O2` | Complete common PSG EEG montage |
| `PSG-2` | `PSG_F3`, `PSG_F4` | Reduced PSG pair with the same feature dimension as the wearable |
| `HB-2` | `HB_1`, `HB_2` | Real wearable headband EEG |

`PSG_F3` maps to `HB_1` and `PSG_F4` maps to `HB_2` by left/right order for the zero-shot comparison. The locations are not equivalent: the wearable channels are approximately AF7/AF8. The resulting contrast combines electrode-location and device effects and must not be interpreted as a pure hardware effect.

Optional EOG, EMG, respiratory, pulse, oxygen-saturation, and motion channels are inventoried but are not permitted in `PSG-6`. Their channel availability and signal types vary, and a common EEG log-bandpower pipeline is not physiologically appropriate for all of them.

## 3. Checks

For every recording pair, verify:

1. both channel sidecars and both EDF files exist;
2. EDF header channel names match their sidecar channel names and order;
3. all six `PSG-6` channels and both `HB-2` channels are present exactly once;
4. required channels use microvolts and 256 Hz in the sidecars;
5. PSG and headband JSON sidecars report the same M1 reference, sampling frequency, and recording duration;
6. all 128 recordings remain eligible for `PSG-6`, `PSG-2`, and `HB-2`; and
7. optional-channel availability is reported rather than used to filter the cohort.

The audit will also retain each distinct PSG and headband channel configuration and the number of recordings using it.

## 4. Decision Rule

- **Pass:** all 128 pairs support `PSG-6`, `PSG-2`, and `HB-2`, and all required header/sidecar checks pass.
- **Revise:** a required channel is absent or inconsistent, but a smaller fixed set can be defined without consulting model performance.
- **No-go:** no scientifically interpretable fixed PSG/wearable mapping covers the frozen cohort.

Any revision requires a new protocol version before feature extraction. Missing optional sensors do not justify recording exclusion.

## 5. Retained Outputs

- `experiments/2026-08-25_block7_channel_compatibility_v0.1/recording_channel_eligibility_v0.1.tsv`
- `experiments/2026-08-25_block7_channel_compatibility_v0.1/channel_availability_v0.1.tsv`
- `experiments/2026-08-25_block7_channel_compatibility_v0.1/channel_configuration_summary_v0.1.tsv`
- `experiments/2026-08-25_block7_channel_compatibility_v0.1/README.md`

Raw EDF files remain outside Git. The retained tables contain only public BOAS recording identifiers and technical metadata.

## 6. Source

The dataset identity and acquisition fields are defined by the BOAS OpenNeuro snapshot: https://doi.org/10.18112/openneuro.ds005555.v1.1.1
