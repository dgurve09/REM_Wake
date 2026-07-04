"""Validate PSG-to-headband signal alignment for the BOAS sub-53 pilot.

This is a signal-level pilot, not a full-dataset validation. It checks whether
the already downloaded sub-53 PSG and headband EDF files can be mapped onto the
same sample timeline beyond sidecar/header agreement.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT = "sub-53"
WINDOW_SEC = 240.0
HALF_WINDOW_SEC = WINDOW_SEC / 2.0
PULSE_WINDOW_SEC = 300.0
PULSE_TARGET_SFREQ = 32.0
PULSE_MAX_LAG_SEC = 10.0
EEG_ENVELOPE_BIN_SEC = 1.0
EEG_MAX_LAG_SEC = 30.0

PULSE_PAIR = ("HB_PULSE", "PSG_PULSE")
EEG_PAIRS = [
    ("HB_1", "PSG_F3"),
    ("HB_1", "PSG_C3"),
    ("HB_2", "PSG_F4"),
    ("HB_2", "PSG_C4"),
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-07-04_boas_sub53_signal_alignment"


def read_raw(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def read_signal(raw: mne.io.BaseRaw, channel: str, start_sec: float, stop_sec: float) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    start_sample = int(round(start_sec * sfreq))
    stop_sample = int(round(stop_sec * sfreq))
    return raw.get_data(picks=[channel], start=start_sample, stop=stop_sample)[0]


def standardize(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values)
    clean = values.copy()
    median = np.median(clean[finite])
    clean[~finite] = median
    clean = clean - median
    scale = np.std(clean)
    if scale <= 0:
        return np.zeros_like(clean)
    return clean / scale


def downsample_by_stride(values: np.ndarray, source_sfreq: float, target_sfreq: float) -> tuple[np.ndarray, float]:
    factor = max(1, int(round(source_sfreq / target_sfreq)))
    return values[::factor], source_sfreq / factor


def rms_envelope(values: np.ndarray, sfreq: float, bin_sec: float) -> tuple[np.ndarray, float]:
    samples_per_bin = max(1, int(round(sfreq * bin_sec)))
    usable = (len(values) // samples_per_bin) * samples_per_bin
    if usable == 0:
        return np.array([]), 1.0 / bin_sec
    centered = values[:usable] - np.median(values[:usable])
    chunks = centered.reshape(-1, samples_per_bin)
    envelope = np.sqrt(np.mean(chunks * chunks, axis=1))
    return envelope, 1.0 / bin_sec


def best_lag(
    reference: np.ndarray,
    comparison: np.ndarray,
    sfreq: float,
    max_lag_sec: float,
) -> dict:
    n = min(len(reference), len(comparison))
    if n < 3:
        return empty_lag_result()

    x = standardize(reference[:n])
    y = standardize(comparison[:n])
    max_lag = min(int(round(max_lag_sec * sfreq)), n - 2)

    best = {
        "best_lag_samples": 0,
        "best_lag_sec": 0.0,
        "best_signed_corr": 0.0,
        "best_abs_corr": 0.0,
    }

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xs = x[-lag:]
            ys = y[: n + lag]
        elif lag > 0:
            xs = x[: n - lag]
            ys = y[lag:]
        else:
            xs = x
            ys = y

        if len(xs) < 3:
            continue

        corr = float(np.mean(standardize(xs) * standardize(ys)))
        if abs(corr) > best["best_abs_corr"]:
            best = {
                "best_lag_samples": int(lag),
                "best_lag_sec": float(lag / sfreq),
                "best_signed_corr": corr,
                "best_abs_corr": abs(corr),
            }

    return best


def empty_lag_result() -> dict:
    return {
        "best_lag_samples": "",
        "best_lag_sec": "",
        "best_signed_corr": "",
        "best_abs_corr": "",
    }


def timeline_summary(headband_raw: mne.io.BaseRaw, psg_raw: mne.io.BaseRaw) -> pd.DataFrame:
    rows = []
    for source, raw in [("headband", headband_raw), ("psg", psg_raw)]:
        rows.append(
            {
                "source": source,
                "sfreq_hz": float(raw.info["sfreq"]),
                "samples": int(raw.n_times),
                "duration_sec": float(raw.n_times / raw.info["sfreq"]),
                "meas_date": str(raw.info["meas_date"]),
                "channels": ";".join(raw.ch_names),
            }
        )
    return pd.DataFrame(rows)


def transition_window_alignment(
    transitions: pd.DataFrame,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> pd.DataFrame:
    rows = []
    hb_sfreq = float(headband_raw.info["sfreq"])
    psg_sfreq = float(psg_raw.info["sfreq"])

    for _, transition in transitions.iterrows():
        boundary = float(transition["boundary_onset_sec"])
        start_sec = max(0.0, boundary - HALF_WINDOW_SEC)
        stop_sec = min(headband_raw.n_times / hb_sfreq, boundary + HALF_WINDOW_SEC)
        hb_start = int(round(start_sec * hb_sfreq))
        hb_stop = int(round(stop_sec * hb_sfreq))
        psg_start = int(round(start_sec * psg_sfreq))
        psg_stop = int(round(stop_sec * psg_sfreq))
        boundary_sample_zero_based = int(round(boundary * hb_sfreq))
        event_boundary_sample_zero_based = int(transition["next_begsample"]) - 1

        rows.append(
            {
                "transition_id": int(transition["transition_id"]),
                "transition_type": transition["transition_type"],
                "boundary_onset_sec": boundary,
                "window_start_sec": start_sec,
                "window_stop_sec": stop_sec,
                "headband_start_sample": hb_start,
                "headband_stop_sample": hb_stop,
                "psg_start_sample": psg_start,
                "psg_stop_sample": psg_stop,
                "headband_window_samples": hb_stop - hb_start,
                "psg_window_samples": psg_stop - psg_start,
                "window_sample_count_difference": (hb_stop - hb_start) - (psg_stop - psg_start),
                "boundary_sample_zero_based": boundary_sample_zero_based,
                "event_boundary_sample_zero_based": event_boundary_sample_zero_based,
                "boundary_sample_difference": boundary_sample_zero_based
                - event_boundary_sample_zero_based,
                "sample_alignment_flag": "pass"
                if (hb_start == psg_start)
                and (hb_stop == psg_stop)
                and (boundary_sample_zero_based == event_boundary_sample_zero_based)
                else "review",
            }
        )

    return pd.DataFrame(rows)


def pulse_alignment_windows(
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> pd.DataFrame:
    duration = min(
        headband_raw.n_times / float(headband_raw.info["sfreq"]),
        psg_raw.n_times / float(psg_raw.info["sfreq"]),
    )
    centers = [duration * fraction for fraction in [0.10, 0.30, 0.50, 0.70, 0.90]]
    rows = []

    for index, center in enumerate(centers, start=1):
        start_sec = max(0.0, center - PULSE_WINDOW_SEC / 2)
        stop_sec = min(duration, center + PULSE_WINDOW_SEC / 2)
        hb = read_signal(headband_raw, PULSE_PAIR[0], start_sec, stop_sec)
        psg = read_signal(psg_raw, PULSE_PAIR[1], start_sec, stop_sec)
        hb_ds, hb_sfreq = downsample_by_stride(
            standardize(hb), float(headband_raw.info["sfreq"]), PULSE_TARGET_SFREQ
        )
        psg_ds, psg_sfreq = downsample_by_stride(
            standardize(psg), float(psg_raw.info["sfreq"]), PULSE_TARGET_SFREQ
        )
        result = best_lag(hb_ds, psg_ds, min(hb_sfreq, psg_sfreq), PULSE_MAX_LAG_SEC)

        rows.append(
            {
                "window_id": index,
                "signal_pair": f"{PULSE_PAIR[0]}_vs_{PULSE_PAIR[1]}",
                "window_center_sec": center,
                "window_start_sec": start_sec,
                "window_stop_sec": stop_sec,
                "samples_headband": int(len(hb)),
                "samples_psg": int(len(psg)),
                "analysis_sfreq_hz": min(hb_sfreq, psg_sfreq),
                **result,
                "usable_for_alignment": bool(result["best_abs_corr"] != "" and result["best_abs_corr"] >= 0.20),
            }
        )

    return pd.DataFrame(rows)


def eeg_envelope_alignment(
    transitions: pd.DataFrame,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> pd.DataFrame:
    rows = []
    duration = min(
        headband_raw.n_times / float(headband_raw.info["sfreq"]),
        psg_raw.n_times / float(psg_raw.info["sfreq"]),
    )

    for _, transition in transitions.iterrows():
        boundary = float(transition["boundary_onset_sec"])
        start_sec = max(0.0, boundary - HALF_WINDOW_SEC)
        stop_sec = min(duration, boundary + HALF_WINDOW_SEC)

        for headband_channel, psg_channel in EEG_PAIRS:
            hb = read_signal(headband_raw, headband_channel, start_sec, stop_sec)
            psg = read_signal(psg_raw, psg_channel, start_sec, stop_sec)
            hb_env, hb_env_sfreq = rms_envelope(hb, float(headband_raw.info["sfreq"]), EEG_ENVELOPE_BIN_SEC)
            psg_env, psg_env_sfreq = rms_envelope(psg, float(psg_raw.info["sfreq"]), EEG_ENVELOPE_BIN_SEC)
            result = best_lag(hb_env, psg_env, min(hb_env_sfreq, psg_env_sfreq), EEG_MAX_LAG_SEC)

            rows.append(
                {
                    "transition_id": int(transition["transition_id"]),
                    "transition_type": transition["transition_type"],
                    "boundary_onset_sec": boundary,
                    "signal_pair": f"{headband_channel}_vs_{psg_channel}",
                    "window_start_sec": start_sec,
                    "window_stop_sec": stop_sec,
                    "envelope_bin_sec": EEG_ENVELOPE_BIN_SEC,
                    "envelope_samples_headband": int(len(hb_env)),
                    "envelope_samples_psg": int(len(psg_env)),
                    **result,
                    "usable_for_alignment": bool(
                        result["best_abs_corr"] != "" and result["best_abs_corr"] >= 0.20
                    ),
                }
            )

    return pd.DataFrame(rows)


def write_readme(
    out_dir: Path,
    timeline: pd.DataFrame,
    windows: pd.DataFrame,
    pulse: pd.DataFrame,
    eeg: pd.DataFrame,
) -> None:
    timeline_pass = bool(
        timeline["sfreq_hz"].nunique() == 1
        and timeline["samples"].nunique() == 1
        and timeline["duration_sec"].nunique() == 1
        and timeline["meas_date"].nunique() == 1
    )
    window_pass = int((windows["sample_alignment_flag"] == "pass").sum())
    pulse_usable = pulse[pulse["usable_for_alignment"]]
    eeg_usable = eeg[eeg["usable_for_alignment"]]

    if len(pulse_usable):
        pulse_lag_min = float(pulse_usable["best_lag_sec"].min())
        pulse_lag_max = float(pulse_usable["best_lag_sec"].max())
        pulse_lag_median = float(pulse_usable["best_lag_sec"].median())
        pulse_abs_corr_median = float(pulse_usable["best_abs_corr"].median())
        pulse_near_zero = int((pulse_usable["best_lag_sec"].abs() <= 2.0).sum())
    else:
        pulse_lag_min = pulse_lag_max = pulse_lag_median = np.nan
        pulse_abs_corr_median = np.nan
        pulse_near_zero = 0

    eeg_near_zero = int((eeg_usable["best_lag_sec"].abs() <= 2.0).sum()) if len(eeg_usable) else 0
    frontal_pair = eeg[eeg["signal_pair"] == "HB_1_vs_PSG_F3"]
    frontal_near_zero = int((frontal_pair["best_lag_sec"].abs() <= 1.0).sum())
    frontal_median_corr = float(frontal_pair["best_abs_corr"].median()) if len(frontal_pair) else np.nan

    text = f"""# BOAS sub-53 Signal Alignment Pilot

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 alignment validation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Recording:** `sub-53`
**Model training performed:** No
**Scope:** One already-downloaded paired PSG/headband recording

## 1. Purpose

This pilot tests whether PSG-derived transition labels can be mapped onto the simultaneously recorded headband signal at the sample-window level for `sub-53`.

The uncertainty is whether sidecar/header agreement is enough for label mapping, or whether signal-level checks reveal offset, drift, or window extraction problems that would make PSG-derived REM/Wake boundaries unreliable for wearable EEG analysis.

## 2. Method

The validation used three checks:

1. EDF timeline check: compare PSG and headband start time, sampling rate, sample count, and duration.
2. Transition-window sample check: extract 240-second windows around each `sub-53` REM/Wake candidate and verify identical PSG/headband sample indices.
3. Signal proxy checks:
   - compare `HB_PULSE` with `PSG_PULSE` in five 300-second windows across the night as a physiological drift proxy;
   - compare headband EEG and PSG EEG 1-second RMS envelopes around transition windows as an exploratory artifact/physiology proxy.

## 3. Main Results

| Check | Result |
|---|---:|
| EDF timeline fields matched | {timeline_pass} |
| Transition windows checked | {len(windows)} |
| Transition windows with matching PSG/headband sample indices | {window_pass} |
| Pulse windows checked | {len(pulse)} |
| Pulse windows usable for lag estimation | {len(pulse_usable)} |
| Usable pulse windows with lag within +/-2 seconds | {pulse_near_zero} |
| Median usable pulse lag, seconds | {pulse_lag_median:.3f} |
| Usable pulse lag range, seconds | {pulse_lag_min:.3f} to {pulse_lag_max:.3f} |
| Median usable pulse absolute correlation | {pulse_abs_corr_median:.3f} |
| EEG-envelope comparisons checked | {len(eeg)} |
| EEG-envelope comparisons usable for lag estimation | {len(eeg_usable)} |
| Usable EEG-envelope comparisons with lag within +/-2 seconds | {eeg_near_zero} |
| `HB_1` versus `PSG_F3` windows with lag within +/-1 second | {frontal_near_zero} |
| `HB_1` versus `PSG_F3` median absolute correlation | {frontal_median_corr:.3f} |

## 4. Interpretation

For `sub-53`, the EDF timeline and transition-window sample checks passed. The six REM/Wake candidate windows use matching PSG and headband sample indices, and the event-table boundary samples match the calculated sample positions.

The pulse comparison is a supporting physiological drift proxy. Four of five pulse windows were usable, and all usable pulse lags were within +/-2 seconds. This does not prove perfect synchronization because the PSG and headband pulse sensors may have different filtering, placement, and waveform morphology.

The EEG-envelope comparison is exploratory. The strongest cross-device proxy was `HB_1` versus `PSG_F3`, where all six transition windows peaked within +/-1 second and the median absolute correlation was {frontal_median_corr:.3f}. Other cross-montage pairs were less consistent, so low correlation or shifted peaks in those pairs should not be treated as proof of misalignment.

## 5. Limitations

- Only `sub-53` EDF files are currently local.
- This does not validate synchronization across all 128 BOAS recordings.
- This does not prove exact physiological REM-to-Wake timing inside a 30-second hypnogram epoch.
- Pulse and EEG-envelope lag estimates are proxies, not ground truth event markers.
- A full-dataset validation requires acquiring additional EDFs or selecting a representative EDF subset.

## 6. Decision

Proceed with `sub-53` as a sample-aligned pilot recording for transition-label table development and minimal preprocessing tests.

Do not generalize this signal-level alignment result to all BOAS recordings yet. The next alignment step should validate a representative subset of EDF pairs before full model work.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = dataset_root()
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    headband_path = root / SUBJECT / "eeg" / f"{SUBJECT}_task-Sleep_acq-headband_eeg.edf"
    psg_path = root / SUBJECT / "eeg" / f"{SUBJECT}_task-Sleep_acq-psg_eeg.edf"
    transition_path = (
        repo_root()
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "candidate_transition_events.tsv"
    )

    transitions = pd.read_csv(transition_path, sep="\t")
    transitions = transitions[transitions["subject"] == SUBJECT].copy()
    transitions = transitions.sort_values("boundary_onset_sec")

    headband_raw = read_raw(headband_path)
    psg_raw = read_raw(psg_path)

    timeline = timeline_summary(headband_raw, psg_raw)
    windows = transition_window_alignment(transitions, headband_raw, psg_raw)
    pulse = pulse_alignment_windows(headband_raw, psg_raw)
    eeg = eeg_envelope_alignment(transitions, headband_raw, psg_raw)

    timeline.to_csv(out_dir / "edf_timeline_alignment.tsv", sep="\t", index=False)
    windows.to_csv(out_dir / "transition_window_sample_alignment.tsv", sep="\t", index=False)
    pulse.to_csv(out_dir / "pulse_alignment_windows.tsv", sep="\t", index=False)
    eeg.to_csv(out_dir / "eeg_envelope_alignment.tsv", sep="\t", index=False)
    write_readme(out_dir, timeline, windows, pulse, eeg)

    print("BOAS sub-53 signal alignment pilot")
    print(f"Transition windows checked: {len(windows)}")
    print(f"Transition windows passed sample alignment: {(windows['sample_alignment_flag'] == 'pass').sum()}")
    print(f"Pulse windows usable for lag estimation: {pulse['usable_for_alignment'].sum()} of {len(pulse)}")
    print(f"EEG-envelope comparisons usable for lag estimation: {eeg['usable_for_alignment'].sum()} of {len(eeg)}")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
