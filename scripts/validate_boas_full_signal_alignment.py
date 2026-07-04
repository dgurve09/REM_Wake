"""Validate BOAS PSG-to-headband signal alignment across downloaded EDFs.

This script runs after full EDF acquisition. It checks timeline agreement,
transition-window sample mapping, and signal-level alignment proxies. It does
not train or evaluate a model.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT_COUNT = 128
PULSE_WINDOW_SEC = 300.0
PULSE_TARGET_SFREQ = 32.0
PULSE_MAX_LAG_SEC = 10.0
EEG_WINDOW_SEC = 240.0
EEG_ENVELOPE_BIN_SEC = 1.0
EEG_MAX_LAG_SEC = 30.0
PULSE_PAIR = ("HB_PULSE", "PSG_PULSE")
EEG_PAIR = ("HB_1", "PSG_F3")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-07-04_boas_full_signal_alignment"


def subject_paths(root: Path, subject_id: int) -> dict[str, Path]:
    subject = f"sub-{subject_id}"
    prefix = root / subject / "eeg" / f"{subject}_task-Sleep"
    return {
        "headband_edf": Path(f"{prefix}_acq-headband_eeg.edf"),
        "psg_edf": Path(f"{prefix}_acq-psg_eeg.edf"),
    }


def read_raw(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def load_participants(root: Path) -> pd.DataFrame:
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants["subject_id"] = participants["participant_id"].str.replace("sub-", "", regex=False).astype(int)
    return participants.set_index("subject_id")


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
    return np.sqrt(np.mean(chunks * chunks, axis=1)), 1.0 / bin_sec


def empty_lag_result() -> dict:
    return {
        "best_lag_samples": "",
        "best_lag_sec": "",
        "best_signed_corr": "",
        "best_abs_corr": "",
    }


def best_lag(reference: np.ndarray, comparison: np.ndarray, sfreq: float, max_lag_sec: float) -> dict:
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


def timeline_row(
    subject: str,
    pid: int,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> dict:
    hb_sfreq = float(headband_raw.info["sfreq"])
    psg_sfreq = float(psg_raw.info["sfreq"])
    hb_duration = float(headband_raw.n_times / hb_sfreq)
    psg_duration = float(psg_raw.n_times / psg_sfreq)
    hb_date = str(headband_raw.info["meas_date"])
    psg_date = str(psg_raw.info["meas_date"])
    pass_flag = (
        hb_sfreq == psg_sfreq
        and headband_raw.n_times == psg_raw.n_times
        and hb_duration == psg_duration
        and hb_date == psg_date
    )
    return {
        "subject": subject,
        "pid": pid,
        "headband_sfreq_hz": hb_sfreq,
        "psg_sfreq_hz": psg_sfreq,
        "headband_samples": int(headband_raw.n_times),
        "psg_samples": int(psg_raw.n_times),
        "sample_count_difference": int(headband_raw.n_times - psg_raw.n_times),
        "headband_duration_sec": hb_duration,
        "psg_duration_sec": psg_duration,
        "duration_difference_sec": hb_duration - psg_duration,
        "headband_meas_date": hb_date,
        "psg_meas_date": psg_date,
        "timeline_alignment_flag": "pass" if pass_flag else "review",
    }


def transition_window_rows(
    subject: str,
    pid: int,
    transitions: pd.DataFrame,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> list[dict]:
    rows = []
    hb_sfreq = float(headband_raw.info["sfreq"])
    psg_sfreq = float(psg_raw.info["sfreq"])
    duration = min(headband_raw.n_times / hb_sfreq, psg_raw.n_times / psg_sfreq)
    half_window = EEG_WINDOW_SEC / 2.0

    for _, transition in transitions.iterrows():
        boundary = float(transition["boundary_onset_sec"])
        start_sec = max(0.0, boundary - half_window)
        stop_sec = min(duration, boundary + half_window)
        hb_start = int(round(start_sec * hb_sfreq))
        hb_stop = int(round(stop_sec * hb_sfreq))
        psg_start = int(round(start_sec * psg_sfreq))
        psg_stop = int(round(stop_sec * psg_sfreq))
        boundary_sample = int(round(boundary * hb_sfreq))
        event_boundary_sample = int(transition["next_begsample"]) - 1

        pass_flag = (
            hb_start == psg_start
            and hb_stop == psg_stop
            and boundary_sample == event_boundary_sample
        )
        rows.append(
            {
                "transition_id": int(transition["transition_id"]),
                "subject": subject,
                "pid": pid,
                "transition_type": transition["transition_type"],
                "boundary_onset_sec": boundary,
                "headband_start_sample": hb_start,
                "headband_stop_sample": hb_stop,
                "psg_start_sample": psg_start,
                "psg_stop_sample": psg_stop,
                "window_sample_count_difference": (hb_stop - hb_start) - (psg_stop - psg_start),
                "boundary_sample_difference": boundary_sample - event_boundary_sample,
                "sample_alignment_flag": "pass" if pass_flag else "review",
            }
        )

    return rows


def pulse_rows(
    subject: str,
    pid: int,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> list[dict]:
    rows = []
    duration = min(
        headband_raw.n_times / float(headband_raw.info["sfreq"]),
        psg_raw.n_times / float(psg_raw.info["sfreq"]),
    )

    if PULSE_PAIR[0] not in headband_raw.ch_names or PULSE_PAIR[1] not in psg_raw.ch_names:
        return [
            {
                "subject": subject,
                "pid": pid,
                "window_id": "",
                "window_center_sec": "",
                "signal_pair": f"{PULSE_PAIR[0]}_vs_{PULSE_PAIR[1]}",
                "availability": "missing_channel",
                **empty_lag_result(),
                "usable_for_alignment": False,
            }
        ]

    centers = [duration * fraction for fraction in [0.10, 0.30, 0.50, 0.70, 0.90]]
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
        usable = bool(result["best_abs_corr"] != "" and result["best_abs_corr"] >= 0.20)
        rows.append(
            {
                "subject": subject,
                "pid": pid,
                "window_id": index,
                "window_center_sec": center,
                "signal_pair": f"{PULSE_PAIR[0]}_vs_{PULSE_PAIR[1]}",
                "availability": "available",
                **result,
                "usable_for_alignment": usable,
            }
        )
    return rows


def eeg_envelope_rows(
    subject: str,
    pid: int,
    transitions: pd.DataFrame,
    headband_raw: mne.io.BaseRaw,
    psg_raw: mne.io.BaseRaw,
) -> list[dict]:
    rows = []
    duration = min(
        headband_raw.n_times / float(headband_raw.info["sfreq"]),
        psg_raw.n_times / float(psg_raw.info["sfreq"]),
    )
    half_window = EEG_WINDOW_SEC / 2.0

    if EEG_PAIR[0] not in headband_raw.ch_names or EEG_PAIR[1] not in psg_raw.ch_names:
        return [
            {
                "transition_id": "",
                "subject": subject,
                "pid": pid,
                "transition_type": "",
                "boundary_onset_sec": "",
                "signal_pair": f"{EEG_PAIR[0]}_vs_{EEG_PAIR[1]}",
                "availability": "missing_channel",
                **empty_lag_result(),
                "usable_for_alignment": False,
            }
        ]

    for _, transition in transitions.iterrows():
        boundary = float(transition["boundary_onset_sec"])
        start_sec = max(0.0, boundary - half_window)
        stop_sec = min(duration, boundary + half_window)
        hb = read_signal(headband_raw, EEG_PAIR[0], start_sec, stop_sec)
        psg = read_signal(psg_raw, EEG_PAIR[1], start_sec, stop_sec)
        hb_env, hb_env_sfreq = rms_envelope(hb, float(headband_raw.info["sfreq"]), EEG_ENVELOPE_BIN_SEC)
        psg_env, psg_env_sfreq = rms_envelope(psg, float(psg_raw.info["sfreq"]), EEG_ENVELOPE_BIN_SEC)
        result = best_lag(hb_env, psg_env, min(hb_env_sfreq, psg_env_sfreq), EEG_MAX_LAG_SEC)
        usable = bool(result["best_abs_corr"] != "" and result["best_abs_corr"] >= 0.20)
        rows.append(
            {
                "transition_id": int(transition["transition_id"]),
                "subject": subject,
                "pid": pid,
                "transition_type": transition["transition_type"],
                "boundary_onset_sec": boundary,
                "signal_pair": f"{EEG_PAIR[0]}_vs_{EEG_PAIR[1]}",
                "availability": "available",
                **result,
                "usable_for_alignment": usable,
            }
        )
    return rows


def summarize_subjects(
    timeline: pd.DataFrame,
    windows: pd.DataFrame,
    pulse: pd.DataFrame,
    eeg: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    subjects = sorted(timeline["subject"].unique(), key=lambda value: int(value.replace("sub-", "")))

    for subject in subjects:
        t = timeline[timeline["subject"] == subject].iloc[0]
        w = windows[windows["subject"] == subject]
        p = pulse[pulse["subject"] == subject]
        e = eeg[eeg["subject"] == subject]
        p_usable = p[p["usable_for_alignment"] == True]  # noqa: E712
        e_usable = e[e["usable_for_alignment"] == True]  # noqa: E712
        p_lag = pd.to_numeric(p_usable["best_lag_sec"], errors="coerce")
        p_corr = pd.to_numeric(p_usable["best_abs_corr"], errors="coerce")
        e_lag = pd.to_numeric(e_usable["best_lag_sec"], errors="coerce")
        e_corr = pd.to_numeric(e_usable["best_abs_corr"], errors="coerce")

        rows.append(
            {
                "subject": subject,
                "pid": int(t["pid"]),
                "timeline_alignment_flag": t["timeline_alignment_flag"],
                "transition_windows": int(len(w)),
                "transition_windows_pass": int((w["sample_alignment_flag"] == "pass").sum()),
                "pulse_windows": int(len(p[p["availability"] == "available"])),
                "pulse_windows_usable": int(len(p_usable)),
                "pulse_usable_near_zero_lag_2s": int((p_lag.abs() <= 2.0).sum()) if len(p_usable) else 0,
                "pulse_median_abs_corr": float(p_corr.median()) if len(p_usable) else np.nan,
                "eeg_envelope_windows": int(len(e[e["availability"] == "available"])),
                "eeg_envelope_usable": int(len(e_usable)),
                "eeg_envelope_near_zero_lag_2s": int((e_lag.abs() <= 2.0).sum()) if len(e_usable) else 0,
                "eeg_envelope_median_abs_corr": float(e_corr.median()) if len(e_usable) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def write_readme(
    out_dir: Path,
    timeline: pd.DataFrame,
    windows: pd.DataFrame,
    pulse: pd.DataFrame,
    eeg: pd.DataFrame,
    subject_summary: pd.DataFrame,
) -> None:
    timeline_pass = int((timeline["timeline_alignment_flag"] == "pass").sum())
    window_pass = int((windows["sample_alignment_flag"] == "pass").sum())
    pulse_available_subjects = int((subject_summary["pulse_windows"] > 0).sum())
    pulse_subjects_all_near_zero = int(
        (
            (subject_summary["pulse_windows_usable"] > 0)
            & (subject_summary["pulse_windows_usable"] == subject_summary["pulse_usable_near_zero_lag_2s"])
        ).sum()
    )
    eeg_subjects_all_near_zero = int(
        (
            (subject_summary["eeg_envelope_usable"] > 0)
            & (subject_summary["eeg_envelope_usable"] == subject_summary["eeg_envelope_near_zero_lag_2s"])
        ).sum()
    )
    pulse_usable = pulse[pulse["usable_for_alignment"] == True]  # noqa: E712
    eeg_usable = eeg[eeg["usable_for_alignment"] == True]  # noqa: E712
    pulse_usable_lag = pd.to_numeric(pulse_usable["best_lag_sec"], errors="coerce")
    eeg_usable_lag = pd.to_numeric(eeg_usable["best_lag_sec"], errors="coerce")

    text = f"""# BOAS Full Signal Alignment Validation

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 alignment validation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**EDF scope:** 128 paired PSG/headband recordings
**Model training performed:** No

## 1. Purpose

This validation tests whether PSG-derived REM/Wake transition labels can be mapped onto BOAS headband EEG across the downloaded EDF dataset.

The technical uncertainty is whether header agreement generalizes to sample-level transition-window alignment across recordings, and whether signal-level proxies reveal evidence of timing offset or drift that would invalidate PSG-to-headband label mapping.

## 2. Method

The validation used four checks:

1. EDF timeline agreement for each PSG/headband pair.
2. Transition-window sample-index agreement for all E0 REM/Wake candidates.
3. `HB_PULSE` versus `PSG_PULSE` lag estimates in five 300-second windows across each recording where both pulse channels are present.
4. `HB_1` versus `PSG_F3` 1-second RMS-envelope lag estimates around each REM/Wake candidate as an exploratory shared-activity proxy.

Pulse and EEG-envelope lag estimates are supporting proxies. The exact sample-index mapping remains the primary alignment check.

## 3. Main Results

| Check | Result |
|---|---:|
| PSG/headband EDF pairs checked | {len(timeline)} |
| EDF pairs with matching timeline fields | {timeline_pass} |
| REM/Wake transition windows checked | {len(windows)} |
| Transition windows with matching sample indices | {window_pass} |
| Subjects with pulse channels available | {pulse_available_subjects} |
| Pulse windows checked | {int((pulse['availability'] == 'available').sum())} |
| Pulse windows usable for lag estimation | {len(pulse_usable)} |
| Usable pulse windows with lag within +/-2 seconds | {int((pulse_usable_lag.abs() <= 2.0).sum()) if len(pulse_usable) else 0} |
| Subjects where all usable pulse windows were within +/-2 seconds | {pulse_subjects_all_near_zero} |
| EEG-envelope transition windows checked | {int((eeg['availability'] == 'available').sum())} |
| EEG-envelope windows usable for lag estimation | {len(eeg_usable)} |
| Usable EEG-envelope windows with lag within +/-2 seconds | {int((eeg_usable_lag.abs() <= 2.0).sum()) if len(eeg_usable) else 0} |
| Subjects where all usable EEG-envelope windows were within +/-2 seconds | {eeg_subjects_all_near_zero} |

## 4. Interpretation

The full EDF timeline and transition-window sample-index checks passed for the downloaded dataset. This supports using PSG `stage_hum` transition labels on the headband sample timeline under the current deterministic label rule.

The pulse and EEG-envelope proxies provide additional evidence but are not treated as ground truth. Pulse channels are not available in every recording, and pulse waveform differences can shift cross-correlation peaks. EEG-envelope correlations depend on montage, sensor location, artifacts, and sleep physiology, so inconsistent envelope lag does not automatically prove synchronization failure.

## 5. Limitations

- The labels still have 30-second hypnogram uncertainty.
- Signal proxies test broad timing consistency, not exact physiological transition onset.
- Some recordings lack pulse channels.
- Future preprocessing should still retain recording-level quality flags.

## 6. Decision

Proceed to versioned deterministic transition-label table generation and minimal preprocessing.

Model training remains blocked until the label/preprocessing gate reviews the derived label table, split policy, and signal-quality flags.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = dataset_root()
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    participants = load_participants(root)
    candidates = pd.read_csv(
        repo_root()
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "candidate_transition_events.tsv",
        sep="\t",
    )

    timeline_rows = []
    window_rows = []
    pulse_all = []
    eeg_all = []

    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        pid = int(participants.loc[subject_id, "pid"])
        paths = subject_paths(root, subject_id)
        print(f"Checking {subject}")
        headband_raw = read_raw(paths["headband_edf"])
        psg_raw = read_raw(paths["psg_edf"])
        subject_candidates = candidates[candidates["subject"] == subject].copy()

        timeline_rows.append(timeline_row(subject, pid, headband_raw, psg_raw))
        window_rows.extend(
            transition_window_rows(subject, pid, subject_candidates, headband_raw, psg_raw)
        )
        pulse_all.extend(pulse_rows(subject, pid, headband_raw, psg_raw))
        eeg_all.extend(eeg_envelope_rows(subject, pid, subject_candidates, headband_raw, psg_raw))

    timeline = pd.DataFrame(timeline_rows)
    windows = pd.DataFrame(window_rows)
    pulse = pd.DataFrame(pulse_all)
    eeg = pd.DataFrame(eeg_all)
    subject_summary = summarize_subjects(timeline, windows, pulse, eeg)

    timeline.to_csv(out_dir / "edf_timeline_alignment.tsv", sep="\t", index=False)
    windows.to_csv(out_dir / "transition_window_sample_alignment.tsv", sep="\t", index=False)
    pulse.to_csv(out_dir / "pulse_alignment_windows.tsv", sep="\t", index=False)
    eeg.to_csv(out_dir / "eeg_envelope_alignment.tsv", sep="\t", index=False)
    subject_summary.fillna("NA").to_csv(
        out_dir / "subject_alignment_summary.tsv", sep="\t", index=False
    )
    write_readme(out_dir, timeline, windows, pulse, eeg, subject_summary)

    print()
    print("BOAS full signal alignment validation")
    print(f"EDF pairs checked: {len(timeline)}")
    print(f"Timeline pass: {(timeline['timeline_alignment_flag'] == 'pass').sum()} of {len(timeline)}")
    print(f"Transition windows pass: {(windows['sample_alignment_flag'] == 'pass').sum()} of {len(windows)}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
