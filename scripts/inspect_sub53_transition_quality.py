"""Inspect signal quality around BOAS sub-53 REM/Wake candidates.

This is a pilot quality check only. It does not train a model or create a
full-dataset transition inventory.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
WINDOW_SECONDS = 120

HEADBAND_CHANNELS = ["HB_1", "HB_2", "HB_PULSE"]
PSG_CHANNELS = [
    "PSG_F3",
    "PSG_F4",
    "PSG_C3",
    "PSG_C4",
    "PSG_O1",
    "PSG_O2",
    "PSG_EOG",
    "PSG_EMG",
]
STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    8: "PSG disconnection",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def read_raw(path: Path) -> mne.io.BaseRaw:
    return mne.io.read_raw_edf(path, preload=False, verbose="ERROR")


def quality_metrics(
    raw: mne.io.BaseRaw,
    channel: str,
    source: str,
    transition: pd.Series,
) -> dict:
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq
    boundary = float(transition["boundary_onset_sec"])
    window_start = max(0.0, boundary - WINDOW_SECONDS)
    window_end = min(duration, boundary + WINDOW_SECONDS)
    start_sample = int(round(window_start * sfreq))
    stop_sample = int(round(window_end * sfreq))
    expected_samples = stop_sample - start_sample

    data = raw.get_data(picks=[channel], start=start_sample, stop=stop_sample)[0]
    finite_mask = np.isfinite(data)
    finite = data[finite_mask]

    row = {
        "transition_id": int(transition["transition_id"]),
        "transition": transition["transition"],
        "boundary_onset_sec": boundary,
        "source": source,
        "channel": channel,
        "window_start_sec": window_start,
        "window_end_sec": window_end,
        "window_seconds": window_end - window_start,
        "sfreq_hz": sfreq,
        "expected_samples": expected_samples,
        "actual_samples": int(data.size),
        "finite_fraction": float(finite_mask.mean()) if data.size else 0.0,
        "missing_samples": int(data.size - finite_mask.sum()),
    }

    issues = []
    if data.size != expected_samples:
        issues.append("sample_count_mismatch")
    if data.size == 0:
        issues.append("empty_window")
    if finite.size == 0:
        issues.append("no_finite_data")
        row.update(empty_signal_metrics())
        row["quality_flag"] = ";".join(issues)
        return row

    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    p01 = float(np.percentile(finite, 1))
    p99 = float(np.percentile(finite, 99))
    p2p = float(np.ptp(finite))
    std = float(np.std(finite))

    if mad > 0:
        outlier_fraction = float(np.mean(np.abs(finite - median) > 10.0 * mad))
    else:
        outlier_fraction = 0.0

    if finite.size > 1:
        robust_range = max(p99 - p01, np.finfo(float).eps)
        diff = np.abs(np.diff(finite))
        low_diff_fraction = float(np.mean(diff <= robust_range * 1e-6))
    else:
        low_diff_fraction = 1.0

    row.update(
        {
            "min_value": float(np.min(finite)),
            "p01_value": p01,
            "median_value": median,
            "p99_value": p99,
            "max_value": float(np.max(finite)),
            "peak_to_peak": p2p,
            "std": std,
            "mad": mad,
            "outlier_fraction_10mad": outlier_fraction,
            "low_diff_fraction": low_diff_fraction,
        }
    )

    if row["finite_fraction"] < 1.0:
        issues.append("nonfinite_samples")
    if p2p <= 0.0:
        issues.append("flatline")
    if std <= 0.0:
        issues.append("zero_std")
    if outlier_fraction > 0.05:
        issues.append("many_extreme_points")

    row["quality_flag"] = "pass_basic" if not issues else ";".join(issues)
    return row


def empty_signal_metrics() -> dict:
    return {
        "min_value": np.nan,
        "p01_value": np.nan,
        "median_value": np.nan,
        "p99_value": np.nan,
        "max_value": np.nan,
        "peak_to_peak": np.nan,
        "std": np.nan,
        "mad": np.nan,
        "outlier_fraction_10mad": np.nan,
        "low_diff_fraction": np.nan,
    }


def stage_window_summary(events: pd.DataFrame, start_sec: float, end_sec: float) -> dict:
    window = events[(events["onset"] >= start_sec) & (events["onset"] < end_sec)]
    counts = window["stage_hum"].value_counts().sort_index()
    count_text = "; ".join(f"{int(stage)}={int(count)}" for stage, count in counts.items())
    sequence = "->".join(STAGE_NAMES.get(int(stage), str(stage)) for stage in window["stage_hum"])
    return {
        "stage_hum_epochs": int(len(window)),
        "stage_hum_counts": count_text,
        "stage_hum_sequence": sequence,
        "psg_disconnection_epochs": int((window["stage_hum"] == 8).sum()),
    }


def transition_summary(channel_quality: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for transition_id, group in channel_quality.groupby("transition_id"):
        first = group.iloc[0]
        window_start = float(first["window_start_sec"])
        window_end = float(first["window_end_sec"])
        hb_eeg = group[group["channel"].isin(["HB_1", "HB_2"])]
        psg_ref = group[group["channel"].isin(PSG_CHANNELS)]
        stage_info = stage_window_summary(events, window_start, window_end)

        issues = []
        if int((hb_eeg["quality_flag"] == "pass_basic").sum()) < 2:
            issues.append("headband_eeg_issue")
        if int((psg_ref["quality_flag"] == "pass_basic").sum()) < len(PSG_CHANNELS):
            issues.append("psg_reference_issue")
        if stage_info["psg_disconnection_epochs"] > 0:
            issues.append("psg_disconnection_in_window")

        rows.append(
            {
                "transition_id": int(transition_id),
                "transition": first["transition"],
                "boundary_onset_sec": first["boundary_onset_sec"],
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "headband_eeg_pass_channels": int((hb_eeg["quality_flag"] == "pass_basic").sum()),
                "headband_eeg_total_channels": int(len(hb_eeg)),
                "psg_reference_pass_channels": int((psg_ref["quality_flag"] == "pass_basic").sum()),
                "psg_reference_total_channels": int(len(psg_ref)),
                **stage_info,
                "pilot_window_decision": "pass_basic" if not issues else ";".join(issues),
            }
        )

    return pd.DataFrame(rows)


def channel_overview(channel_quality: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (source, channel), group in channel_quality.groupby(["source", "channel"]):
        flag_counts = group["quality_flag"].value_counts().sort_index()
        rows.append(
            {
                "source": source,
                "channel": channel,
                "windows_checked": int(len(group)),
                "pass_basic_windows": int((group["quality_flag"] == "pass_basic").sum()),
                "nonpass_windows": int((group["quality_flag"] != "pass_basic").sum()),
                "quality_flags": "; ".join(
                    f"{flag}={int(count)}" for flag, count in flag_counts.items()
                ),
                "min_finite_fraction": float(group["finite_fraction"].min()),
                "max_outlier_fraction_10mad": float(group["outlier_fraction_10mad"].max()),
                "min_peak_to_peak": float(group["peak_to_peak"].min()),
                "max_peak_to_peak": float(group["peak_to_peak"].max()),
            }
        )

    return pd.DataFrame(rows).sort_values(["source", "channel"])


def main() -> None:
    root = dataset_root()
    out_dir = (
        repo_root()
        / "experiments"
        / "2026-06-25_to_2026-06-28_boas_sub53_transition_quality"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    transition_path = (
        repo_root()
        / "experiments"
        / "2026-06-24_boas_sub53_pilot"
        / "sub53_stage_hum_transition_candidates.tsv"
    )
    transitions = pd.read_csv(transition_path, sep="\t")
    transitions.insert(0, "transition_id", range(1, len(transitions) + 1))

    headband_raw = read_raw(root / "sub-53/eeg/sub-53_task-Sleep_acq-headband_eeg.edf")
    psg_raw = read_raw(root / "sub-53/eeg/sub-53_task-Sleep_acq-psg_eeg.edf")
    psg_events = pd.read_csv(root / "sub-53/eeg/sub-53_task-Sleep_acq-psg_events.tsv", sep="\t")

    rows = []
    for _, transition in transitions.iterrows():
        for channel in HEADBAND_CHANNELS:
            rows.append(quality_metrics(headband_raw, channel, "headband", transition))
        for channel in PSG_CHANNELS:
            rows.append(quality_metrics(psg_raw, channel, "psg", transition))

    channel_quality = pd.DataFrame(rows)
    summary = transition_summary(channel_quality, psg_events)
    overview = channel_overview(channel_quality)

    channel_quality.to_csv(out_dir / "transition_channel_quality.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "transition_window_summary.tsv", sep="\t", index=False)
    overview.to_csv(out_dir / "channel_quality_overview.tsv", sep="\t", index=False)
    transitions.to_csv(out_dir / "input_transition_candidates.tsv", sep="\t", index=False)

    print("Transition-window summary")
    print(summary)
    print()
    print("Channel overview")
    print(overview)
    print()
    print("Channel quality flags")
    print(channel_quality["quality_flag"].value_counts())
    print()
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
