"""Assess BOAS headband EEG quality in reviewed transition/background windows.

This is a signal-quality feasibility assessment. It does not preprocess data,
assign participant splits, train a model, or estimate model performance.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
CHANNELS = ["HB_1", "HB_2"]
EXPECTED_SFREQ_HZ = 256.0
EXPECTED_WINDOW_SEC = 240.0
UNCHANGED_TOLERANCE_UV = 0.01
CRITICAL_FLATLINE_SEC = 5.0
CRITICAL_LOW_ROBUST_RANGE_UV = 1.0
REVIEW_HIGH_ROBUST_RANGE_UV = 1000.0
REVIEW_HIGH_PEAK_TO_PEAK_UV = 5000.0
REVIEW_OUTLIER_FRACTION = 0.01
ABRUPT_JUMP_UV = 500.0
REVIEW_ABRUPT_JUMP_FRACTION = 0.001
REVIEW_ENDPOINT_FRACTION = 0.01


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-07-11_boas_headband_window_quality"


def headband_edf(root: Path, subject: str) -> Path:
    return root / subject / "eeg" / f"{subject}_task-Sleep_acq-headband_eeg.edf"


def read_inputs(root: Path) -> pd.DataFrame:
    quality_root = root / "labels" / "signal_quality_flags_v0.1"
    transitions = pd.read_csv(
        quality_root / "transition_window_quality_flags_v0.1.tsv", sep="\t"
    )
    backgrounds = pd.read_csv(
        quality_root / "background_window_quality_flags_v0.1.tsv", sep="\t"
    )

    transition_windows = transitions[
        [
            "transition_id",
            "subject",
            "participant_id",
            "pid",
            "transition_type",
            "is_primary_label",
            "window_start_sample",
            "window_stop_sample",
            "window_decision",
        ]
    ].copy()
    transition_windows.insert(0, "window_source", "transition")
    transition_windows.insert(
        1, "window_id", transition_windows["transition_id"].map(lambda value: f"T{int(value):04d}")
    )
    transition_windows["label_class"] = transition_windows["transition_type"]
    transition_windows["primary_target"] = transition_windows["is_primary_label"].astype(str)

    background_windows = backgrounds[
        [
            "background_review_id",
            "subject",
            "participant_id",
            "pid",
            "window_start_sample",
            "window_stop_sample",
            "window_decision",
        ]
    ].copy()
    background_windows.insert(0, "window_source", "background_review")
    background_windows.insert(
        1,
        "window_id",
        background_windows["background_review_id"].map(lambda value: f"B{int(value):05d}"),
    )
    background_windows["label_class"] = "background_review"
    background_windows["primary_target"] = "False"

    common = [
        "window_source",
        "window_id",
        "subject",
        "participant_id",
        "pid",
        "label_class",
        "primary_target",
        "window_start_sample",
        "window_stop_sample",
        "window_decision",
    ]
    return pd.concat(
        [transition_windows[common], background_windows[common]], ignore_index=True
    ).sort_values(["subject", "window_start_sample", "window_source"])


def longest_true_run(values: np.ndarray) -> int:
    if values.size == 0 or not values.any():
        return 0
    padded = np.concatenate(([False], values, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(np.max(stops - starts))


def flag_text(flags: list[str]) -> str:
    return "pass" if not flags else ";".join(flags)


def channel_metrics(
    row: pd.Series, channel: str, data_volts: np.ndarray, sfreq: float
) -> dict:
    data_uv = np.asarray(data_volts, dtype=np.float64) * 1e6
    finite_mask = np.isfinite(data_uv)
    finite = data_uv[finite_mask]
    expected_samples = int(row["window_stop_sample"] - row["window_start_sample"])
    critical: list[str] = []
    review: list[str] = []

    if data_uv.size != expected_samples:
        critical.append("sample_count_mismatch")
    if data_uv.size == 0:
        critical.append("empty_window")
    if finite.size != data_uv.size:
        critical.append("nonfinite_samples")

    metrics = {
        "actual_samples": int(data_uv.size),
        "expected_samples": expected_samples,
        "finite_fraction": float(finite_mask.mean()) if data_uv.size else 0.0,
        "min_uv": np.nan,
        "p01_uv": np.nan,
        "median_uv": np.nan,
        "p99_uv": np.nan,
        "max_uv": np.nan,
        "robust_range_uv": np.nan,
        "peak_to_peak_uv": np.nan,
        "std_uv": np.nan,
        "mad_uv": np.nan,
        "outlier_fraction_10mad": np.nan,
        "longest_unchanged_sec": np.nan,
        "abrupt_jump_fraction": np.nan,
        "endpoint_fraction": np.nan,
    }

    if finite.size:
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        median = float(np.median(finite))
        p01 = float(np.percentile(finite, 1))
        p99 = float(np.percentile(finite, 99))
        peak_to_peak = maximum - minimum
        robust_range = p99 - p01
        mad = float(np.median(np.abs(finite - median)))
        if mad > 0:
            outlier_fraction = float(np.mean(np.abs(finite - median) > 10.0 * mad))
        else:
            outlier_fraction = 0.0

        differences = np.abs(np.diff(data_uv))
        finite_differences = differences[np.isfinite(differences)]
        unchanged_run = longest_true_run(
            finite_differences <= UNCHANGED_TOLERANCE_UV
        )
        longest_unchanged_sec = (
            (unchanged_run + 1) / sfreq if unchanged_run > 0 else 0.0
        )
        abrupt_fraction = (
            float(np.mean(finite_differences > ABRUPT_JUMP_UV))
            if finite_differences.size
            else 0.0
        )
        endpoint_fraction = float(np.mean((finite == minimum) | (finite == maximum)))

        metrics.update(
            {
                "min_uv": minimum,
                "p01_uv": p01,
                "median_uv": median,
                "p99_uv": p99,
                "max_uv": maximum,
                "robust_range_uv": robust_range,
                "peak_to_peak_uv": peak_to_peak,
                "std_uv": float(np.std(finite)),
                "mad_uv": mad,
                "outlier_fraction_10mad": outlier_fraction,
                "longest_unchanged_sec": longest_unchanged_sec,
                "abrupt_jump_fraction": abrupt_fraction,
                "endpoint_fraction": endpoint_fraction,
            }
        )

        if peak_to_peak == 0.0:
            critical.append("constant_signal")
        if robust_range < CRITICAL_LOW_ROBUST_RANGE_UV:
            critical.append("near_flat_robust_range")
        if longest_unchanged_sec >= CRITICAL_FLATLINE_SEC:
            critical.append("flatline_at_least_5_sec")
        if robust_range > REVIEW_HIGH_ROBUST_RANGE_UV:
            review.append("high_robust_range")
        if peak_to_peak > REVIEW_HIGH_PEAK_TO_PEAK_UV:
            review.append("high_peak_to_peak")
        if outlier_fraction > REVIEW_OUTLIER_FRACTION:
            review.append("many_10mad_outliers")
        if abrupt_fraction > REVIEW_ABRUPT_JUMP_FRACTION:
            review.append("many_abrupt_jumps")
        if endpoint_fraction > REVIEW_ENDPOINT_FRACTION:
            review.append("repeated_endpoints")

    decision = "exclude" if critical else ("review" if review else "include")
    return {
        "window_source": row["window_source"],
        "window_id": row["window_id"],
        "subject": row["subject"],
        "participant_id": row["participant_id"],
        "pid": int(row["pid"]),
        "label_class": row["label_class"],
        "primary_target": row["primary_target"],
        "channel": channel,
        "sfreq_hz": sfreq,
        "window_start_sample": int(row["window_start_sample"]),
        "window_stop_sample": int(row["window_stop_sample"]),
        **metrics,
        "critical_flags": flag_text(critical),
        "review_flags": flag_text(review),
        "channel_decision": decision,
    }


def assess_windows(windows: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    subjects = sorted(
        windows["subject"].unique(), key=lambda value: int(value.replace("sub-", ""))
    )
    for index, subject in enumerate(subjects, start=1):
        path = headband_edf(root, subject)
        if not path.exists():
            raise FileNotFoundError(f"Missing headband EDF: {path}")
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        missing_channels = [channel for channel in CHANNELS if channel not in raw.ch_names]
        if missing_channels:
            raise ValueError(f"{subject} missing channels: {missing_channels}")
        sfreq = float(raw.info["sfreq"])
        if sfreq != EXPECTED_SFREQ_HZ:
            raise ValueError(f"{subject} sampling frequency is {sfreq}, expected 256 Hz")

        subject_windows = windows[windows["subject"] == subject]
        for _, window in subject_windows.iterrows():
            start = int(window["window_start_sample"])
            stop = int(window["window_stop_sample"])
            data = raw.get_data(picks=CHANNELS, start=start, stop=stop)
            for channel_index, channel in enumerate(CHANNELS):
                rows.append(channel_metrics(window, channel, data[channel_index], sfreq))
        print(f"Checked {subject} ({index}/{len(subjects)})")
    return pd.DataFrame(rows)


def build_window_decisions(channel_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "window_source",
        "window_id",
        "subject",
        "participant_id",
        "pid",
        "label_class",
        "primary_target",
    ]
    for keys, group in channel_table.groupby(group_columns, sort=False):
        decisions = set(group["channel_decision"])
        critical_flags = sorted(
            {
                flag
                for text in group.loc[group["critical_flags"] != "pass", "critical_flags"]
                for flag in text.split(";")
            }
        )
        review_flags = sorted(
            {
                flag
                for text in group.loc[group["review_flags"] != "pass", "review_flags"]
                for flag in text.split(";")
            }
        )
        if "exclude" in decisions:
            decision = "exclude"
        elif "review" in decisions:
            decision = "review"
        else:
            decision = "include"

        targeted_review_flags = [
            flag for flag in review_flags if flag != "many_10mad_outliers"
        ]
        if decision == "exclude":
            review_priority = "critical_exclusion"
        elif targeted_review_flags:
            review_priority = "targeted_review"
        elif review_flags == ["many_10mad_outliers"]:
            review_priority = "mad_only_review"
        else:
            review_priority = "no_review_flag"
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "channels_checked": int(len(group)),
                "include_channels": int((group["channel_decision"] == "include").sum()),
                "review_channels": int((group["channel_decision"] == "review").sum()),
                "exclude_channels": int((group["channel_decision"] == "exclude").sum()),
                "critical_flags": ";".join(critical_flags) or "pass",
                "review_flags": ";".join(review_flags) or "pass",
                "window_signal_decision": decision,
                "review_priority": review_priority,
            }
        )
    return pd.DataFrame(rows).sort_values(["window_source", "subject", "window_id"])


def build_recording_summary(windows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        windows.groupby(["subject", "participant_id", "pid", "window_source", "window_signal_decision"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ["include", "review", "exclude"]:
        if column not in summary:
            summary[column] = 0
    summary.columns.name = None
    return summary.sort_values(["subject", "window_source"])


def build_aggregate_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby(
            ["window_source", "label_class", "window_signal_decision", "review_priority"]
        )
        .size()
        .rename("windows")
        .reset_index()
        .sort_values(["window_source", "window_signal_decision"])
    )


def write_readme(destination: Path, windows: pd.DataFrame, channels: pd.DataFrame) -> None:
    counts = windows.groupby(["window_source", "window_signal_decision"]).size().to_dict()
    value = lambda source, decision: int(counts.get((source, decision), 0))
    critical_windows = int((windows["window_signal_decision"] == "exclude").sum())
    review_windows = int((windows["window_signal_decision"] == "review").sum())
    reviewed_pid = int(
        windows.loc[windows["window_signal_decision"] != "include", "pid"].nunique()
    )
    mad_only_windows = int((windows["review_priority"] == "mad_only_review").sum())
    targeted_review_windows = int((windows["review_priority"] == "targeted_review").sum())
    primary = windows[windows["primary_target"].astype(str).str.lower() == "true"]
    primary_excluded = int((primary["window_signal_decision"] == "exclude").sum())
    primary_retained = int((primary["window_signal_decision"] != "exclude").sum())
    primary_pid_retained = int(
        primary.loc[primary["window_signal_decision"] != "exclude", "pid"].nunique()
    )
    decision = (
        "The critical failure rate requires revision of the preprocessing candidate set."
        if critical_windows
        else "No critical signal failure was found under the predeclared rules; retain all windows, with review flags preserved."
    )

    text = f"""# BOAS Headband Window Signal-Quality Assessment

**Work date:** 2026-07-11
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Protocol:** `docs/feasibility/headband_window_signal_quality_protocol_v0.1.md`
**Model training performed:** No

## 1. Question

Do the reviewed REM/Wake transition and background windows contain gross `HB_1` or `HB_2` amplitude failures that change the feasibility decision or require exclusion before grouped split design?

## 2. Result

| Window source | Include | Review | Exclude |
|---|---:|---:|---:|
| Transition | {value('transition', 'include')} | {value('transition', 'review')} | {value('transition', 'exclude')} |
| Background review | {value('background_review', 'include')} | {value('background_review', 'review')} | {value('background_review', 'exclude')} |

- Channel-window measurements: {len(channels):,}
- Window decisions: {len(windows):,}
- Windows with review flags: {review_windows:,}
- Windows with critical exclusion flags: {critical_windows:,}
- Unique `pid` values with a review or exclusion: {reviewed_pid:,}
- Windows flagged only by the nonspecific 10-MAD rule: {mad_only_windows:,}
- Windows with a targeted amplitude, jump, or endpoint review flag: {targeted_review_windows:,}
- Primary REM-to-Wake windows retained after critical exclusions: {primary_retained:,} of {len(primary):,}
- Primary REM-to-Wake windows critically excluded: {primary_excluded:,}
- Unique `pid` values retaining at least one primary REM-to-Wake window: {primary_pid_retained:,}

## 3. Decision

{decision}

Review flags are not silently converted into exclusions. They remain available for sensitivity analysis and targeted visual inspection during the label/preprocessing gate.

The predeclared 10-MAD rule proved too nonspecific as a stand-alone review criterion because raw EEG is heavy-tailed and the rule reached all participant groups. Its result is preserved as `mad_only_review`, but it is not used as an exclusion. Targeted review is instead prioritized using amplitude range, peak-to-peak, abrupt-jump, and repeated-endpoint flags.

## 4. Outputs

| File | Purpose |
|---|---|
| `headband_channel_window_metrics_v0.1.tsv` | Per-channel amplitude and continuity measurements |
| `headband_window_signal_decisions_v0.1.tsv` | Combined two-channel decision for each reviewed window |
| `recording_window_signal_quality_summary_v0.1.tsv` | Counts by recording, source, and decision |
| `headband_window_signal_quality_summary_v0.1.tsv` | Aggregate decision counts |

## 5. Limitations

- Operational thresholds are conservative engineering screening rules, not validated clinical EEG-quality criteria.
- Endpoint repetition is only a clipping proxy because the physical device saturation limits are not independently validated here.
- A review flag does not prove that a window is unusable; later sensitivity analysis must test whether conclusions change when reviewed windows are excluded.
- This assessment does not evaluate model performance.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    data = dataset_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    windows = read_inputs(root)
    expected_rows = 476 + 4302
    if len(windows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} reviewed windows, found {len(windows)}")

    channel_table = assess_windows(windows, data)
    window_table = build_window_decisions(channel_table)
    recording_summary = build_recording_summary(window_table)
    aggregate_summary = build_aggregate_summary(window_table)

    channel_table.to_csv(
        destination / "headband_channel_window_metrics_v0.1.tsv", sep="\t", index=False
    )
    window_table.to_csv(
        destination / "headband_window_signal_decisions_v0.1.tsv", sep="\t", index=False
    )
    recording_summary.to_csv(
        destination / "recording_window_signal_quality_summary_v0.1.tsv", sep="\t", index=False
    )
    aggregate_summary.to_csv(
        destination / "headband_window_signal_quality_summary_v0.1.tsv", sep="\t", index=False
    )
    write_readme(destination, window_table, channel_table)

    print(aggregate_summary.to_string(index=False))
    print(f"Wrote signal-quality assessment to {destination}")


if __name__ == "__main__":
    main()
