"""Validate minimal wearable EEG preprocessing on train recordings only.

The script filters continuous train recordings, fits train-only robust scaling,
and writes compact checks. It does not save signal arrays or train a model.
"""

from __future__ import annotations

import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt, sosfreqz


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
PREPROCESSING_VERSION = "v0.1"
SPLIT_VERSION = "v0.1"
QUALITY_VERSION = "v0.2"
EXPERIMENT_DIR_NAME = "2026-07-15_minimal_preprocessing_v0.1"
SPECIFICATION_PATH = "docs/preprocessing/minimal_wearable_eeg_preprocessing_spec_v0.1.md"
EXPECTED_TRAIN_TRANSITIONS = 304
EXPECTED_TRAIN_BACKGROUNDS = 2761
CHANNELS = ["HB_1", "HB_2"]
INPUT_SFREQ_HZ = 256.0
OUTPUT_SFREQ_HZ = 128.0
LOW_CUT_HZ = 0.3
HIGH_CUT_HZ = 35.0
FILTER_ORDER = 4
WINDOW_SEC = 240.0
EXPECTED_INPUT_SAMPLES = int(WINDOW_SEC * INPUT_SFREQ_HZ)
EXPECTED_OUTPUT_SAMPLES = int(WINDOW_SEC * OUTPUT_SFREQ_HZ)
SCALER_SAMPLE_STRIDE = int(OUTPUT_SFREQ_HZ)
ROBUST_SCALE_FACTOR = 1.4826


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR_NAME


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def headband_path(data: Path, subject: str) -> Path:
    return data / subject / "eeg" / f"{subject}_task-Sleep_acq-headband_eeg.edf"


def build_filter() -> np.ndarray:
    return butter(
        FILTER_ORDER,
        [LOW_CUT_HZ, HIGH_CUT_HZ],
        btype="bandpass",
        fs=INPUT_SFREQ_HZ,
        output="sos",
    )


def load_train_assignments(root: Path) -> pd.DataFrame:
    path = root / "splits" / "grouped_pid_split_v0.1" / "pid_split_assignments_v0.1.tsv"
    assignments = read_tsv(path)
    train = assignments[assignments["partition"] == "train"].copy()
    if len(train) != 64 or int(train["recording_count"].sum()) != 82:
        raise ValueError("Frozen train partition does not contain 64 pid / 82 recordings")
    return train


def load_train_windows(root: Path, train_pid: set[int]) -> pd.DataFrame:
    quality = root / "labels" / f"signal_quality_flags_{QUALITY_VERSION}"
    transitions = read_tsv(
        quality / f"transition_window_quality_flags_{QUALITY_VERSION}.tsv"
    )
    backgrounds = read_tsv(
        quality / f"background_window_quality_flags_{QUALITY_VERSION}.tsv"
    )

    transitions = transitions[
        transitions["pid"].isin(train_pid)
        & (transitions["preprocessing_decision"] != "exclude_critical")
    ].copy()
    transitions["window_source"] = "transition"
    transitions["window_id"] = transitions["transition_id"].map(
        lambda value: f"T{int(value):04d}"
    )
    transitions["label_class"] = transitions["transition_type"]

    backgrounds = backgrounds[
        backgrounds["pid"].isin(train_pid)
        & (backgrounds["preprocessing_decision"] != "exclude_critical")
    ].copy()
    backgrounds["window_source"] = "background_review"
    backgrounds["window_id"] = backgrounds["background_review_id"].map(
        lambda value: f"B{int(value):05d}"
    )
    backgrounds["label_class"] = backgrounds["background_tier"]

    columns = [
        "window_source",
        "window_id",
        "subject",
        "participant_id",
        "pid",
        "label_class",
        "window_start_sample",
        "window_stop_sample",
        "preprocessing_decision",
    ]
    windows = pd.concat([transitions[columns], backgrounds[columns]], ignore_index=True)
    if len(transitions) != EXPECTED_TRAIN_TRANSITIONS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_TRANSITIONS} retained train transition windows, "
            f"found {len(transitions)}"
        )
    if len(backgrounds) != EXPECTED_TRAIN_BACKGROUNDS:
        raise ValueError(
            f"Expected {EXPECTED_TRAIN_BACKGROUNDS:,} retained train background windows, "
            f"found {len(backgrounds)}"
        )
    if not set(windows["pid"]).issubset(train_pid):
        raise ValueError("Non-train pid found in preprocessing windows")
    return windows.sort_values(["subject", "window_start_sample", "window_source"])


def signal_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_fraction": 0.0,
            "p01": np.nan,
            "median": np.nan,
            "p99": np.nan,
            "std": np.nan,
            "mad": np.nan,
            "peak_to_peak": np.nan,
        }
    median = float(np.median(finite))
    return {
        "finite_fraction": float(np.isfinite(values).mean()),
        "p01": float(np.percentile(finite, 1)),
        "median": median,
        "p99": float(np.percentile(finite, 99)),
        "std": float(np.std(finite)),
        "mad": float(np.median(np.abs(finite - median))),
        "peak_to_peak": float(np.ptp(finite)),
    }


def synthetic_checks(sos: np.ndarray) -> pd.DataFrame:
    duration_sec = 120.0
    time = np.arange(int(duration_sec * INPUT_SFREQ_HZ)) / INPUT_SFREQ_HZ
    rows = []
    for frequency, expected in [(0.05, "attenuate"), (10.0, "retain"), (50.0, "attenuate")]:
        signal = np.sin(2.0 * np.pi * frequency * time)
        filtered = sosfiltfilt(sos, signal)
        output = resample_poly(filtered, up=1, down=2)
        trim = int(10 * OUTPUT_SFREQ_HZ)
        output_center = output[trim:-trim]
        input_rms = float(np.sqrt(np.mean(signal**2)))
        output_rms = float(np.sqrt(np.mean(output_center**2)))
        gain = output_rms / input_rms
        passed = gain >= 0.90 if expected == "retain" else gain <= 0.10
        rows.append(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "frequency_hz": frequency,
                "expected_behavior": expected,
                "rms_gain": gain,
                "criterion": ">=0.90" if expected == "retain" else "<=0.10",
                "check_decision": "pass" if passed else "fail",
            }
        )
    return pd.DataFrame(rows)


def filter_response(sos: np.ndarray) -> pd.DataFrame:
    frequencies = np.array([0.05, 0.1, 0.3, 1.0, 10.0, 35.0, 50.0, 64.0])
    _, response = sosfreqz(sos, worN=frequencies, fs=INPUT_SFREQ_HZ)
    single_pass = np.abs(response)
    effective = single_pass**2
    return pd.DataFrame(
        {
            "preprocessing_version": PREPROCESSING_VERSION,
            "frequency_hz": frequencies,
            "single_pass_gain": single_pass,
            "forward_backward_gain": effective,
            "forward_backward_db": 20.0 * np.log10(np.maximum(effective, 1e-12)),
        }
    )


def process_train_recordings(
    data: Path,
    assignments: pd.DataFrame,
    windows: pd.DataFrame,
    sos: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[list[np.ndarray]]]:
    recording_rows = []
    window_rows = []
    scaler_samples: list[list[np.ndarray]] = [[], []]

    train_subjects = []
    for _, assignment in assignments.iterrows():
        for subject in str(assignment["subjects"]).split(";"):
            train_subjects.append((subject, int(assignment["pid"])))
    train_subjects.sort(key=lambda item: int(item[0].replace("sub-", "")))

    for index, (subject, pid) in enumerate(train_subjects, start=1):
        path = headband_path(data, subject)
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        missing = [channel for channel in CHANNELS if channel not in raw.ch_names]
        sfreq = float(raw.info["sfreq"])
        if missing:
            raise ValueError(f"{subject} missing required channels: {missing}")
        if sfreq != INPUT_SFREQ_HZ:
            raise ValueError(f"{subject} has sampling frequency {sfreq}")

        raw_uv = raw.get_data(picks=CHANNELS) * 1e6
        raw_finite_fraction = float(np.isfinite(raw_uv).mean())
        if raw_finite_fraction != 1.0:
            raise ValueError(f"{subject} contains nonfinite continuous EEG samples")

        filtered_uv = sosfiltfilt(sos, raw_uv, axis=1)
        resampled_uv = resample_poly(filtered_uv, up=1, down=2, axis=1)
        expected_output = int(np.ceil(raw_uv.shape[1] / 2.0))
        duration_error_sec = abs(
            raw_uv.shape[1] / INPUT_SFREQ_HZ
            - resampled_uv.shape[1] / OUTPUT_SFREQ_HZ
        )
        recording_pass = (
            resampled_uv.shape[1] == expected_output
            and np.isfinite(resampled_uv).all()
            and duration_error_sec <= 1.0 / OUTPUT_SFREQ_HZ
        )

        for channel_index in range(len(CHANNELS)):
            scaler_samples[channel_index].append(
                resampled_uv[channel_index, ::SCALER_SAMPLE_STRIDE].copy()
            )

        recording_rows.append(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "subject": subject,
                "pid": pid,
                "input_samples": int(raw_uv.shape[1]),
                "output_samples": int(resampled_uv.shape[1]),
                "input_duration_sec": raw_uv.shape[1] / INPUT_SFREQ_HZ,
                "output_duration_sec": resampled_uv.shape[1] / OUTPUT_SFREQ_HZ,
                "duration_error_sec": duration_error_sec,
                "raw_finite_fraction": raw_finite_fraction,
                "filtered_finite_fraction": float(np.isfinite(resampled_uv).mean()),
                "recording_check_decision": "pass" if recording_pass else "fail",
            }
        )

        subject_windows = windows[windows["subject"] == subject]
        for _, window in subject_windows.iterrows():
            input_start = int(window["window_start_sample"])
            input_stop = int(window["window_stop_sample"])
            if input_start % 2 or input_stop % 2:
                raise ValueError(f"{window['window_id']} is not aligned to 128 Hz")
            output_start = input_start // 2
            output_stop = input_stop // 2
            raw_window = raw_uv[:, input_start:input_stop]
            filtered_window = resampled_uv[:, output_start:output_stop]

            row = {
                "preprocessing_version": PREPROCESSING_VERSION,
                "split_version": SPLIT_VERSION,
                "quality_version": QUALITY_VERSION,
                "window_source": window["window_source"],
                "window_id": window["window_id"],
                "subject": subject,
                "participant_id": window["participant_id"],
                "pid": int(window["pid"]),
                "label_class": window["label_class"],
                "quality_decision": window["preprocessing_decision"],
                "input_samples_per_channel": int(raw_window.shape[1]),
                "output_samples_per_channel": int(filtered_window.shape[1]),
            }
            window_pass = (
                raw_window.shape[1] == EXPECTED_INPUT_SAMPLES
                and filtered_window.shape[1] == EXPECTED_OUTPUT_SAMPLES
                and np.isfinite(raw_window).all()
                and np.isfinite(filtered_window).all()
            )
            for channel_index, channel in enumerate(CHANNELS):
                raw_metrics = signal_summary(raw_window[channel_index])
                filtered_metrics = signal_summary(filtered_window[channel_index])
                for name, value in raw_metrics.items():
                    row[f"{channel}_raw_{name}"] = value
                for name, value in filtered_metrics.items():
                    row[f"{channel}_filtered_{name}"] = value
            row["window_check_decision"] = "pass" if window_pass else "fail"
            window_rows.append(row)

        print(f"Preprocessed {subject} ({index}/{len(train_subjects)})")
        del raw_uv, filtered_uv, resampled_uv

    return pd.DataFrame(recording_rows), pd.DataFrame(window_rows), scaler_samples


def fit_scaler(samples: list[list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    for channel, chunks in zip(CHANNELS, samples):
        values = np.concatenate(chunks)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = ROBUST_SCALE_FACTOR * mad
        if not np.isfinite(robust_scale) or robust_scale <= 0:
            raise ValueError(f"Invalid robust scale for {channel}: {robust_scale}")
        rows.append(
            {
                "preprocessing_version": PREPROCESSING_VERSION,
                "fit_partition": "train",
                "channel": channel,
                "subsample_frequency_hz": OUTPUT_SFREQ_HZ / SCALER_SAMPLE_STRIDE,
                "samples_used": int(len(values)),
                "median_uv": median,
                "mad_uv": mad,
                "robust_scale_uv": robust_scale,
            }
        )
    return pd.DataFrame(rows)


def add_normalized_metrics(windows: pd.DataFrame, scaler: pd.DataFrame) -> pd.DataFrame:
    windows = windows.copy()
    parameters = scaler.set_index("channel").to_dict("index")
    for channel in CHANNELS:
        center = parameters[channel]["median_uv"]
        scale = parameters[channel]["robust_scale_uv"]
        windows[f"{channel}_normalized_p01"] = (
            windows[f"{channel}_filtered_p01"] - center
        ) / scale
        windows[f"{channel}_normalized_median"] = (
            windows[f"{channel}_filtered_median"] - center
        ) / scale
        windows[f"{channel}_normalized_p99"] = (
            windows[f"{channel}_filtered_p99"] - center
        ) / scale
        windows[f"{channel}_normalized_std"] = windows[f"{channel}_filtered_std"] / scale
        windows[f"{channel}_normalized_mad"] = windows[f"{channel}_filtered_mad"] / scale

    normalized_columns = [column for column in windows if "_normalized_" in column]
    finite_normalized = np.isfinite(windows[normalized_columns].to_numpy()).all(axis=1)
    windows.loc[~finite_normalized, "window_check_decision"] = "fail"
    return windows


def build_window_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby(
            ["window_source", "label_class", "quality_decision", "window_check_decision"]
        )
        .size()
        .rename("windows")
        .reset_index()
        .sort_values(["window_source", "label_class", "quality_decision"])
    )


def write_readme(
    destination: Path,
    recordings: pd.DataFrame,
    windows: pd.DataFrame,
    scaler: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> None:
    all_checks_pass = (
        (recordings["recording_check_decision"] == "pass").all()
        and (windows["window_check_decision"] == "pass").all()
        and (synthetic["check_decision"] == "pass").all()
    )
    decision_text = (
        "Pass the mechanical preprocessing validation for the declared input artifact."
        if all_checks_pass
        else "Fail the mechanical preprocessing validation. Preserve the failed rows and revise the input-quality or preprocessing rule before the gate."
    )
    scaler_rows = []
    for _, row in scaler.iterrows():
        scaler_rows.append(
            f"| {row['channel']} | {row['samples_used']:,} | {row['median_uv']:.6f} | "
            f"{row['mad_uv']:.6f} | {row['robust_scale_uv']:.6f} |"
        )
    scaler_text = "\n".join(scaler_rows)
    text = f"""# Minimal Wearable EEG Preprocessing Validation {PREPROCESSING_VERSION}

**Work date:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `{SPECIFICATION_PATH}`
**Partition processed:** Train only
**Model training performed:** No

## 1. Result

- Train `pid` values processed: 64
- Train recordings processed: {len(recordings)}
- Retained transition windows processed: {(windows['window_source'] == 'transition').sum()}
- Retained background review windows processed: {(windows['window_source'] == 'background_review').sum()}
- Recording checks passed: {(recordings['recording_check_decision'] == 'pass').sum()} of {len(recordings)}
- Window checks passed: {(windows['window_check_decision'] == 'pass').sum()} of {len(windows)}
- Synthetic frequency checks passed: {(synthetic['check_decision'] == 'pass').sum()} of {len(synthetic)}
- Validation/test recordings read: 0

## 2. Train-Only Robust Scaling

| Channel | 1 Hz samples | Median, uV | MAD, uV | Robust scale, uV |
|---|---:|---:|---:|---:|
{scaler_text}

## 3. Interpretation

The continuous-recording filter, 256-to-128 Hz resampling, window mapping, and train-only robust scaling are mechanically reproducible for the retained train candidates if all checks above pass. This validation establishes preprocessing integrity only; it does not establish that the choices improve REM-to-Wake detection.

Targeted-review windows were processed to test pipeline stability but remain separately flagged. Their use in the primary analysis is not decided here.

## 4. Decision

{decision_text}

## 5. Outputs

| File | Purpose |
|---|---|
| `filter_response_{PREPROCESSING_VERSION}.tsv` | Measured single-pass and forward-backward filter response |
| `synthetic_frequency_checks_{PREPROCESSING_VERSION}.tsv` | Predeclared retain/attenuate checks |
| `train_robust_scaler_{PREPROCESSING_VERSION}.tsv` | Scaling fitted from train recordings only |
| `train_recording_preprocessing_checks_{PREPROCESSING_VERSION}.tsv` | Continuous-recording integrity checks |
| `train_window_preprocessing_checks_{PREPROCESSING_VERSION}.tsv` | Compact raw, filtered, and normalized window summaries |
| `train_window_preprocessing_summary_{PREPROCESSING_VERSION}.tsv` | Counts by window and quality class |

## 6. Decision Boundary

Validation and test signals remain untouched. Model training remains blocked until targeted-review treatment is frozen and the label/preprocessing gate decision is recorded.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    data = dataset_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    assignments = load_train_assignments(root)
    train_pid = set(assignments["pid"].astype(int))
    windows = load_train_windows(root, train_pid)
    sos = build_filter()
    synthetic = synthetic_checks(sos)
    response = filter_response(sos)

    recordings, window_checks, samples = process_train_recordings(
        data, assignments, windows, sos
    )
    scaler = fit_scaler(samples)
    window_checks = add_normalized_metrics(window_checks, scaler)
    window_summary = build_window_summary(window_checks)

    response.to_csv(
        destination / f"filter_response_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    synthetic.to_csv(
        destination / f"synthetic_frequency_checks_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    scaler.to_csv(
        destination / f"train_robust_scaler_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    recordings.to_csv(
        destination / f"train_recording_preprocessing_checks_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    window_checks.to_csv(
        destination / f"train_window_preprocessing_checks_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    window_summary.to_csv(
        destination / f"train_window_preprocessing_summary_{PREPROCESSING_VERSION}.tsv",
        sep="\t",
        index=False,
    )
    write_readme(destination, recordings, window_checks, scaler, synthetic)

    print(synthetic.to_string(index=False))
    print(scaler.to_string(index=False))
    print(window_summary.to_string(index=False))
    print(f"Wrote preprocessing validation to {destination}")


if __name__ == "__main__":
    main()
