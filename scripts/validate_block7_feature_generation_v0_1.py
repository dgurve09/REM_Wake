"""Validate train-only Block 7 PSG and wearable feature generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, resample_poly, sosfiltfilt, welch

from reviewed_output import verify_or_create_tsv


# Section 1: frozen configuration

VERSION = "v0.1"
DATASET = "ds005555"
SNAPSHOT = "1.1.1"
EXPERIMENT_DIR = "2026-09-06_block7_feature_generation_validation_v0.1"
DERIVED_DIR = "block7_feature_generation_validation_v0.1"
PROTOCOL_COMMIT = "1f6797f"
FEATURE_PLAN_COMMIT = "71ddc92"
INITIAL_IMPLEMENTATION_COMMIT = "6c38a64"
EXPECTED_TRAIN_RECORDINGS = 82
EXPECTED_TRAIN_PIDS = 64
INPUT_SFREQ = 256.0
OUTPUT_SFREQ = 128.0
SCALER_SAMPLE_STRIDE = 128
ROBUST_SCALE_FACTOR = 1.4826
EPOCH_SEC = 30.0
EPOCH_SAMPLES = int(EPOCH_SEC * OUTPUT_SFREQ)
VALID_STAGES = [0, 1, 2, 3, 4]
PSG6 = ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"]
PSG2 = ["PSG_F3", "PSG_F4"]
HB2 = ["HB_1", "HB_2"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, EPOCH_SEC)
PSG_PARITY_TOLERANCE = 1e-10
HB_REPRODUCTION_TOLERANCE = 1e-6


# Section 2: paths and frozen train assignment

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def dataset_root() -> Path:
    return data_parent() / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def train_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    train = split[split["partition"].eq("train")].copy()
    rows = []
    for item in train.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_TRAIN_RECORDINGS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_RECORDINGS} train recordings")
    if result["pid"].nunique() != EXPECTED_TRAIN_PIDS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_PIDS} train pid groups")
    if result["subject"].duplicated().any() or not result["partition"].eq("train").all():
        raise ValueError("Invalid frozen train assignment")
    return result.sort_values("subject", key=lambda values: values.map(subject_number))


def edf_path(subject: str, acquisition: str) -> Path:
    return (
        dataset_root()
        / subject
        / "eeg"
        / f"{subject}_task-Sleep_acq-{acquisition}_eeg.edf"
    )


def event_path(subject: str) -> Path:
    return dataset_root() / subject / "eeg" / f"{subject}_task-Sleep_acq-psg_events.tsv"


def reference_hb_path(subject: str) -> Path:
    return (
        data_parent()
        / "derived"
        / "stage_first_feature_baseline_v0.1"
        / "recording_features"
        / f"{subject}_features_v0.1.npz"
    )


def feature_path(subject: str, modality: str) -> Path:
    folder = modality.lower().replace("-", "")
    return derived_dir() / "recording_features" / folder / f"{subject}_features_v0.1.npz"


# Section 3: shared signal and spectral functions

def filter_sos() -> np.ndarray:
    return butter(4, [0.3, 35.0], btype="bandpass", fs=INPUT_SFREQ, output="sos")


def read_uv(subject: str, acquisition: str, channels: list[str]) -> np.ndarray:
    path = edf_path(subject, acquisition)
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if float(raw.info["sfreq"]) != INPUT_SFREQ:
        raise ValueError(f"{subject} {acquisition} sampling frequency is not 256 Hz")
    missing = [channel for channel in channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{subject} {acquisition} missing channels: {missing}")
    signal = raw.get_data(picks=channels) * 1e6
    if not np.isfinite(signal).all():
        raise ValueError(f"{subject} {acquisition} contains nonfinite samples")
    return signal


def filter_resample(signal_uv: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered = sosfiltfilt(sos, signal_uv, axis=1)
    resampled = resample_poly(filtered, up=1, down=2, axis=1)
    expected = int(np.ceil(signal_uv.shape[1] / 2.0))
    if resampled.shape[1] != expected or not np.isfinite(resampled).all():
        raise ValueError("Filter/resample output failed length or finite-value check")
    return resampled


def normalize(
    resampled_uv: np.ndarray,
    channels: list[str],
    scaler: dict[str, dict[str, float]],
) -> np.ndarray:
    result = resampled_uv.copy()
    for index, channel in enumerate(channels):
        center = float(scaler[channel]["median_uv"])
        scale = float(scaler[channel]["robust_scale_uv"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid robust scale for {channel}: {scale}")
        result[index] = (result[index] - center) / scale
    return result


def feature_names(channels: list[str]) -> list[str]:
    return [f"{channel}_{band}_log10_mean_psd" for channel in channels for band in BANDS]


def valid_events(subject: str) -> pd.DataFrame:
    events = pd.read_csv(
        event_path(subject),
        sep="\t",
        usecols=["onset", "duration", "stage_hum"],
    ).sort_values("onset")
    if not np.isclose(events["duration"].astype(float), EPOCH_SEC).all():
        raise ValueError(f"{subject} contains a non-30-second scoring epoch")
    return events[events["stage_hum"].isin(VALID_STAGES)].reset_index(drop=True)


def epoch_features(
    normalized: np.ndarray,
    channels: list[str],
    events: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    epochs = []
    onsets = []
    stages = []
    for item in events.itertuples(index=False):
        start = int(round(float(item.onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        if start < 0 or stop > normalized.shape[1]:
            continue
        epoch = normalized[:, start:stop]
        if epoch.shape[1] == EPOCH_SAMPLES and np.isfinite(epoch).all():
            epochs.append(epoch)
            onsets.append(float(item.onset))
            stages.append(int(item.stage_hum))
    if not epochs:
        raise ValueError("No complete valid epochs were retained")

    epoch_array = np.stack(epochs)
    frequencies, density = welch(
        epoch_array,
        fs=OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    columns = []
    for channel_index in range(len(channels)):
        for low, high in BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            power = density[:, channel_index, :][:, mask].mean(axis=1)
            columns.append(np.log10(np.maximum(power, np.finfo(float).eps)))
    features = np.column_stack(columns)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite spectral features were produced")
    return (
        np.asarray(onsets, dtype=np.float64),
        np.asarray(stages, dtype=np.int8),
        features,
        feature_names(channels),
    )


def context_onsets(onsets: np.ndarray) -> np.ndarray:
    retained = []
    for start in range(0, len(onsets) - len(CONTEXT_OFFSETS) + 1):
        local = onsets[start : start + len(CONTEXT_OFFSETS)]
        if np.allclose(np.diff(local), EPOCH_SEC, atol=1e-9, rtol=0.0):
            retained.append(float(local[4]))
    return np.asarray(retained, dtype=np.float64)


# Section 4: train-only scaler fitting

def fit_modality_scaler(
    assignments: pd.DataFrame,
    acquisition: str,
    channels: list[str],
    owner: str,
    sos: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    samples = {channel: [] for channel in channels}
    recording_rows = []
    for index, item in enumerate(assignments.itertuples(index=False), start=1):
        signal = read_uv(item.subject, acquisition, channels)
        resampled = filter_resample(signal, sos)
        for channel_index, channel in enumerate(channels):
            samples[channel].append(resampled[channel_index, ::SCALER_SAMPLE_STRIDE].copy())
        input_duration = signal.shape[1] / INPUT_SFREQ
        output_duration = resampled.shape[1] / OUTPUT_SFREQ
        duration_error = abs(input_duration - output_duration)
        recording_pass = duration_error <= 1.0 / OUTPUT_SFREQ
        recording_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": "train",
                "scaler_owner": owner,
                "acquisition": acquisition,
                "input_samples": int(signal.shape[1]),
                "output_samples": int(resampled.shape[1]),
                "duration_error_sec": duration_error,
                "finite_values": True,
                "recording_check": "pass" if recording_pass else "fail",
            }
        )
        print(f"Scaler {owner}: {item.subject} ({index}/{len(assignments)})")
        del signal, resampled

    scaler_rows = []
    for channel in channels:
        values = np.concatenate(samples[channel])
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scale = ROBUST_SCALE_FACTOR * mad
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid fitted robust scale for {channel}: {scale}")
        scaler_rows.append(
            {
                "scaler_owner": owner,
                "fit_partition": "train",
                "channel": channel,
                "subsample_frequency_hz": OUTPUT_SFREQ / SCALER_SAMPLE_STRIDE,
                "samples_used": int(len(values)),
                "median_uv": median,
                "mad_uv": mad,
                "robust_scale_uv": scale,
            }
        )
    return pd.DataFrame(scaler_rows), pd.DataFrame(recording_rows)


def add_hb_reference_check(scalers: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(
        repo_root()
        / "experiments/2026-07-15_minimal_preprocessing_v0.2/train_robust_scaler_v0.2.tsv",
        sep="\t",
    ).set_index("channel")
    result = scalers.copy()
    median_differences = []
    scale_differences = []
    matches = []
    reference_sources = []
    for item in result.itertuples(index=False):
        if item.channel in HB2:
            median_difference = abs(item.median_uv - float(reference.loc[item.channel, "median_uv"]))
            scale_difference = abs(
                item.robust_scale_uv - float(reference.loc[item.channel, "robust_scale_uv"])
            )
            reference_sources.append("minimal_preprocessing_v0.2")
            median_differences.append(median_difference)
            scale_differences.append(scale_difference)
            matches.append(median_difference <= 1e-12 and scale_difference <= 1e-12)
        else:
            reference_sources.append("not_applicable")
            median_differences.append(np.nan)
            scale_differences.append(np.nan)
            matches.append(True)
    result["reference_source"] = reference_sources
    result["reference_median_abs_diff"] = median_differences
    result["reference_scale_abs_diff"] = scale_differences
    result["reference_check_pass"] = matches
    return result


def scaler_map(scalers: pd.DataFrame, owner: str) -> dict[str, dict[str, float]]:
    selected = scalers[scalers["scaler_owner"].eq(owner)].set_index("channel")
    return selected[["median_uv", "robust_scale_uv"]].to_dict("index")


# Section 5: synthetic spectral validation

def synthetic_checks(sos: np.ndarray) -> pd.DataFrame:
    representatives = {
        "delta": 2.0,
        "theta": 6.0,
        "alpha": 10.0,
        "sigma": 14.0,
        "beta": 22.0,
    }
    band_names = list(BANDS)
    time = np.arange(int(EPOCH_SEC * INPUT_SFREQ)) / INPUT_SFREQ
    rows = []
    for intended, frequency in representatives.items():
        signal = np.sin(2.0 * np.pi * frequency * time)[None, :]
        resampled = filter_resample(signal, sos)
        events = pd.DataFrame({"onset": [0.0], "stage_hum": [0]})
        _, _, features, _ = epoch_features(resampled, ["synthetic"], events)
        powers = dict(zip(band_names, features[0]))
        intended_index = band_names.index(intended)
        nonadjacent = [
            band
            for index, band in enumerate(band_names)
            if abs(index - intended_index) > 1
        ]
        margin = min(powers[intended] - powers[band] for band in nonadjacent)
        rows.append(
            {
                "intended_band": intended,
                "frequency_hz": frequency,
                "intended_log10_power": powers[intended],
                "minimum_nonadjacent_margin": margin,
                "check_pass": margin > 0,
            }
        )
    return pd.DataFrame(rows)


# Section 6: immutable external feature arrays

def verify_or_create_npz(
    path: Path,
    onsets: np.ndarray,
    stages: np.ndarray,
    features: np.ndarray,
    names: list[str],
) -> None:
    expected_features = features.astype(np.float32)
    expected_names = np.asarray(names)
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            checks = [
                np.array_equal(values["onset"], onsets),
                np.array_equal(values["stage"], stages),
                np.array_equal(values["features"], expected_features),
                np.array_equal(values["feature_names"], expected_names),
            ]
        if not all(checks):
            raise RuntimeError(f"External feature artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        onset=onsets,
        stage=stages,
        features=expected_features,
        feature_names=expected_names,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference_hb(subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    path = reference_hb_path(subject)
    with np.load(path, allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["stage"].astype(np.int8),
            values["features"].astype(np.float64),
            values["feature_names"].astype(str).tolist(),
        )


# Section 7: per-recording feature and context validation

def validate_recording_features(
    assignments: pd.DataFrame,
    scalers: pd.DataFrame,
    sos: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    psg_scaler = scaler_map(scalers, "PSG-6")
    hb_scaler = scaler_map(scalers, "HB-2")
    rows = []
    artifact_rows = []
    for index, item in enumerate(assignments.itertuples(index=False), start=1):
        events = valid_events(item.subject)

        psg_raw = read_uv(item.subject, "psg", PSG6)
        psg6_signal = normalize(filter_resample(psg_raw, sos), PSG6, psg_scaler)
        psg6_onset, psg6_stage, psg6_features, psg6_names = epoch_features(
            psg6_signal, PSG6, events
        )
        psg2_signal = normalize(filter_resample(psg_raw[:2], sos), PSG2, psg_scaler)
        psg2_onset, psg2_stage, psg2_features, psg2_names = epoch_features(
            psg2_signal, PSG2, events
        )
        del psg_raw, psg6_signal, psg2_signal

        hb_raw = read_uv(item.subject, "headband", HB2)
        hb_signal = normalize(filter_resample(hb_raw, sos), HB2, hb_scaler)
        hb_onset, hb_stage, hb_features, hb_names = epoch_features(hb_signal, HB2, events)
        del hb_raw, hb_signal

        reference_onset, reference_stage, reference_features, reference_names = (
            load_reference_hb(item.subject)
        )
        onset_match = (
            np.array_equal(psg6_onset, psg2_onset)
            and np.array_equal(psg6_onset, hb_onset)
            and np.array_equal(hb_onset, reference_onset)
        )
        stage_match = (
            np.array_equal(psg6_stage, psg2_stage)
            and np.array_equal(psg6_stage, hb_stage)
            and np.array_equal(hb_stage, reference_stage)
        )
        psg_overlap_difference = float(np.max(np.abs(psg6_features[:, :10] - psg2_features)))
        hb_difference = float(np.max(np.abs(hb_features - reference_features)))
        psg_parity = (
            psg6_names[:10] == psg2_names
            and psg_overlap_difference <= PSG_PARITY_TOLERANCE
        )
        hb_reproduction = (
            hb_names == reference_names and hb_difference <= HB_REPRODUCTION_TOLERANCE
        )

        contexts = {
            "PSG-6": context_onsets(psg6_onset),
            "PSG-2": context_onsets(psg2_onset),
            "HB-2": context_onsets(hb_onset),
        }
        context_match = np.array_equal(contexts["PSG-6"], contexts["PSG-2"]) and np.array_equal(
            contexts["PSG-6"], contexts["HB-2"]
        )
        all_finite = all(
            np.isfinite(array).all()
            for array in [psg6_features, psg2_features, hb_features]
        )

        modality_values = {
            "PSG-6": (psg6_onset, psg6_stage, psg6_features, psg6_names),
            "PSG-2": (psg2_onset, psg2_stage, psg2_features, psg2_names),
            "HB-2": (hb_onset, hb_stage, hb_features, hb_names),
        }
        hashes = {}
        for modality, values in modality_values.items():
            path = feature_path(item.subject, modality)
            verify_or_create_npz(path, *values)
            relative = path.relative_to(data_parent()).as_posix()
            hashes[modality] = sha256(path)
            artifact_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "partition": "train",
                    "modality": modality,
                    "relative_external_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": hashes[modality],
                }
            )

        recording_pass = all(
            [
                onset_match,
                stage_match,
                psg_parity,
                hb_reproduction,
                context_match,
                all_finite,
            ]
        )
        rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": "train",
                "base_epochs": len(psg6_onset),
                "context_rows": len(contexts["PSG-6"]),
                "psg6_input_features": len(PSG6) * len(BANDS) * len(CONTEXT_OFFSETS),
                "psg2_input_features": len(PSG2) * len(BANDS) * len(CONTEXT_OFFSETS),
                "hb2_input_features": len(HB2) * len(BANDS) * len(CONTEXT_OFFSETS),
                "onset_match": onset_match,
                "stage_match": stage_match,
                "psg_overlap_max_abs_diff": psg_overlap_difference,
                "psg_overlap_pass": psg_parity,
                "hb_reference_max_abs_diff": hb_difference,
                "hb_reproduction_pass": hb_reproduction,
                "context_onset_match": context_match,
                "all_features_finite": all_finite,
                "psg6_sha256": hashes["PSG-6"],
                "psg2_sha256": hashes["PSG-2"],
                "hb2_sha256": hashes["HB-2"],
                "recording_check": "pass" if recording_pass else "fail",
            }
        )
        print(f"Features: {item.subject} ({index}/{len(assignments)}) - {'pass' if recording_pass else 'fail'}")
    return pd.DataFrame(rows), pd.DataFrame(artifact_rows)


# Section 8: reviewed outputs and gate

def verify_or_create_text(path: Path, expected: str) -> None:
    normalized = expected.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != normalized:
            raise RuntimeError(f"Reviewed output changed; create a new version: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(normalized, encoding="utf-8")


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "mne": mne.__version__,
    }


def gate_checks(
    assignments: pd.DataFrame,
    signal_checks: pd.DataFrame,
    scalers: pd.DataFrame,
    synthetic: pd.DataFrame,
    recordings: pd.DataFrame,
    artifacts: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("train_assignment_only", len(assignments) == 82 and assignments["partition"].eq("train").all()),
        ("signal_path_all_train_recordings", len(signal_checks) == 164 and signal_checks["recording_check"].eq("pass").all()),
        ("scalers_finite_positive", len(scalers) == 8 and np.isfinite(scalers["robust_scale_uv"]).all() and scalers["robust_scale_uv"].gt(0).all()),
        ("wearable_scaler_reproduced", scalers["reference_check_pass"].all()),
        ("synthetic_spectral_checks", len(synthetic) == 5 and synthetic["check_pass"].all()),
        ("psg6_psg2_overlap_parity", recordings["psg_overlap_pass"].all()),
        ("wearable_feature_reproduction", recordings["hb_reproduction_pass"].all()),
        ("context_onsets_match", recordings["context_onset_match"].all()),
        ("context_feature_dimensions", recordings["psg6_input_features"].eq(240).all() and recordings["psg2_input_features"].eq(80).all() and recordings["hb2_input_features"].eq(80).all()),
        ("all_features_finite", recordings["all_features_finite"].all()),
        ("all_train_recordings_pass", len(recordings) == 82 and recordings["recording_check"].eq("pass").all()),
        ("external_feature_hashes_recorded", len(artifacts) == 246 and artifacts["sha256"].str.len().eq(64).all()),
        ("validation_test_access", True),
    ]
    rows = []
    for name, passed in checks:
        detail = "train_only_no_validation_or_test_paths" if name == "validation_test_access" else "prespecified_check"
        rows.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})
    return pd.DataFrame(rows)


def write_reviewed_outputs(
    result_code_commit: str,
    signal_checks: pd.DataFrame,
    scalers: pd.DataFrame,
    synthetic: pd.DataFrame,
    recordings: pd.DataFrame,
    artifacts: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    out = output_dir()
    verify_or_create_tsv(signal_checks, out / "train_recording_signal_checks_v0.1.tsv")
    verify_or_create_tsv(scalers, out / "train_robust_scalers_v0.1.tsv")
    verify_or_create_tsv(synthetic, out / "synthetic_spectral_checks_v0.1.tsv")
    verify_or_create_tsv(recordings, out / "recording_feature_validation_v0.1.tsv")
    verify_or_create_tsv(artifacts, out / "external_feature_manifest_v0.1.tsv")
    verify_or_create_tsv(checks, out / "feature_gate_checks_v0.1.tsv")
    versions = json.dumps(software_versions(), indent=2, sort_keys=True) + "\n"
    verify_or_create_text(out / "software_versions_v0.1.json", versions)

    psg_max = recordings["psg_overlap_max_abs_diff"].max()
    hb_max = recordings["hb_reference_max_abs_diff"].max()
    readme = f"""# Block 7 Feature-Generation Validation v0.1

**Work date:** 2026-09-06
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Feature-plan commit:** `{FEATURE_PLAN_COMMIT}`
**Initial implementation commit:** `{INITIAL_IMPLEMENTATION_COMMIT}`
**Execution code commit:** `{result_code_commit}`
**Dataset:** BOAS OpenNeuro `{DATASET}`, snapshot `{SNAPSHOT}`
**Partition processed:** Train only
**Model training performed:** No
**Validation or test signals accessed:** No

## Result

| Check | Result |
|---|---:|
| Train `pid` groups | {train_assignments()['pid'].nunique()} |
| Train recordings | {len(recordings)} |
| Train signal-path checks | {int(signal_checks['recording_check'].eq('pass').sum())}/{len(signal_checks)} |
| Synthetic spectral checks | {int(synthetic['check_pass'].sum())}/{len(synthetic)} |
| PSG-6/PSG-2 overlap maximum absolute difference | {psg_max:.12g} |
| Wearable reproduction maximum absolute difference | {hb_max:.12g} |
| Recording feature/context checks | {int(recordings['recording_check'].eq('pass').sum())}/{len(recordings)} |
| Gate checks | {int(checks['status'].eq('pass').sum())}/{len(checks)} |
| Gate decision | **{'pass' if checks['status'].eq('pass').all() else 'fail'}** |

The gate tests whether the three Block 7 feature paths are mechanically comparable. It does not test event performance or device transfer. Full feature arrays remain outside Git; their relative paths and SHA-256 values are retained in the external manifest.

No validation or test recording, feature, label row, score, model, threshold, alarm, or metric was accessed.
"""
    verify_or_create_text(out / "README.md", readme)


# Section 9: command entry point

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-code-commit",
        required=True,
        help="Committed code hash used for this reviewed run",
    )
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A valid committed code hash is required")
    if not dataset_root().exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root()}")

    assignments = train_assignments()
    sos = filter_sos()
    psg_scaler, psg_signal_checks = fit_modality_scaler(
        assignments, "psg", PSG6, "PSG-6", sos
    )
    hb_scaler, hb_signal_checks = fit_modality_scaler(
        assignments, "headband", HB2, "HB-2", sos
    )
    scalers = add_hb_reference_check(pd.concat([psg_scaler, hb_scaler], ignore_index=True))
    signal_checks = pd.concat([psg_signal_checks, hb_signal_checks], ignore_index=True)
    synthetic = synthetic_checks(sos)
    recordings, artifacts = validate_recording_features(assignments, scalers, sos)
    checks = gate_checks(
        assignments,
        signal_checks,
        scalers,
        synthetic,
        recordings,
        artifacts,
    )
    write_reviewed_outputs(
        args.result_code_commit,
        signal_checks,
        scalers,
        synthetic,
        recordings,
        artifacts,
        checks,
    )
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("Block 7 feature-generation gate failed")
    print("Block 7 feature-generation gate passed")


if __name__ == "__main__":
    main()
