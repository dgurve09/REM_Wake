"""Run the frozen paired enriched-feature and pruning experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import warnings
from io import BytesIO
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import butter, csd, resample_poly, sosfiltfilt, welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import (
    evaluate_events,
    metric_values,
    optimal_matches,
)


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-10-04_paired_enriched_feature_pruning_nested_v0.1"
DERIVED_DIR = "paired_enriched_feature_pruning_nested_v0.1"
PROTOCOL_COMMIT = "7cc6253"
BASE_SEED = 20261004
OUTER_FOLDS = 5
EPOCH_SEC = 30.0
INPUT_SFREQ = 256.0
OUTPUT_SFREQ = 128.0
EPOCH_SAMPLES = int(EPOCH_SEC * OUTPUT_SFREQ)
CONTEXT_EPOCHS = 8
BASE_FEATURES_PER_EPOCH = 10
RICH_FEATURES_PER_EPOCH = 45
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 1e-12
BOOTSTRAP_RESAMPLES = 2000
MEANINGFUL_F1_GAIN = 0.05
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
PIPELINES = ["F0-L2", "F1-L2", "F1-EN"]
EN_C_VALUES = [0.01, 0.1, 1.0]
EPSILON = np.finfo(float).eps
PSG2 = ["PSG_F3", "PSG_F4"]
HB2 = ["HB_1", "HB_2"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}

MODALITIES = {
    "P2": {"acquisition": "psg", "channels": PSG2, "folder": "psg2", "scaler": "PSG-6"},
    "H2": {"acquisition": "headband", "channels": HB2, "folder": "hb2", "scaler": "HB-2"},
}


# Section 2: paths and immutable writers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return repo_root().parent / "REM_W_data"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def dataset_root() -> Path:
    return data_parent() / "boas_ds005555_v1.1.1"


def cache_path(subject: str, modality: str) -> Path:
    return derived_dir() / "recording_features" / modality.lower() / f"{subject}_features_v0.1.npz"


def block7_feature_path(subject: str, modality: str) -> Path:
    folder = MODALITIES[modality]["folder"]
    return (
        data_parent()
        / "derived/block7_feature_generation_validation_v0.1/recording_features"
        / folder
        / f"{subject}_features_v0.1.npz"
    )


def safe_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def model_path(modality: str, pipeline: str, outer: int, phase: str, c_value: float) -> Path:
    return (
        derived_dir()
        / "models"
        / modality.lower()
        / pipeline.lower().replace("-", "_")
        / f"outer_{outer}_{phase}_c{safe_name(c_value)}_v0.1.json.gz"
    )


def score_path(modality: str, pipeline: str, outer: int, phase: str, c_value: float) -> Path:
    return (
        derived_dir()
        / "scores"
        / modality.lower()
        / pipeline.lower().replace("-", "_")
        / f"outer_{outer}_{phase}_c{safe_name(c_value)}_v0.1.tsv.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_gzip(value: bytes) -> bytes:
    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(value)
    return output.getvalue()


def verify_or_create_gzip_tsv(path: Path, frame: pd.DataFrame) -> None:
    expected = deterministic_gzip(
        frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    )
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External score artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def verify_or_create_json_gzip(path: Path, payload: dict) -> None:
    expected = deterministic_gzip(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External model artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
    ).strip()


# Section 3: frozen membership and event helpers

def train_assignments() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in source[source["partition"].eq("train")].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    result = pd.DataFrame(rows)
    if len(result) != 82 or result["pid"].nunique() != 64 or result["subject"].duplicated().any():
        raise ValueError("Unexpected frozen train membership")
    return result.sort_values("subject", key=lambda values: values.str.replace("sub-", "").astype(int))

def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def time_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


def context_start_indices(onsets: np.ndarray) -> np.ndarray:
    if len(onsets) < CONTEXT_EPOCHS:
        return np.asarray([], dtype=int)
    contiguous = np.isclose(np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(CONTEXT_EPOCHS - 1, dtype=np.int8), mode="valid")
    return np.flatnonzero(counts == CONTEXT_EPOCHS - 1)


def frozen_fold_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    path = (
        repo_root()
        / "experiments/2026-09-12_block8_noise_augmented_training_v0.1"
        / "train_oof_fold_assignments_v0.1.tsv"
    )
    folds = pd.read_csv(path, sep="\t").sort_values(["fold", "pid"]).reset_index(drop=True)
    if len(folds) != 64 or folds["pid"].nunique() != 64:
        raise ValueError("Unexpected frozen fold membership")
    if set(folds["pid"].astype(int)) != set(assignments["pid"].astype(int)):
        raise ValueError("Frozen folds do not cover the train participants")
    return folds


def reference_events(assignments: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(
        repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    rows = membership[
        membership["subject"].isin(set(assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    if set(rows["partition"]) != {"train"}:
        raise ValueError("Unauthorized reference-event partition")
    return rows


def labeled_candidates(assignments: pd.DataFrame) -> pd.DataFrame:
    positive = reference_events(assignments)
    positive = positive[truth(positive["primary_analysis_eligible"])].copy()
    positive_rows = pd.DataFrame(
        {
            "sample_id": "transition_" + positive["transition_id"].astype(str),
            "subject": positive["subject"],
            "pid": positive["pid"].astype(int),
            "partition": positive["partition"],
            "candidate_time_sec": positive["nominal_boundary_sec"].astype(float),
            "label": 1,
            "source_tier": "REM_to_Wake",
        }
    )
    membership = pd.read_csv(
        repo_root()
        / "labels/quality_analysis_membership_v0.1/background_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    detail = pd.read_csv(
        repo_root() / "labels/background_windows_v0.1/background_review_windows_v0.1.tsv",
        sep="\t",
        usecols=["background_review_id", "center_sec"],
    )
    negative = membership[
        membership["subject"].isin(set(assignments["subject"]))
        & truth(membership["primary_analysis_eligible"])
    ].merge(detail, on="background_review_id", validate="one_to_one")
    negative_rows = pd.DataFrame(
        {
            "sample_id": "background_" + negative["background_review_id"].astype(str),
            "subject": negative["subject"],
            "pid": negative["pid"].astype(int),
            "partition": negative["partition"],
            "candidate_time_sec": negative["center_sec"].astype(float),
            "label": 0,
            "source_tier": negative["background_tier"],
        }
    )
    result = pd.concat([positive_rows, negative_rows], ignore_index=True)
    if result["sample_id"].duplicated().any() or set(result["partition"]) != {"train"}:
        raise ValueError("Invalid labeled candidate membership")
    return result.sort_values(["subject", "candidate_time_sec", "label"])


def local_event_inputs(
    references: pd.DataFrame, membership: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = (
        "primary_analysis_eligible"
        if membership == "primary"
        else "expanded_quality_analysis_eligible"
    )
    eligible = truth(references[column])
    columns = ["subject", "pid", "event_time_sec"]
    return references.loc[eligible, columns], references.loc[~eligible, columns]


def threshold_grid() -> np.ndarray:
    coarse = np.arange(0.01, 0.951, 0.05)
    logits = np.arange(3.0, 14.001, 0.25)
    return np.unique(np.concatenate([coarse, 1.0 / (1.0 + np.exp(-logits))]))


THRESHOLDS = threshold_grid()


def collapsed_times(group: pd.DataFrame, threshold: float) -> np.ndarray:
    times = group["candidate_time_sec"].to_numpy(dtype=float)
    probabilities = group["probability"].to_numpy(dtype=float)
    selected = np.flatnonzero(probabilities >= threshold)
    if len(selected) == 0:
        return np.asarray([], dtype=float)
    splits = np.flatnonzero(np.diff(times[selected]) > EPOCH_SEC + 1e-6) + 1
    result = []
    for run in np.split(selected, splits):
        local = probabilities[run]
        best = run[np.flatnonzero(np.isclose(local, local.max()))[0]]
        result.append(times[best])
    return np.asarray(result, dtype=float)


def fast_primary_summary(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    threshold: float,
) -> dict:
    eligible, ignored = local_event_inputs(references, "primary")
    score_groups = {
        subject: group.sort_values("candidate_time_sec")
        for subject, group in scores.groupby("subject")
    }
    eligible_groups = {
        subject: group["event_time_sec"].to_numpy(dtype=float)
        for subject, group in eligible.groupby("subject")
    }
    ignored_groups = {
        subject: group["event_time_sec"].to_numpy(dtype=float)
        for subject, group in ignored.groupby("subject")
    }
    tp = fp = fn = ignored_count = predicted = references_count = 0
    for item in support.itertuples(index=False):
        predictions = collapsed_times(score_groups[item.subject], threshold)
        refs = eligible_groups.get(item.subject, np.asarray([], dtype=float))
        ignored_refs = ignored_groups.get(item.subject, np.asarray([], dtype=float))
        matches = optimal_matches(refs, predictions, 15.0)
        matched_predictions = {value[1] for value in matches}
        unmatched = [index for index in range(len(predictions)) if index not in matched_predictions]
        ignored_matches = optimal_matches(ignored_refs, predictions[unmatched], 15.0)
        tp += len(matches)
        fn += len(refs) - len(matches)
        fp += len(unmatched) - len(ignored_matches)
        ignored_count += len(ignored_matches)
        predicted += len(predictions)
        references_count += len(refs)
    hours = float(support["supported_hours"].sum())
    return {
        "recordings": len(support),
        "pid": support["pid"].nunique(),
        "reference_events": references_count,
        "predicted_events": predicted,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "ignored_predictions": ignored_count,
        "supported_hours": hours,
        **metric_values(tp, fp, fn, hours),
    }


# Section 4: feature extraction

def edf_path(subject: str, acquisition: str) -> Path:
    return dataset_root() / subject / "eeg" / f"{subject}_task-Sleep_acq-{acquisition}_eeg.edf"


def read_uv(subject: str, acquisition: str, channels: list[str]) -> np.ndarray:
    edf = edfio.read_edf(edf_path(subject, acquisition), lazy_load_data=True)
    if any(channel not in edf.labels for channel in channels):
        raise ValueError(f"{subject} is missing a required {acquisition} channel")
    signals = []
    for channel in channels:
        signal = edf.get_signal(channel)
        if float(signal.sampling_frequency) != INPUT_SFREQ:
            raise ValueError(f"{subject} {channel} sampling frequency is not 256 Hz")
        if str(signal.physical_dimension).lower() not in {"uv", "µv"}:
            raise ValueError(f"{subject} {channel} unexpected unit: {signal.physical_dimension}")
        signals.append(np.asarray(signal.data, dtype=np.float64))
    result = np.stack(signals)
    if not np.isfinite(result).all():
        raise ValueError(f"{subject} {acquisition} contains nonfinite samples")
    return result


def filter_sos() -> np.ndarray:
    return butter(4, [0.3, 35.0], btype="bandpass", fs=INPUT_SFREQ, output="sos")


def filter_resample(signal: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered = sosfiltfilt(sos, signal, axis=1)
    result = resample_poly(filtered, up=1, down=2, axis=1)
    expected = int(np.ceil(signal.shape[1] / 2.0))
    if result.shape[1] != expected or not np.isfinite(result).all():
        raise ValueError("Filter/resample output failed validation")
    return result


def normalize(
    signal: np.ndarray,
    channels: list[str],
    scaler: dict[str, dict[str, float]],
) -> np.ndarray:
    result = signal.copy()
    for index, channel in enumerate(channels):
        center = float(scaler[channel]["median_uv"])
        scale = float(scaler[channel]["robust_scale_uv"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid robust scale for {channel}")
        result[index] = (result[index] - center) / scale
    return result

def frozen_scalers() -> dict[str, dict[str, dict[str, float]]]:
    table = pd.read_csv(
        repo_root()
        / "experiments/2026-09-06_block7_feature_generation_validation_v0.1"
        / "train_robust_scalers_v0.1.tsv",
        sep="\t",
    )
    result = {}
    for modality, config in MODALITIES.items():
        local = table[table["scaler_owner"].eq(config["scaler"])].set_index("channel")
        result[modality] = {
            channel: {
                "median_uv": float(local.loc[channel, "median_uv"]),
                "robust_scale_uv": float(local.loc[channel, "robust_scale_uv"]),
            }
            for channel in config["channels"]
        }
    return result


def channel_feature_names(channel: str) -> list[str]:
    names = [f"{channel}_{band}_log10_mean_psd" for band in BANDS]
    names += [f"{channel}_{band}_relative_power" for band in BANDS]
    names += [
        f"{channel}_spectral_entropy",
        f"{channel}_spectral_edge_95_hz",
        f"{channel}_log10_rms",
        f"{channel}_log10_mean_line_length",
        f"{channel}_zero_crossing_rate",
        f"{channel}_hjorth_mobility",
        f"{channel}_hjorth_complexity",
    ]
    return names


def rich_feature_names(channels: list[str]) -> list[str]:
    names = []
    for channel in channels:
        names.extend(channel_feature_names(channel))
    names.append(f"{channels[0]}_{channels[1]}_correlation")
    names += [f"{channels[0]}_{channels[1]}_{band}_coherence" for band in BANDS]
    names += [f"{channels[0]}_minus_{channels[1]}_{band}_log_power" for band in BANDS]
    if len(names) != RICH_FEATURES_PER_EPOCH:
        raise ValueError("Enriched feature schema length changed")
    return names


def base_indices() -> np.ndarray:
    return np.asarray(list(range(5)) + list(range(17, 22)), dtype=int)


def epoch_blocks(signal: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    blocks = []
    for onset in onsets:
        start = int(round(float(onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        block = signal[:, start:stop]
        if block.shape != (2, EPOCH_SAMPLES):
            raise ValueError(f"Incomplete epoch at {onset:g} seconds")
        blocks.append(block)
    result = np.stack(blocks)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite epoch samples")
    return result


def calculate_rich_features(blocks: np.ndarray, channels: list[str]) -> np.ndarray:
    frequencies, density = welch(
        blocks,
        fs=OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    spectral_mask = (frequencies >= 0.5) & (frequencies < 30.0)
    spectral_frequencies = frequencies[spectral_mask]
    columns = []
    channel_log_power = []
    for channel_index in range(2):
        local_density = density[:, channel_index, :]
        band_power = []
        for low, high in BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            band_power.append(local_density[:, mask].mean(axis=1))
        band_power = np.column_stack(band_power)
        log_power = np.log10(np.maximum(band_power, EPSILON))
        channel_log_power.append(log_power)
        columns.extend(log_power[:, index] for index in range(5))

        spectral_density = np.maximum(local_density[:, spectral_mask], 0.0)
        total_power = np.maximum(spectral_density.sum(axis=1), EPSILON)
        for index in range(5):
            low, high = list(BANDS.values())[index]
            mask = (frequencies >= low) & (frequencies < high)
            numerator = np.maximum(local_density[:, mask], 0.0).sum(axis=1)
            columns.append(numerator / total_power)

        probability = spectral_density / total_power[:, None]
        entropy = -(probability * np.log(np.maximum(probability, EPSILON))).sum(axis=1)
        entropy /= np.log(probability.shape[1])
        columns.append(entropy)

        cumulative = np.cumsum(spectral_density, axis=1)
        edge_index = np.argmax(cumulative >= 0.95 * total_power[:, None], axis=1)
        columns.append(spectral_frequencies[edge_index])

        values = blocks[:, channel_index, :]
        difference_1 = np.diff(values, axis=1)
        difference_2 = np.diff(difference_1, axis=1)
        variance_0 = np.maximum(np.var(values, axis=1), EPSILON)
        variance_1 = np.maximum(np.var(difference_1, axis=1), EPSILON)
        variance_2 = np.maximum(np.var(difference_2, axis=1), EPSILON)
        rms = np.sqrt(np.mean(values * values, axis=1))
        line_length = np.mean(np.abs(difference_1), axis=1)
        zero_crossing = np.mean(np.signbit(values[:, 1:]) != np.signbit(values[:, :-1]), axis=1)
        mobility = np.sqrt(variance_1 / variance_0)
        complexity = np.sqrt(variance_2 / variance_1) / np.maximum(mobility, EPSILON)
        columns += [
            np.log10(np.maximum(rms, EPSILON)),
            np.log10(np.maximum(line_length, EPSILON)),
            zero_crossing,
            mobility,
            complexity,
        ]

    left = blocks[:, 0, :] - blocks[:, 0, :].mean(axis=1, keepdims=True)
    right = blocks[:, 1, :] - blocks[:, 1, :].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    correlation = np.sum(left * right, axis=1) / np.maximum(denominator, EPSILON)
    columns.append(np.clip(correlation, -1.0, 1.0))

    coherence_frequency, cross_density = csd(
        blocks[:, 0, :],
        blocks[:, 1, :],
        fs=OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    coherence_density = np.abs(cross_density) ** 2 / np.maximum(
        density[:, 0, :] * density[:, 1, :], EPSILON
    )
    coherence_density = np.clip(coherence_density, 0.0, 1.0)
    for low, high in BANDS.values():
        mask = (coherence_frequency >= low) & (coherence_frequency < high)
        columns.append(coherence_density[:, mask].mean(axis=1))

    asymmetry = channel_log_power[0] - channel_log_power[1]
    columns.extend(asymmetry[:, index] for index in range(5))
    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (len(blocks), RICH_FEATURES_PER_EPOCH):
        raise ValueError(f"Unexpected enriched feature shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite enriched features")
    return result


def feature_bounds_pass(features: np.ndarray) -> bool:
    names = rich_feature_names(["C1", "C2"])
    columns = {name: index for index, name in enumerate(names)}
    relative = [index for name, index in columns.items() if name.endswith("_relative_power")]
    entropy = [index for name, index in columns.items() if name.endswith("_spectral_entropy")]
    edge = [index for name, index in columns.items() if name.endswith("_spectral_edge_95_hz")]
    zero = [index for name, index in columns.items() if name.endswith("_zero_crossing_rate")]
    coherence_columns = [index for name, index in columns.items() if name.endswith("_coherence")]
    correlation_column = columns["C1_C2_correlation"]
    return bool(
        np.logical_and(features[:, relative] >= 0.0, features[:, relative] <= 1.0).all()
        and np.logical_and(features[:, entropy] >= 0.0, features[:, entropy] <= 1.0).all()
        and np.logical_and(features[:, edge] >= 0.5, features[:, edge] < 30.0).all()
        and np.logical_and(features[:, zero] >= 0.0, features[:, zero] <= 1.0).all()
        and np.logical_and(features[:, coherence_columns] >= 0.0, features[:, coherence_columns] <= 1.0).all()
        and np.logical_and(features[:, correlation_column] >= -1.0, features[:, correlation_column] <= 1.0).all()
    )


def load_or_create_features(
    subject: str, modality: str, scaler: dict[str, dict[str, float]]
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    reviewed_path = block7_feature_path(subject, modality)
    with np.load(reviewed_path, allow_pickle=False) as values:
        onsets = values["onset"].astype(np.float64)
        reviewed_base = values["features"].astype(np.float64)
        reviewed_names = values["feature_names"].astype(str).tolist()

    destination = cache_path(subject, modality)
    names = rich_feature_names(MODALITIES[modality]["channels"])
    if destination.exists():
        with np.load(destination, allow_pickle=False) as values:
            cached_onsets = values["onset"].astype(np.float64)
            rich = values["features"].astype(np.float32)
            cached_names = values["feature_names"].astype(str).tolist()
        if not np.array_equal(onsets, cached_onsets) or names != cached_names:
            raise ValueError(f"Cached feature schema changed: {subject}, {modality}")
        status = "reused"
    else:
        config = MODALITIES[modality]
        raw = read_uv(subject, config["acquisition"], config["channels"])
        signal = normalize(filter_resample(raw, filter_sos()), config["channels"], scaler)
        rich = calculate_rich_features(epoch_blocks(signal, onsets), config["channels"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            onset=onsets,
            features=rich,
            feature_names=np.asarray(names),
        )
        status = "created"

    extracted_base = rich[:, base_indices()].astype(np.float64)
    maximum_difference = float(np.max(np.abs(extracted_base - reviewed_base)))
    expected_names = [names[index] for index in base_indices()]
    if reviewed_names != expected_names or maximum_difference > 1e-5:
        raise ValueError(f"F0 feature parity failed: {subject}, {modality}, {maximum_difference}")
    if not feature_bounds_pass(rich):
        raise ValueError(f"Enriched feature bounds failed: {subject}, {modality}")
    summary = {
        "subject": subject,
        "modality": modality,
        "epochs": len(onsets),
        "features_per_epoch": rich.shape[1],
        "maximum_f0_absolute_difference": maximum_difference,
        "all_values_finite": bool(np.isfinite(rich).all()),
        "feature_bounds_pass": True,
        "cache_status": status,
        "cache_bytes": destination.stat().st_size,
        "cache_sha256": sha256(destination),
    }
    return onsets, rich, names, summary


def context_matrix(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = context_start_indices(onsets)
    indices = starts[:, None] + np.arange(CONTEXT_EPOCHS)[None, :]
    matrix = features[indices].reshape(len(starts), -1)
    centers = onsets[starts + 4]
    return centers, matrix.astype(np.float32, copy=False)


def prepare_recordings(assignments: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    scalers = frozen_scalers()
    recordings = {modality: {} for modality in MODALITIES}
    summaries = []
    schema_rows = []
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        modality_centers = {}
        for modality in MODALITIES:
            onsets, rich, names, summary = load_or_create_features(
                item.subject, modality, scalers[modality]
            )
            centers, rich_context = context_matrix(onsets, rich)
            base_context = rich_context.reshape(len(centers), CONTEXT_EPOCHS, -1)[
                :, :, base_indices()
            ].reshape(len(centers), -1)
            recordings[modality][item.subject] = {
                "pid": int(item.pid),
                "centers": centers,
                "F0": base_context,
                "F1": rich_context,
            }
            modality_centers[modality] = centers
            summaries.append({"pid": int(item.pid), **summary, "supported_boundaries": len(centers)})
            if number == 1:
                for epoch_offset in range(CONTEXT_EPOCHS):
                    for feature_index, feature_name in enumerate(names):
                        schema_rows.append(
                            {
                                "modality": modality,
                                "context_epoch": epoch_offset,
                                "relative_time_sec": (epoch_offset - 4) * EPOCH_SEC,
                                "feature_index": epoch_offset * len(names) + feature_index,
                                "feature_name": feature_name,
                                "is_f0_feature": feature_index in set(base_indices()),
                            }
                        )
        if not np.array_equal(modality_centers["P2"], modality_centers["H2"]):
            raise ValueError(f"P2/H2 context-center mismatch: {item.subject}")
        print(f"features {number:02d}/{len(assignments)} {item.subject}", flush=True)
    return recordings, pd.DataFrame(summaries), pd.DataFrame(schema_rows)


# Section 5: candidate matrices and fold-local models

def build_candidate_matrices(candidates: pd.DataFrame, recordings: dict) -> tuple[dict, pd.DataFrame]:
    rows = []
    retained_indices = {modality: [] for modality in MODALITIES}
    for item in candidates.itertuples(index=False):
        indices = {}
        for modality in MODALITIES:
            centers = recordings[modality][item.subject]["centers"]
            lookup = {time_key(value): index for index, value in enumerate(centers)}
            indices[modality] = lookup.get(time_key(item.candidate_time_sec))
        if indices["P2"] != indices["H2"]:
            raise ValueError(f"Candidate parity failed: {item.sample_id}")
        retained = indices["P2"] is not None
        rows.append(
            {
                "sample_id": item.sample_id,
                "subject": item.subject,
                "pid": int(item.pid),
                "candidate_time_sec": float(item.candidate_time_sec),
                "label": int(item.label),
                "source_tier": item.source_tier,
                "retained": retained,
                "drop_reason": "" if retained else "missing_required_context",
            }
        )
        if retained:
            for modality in MODALITIES:
                retained_indices[modality].append((item.subject, indices[modality]))
    construction = pd.DataFrame(rows)
    retained = construction[truth(construction["retained"])].reset_index(drop=True)
    matrices = {}
    for modality in MODALITIES:
        matrices[modality] = {
            kind: np.stack(
                [recordings[modality][subject][kind][index] for subject, index in retained_indices[modality]]
            ).astype(np.float32)
            for kind in ["F0", "F1"]
        }
    if len(retained) != 2743 or int(retained["label"].sum()) != 180:
        raise ValueError("Frozen candidate count changed")
    return matrices, construction


def flattened_feature_names(schema: pd.DataFrame, modality: str, kind: str) -> list[str]:
    local = schema[schema["modality"].eq(modality)].copy()
    if kind == "F0":
        local = local[truth(local["is_f0_feature"])]
    return [f"t{int(row.relative_time_sec):+d}_{row.feature_name}" for row in local.itertuples(index=False)]


def correlation_prune(values: np.ndarray) -> np.ndarray:
    variance = np.var(values, axis=0)
    candidates = np.flatnonzero(variance > VARIANCE_THRESHOLD)
    if len(candidates) == 0:
        raise ValueError("Variance pruning removed every feature")
    correlation_matrix = np.corrcoef(values[:, candidates], rowvar=False)
    retained_local = []
    for local_index in range(len(candidates)):
        if not retained_local:
            retained_local.append(local_index)
            continue
        correlations = np.abs(correlation_matrix[local_index, retained_local])
        if not np.any(correlations >= CORRELATION_THRESHOLD):
            retained_local.append(local_index)
    return candidates[np.asarray(retained_local, dtype=int)]


def fit_model(
    values: np.ndarray,
    labels: np.ndarray,
    pipeline: str,
    c_value: float,
    feature_names: list[str],
) -> tuple[dict, dict]:
    selected = (
        correlation_prune(values)
        if pipeline == "F1-EN"
        else np.arange(values.shape[1], dtype=int)
    )
    scaler = StandardScaler().fit(values[:, selected])
    penalty = "elasticnet" if pipeline == "F1-EN" else "l2"
    solver = "saga" if pipeline == "F1-EN" else "lbfgs"
    settings = {
        "C": c_value,
        "class_weight": "balanced",
        "solver": solver,
        "penalty": penalty,
        "max_iter": 3000,
        "tol": 1e-4,
        "random_state": BASE_SEED,
    }
    if pipeline == "F1-EN":
        settings["l1_ratio"] = 0.5
    model = LogisticRegression(**settings)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(values[:, selected]), labels)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    nonzero = np.flatnonzero(np.abs(model.coef_[0]) > 1e-12)
    payload = {
        "configuration": {
            "pipeline": pipeline,
            "C": c_value,
            "penalty": penalty,
            "solver": solver,
            "l1_ratio": 0.5 if pipeline == "F1-EN" else None,
            "variance_threshold": VARIANCE_THRESHOLD,
            "correlation_threshold": CORRELATION_THRESHOLD,
        },
        "selected_indices": selected.tolist(),
        "selected_feature_names": [feature_names[index] for index in selected],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
    }
    fitted = {"selected": selected, "scaler": scaler, "model": model}
    summary = {
        "input_features": values.shape[1],
        "retained_after_pruning": len(selected),
        "nonzero_coefficients": len(nonzero),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
        "payload": payload,
    }
    return fitted, summary


def score_model(fitted: dict, values: np.ndarray) -> np.ndarray:
    selected = fitted["selected"]
    scaled = fitted["scaler"].transform(values[:, selected])
    probability = fitted["model"].predict_proba(scaled)[:, 1]
    if not np.logical_and(probability >= 0.0, probability <= 1.0).all():
        raise ValueError("Invalid classifier probability")
    return probability


def score_assignments(
    fitted: dict,
    heldout: pd.DataFrame,
    recordings: dict,
    modality: str,
    pipeline: str,
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kind = "F0" if pipeline == "F0-L2" else "F1"
    score_rows = []
    support_rows = []
    for item in heldout.itertuples(index=False):
        recording = recordings[modality][item.subject]
        probability = score_model(fitted, recording[kind])
        score_rows.append(
            pd.DataFrame(
                {
                    "modality": modality,
                    "pipeline": f"{modality}-{pipeline}",
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["centers"],
                    "probability": probability,
                }
            )
        )
        support_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(probability),
                "supported_hours": len(probability) * EPOCH_SEC / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def run_fit(
    modality: str,
    pipeline: str,
    outer: int,
    phase: str,
    c_value: float,
    fit_mask: np.ndarray,
    heldout: pd.DataFrame,
    matrices: dict,
    labels: np.ndarray,
    recordings: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    kind = "F0" if pipeline == "F0-L2" else "F1"
    model_file = model_path(modality, pipeline, outer, phase, c_value)
    score_file = score_path(modality, pipeline, outer, phase, c_value)
    if model_file.exists() != score_file.exists():
        raise RuntimeError(f"Incomplete cached fit artifacts: {model_file} / {score_file}")
    if model_file.exists():
        with gzip.open(model_file, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        scores = pd.read_csv(score_file, sep="\t", compression="gzip")
        expected_pipeline = f"{modality}-{pipeline}"
        if (
            payload["configuration"]["pipeline"] != pipeline
            or float(payload["configuration"]["C"]) != float(c_value)
            or not scores["pipeline"].eq(expected_pipeline).all()
            or not scores["outer_fold"].eq(outer).all()
            or not scores["phase"].eq(phase).all()
            or set(scores["subject"]) != set(heldout["subject"])
        ):
            raise RuntimeError(f"Cached fit identity mismatch: {model_file}")
        support = (
            scores.groupby(["subject", "pid", "outer_fold"], as_index=False)
            .size()
            .rename(columns={"size": "supported_boundaries"})
        )
        support["supported_hours"] = support["supported_boundaries"] * EPOCH_SEC / 3600.0
        coefficients = np.asarray(payload["coefficient"], dtype=float)
        summary = {
            "input_features": matrices[modality][kind].shape[1],
            "retained_after_pruning": len(payload["selected_indices"]),
            "nonzero_coefficients": int(np.sum(np.abs(coefficients) > 1e-12)),
            "iterations": int(payload["iterations"]),
            "converged": bool(payload["converged"]),
            "model_relative_path": model_file.relative_to(data_parent()).as_posix(),
            "model_sha256": sha256(model_file),
            "score_relative_path": score_file.relative_to(data_parent()).as_posix(),
            "score_sha256": sha256(score_file),
        }
        return scores, support, summary

    fitted, summary = fit_model(
        matrices[modality][kind][fit_mask], labels[fit_mask], pipeline, c_value, feature_names
    )
    scores, support = score_assignments(
        fitted, heldout, recordings, modality, pipeline, outer, phase
    )
    verify_or_create_json_gzip(model_file, summary.pop("payload"))
    verify_or_create_gzip_tsv(score_file, scores)
    summary.update(
        {
            "model_relative_path": model_file.relative_to(data_parent()).as_posix(),
            "model_sha256": sha256(model_file),
            "score_relative_path": score_file.relative_to(data_parent()).as_posix(),
            "score_sha256": sha256(score_file),
        }
    )
    return scores, support, summary


# Section 6: event evaluation and uncertainty

def collapse_outer_events(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in selections.itertuples(index=False):
        local = scores[
            scores["modality"].eq(item.modality)
            & scores["pipeline"].eq(f"{item.modality}-{item.candidate}")
            & scores["outer_fold"].eq(item.outer_fold)
        ]
        for (subject, pid), group in local.groupby(["subject", "pid"], sort=True):
            predictions = collapsed_times(group.sort_values("candidate_time_sec"), float(item.threshold))
            for event_time in predictions:
                rows.append(
                    {
                        "modality": item.modality,
                        "pipeline": f"{item.modality}-{item.candidate}",
                        "outer_fold": int(item.outer_fold),
                        "subject": subject,
                        "pid": int(pid),
                        "event_time_sec": float(event_time),
                        "threshold": float(item.threshold),
                        "C": float(item.C),
                    }
                )
    columns = ["modality", "pipeline", "outer_fold", "subject", "pid", "event_time_sec", "threshold", "C"]
    return pd.DataFrame(rows, columns=columns)


def evaluate_all(
    events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    participant_rows = []
    match_rows = []
    for modality in MODALITIES:
        for candidate in PIPELINES:
            pipeline = f"{modality}-{candidate}"
            predictions = events[events["pipeline"].eq(pipeline)]
            for membership in MEMBERSHIPS:
                eligible, ignored = local_event_inputs(references, membership)
                for tolerance in TOLERANCES:
                    _, participants, matches, summary = evaluate_events(
                        eligible,
                        predictions[["subject", "pid", "event_time_sec"]],
                        ignored,
                        support[["subject", "pid", "supported_hours"]],
                        tolerance,
                    )
                    config = {
                        "modality": modality,
                        "pipeline": pipeline,
                        "partition": "train_nested_oof",
                        "membership": membership,
                        "tolerance_sec": tolerance,
                    }
                    metric_rows.append({**config, **summary})
                    for frame, destination in [(participants, participant_rows), (matches, match_rows)]:
                        if len(frame):
                            local = frame.copy()
                            for key, value in reversed(list(config.items())):
                                if key not in local.columns:
                                    local.insert(0, key, value)
                            destination.append(local)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "participants": pd.concat(participant_rows, ignore_index=True),
        "matches": pd.concat(match_rows, ignore_index=True),
    }


def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        participants["membership"].eq("primary") & participants["tolerance_sec"].eq(15.0)
    ]
    comparisons = [
        ("F1-L2", "F0-L2"),
        ("F1-EN", "F0-L2"),
        ("F1-EN", "F1-L2"),
    ]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    rows = []
    for modality in MODALITIES:
        for left_name, right_name in comparisons:
            left = primary[primary["pipeline"].eq(f"{modality}-{left_name}")][columns]
            right = primary[primary["pipeline"].eq(f"{modality}-{right_name}")][columns]
            paired = left.merge(right, on="pid", suffixes=("_left", "_right"), validate="one_to_one")
            if len(paired) != 64:
                raise ValueError("Incomplete participant pairing")
            rng = np.random.default_rng(BASE_SEED)
            samples = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
                values = {}
                for side in ["left", "right"]:
                    values[side] = metric_values(
                        int(sample[f"true_positive_{side}"].sum()),
                        int(sample[f"false_positive_{side}"].sum()),
                        int(sample[f"false_negative_{side}"].sum()),
                        float(sample[f"supported_hours_{side}"].sum()),
                    )
                samples.append(
                    {
                        "event_f1_difference": values["left"]["f1"] - values["right"]["f1"],
                        "false_alarms_per_hour_difference": values["left"]["false_alarms_per_hour"]
                        - values["right"]["false_alarms_per_hour"],
                    }
                )
            sample_frame = pd.DataFrame(samples)
            point = {}
            for side, frame in [("left", left), ("right", right)]:
                point[side] = metric_values(
                    int(frame["true_positive"].sum()),
                    int(frame["false_positive"].sum()),
                    int(frame["false_negative"].sum()),
                    float(frame["supported_hours"].sum()),
                )
            differences = {
                "event_f1_difference": point["left"]["f1"] - point["right"]["f1"],
                "false_alarms_per_hour_difference": point["left"]["false_alarms_per_hour"]
                - point["right"]["false_alarms_per_hour"],
            }
            for metric, value in differences.items():
                rows.append(
                    {
                        "modality": modality,
                        "comparison": f"{left_name}_minus_{right_name}",
                        "metric": metric,
                        "point_difference": value,
                        "resamples": BOOTSTRAP_RESAMPLES,
                        "seed": BASE_SEED,
                        "lower_95": float(sample_frame[metric].quantile(0.025)),
                        "median": float(sample_frame[metric].quantile(0.5)),
                        "upper_95": float(sample_frame[metric].quantile(0.975)),
                    }
                )
    return pd.DataFrame(rows)


def decision_table(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        metrics["membership"].eq("primary") & metrics["tolerance_sec"].eq(15.0)
    ].set_index("pipeline")
    rows = []
    for modality in MODALITIES:
        for candidate in ["F1-L2", "F1-EN"]:
            comparison = f"{candidate}_minus_F0-L2"
            f1_difference = float(
                primary.loc[f"{modality}-{candidate}", "f1"]
                - primary.loc[f"{modality}-F0-L2", "f1"]
            )
            far_difference = float(
                primary.loc[f"{modality}-{candidate}", "false_alarms_per_hour"]
                - primary.loc[f"{modality}-F0-L2", "false_alarms_per_hour"]
            )
            local = bootstrap[
                bootstrap["modality"].eq(modality) & bootstrap["comparison"].eq(comparison)
            ].set_index("metric")
            point_pass = f1_difference >= MEANINGFUL_F1_GAIN and far_difference <= 0.0
            directional = bool(
                local.loc["event_f1_difference", "lower_95"] > 0.0
                and local.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
            )
            material = bool(
                local.loc["event_f1_difference", "lower_95"] >= MEANINGFUL_F1_GAIN
                and local.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
            )
            rows.append(
                {
                    "modality": modality,
                    "hypothesis": f"H-{candidate}",
                    "comparison": comparison,
                    "f1_difference": f1_difference,
                    "required_f1_difference": MEANINGFUL_F1_GAIN,
                    "false_alarms_per_hour_difference": far_difference,
                    "maximum_far_difference": 0.0,
                    "point_gate_pass": point_pass,
                    "directional_participant_support": directional,
                    "material_participant_support": material,
                    "decision": "freeze_for_new_confirmation" if point_pass else "stop_v0.1",
                }
            )
    return pd.DataFrame(rows)


# Section 7: complete nested experiment

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    assignments = train_assignments()
    folds = frozen_fold_assignments(assignments)
    candidates = labeled_candidates(assignments)
    references = reference_events(assignments)
    recordings, feature_summary, schema = prepare_recordings(assignments)
    matrices, construction = build_candidate_matrices(candidates, recordings)
    retained = construction[truth(construction["retained"])].reset_index(drop=True)
    labels = retained["label"].to_numpy(dtype=int)
    groups = retained["pid"].to_numpy(dtype=int)

    fit_rows = []
    threshold_curve_rows = []
    selection_rows = []
    outer_score_rows = []
    outer_support_rows = []

    for modality in MODALITIES:
        for outer in range(1, OUTER_FOLDS + 1):
            outer_pid = set(folds[folds["fold"].eq(outer)]["pid"].astype(int))
            for pipeline in PIPELINES:
                c_values = EN_C_VALUES if pipeline == "F1-EN" else [1.0]
                candidate_curves = []
                for c_value in c_values:
                    inner_scores = []
                    inner_support = []
                    for inner in sorted(set(range(1, OUTER_FOLDS + 1)) - {outer}):
                        inner_pid = set(folds[folds["fold"].eq(inner)]["pid"].astype(int))
                        fit_mask = ~np.isin(groups, list(outer_pid | inner_pid))
                        heldout = assignments[assignments["pid"].isin(inner_pid)].copy()
                        kind = "F0" if pipeline == "F0-L2" else "F1"
                        names = flattened_feature_names(schema, modality, kind)
                        scores, support, summary = run_fit(
                            modality,
                            pipeline,
                            outer,
                            f"inner_{inner}",
                            c_value,
                            fit_mask,
                            heldout,
                            matrices,
                            labels,
                            recordings,
                            names,
                        )
                        fit_rows.append(
                            {
                                "modality": modality,
                                "candidate": pipeline,
                                "outer_fold": outer,
                                "inner_fold": inner,
                                "phase": f"inner_{inner}",
                                "C": c_value,
                                **summary,
                            }
                        )
                        inner_scores.append(scores)
                        inner_support.append(support)
                    pooled_scores = pd.concat(inner_scores, ignore_index=True)
                    pooled_support = pd.concat(inner_support, ignore_index=True)
                    for threshold in THRESHOLDS:
                        candidate_curves.append(
                            {
                                "modality": modality,
                                "candidate": pipeline,
                                "outer_fold": outer,
                                "C": c_value,
                                "threshold": float(threshold),
                                **fast_primary_summary(
                                    pooled_scores, pooled_support, references, float(threshold)
                                ),
                            }
                        )
                curve = pd.DataFrame(candidate_curves)
                selected = curve.sort_values(
                    ["f1", "false_alarms_per_hour", "recall", "threshold", "C"],
                    ascending=[False, True, False, False, True],
                    kind="stable",
                ).iloc[0].to_dict()
                selected["selection_rule"] = (
                    "max_f1_then_min_far_then_max_recall_then_max_threshold_then_min_C"
                )
                threshold_curve_rows.append(curve)
                selection_rows.append(selected)

                fit_mask = ~np.isin(groups, list(outer_pid))
                heldout = assignments[assignments["pid"].isin(outer_pid)].copy()
                kind = "F0" if pipeline == "F0-L2" else "F1"
                names = flattened_feature_names(schema, modality, kind)
                scores, support, summary = run_fit(
                    modality,
                    pipeline,
                    outer,
                    "outer_final",
                    float(selected["C"]),
                    fit_mask,
                    heldout,
                    matrices,
                    labels,
                    recordings,
                    names,
                )
                fit_rows.append(
                    {
                        "modality": modality,
                        "candidate": pipeline,
                        "outer_fold": outer,
                        "inner_fold": 9,
                        "phase": "outer_final",
                        "C": float(selected["C"]),
                        **summary,
                    }
                )
                outer_score_rows.append(scores)
                outer_support_rows.append(support)
                print(f"completed {modality} outer {outer} {pipeline}", flush=True)

    fit_summary = pd.DataFrame(fit_rows)
    selections = pd.DataFrame(selection_rows)
    outer_scores = pd.concat(outer_score_rows, ignore_index=True)
    support_all = pd.concat(outer_support_rows, ignore_index=True)
    support = support_all.drop_duplicates(["subject", "pid", "outer_fold"]).sort_values("subject")
    events = collapse_outer_events(outer_scores, selections)
    evaluation = evaluate_all(events, support, references)
    bootstrap = paired_bootstrap(evaluation["participants"])
    decisions = decision_table(evaluation["metrics"], bootstrap)

    external_files = sorted(path for path in derived_dir().rglob("*") if path.is_file())
    manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in external_files
        ]
    )

    convergence_failures = int((~fit_summary["converged"].astype(bool)).sum())
    outer_convergence_failures = int(
        (~fit_summary.loc[fit_summary["phase"].eq("outer_final"), "converged"].astype(bool)).sum()
    )
    checks = pd.DataFrame(
        [
            ("train_scope", len(assignments) == 82 and assignments["pid"].nunique() == 64, "82 recordings; 64 pid"),
            ("fold_scope", len(folds) == 64 and folds["fold"].nunique() == 5, "five frozen pid folds"),
            ("candidate_accounting", len(retained) == 2743 and int(labels.sum()) == 180, "2743 candidates; 180 positives"),
            ("feature_parity", feature_summary["maximum_f0_absolute_difference"].max() <= 1e-5, f"max={feature_summary['maximum_f0_absolute_difference'].max():.3g}"),
            ("feature_bounds", feature_summary["feature_bounds_pass"].astype(bool).all(), "all 164 modality-recording rows"),
            (
                "fit_convergence",
                convergence_failures == 0,
                f"{convergence_failures} of {len(fit_summary)} fits reached max_iter; "
                f"{outer_convergence_failures} outer-final failures",
            ),
            ("threshold_selection", len(selections) == 30, "2 modalities x 5 folds x 3 candidates"),
            ("outer_coverage", outer_scores.groupby("pipeline")["pid"].nunique().eq(64).all(), "64 pid per pipeline"),
            ("external_scope", not manifest["relative_path"].str.contains("validation|test", case=False, regex=True).any(), f"{len(manifest)} external artifacts"),
            ("decision_count", len(decisions) == 4, "two hypotheses per modality"),
        ],
        columns=["check", "status", "detail"],
    )
    checks["status"] = np.where(checks["status"], "pass", "fail")
    fatal_checks = checks[~checks["check"].eq("fit_convergence")]
    if not fatal_checks["status"].eq("pass").all():
        raise ValueError("Experiment checks failed:\n" + checks.to_string(index=False))

    outer_final = fit_summary[fit_summary["phase"].eq("outer_final")]
    stability_rows = []
    for item in outer_final.itertuples(index=False):
        path = data_parent() / item.model_relative_path
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        selected_names = set(payload["selected_feature_names"])
        nonzero_names = {
            name
            for name, value in zip(payload["selected_feature_names"], payload["coefficient"])
            if abs(float(value)) > 1e-12
        }
        kind = "F0" if item.candidate == "F0-L2" else "F1"
        for name in flattened_feature_names(schema, item.modality, kind):
            stability_rows.append(
                {
                    "modality": item.modality,
                    "candidate": item.candidate,
                    "outer_fold": int(item.outer_fold),
                    "feature_name": name,
                    "retained_after_pruning": name in selected_names,
                    "nonzero_coefficient": name in nonzero_names,
                }
            )
    stability = pd.DataFrame(stability_rows)
    stability_summary = (
        stability.groupby(["modality", "candidate", "feature_name"], as_index=False)
        .agg(
            outer_folds=("outer_fold", "nunique"),
            retained_folds=("retained_after_pruning", "sum"),
            nonzero_folds=("nonzero_coefficient", "sum"),
        )
    )

    output = output_dir()
    verify_or_create_tsv(feature_summary, output / "feature_generation_summary_v0.1.tsv")
    verify_or_create_tsv(schema, output / "feature_schema_v0.1.tsv")
    verify_or_create_tsv(construction, output / "candidate_construction_v0.1.tsv")
    verify_or_create_tsv(fit_summary, output / "model_fit_summary_v0.1.tsv")
    verify_or_create_tsv(
        pd.concat(threshold_curve_rows, ignore_index=True),
        output / "inner_threshold_curves_v0.1.tsv",
    )
    verify_or_create_tsv(selections, output / "inner_threshold_selections_v0.1.tsv")
    verify_or_create_tsv(support, output / "outer_support_v0.1.tsv")
    verify_or_create_tsv(events, output / "outer_predicted_events_v0.1.tsv")
    verify_or_create_tsv(evaluation["metrics"], output / "outer_event_metrics_v0.1.tsv")
    verify_or_create_tsv(evaluation["participants"], output / "outer_event_participants_v0.1.tsv")
    verify_or_create_tsv(evaluation["matches"], output / "outer_event_matches_v0.1.tsv")
    verify_or_create_tsv(bootstrap, output / "paired_participant_bootstrap_v0.1.tsv")
    verify_or_create_tsv(decisions, output / "hypothesis_decisions_v0.1.tsv")
    verify_or_create_tsv(stability_summary, output / "outer_feature_selection_stability_v0.1.tsv")
    verify_or_create_tsv(manifest, output / "external_artifact_manifest_v0.1.tsv")
    verify_or_create_tsv(checks, output / "in_run_checks_v0.1.tsv")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": git_commit,
        "protocol_commit": PROTOCOL_COMMIT,
    }
    verify_or_create_text(
        output / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )

    primary = evaluation["metrics"][
        evaluation["metrics"]["membership"].eq("primary")
        & evaluation["metrics"]["tolerance_sec"].eq(15.0)
    ].set_index("pipeline")
    lines = [
        "# Paired Enriched-Feature and Pruning Experiment v0.1",
        "",
        "Train-only participant-grouped nested results using the frozen feature and pruning protocol.",
        "",
        "| Pipeline | Precision | Recall | F1 | False alarms/hour |",
        "|---|---:|---:|---:|---:|",
    ]
    for modality in MODALITIES:
        for candidate in PIPELINES:
            row = primary.loc[f"{modality}-{candidate}"]
            lines.append(
                f"| {modality}-{candidate} | {row.precision:.4f} | {row.recall:.4f} | "
                f"{row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
            )
    lines += [
        "",
        f"Convergence check: {convergence_failures} of {len(fit_summary)} fits reached the predeclared "
        f"3,000-iteration limit; {outer_convergence_failures} were outer-final fits. Settings were not changed after inspection.",
        "",
        "The advancement decisions are recorded in `hypothesis_decisions_v0.1.tsv`.",
        "Validation and current-test data remained closed. Enriched arrays, fitted models, and full-night scores remain under `REM_W_data` outside Git.",
        "",
    ]
    verify_or_create_text(output / "README.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
