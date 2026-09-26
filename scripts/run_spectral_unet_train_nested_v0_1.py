"""Run the frozen train-only spectral U-Net experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values, optimal_matches


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-25_spectral_unet_train_nested_v0.1"
DERIVED_DIR = "spectral_unet_train_nested_v0.1"
PROTOCOL_COMMIT = "03d0e0c"
BASE_SEED = 20260925
OUTER_FOLDS = 5
INNER_FOLDS = 4
EPOCH_SEC = 30.0
INPUT_SFREQ = 256.0
OUTPUT_SFREQ = 128.0
EPOCH_SAMPLES = int(EPOCH_SEC * OUTPUT_SFREQ)
HB2 = ["HB_1", "HB_2"]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
SUBEPOCH_SEC = 2.0
SUBEPOCH_SAMPLES = int(SUBEPOCH_SEC * OUTPUT_SFREQ)
SUBEPOCHS_PER_EPOCH = int(EPOCH_SEC / SUBEPOCH_SEC)
SEQUENCE_EPOCHS = 8
SEQUENCE_BINS = SEQUENCE_EPOCHS * SUBEPOCHS_PER_EPOCH
FEATURE_COUNT = 10
CENTER_START = 52
CENTER_STOP = 68
TRAINING_EPOCHS = 30
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
GRADIENT_CLIP = 5.0
BOOTSTRAP_RESAMPLES = 2000
MEANINGFUL_F1_GAIN = 0.05
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, EPOCH_SEC)
PIPELINES = ["NESTED-BLSTM-CRF", "SPECTRAL-UNET"]


# Section 2: paths and immutable writers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def cache_path(subject: str) -> Path:
    return derived_dir() / "subepoch_features" / f"{subject}_spectral_v0.1.npz"


def fit_name(outer: int, phase: str) -> str:
    return f"outer_{outer}_{phase}"


def model_path(outer: int, phase: str) -> Path:
    return derived_dir() / "models" / f"{fit_name(outer, phase)}_v0.1.json.gz"


def score_path(outer: int, phase: str) -> Path:
    return derived_dir() / "scores" / f"{fit_name(outer, phase)}_v0.1.tsv.gz"


def outer_score_path() -> Path:
    return derived_dir() / "scores" / "spectral_unet_outer_scores_v0.1.tsv.gz"


def deterministic_gzip(value: bytes) -> bytes:
    from io import BytesIO

    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(value)
    return output.getvalue()


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def verify_or_create_gzip_tsv(path: Path, frame: pd.DataFrame) -> None:
    value = frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    expected = deterministic_gzip(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External score artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def verify_or_create_json_gzip(path: Path, payload: dict) -> None:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = deterministic_gzip(value.encode("utf-8"))
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External model artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
    ).strip()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


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
    result = pd.DataFrame(rows).sort_values(
        "subject", key=lambda value: value.map(subject_number)
    )
    if len(result) != 82 or result["pid"].nunique() != 64:
        raise ValueError("Unexpected frozen train membership")
    if result["subject"].duplicated().any():
        raise ValueError("Duplicate train recording")
    return result.reset_index(drop=True)


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


def block7_feature_path(subject: str) -> Path:
    return (
        data_parent()
        / "derived/block7_feature_generation_validation_v0.1/recording_features/hb2"
        / f"{subject}_features_v0.1.npz"
    )


def edf_path(subject: str) -> Path:
    return (
        data_parent()
        / "boas_ds005555_v1.1.1"
        / subject
        / "eeg"
        / f"{subject}_task-Sleep_acq-headband_eeg.edf"
    )


def read_uv(subject: str, acquisition: str, channels: list[str]) -> np.ndarray:
    if acquisition != "headband":
        raise ValueError("This experiment permits headband EDF access only")
    edf = edfio.read_edf(edf_path(subject), lazy_load_data=True)
    if any(channel not in edf.labels for channel in channels):
        raise ValueError(f"{subject} is missing a required headband channel")
    signals = []
    for channel in channels:
        signal = edf.get_signal(channel)
        if float(signal.sampling_frequency) != INPUT_SFREQ:
            raise ValueError(f"{subject} sampling frequency is not 256 Hz")
        if str(signal.physical_dimension).lower() not in {"uv", "µv"}:
            raise ValueError(f"{subject} unexpected physical unit: {signal.physical_dimension}")
        signals.append(np.asarray(signal.data, dtype=np.float64))
    result = np.stack(signals)
    if not np.isfinite(result).all():
        raise ValueError(f"{subject} contains nonfinite samples")
    return result


def filter_sos() -> np.ndarray:
    return butter(4, [0.3, 35.0], btype="bandpass", fs=INPUT_SFREQ, output="sos")


def filter_resample(signal_uv: np.ndarray, sos: np.ndarray) -> np.ndarray:
    filtered = sosfiltfilt(sos, signal_uv, axis=1)
    result = resample_poly(filtered, up=1, down=2, axis=1)
    expected = int(np.ceil(signal_uv.shape[1] / 2.0))
    if result.shape[1] != expected or not np.isfinite(result).all():
        raise ValueError("Filter/resample output failed validation")
    return result


def normalize(signal_uv: np.ndarray, channels: list[str], scaler: dict) -> np.ndarray:
    result = signal_uv.copy()
    for index, channel in enumerate(channels):
        center = float(scaler[channel]["median_uv"])
        scale = float(scaler[channel]["robust_scale_uv"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid robust scale for {channel}")
        result[index] = (result[index] - center) / scale
    return result


def context_start_indices(onsets: np.ndarray) -> np.ndarray:
    if len(onsets) < len(CONTEXT_OFFSETS):
        return np.asarray([], dtype=int)
    contiguous = np.isclose(
        np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0
    ).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(7, dtype=np.int8), mode="valid")
    return np.flatnonzero(counts == 7)


def time_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


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


def collapse_events(scores: pd.DataFrame, threshold: float, pipeline: str) -> pd.DataFrame:
    rows = []
    for (subject, pid, outer), group in scores.groupby(
        ["subject", "pid", "outer_fold"], sort=True
    ):
        group = group.sort_values("candidate_time_sec")
        times = group["candidate_time_sec"].to_numpy(dtype=float)
        probabilities = group["probability"].to_numpy(dtype=float)
        selected = np.flatnonzero(probabilities >= threshold)
        if len(selected) == 0:
            continue
        splits = np.flatnonzero(np.diff(times[selected]) > EPOCH_SEC + 1e-6) + 1
        for run in np.split(selected, splits):
            local = probabilities[run]
            best = run[np.flatnonzero(np.isclose(local, local.max()))[0]]
            rows.append(
                {
                    "pipeline": pipeline,
                    "outer_fold": int(outer),
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(times[best]),
                    "probability": float(probabilities[best]),
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(rows)


# Section 3: two-second spectral representation

def feature_names() -> list[str]:
    return [f"{channel}_{band}_log10_mean_psd" for channel in HB2 for band in BANDS]


def frozen_hb_scaler() -> dict[str, dict[str, float]]:
    table = pd.read_csv(
        repo_root()
        / "experiments/2026-09-06_block7_feature_generation_validation_v0.1"
        / "train_robust_scalers_v0.1.tsv",
        sep="\t",
    )
    table = table[table["scaler_owner"].eq("HB-2")].set_index("channel")
    if list(table.loc[HB2].index) != HB2:
        raise ValueError("Frozen HB-2 scaler schema changed")
    return {
        channel: {
            "median_uv": float(table.loc[channel, "median_uv"]),
            "robust_scale_uv": float(table.loc[channel, "robust_scale_uv"]),
        }
        for channel in HB2
    }


def calculate_subepoch_features(normalized: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    epochs = []
    for onset in onsets:
        start = int(round(float(onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        epoch = normalized[:, start:stop]
        if epoch.shape != (len(HB2), EPOCH_SAMPLES):
            raise ValueError(f"Incomplete epoch at {onset:g} seconds")
        epochs.append(epoch.reshape(len(HB2), SUBEPOCHS_PER_EPOCH, SUBEPOCH_SAMPLES))
    blocks = np.stack(epochs)
    frequencies, density = welch(
        blocks,
        fs=OUTPUT_SFREQ,
        window="hann",
        nperseg=SUBEPOCH_SAMPLES,
        noverlap=0,
        axis=-1,
    )
    columns = []
    for channel_index in range(len(HB2)):
        for low, high in BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            power = density[:, channel_index, :, :][:, :, mask].mean(axis=-1)
            columns.append(np.log10(np.maximum(power, np.finfo(float).eps)))
    result = np.stack(columns, axis=-1).astype(np.float32)
    expected = (len(onsets), SUBEPOCHS_PER_EPOCH, FEATURE_COUNT)
    if result.shape != expected or not np.isfinite(result).all():
        raise ValueError("Invalid two-second spectral feature output")
    return result


def load_or_create_cache(subject: str, scaler: dict) -> tuple[np.ndarray, np.ndarray, str]:
    path = cache_path(subject)
    block7_path = block7_feature_path(subject)
    with np.load(block7_path, allow_pickle=False) as values:
        onsets = values["onset"].astype(np.float64)
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            cached_onsets = values["onset"].astype(np.float64)
            features = values["features"].astype(np.float32)
            names = values["feature_names"].astype(str).tolist()
        if not np.array_equal(onsets, cached_onsets) or names != feature_names():
            raise ValueError(f"Cached schema or onset mismatch: {subject}")
        if features.shape != (len(onsets), SUBEPOCHS_PER_EPOCH, FEATURE_COUNT):
            raise ValueError(f"Cached shape mismatch: {subject}")
        return onsets, features, "reused"

    signal = read_uv(subject, "headband", HB2)
    normalized = normalize(filter_resample(signal, filter_sos()), HB2, scaler)
    features = calculate_subepoch_features(normalized, onsets)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        onset=onsets,
        features=features,
        feature_names=np.asarray(feature_names()),
    )
    return onsets, features, "created"


def prepare_recordings(assignments: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame]:
    scaler = frozen_hb_scaler()
    recordings = {}
    rows = []
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        onsets, features, status = load_or_create_cache(item.subject, scaler)
        starts = context_start_indices(onsets)
        centers = onsets[starts + 4]
        recordings[item.subject] = {
            "pid": int(item.pid),
            "onsets": onsets,
            "features": features,
            "starts": starts,
            "centers": centers,
        }
        rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "epochs": len(onsets),
                "supported_boundaries": len(starts),
                "cache_status": status,
                "cache_bytes": cache_path(item.subject).stat().st_size,
                "cache_sha256": sha256(cache_path(item.subject)),
            }
        )
        print(f"features {number:02d}/{len(assignments)} {item.subject} {status}", flush=True)
    return recordings, pd.DataFrame(rows)


def recording_sequences(recording: dict) -> np.ndarray:
    offsets = np.arange(SEQUENCE_EPOCHS)
    epochs = recording["features"][recording["starts"][:, None] + offsets[None, :]]
    result = epochs.reshape(len(epochs), SEQUENCE_BINS, FEATURE_COUNT)
    if result.shape[1:] != (SEQUENCE_BINS, FEATURE_COUNT):
        raise ValueError("Recording sequence shape changed")
    return result.astype(np.float32, copy=False)


def build_labeled_sequences(
    candidates: pd.DataFrame, recordings: dict[str, dict]
) -> tuple[np.ndarray, pd.DataFrame]:
    matrices = []
    rows = []
    for subject, group in candidates.groupby("subject", sort=True):
        recording = recordings[subject]
        sequences = recording_sequences(recording)
        lookup = {time_key(value): index for index, value in enumerate(recording["centers"])}
        for item in group.itertuples(index=False):
            index = lookup.get(time_key(item.candidate_time_sec))
            retained = index is not None
            rows.append(
                {
                    "sample_id": item.sample_id,
                    "subject": subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": float(item.candidate_time_sec),
                    "label": int(item.label),
                    "source_tier": item.source_tier,
                    "retained": retained,
                    "drop_reason": "" if retained else "missing_required_context",
                }
            )
            if retained:
                matrices.append(sequences[index])
    construction = pd.DataFrame(rows)
    retained_rows = construction[truth(construction["retained"])].reset_index(drop=True)
    if len(matrices) != len(retained_rows):
        raise ValueError("Candidate construction accounting failed")
    return np.stack(matrices).astype(np.float32), construction


# Section 4: fixed 1D U-Net

class ConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=5, padding=2),
            nn.GroupNorm(4, output_channels),
            nn.GELU(),
            nn.Conv1d(output_channels, output_channels, kernel_size=5, padding=2),
            nn.GroupNorm(4, output_channels),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


class SpectralUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = ConvBlock(FEATURE_COUNT, 8)
        self.encoder_2 = ConvBlock(8, 16)
        self.encoder_3 = ConvBlock(16, 32)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = ConvBlock(32, 64)
        self.up_3 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)
        self.decoder_3 = ConvBlock(64, 32)
        self.up_2 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)
        self.decoder_2 = ConvBlock(32, 16)
        self.up_1 = nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2)
        self.decoder_1 = ConvBlock(16, 8)
        self.output = nn.Conv1d(8, 1, kernel_size=1)

    def dense_logits(self, values: torch.Tensor) -> torch.Tensor:
        level_1 = self.encoder_1(values)
        level_2 = self.encoder_2(self.pool(level_1))
        level_3 = self.encoder_3(self.pool(level_2))
        encoded = self.bottleneck(self.pool(level_3))
        decoded_3 = self.decoder_3(torch.cat([self.up_3(encoded), level_3], dim=1))
        decoded_2 = self.decoder_2(torch.cat([self.up_2(decoded_3), level_2], dim=1))
        decoded_1 = self.decoder_1(torch.cat([self.up_1(decoded_2), level_1], dim=1))
        return self.output(decoded_1).squeeze(1)

    def candidate_logits(self, values: torch.Tensor) -> torch.Tensor:
        return self.dense_logits(values)[:, CENTER_START:CENTER_STOP].mean(dim=1)

    def probabilities(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.candidate_logits(values))


def configure_torch(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def synthetic_checks() -> pd.DataFrame:
    configure_torch(BASE_SEED)
    model = SpectralUNet()
    values = torch.randn(4, FEATURE_COUNT, SEQUENCE_BINS, requires_grad=True)
    dense = model.dense_logits(values)
    probability = model.probabilities(values)
    probability.mean().backward()
    gradients_finite = all(
        value.grad is not None and torch.isfinite(value.grad).all()
        for value in model.parameters()
    )
    time = np.arange(int(EPOCH_SEC * OUTPUT_SFREQ)) / OUTPUT_SFREQ
    synthetic_signal = np.stack(
        [np.sin(2 * np.pi * 10.0 * time), np.sin(2 * np.pi * 6.0 * time)]
    )
    spectral = calculate_subepoch_features(synthetic_signal, np.asarray([0.0]))
    first_channel_peak = int(np.argmax(spectral[0, 0, :5]))
    second_channel_peak = int(np.argmax(spectral[0, 0, 5:]))
    receptive_field = 1
    jump = 1
    for _ in range(3):
        receptive_field += 4 * jump
        receptive_field += 4 * jump
        receptive_field += jump
        jump *= 2
    receptive_field += 4 * jump
    receptive_field += 4 * jump
    rows = [
        {
            "check": "dense_output_shape",
            "value": str(tuple(dense.shape)),
            "expected": str((4, SEQUENCE_BINS)),
            "status": "pass" if dense.shape == (4, SEQUENCE_BINS) else "fail",
        },
        {
            "check": "candidate_probability_shape_and_range",
            "value": str(tuple(probability.shape)),
            "expected": str((4,)),
            "status": "pass"
            if probability.shape == (4,)
            and torch.isfinite(probability).all()
            and torch.logical_and(probability >= 0, probability <= 1).all()
            else "fail",
        },
        {
            "check": "all_parameter_gradients_finite",
            "value": str(bool(gradients_finite)).lower(),
            "expected": "true",
            "status": "pass" if gradients_finite else "fail",
        },
        {
            "check": "center_aggregation_width",
            "value": CENTER_STOP - CENTER_START,
            "expected": 16,
            "status": "pass" if CENTER_STOP - CENTER_START == 16 else "fail",
        },
        {
            "check": "synthetic_10hz_alpha_peak",
            "value": first_channel_peak,
            "expected": 2,
            "status": "pass" if first_channel_peak == 2 else "fail",
        },
        {
            "check": "synthetic_6hz_theta_peak",
            "value": second_channel_peak,
            "expected": 1,
            "status": "pass" if second_channel_peak == 1 else "fail",
        },
        {
            "check": "encoder_receptive_field_covers_context",
            "value": receptive_field,
            "expected": SEQUENCE_BINS,
            "status": "pass" if receptive_field >= SEQUENCE_BINS else "fail",
        },
        {
            "check": "trainable_parameter_count",
            "value": parameter_count(model),
            "expected": parameter_count(model),
            "status": "pass",
        },
    ]
    return pd.DataFrame(rows)


# Section 5: fitting and full-night scoring

def fit_model(
    sequences: np.ndarray,
    labels: np.ndarray,
    seed: int,
    name: str,
) -> tuple[SpectralUNet, np.ndarray, np.ndarray, dict]:
    configure_torch(seed)
    flattened = sequences.reshape(-1, FEATURE_COUNT).astype(np.float64)
    mean = flattened.mean(axis=0)
    scale = flattened.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError(f"Invalid fit scaler: {name}")
    scaled = ((sequences - mean) / scale).astype(np.float32).transpose(0, 2, 1)
    target = labels.astype(np.float32)
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())
    positive_weight = negative_count / positive_count
    dataset = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(target))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = SpectralUNet()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first_loss = None
    final_loss = None
    maximum_gradient = 0.0
    positive_tensor = torch.tensor(positive_weight, dtype=torch.float32)
    for epoch in range(1, TRAINING_EPOCHS + 1):
        model.train()
        losses = []
        for values, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model.candidate_logits(values)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, batch_labels, pos_weight=positive_tensor
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {name}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            maximum_gradient = max(maximum_gradient, float(gradient.detach().item()))
        mean_loss = float(np.mean(losses))
        first_loss = mean_loss if first_loss is None else first_loss
        final_loss = mean_loss
    if not all(torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError(f"Non-finite parameter: {name}")
    summary = {
        "fit": name,
        "seed": seed,
        "fit_rows": len(labels),
        "fit_positive": positive_count,
        "fit_negative": negative_count,
        "positive_weight": positive_weight,
        "trainable_parameters": parameter_count(model),
        "epochs_completed": TRAINING_EPOCHS,
        "first_epoch_loss": first_loss,
        "final_epoch_loss": final_loss,
        "maximum_preclip_gradient_norm": maximum_gradient,
    }
    return model, mean, scale, summary


def score_sequences(
    model: SpectralUNet, mean: np.ndarray, scale: np.ndarray, sequences: np.ndarray
) -> np.ndarray:
    scaled = ((sequences - mean) / scale).astype(np.float32).transpose(0, 2, 1)
    values = torch.from_numpy(scaled)
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 512):
            rows.append(model.probabilities(values[start : start + 512]))
    result = torch.cat(rows).cpu().numpy().astype(float)
    if not np.isfinite(result).all() or not np.logical_and(result >= 0, result <= 1).all():
        raise ValueError("Invalid U-Net probabilities")
    return result


def score_assignments(
    model: SpectralUNet,
    mean: np.ndarray,
    scale: np.ndarray,
    assignments: pd.DataFrame,
    recordings: dict[str, dict],
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        recording = recordings[item.subject]
        probability = score_sequences(model, mean, scale, recording_sequences(recording))
        score_rows.append(
            pd.DataFrame(
                {
                    "pipeline": "SPECTRAL-UNET",
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


def model_payload(
    model: SpectralUNet,
    mean: np.ndarray,
    scale: np.ndarray,
    summary: dict,
) -> dict:
    return {
        "configuration": {
            "version": VERSION,
            "feature_count": FEATURE_COUNT,
            "sequence_bins": SEQUENCE_BINS,
            "center_start": CENTER_START,
            "center_stop": CENTER_STOP,
            "training_epochs": TRAINING_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": GRADIENT_CLIP,
        },
        "feature_names": feature_names(),
        "scaler_mean": mean.tolist(),
        "scaler_scale": scale.tolist(),
        "fit_summary": summary,
        "state_dict": {
            key: value.detach().cpu().numpy().tolist()
            for key, value in sorted(model.state_dict().items())
        },
    }


def run_fit(
    outer: int,
    phase: str,
    seed: int,
    fit_mask: np.ndarray,
    heldout: pd.DataFrame,
    sequences: np.ndarray,
    labels: np.ndarray,
    recordings: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    score_file = score_path(outer, phase)
    model_file = model_path(outer, phase)
    if score_file.exists() and model_file.exists():
        scores = pd.read_csv(score_file, sep="\t")
        with gzip.open(model_file, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        support = (
            scores.groupby(["subject", "pid", "outer_fold"], as_index=False)
            .size()
            .rename(columns={"size": "supported_boundaries"})
        )
        support["supported_hours"] = support["supported_boundaries"] * EPOCH_SEC / 3600.0
        return scores, support, payload["fit_summary"]

    model, mean, scale, summary = fit_model(
        sequences[fit_mask], labels[fit_mask], seed, fit_name(outer, phase)
    )
    scores, support = score_assignments(
        model, mean, scale, heldout, recordings, outer, phase
    )
    verify_or_create_json_gzip(model_file, model_payload(model, mean, scale, summary))
    verify_or_create_gzip_tsv(score_file, scores)
    return scores, support, summary


# Section 6: event evaluation and participant uncertainty

def evaluate_pipelines(
    events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for pipeline in PIPELINES:
        predictions = events[events["pipeline"].eq(pipeline)]
        for membership in MEMBERSHIPS:
            eligible, ignored = local_event_inputs(references, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible,
                    predictions[["subject", "pid", "event_time_sec"]],
                    ignored,
                    support[["subject", "pid", "supported_hours"]],
                    tolerance,
                )
                config = {
                    "pipeline": pipeline,
                    "partition": "train_nested_oof",
                    "membership": membership,
                    "tolerance_sec": tolerance,
                }
                metric_rows.append({**config, **summary})
                for frame, rows in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        rows.append(local)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "recordings": pd.concat(recording_rows, ignore_index=True),
        "participants": pd.concat(participant_rows, ignore_index=True),
        "matches": pd.concat(match_rows, ignore_index=True),
    }


def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    left = primary[primary["pipeline"] == "SPECTRAL-UNET"][columns]
    right = primary[primary["pipeline"] == "NESTED-BLSTM-CRF"][columns]
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
    points = {}
    for side, frame in [("left", left), ("right", right)]:
        points[side] = metric_values(
            int(frame["true_positive"].sum()),
            int(frame["false_positive"].sum()),
            int(frame["false_negative"].sum()),
            float(frame["supported_hours"].sum()),
        )
    differences = {
        "event_f1_difference": points["left"]["f1"] - points["right"]["f1"],
        "false_alarms_per_hour_difference": points["left"]["false_alarms_per_hour"]
        - points["right"]["false_alarms_per_hour"],
    }
    return pd.DataFrame(
        [
            {
                "comparison": "SPECTRAL-UNET_minus_NESTED-BLSTM-CRF",
                "metric": metric,
                "point_difference": value,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BASE_SEED,
                "lower_95": float(sample_frame[metric].quantile(0.025)),
                "median": float(sample_frame[metric].quantile(0.5)),
                "upper_95": float(sample_frame[metric].quantile(0.975)),
            }
            for metric, value in differences.items()
        ]
    )


def decision_table(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)
    ].set_index("pipeline")
    f1_difference = float(primary.loc["SPECTRAL-UNET", "f1"] - primary.loc["NESTED-BLSTM-CRF", "f1"])
    far_difference = float(
        primary.loc["SPECTRAL-UNET", "false_alarms_per_hour"]
        - primary.loc["NESTED-BLSTM-CRF", "false_alarms_per_hour"]
    )
    intervals = bootstrap.set_index("metric")
    point_pass = f1_difference >= MEANINGFUL_F1_GAIN and far_difference <= 0.0
    direction = bool(
        intervals.loc["event_f1_difference", "lower_95"] > 0.0
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-UN1_spectral_unet_advancement",
                "f1_difference": f1_difference,
                "required_f1_difference": MEANINGFUL_F1_GAIN,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": point_pass,
                "decision": "freeze_for_new_confirmation" if point_pass else "stop_v0.1",
            },
            {
                "hypothesis": "participant_direction_support",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.0,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": direction,
                "decision": "supported" if direction else "not_supported",
            },
        ]
    )


# Section 7: complete experiment

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    checks = synthetic_checks()
    if not checks["status"].eq("pass").all():
        raise ValueError("Synthetic U-Net checks failed")

    assignments = train_assignments()
    folds = frozen_fold_assignments(assignments)
    recordings, feature_summary = prepare_recordings(assignments)
    candidates = labeled_candidates(assignments)
    sequences, construction = build_labeled_sequences(candidates, recordings)
    retained = construction[truth(construction["retained"])].reset_index(drop=True)
    labels = retained["label"].to_numpy(dtype=int)
    groups = retained["pid"].to_numpy(dtype=int)

    fit_rows = []
    threshold_curve_rows = []
    threshold_rows = []
    outer_score_rows = []
    outer_support_rows = []

    for outer in range(1, OUTER_FOLDS + 1):
        outer_pid = set(folds[folds["fold"] == outer]["pid"].astype(int))
        inner_scores = []
        inner_support = []
        for inner in sorted(set(range(1, OUTER_FOLDS + 1)) - {outer}):
            inner_pid = set(folds[folds["fold"] == inner]["pid"].astype(int))
            fit_mask = ~np.isin(groups, list(outer_pid | inner_pid))
            heldout = assignments[assignments["pid"].isin(inner_pid)].copy()
            scores, support, summary = run_fit(
                outer,
                f"inner_{inner}",
                BASE_SEED + 100 * outer + 10 * inner,
                fit_mask,
                heldout,
                sequences,
                labels,
                recordings,
            )
            fit_rows.append({"outer_fold": outer, "inner_fold": inner, "phase": f"inner_{inner}", **summary})
            inner_scores.append(scores)
            inner_support.append(support)
            print(f"completed outer {outer} inner {inner}", flush=True)

        pooled_scores = pd.concat(inner_scores, ignore_index=True)
        pooled_support = pd.concat(inner_support, ignore_index=True)
        if pooled_support["subject"].duplicated().any():
            raise ValueError(f"Duplicate inner support in outer fold {outer}")
        curve = []
        references = reference_events(assignments)
        for threshold in THRESHOLDS:
            curve.append(
                {
                    "outer_fold": outer,
                    "pipeline": "SPECTRAL-UNET",
                    "threshold": float(threshold),
                    **fast_primary_summary(
                        pooled_scores, pooled_support, references, float(threshold)
                    ),
                }
            )
        curve = pd.DataFrame(curve)
        selected = curve.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0].to_dict()
        selected["selection_rule"] = "max_f1_then_min_far_then_max_recall_then_max_threshold"
        threshold_curve_rows.append(curve)
        threshold_rows.append(selected)

        fit_mask = ~np.isin(groups, list(outer_pid))
        heldout = assignments[assignments["pid"].isin(outer_pid)].copy()
        scores, support, summary = run_fit(
            outer,
            "outer_final",
            BASE_SEED + 100 * outer + 90,
            fit_mask,
            heldout,
            sequences,
            labels,
            recordings,
        )
        fit_rows.append({"outer_fold": outer, "inner_fold": 9, "phase": "outer_final", **summary})
        outer_score_rows.append(scores)
        outer_support_rows.append(support)
        print(f"completed outer {outer} final", flush=True)

    outer_scores = pd.concat(outer_score_rows, ignore_index=True)
    outer_support = pd.concat(outer_support_rows, ignore_index=True).sort_values("subject")
    selections = pd.DataFrame(threshold_rows)
    verify_or_create_gzip_tsv(outer_score_path(), outer_scores)

    unet_events = []
    for outer in range(1, OUTER_FOLDS + 1):
        threshold = float(selections[selections["outer_fold"] == outer].iloc[0].threshold)
        unet_events.append(
            collapse_events(
                outer_scores[outer_scores["outer_fold"] == outer], threshold, "SPECTRAL-UNET"
            )
        )
    unet_events = pd.concat(unet_events, ignore_index=True)
    baseline_events = pd.read_csv(
        repo_root()
        / "experiments/2026-09-22_deep_temporal_nested_cv_v0.1"
        / "outer_predicted_events_v0.1.tsv",
        sep="\t",
    )
    baseline_events = baseline_events[baseline_events["pipeline"].eq("NESTED-BLSTM-CRF")]
    events = pd.concat([baseline_events, unet_events], ignore_index=True, sort=False)
    references = reference_events(assignments)
    evaluation = evaluate_pipelines(events, outer_support, references)
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
    output = output_dir()
    verify_or_create_tsv(checks, output / "synthetic_model_checks_v0.1.tsv")
    verify_or_create_tsv(feature_summary, output / "feature_generation_summary_v0.1.tsv")
    verify_or_create_tsv(
        pd.DataFrame({"feature_index": range(FEATURE_COUNT), "feature_name": feature_names()}),
        output / "feature_schema_v0.1.tsv",
    )
    verify_or_create_tsv(construction, output / "candidate_construction_v0.1.tsv")
    verify_or_create_tsv(pd.DataFrame(fit_rows), output / "model_fit_summary_v0.1.tsv")
    verify_or_create_tsv(
        pd.concat(threshold_curve_rows, ignore_index=True),
        output / "inner_threshold_curves_v0.1.tsv",
    )
    verify_or_create_tsv(selections, output / "inner_threshold_selections_v0.1.tsv")
    verify_or_create_tsv(outer_support, output / "outer_support_v0.1.tsv")
    verify_or_create_tsv(events, output / "outer_predicted_events_v0.1.tsv")
    verify_or_create_tsv(evaluation["metrics"], output / "outer_event_metrics_v0.1.tsv")
    verify_or_create_tsv(evaluation["recordings"], output / "outer_event_recordings_v0.1.tsv")
    verify_or_create_tsv(evaluation["participants"], output / "outer_event_participants_v0.1.tsv")
    verify_or_create_tsv(evaluation["matches"], output / "outer_event_matches_v0.1.tsv")
    verify_or_create_tsv(bootstrap, output / "paired_participant_bootstrap_v0.1.tsv")
    verify_or_create_tsv(decisions, output / "hypothesis_decisions_v0.1.tsv")
    verify_or_create_tsv(manifest, output / "external_artifact_manifest_v0.1.tsv")
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "edfio": edfio.__version__,
        "git_commit": git_commit,
        "protocol_commit": PROTOCOL_COMMIT,
    }
    verify_or_create_text(
        output / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )

    primary = evaluation["metrics"][
        (evaluation["metrics"]["membership"] == "primary")
        & (evaluation["metrics"]["tolerance_sec"] == 15.0)
    ].set_index("pipeline")
    readme = f"""# Spectral U-Net Train-Only Nested Experiment v0.1

This directory contains compact reviewed outputs for the frozen 2026-09-25 protocol.

| Pipeline | F1 | Recall | Precision | False alarms/hour |
|---|---:|---:|---:|---:|
| Nested BLSTM-CRF | {primary.loc['NESTED-BLSTM-CRF', 'f1']:.4f} | {primary.loc['NESTED-BLSTM-CRF', 'recall']:.4f} | {primary.loc['NESTED-BLSTM-CRF', 'precision']:.4f} | {primary.loc['NESTED-BLSTM-CRF', 'false_alarms_per_hour']:.4f} |
| Spectral U-Net | {primary.loc['SPECTRAL-UNET', 'f1']:.4f} | {primary.loc['SPECTRAL-UNET', 'recall']:.4f} | {primary.loc['SPECTRAL-UNET', 'precision']:.4f} | {primary.loc['SPECTRAL-UNET', 'false_alarms_per_hour']:.4f} |

Decision: `{decisions.iloc[0].decision}`.

This is reused-cohort train-only model development, not independent confirmation. Raw EDFs, generated tensors, model states, and full-night scores remain under `REM_W_data` outside Git.
"""
    verify_or_create_text(output / "README.md", readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
