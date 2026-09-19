"""Run the frozen train-only LSTM-CRF temporal-representation experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import platform
import subprocess
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reviewed_output import verify_or_create_tsv
from run_block7_transfer_validation_v0_1 import (
    context_start_indices,
    feature_path,
    labeled_candidates,
    local_event_inputs,
    reference_events,
    sha256,
    time_key,
)
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen experiment configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-19_lstm_crf_train_oof_v0.1"
DERIVED_DIR = "lstm_crf_train_oof_v0.1"
PROTOCOL_COMMIT = "2f9d961"
BASE_SEED = 20260919
FOLDS = 5
EPOCH_SEC = 30.0
SEQUENCE_LENGTH = 8
FEATURE_COUNT = 10
STATE_COUNT = 3
HIDDEN_SIZE = 16
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 128
TRAINING_EPOCHS = 60
GRADIENT_CLIP = 5.0
BOOTSTRAP_RESAMPLES = 2000
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
MEANINGFUL_F1_GAIN = 0.05
ENDPOINT_PATH = np.asarray([0, 0, 0, 1, 2, 0, 0, 0], dtype=np.int64)
BACKGROUND_PATH = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)


# Section 2: paths and immutable output helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def score_path() -> Path:
    return derived_dir() / "candidate_scores" / "train_oof_scores_v0.1.tsv.gz"


def model_path(model: str, fold: int) -> Path:
    name = model.lower().replace("-", "_")
    return derived_dir() / "models" / f"{name}_fold_{fold}_v0.1.json.gz"


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def deterministic_gzip_bytes(value: bytes) -> bytes:
    from io import BytesIO

    output = BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(value)
    return output.getvalue()


def verify_or_create_json_gzip(path: Path, payload: dict) -> None:
    value = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    expected = deterministic_gzip_bytes(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External model artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def verify_or_create_gzip_tsv(path: Path, frame: pd.DataFrame) -> None:
    value = frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")
    expected = deterministic_gzip_bytes(value)
    if path.exists():
        if path.read_bytes() != expected:
            raise RuntimeError(f"External score artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(expected)


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
    ).strip()


def canonical_state_hash(model: nn.Module) -> str:
    payload = {
        key: value.detach().cpu().numpy().tolist()
        for key, value in sorted(model.state_dict().items())
    }
    value = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


# Section 3: train membership, folds, and sequence construction

def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def train_assignments() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    source = source[source["partition"] == "train"].copy()
    rows = []
    for item in source.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    result = pd.DataFrame(rows).sort_values(
        "subject", key=lambda value: value.map(subject_number)
    )
    if len(result) != 82 or result["pid"].nunique() != 64:
        raise ValueError("Unexpected train membership")
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
    if set(folds["fold"].astype(int)) != set(range(1, FOLDS + 1)):
        raise ValueError("Unexpected fold identifiers")
    return folds


def load_recording_sequences(
    assignments: pd.DataFrame,
) -> tuple[dict[str, dict], pd.DataFrame]:
    recordings = {}
    schema = None
    for item in assignments.itertuples(index=False):
        path = feature_path(item.subject, "train", "HB-2")
        with np.load(path, allow_pickle=False) as values:
            onsets = values["onset"].astype(np.float64)
            features = values["features"].astype(np.float32)
            names = values["feature_names"].astype(str).tolist()
        if features.shape[1] != FEATURE_COUNT or not np.isfinite(features).all():
            raise ValueError(f"Invalid features for {item.subject}")
        if schema is None:
            schema = names
        elif names != schema:
            raise ValueError(f"Feature schema changed for {item.subject}")
        starts = context_start_indices(onsets)
        sequences = np.stack(
            [features[starts + offset] for offset in range(SEQUENCE_LENGTH)], axis=1
        )
        centers = onsets[starts + 4]
        recordings[item.subject] = {
            "pid": int(item.pid),
            "centers": centers,
            "sequences": sequences,
            "path": path,
        }
    schema_frame = pd.DataFrame(
        {"feature_index": np.arange(FEATURE_COUNT), "feature_name": schema}
    )
    return recordings, schema_frame


def build_labeled_sequences(
    candidates: pd.DataFrame, recordings: dict[str, dict]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    matrices = []
    tags = []
    rows = []
    for subject, group in candidates.groupby("subject", sort=True):
        recording = recordings[subject]
        lookup = {
            time_key(value): index
            for index, value in enumerate(recording["centers"])
        }
        for item in group.itertuples(index=False):
            index = lookup.get(time_key(item.candidate_time_sec))
            retained = index is not None
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
                matrices.append(recording["sequences"][index])
                tags.append(ENDPOINT_PATH if int(item.label) == 1 else BACKGROUND_PATH)
    metadata = pd.DataFrame(rows)
    retained = metadata[metadata["retained"]].reset_index(drop=True)
    if len(retained) != len(matrices):
        raise ValueError("Candidate construction accounting failed")
    return (
        np.stack(matrices).astype(np.float32),
        np.stack(tags).astype(np.int64),
        metadata,
    )


# Section 4: transparent linear-chain CRF and LSTM emissions

class LinearChainCRF(nn.Module):
    def __init__(self, state_count: int):
        super().__init__()
        self.state_count = state_count
        self.start = nn.Parameter(torch.empty(state_count))
        self.transitions = nn.Parameter(torch.empty(state_count, state_count))
        self.end = nn.Parameter(torch.empty(state_count))
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)

    def path_energy(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        batch_index = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start[tags[:, 0]]
        score = score + emissions[batch_index, 0, tags[:, 0]]
        for step in range(1, emissions.shape[1]):
            score = score + self.transitions[tags[:, step - 1], tags[:, step]]
            score = score + emissions[batch_index, step, tags[:, step]]
        return score + self.end[tags[:, -1]]

    def log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for step in range(1, emissions.shape[1]):
            scores = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)
            alpha = torch.logsumexp(scores, dim=1) + emissions[:, step]
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def negative_log_likelihood(
        self, emissions: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        return self.log_partition(emissions) - self.path_energy(emissions, tags)


class BiLstmCrf(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.emission = nn.Linear(HIDDEN_SIZE * 2, STATE_COUNT)
        self.crf = LinearChainCRF(STATE_COUNT)

    def emissions(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.emission(encoded)

    def negative_log_likelihood(
        self, values: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        return self.crf.negative_log_likelihood(self.emissions(values), tags)

    def event_probability(self, values: torch.Tensor) -> torch.Tensor:
        emissions = self.emissions(values)
        positive = torch.as_tensor(ENDPOINT_PATH, device=values.device).repeat(
            len(values), 1
        )
        negative = torch.as_tensor(BACKGROUND_PATH, device=values.device).repeat(
            len(values), 1
        )
        margin = self.crf.path_energy(emissions, positive) - self.crf.path_energy(
            emissions, negative
        )
        return torch.sigmoid(margin)


def configure_torch(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def crf_synthetic_checks() -> pd.DataFrame:
    configure_torch(BASE_SEED)
    crf = LinearChainCRF(STATE_COUNT)
    emissions = torch.randn(2, 3, STATE_COUNT, requires_grad=True)
    tags = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    paths = torch.tensor(
        list(itertools.product(range(STATE_COUNT), repeat=3)), dtype=torch.long
    )
    exhaustive = []
    for index in range(len(emissions)):
        repeated = emissions[index : index + 1].repeat(len(paths), 1, 1)
        exhaustive.append(torch.logsumexp(crf.path_energy(repeated, paths), dim=0))
    exhaustive = torch.stack(exhaustive)
    forward = crf.log_partition(emissions)
    partition_error = float(torch.max(torch.abs(exhaustive - forward)).item())
    nll = crf.negative_log_likelihood(emissions, tags)
    expected_nll = exhaustive - crf.path_energy(emissions, tags)
    nll_error = float(torch.max(torch.abs(nll - expected_nll)).item())
    nll.mean().backward()
    gradients_finite = bool(
        torch.isfinite(emissions.grad).all()
        and all(torch.isfinite(value.grad).all() for value in crf.parameters())
    )
    return pd.DataFrame(
        [
            {
                "check": "forward_partition_matches_exhaustive_enumeration",
                "value": partition_error,
                "tolerance": 1e-6,
                "status": "pass" if partition_error <= 1e-6 else "fail",
            },
            {
                "check": "negative_log_likelihood_matches_definition",
                "value": nll_error,
                "tolerance": 1e-6,
                "status": "pass" if nll_error <= 1e-6 else "fail",
            },
            {
                "check": "synthetic_gradients_finite",
                "value": float(gradients_finite),
                "tolerance": 1.0,
                "status": "pass" if gradients_finite else "fail",
            },
        ]
    )


# Section 5: fixed model fitting and scoring

def fit_lstm_crf(
    sequences: np.ndarray,
    tags: np.ndarray,
    labels: np.ndarray,
    seed: int,
    fold_name: str,
) -> tuple[BiLstmCrf, StandardScaler, pd.DataFrame]:
    configure_torch(seed)
    scaler = StandardScaler().fit(sequences.reshape(-1, FEATURE_COUNT))
    scaled = scaler.transform(sequences.reshape(-1, FEATURE_COUNT)).reshape(
        sequences.shape
    )
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())
    positive_weight = negative_count / positive_count
    weights = np.where(labels == 1, positive_weight, 1.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(scaled.astype(np.float32)),
        torch.from_numpy(tags.astype(np.int64)),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = BiLstmCrf()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    history = []
    for epoch in range(1, TRAINING_EPOCHS + 1):
        model.train()
        losses = []
        maximum_gradient_norm = 0.0
        for batch_values, batch_tags, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            losses_per_sequence = model.negative_log_likelihood(
                batch_values, batch_tags
            )
            loss = torch.sum(losses_per_sequence * batch_weights) / torch.sum(
                batch_weights
            )
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in {fold_name}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), GRADIENT_CLIP
            )
            optimizer.step()
            losses.append(float(loss.detach().item()))
            maximum_gradient_norm = max(
                maximum_gradient_norm, float(gradient_norm.detach().item())
            )
        history.append(
            {
                "fit": fold_name,
                "epoch": epoch,
                "mean_batch_loss": float(np.mean(losses)),
                "maximum_preclip_gradient_norm": maximum_gradient_norm,
            }
        )
    if not all(torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError(f"Non-finite fitted parameter in {fold_name}")
    return model, scaler, pd.DataFrame(history)


def fit_logistic(
    sequences: np.ndarray, labels: np.ndarray
) -> tuple[object, int, int]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            random_state=BASE_SEED,
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(sequences.reshape(len(sequences), -1), labels)
    warning_count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    iterations = int(model.named_steps["logisticregression"].n_iter_.max())
    return model, warning_count, iterations


def lstm_payload(
    model: BiLstmCrf, scaler: StandardScaler, fold: int, feature_names: list[str]
) -> dict:
    return {
        "model": "LC-1",
        "fold": fold,
        "configuration": {
            "hidden_size_per_direction": HIDDEN_SIZE,
            "states": STATE_COUNT,
            "training_epochs": TRAINING_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "gradient_clip": GRADIENT_CLIP,
            "seed": BASE_SEED + fold,
        },
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "state_dict": {
            key: value.detach().cpu().numpy().tolist()
            for key, value in sorted(model.state_dict().items())
        },
    }


def logistic_payload(model, fold: int, feature_names: list[str]) -> dict:
    scaler = model.named_steps["standardscaler"]
    classifier = model.named_steps["logisticregression"]
    return {
        "model": "LR-OOF",
        "fold": fold,
        "configuration": {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "max_iter": 500,
            "tol": 1e-4,
            "random_state": BASE_SEED,
        },
        "flattened_feature_names": [
            f"offset_{offset}_{name}"
            for offset in [-120, -90, -60, -30, 0, 30, 60, 90]
            for name in feature_names
        ],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(),
        "iterations": classifier.n_iter_.tolist(),
    }


def score_lstm(
    model: BiLstmCrf, scaler: StandardScaler, sequences: np.ndarray
) -> np.ndarray:
    scaled = scaler.transform(sequences.reshape(-1, FEATURE_COUNT)).reshape(
        sequences.shape
    )
    values = torch.from_numpy(scaled.astype(np.float32))
    probabilities = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            probabilities.append(model.event_probability(values[start : start + 1024]))
    return torch.cat(probabilities).cpu().numpy().astype(float)


def score_heldout_recordings(
    model_name: str,
    model,
    scaler,
    heldout: pd.DataFrame,
    recordings: dict[str, dict],
    fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        sequences = recording["sequences"]
        if model_name == "LC-1":
            probability = score_lstm(model, scaler, sequences)
        else:
            probability = model.predict_proba(sequences.reshape(len(sequences), -1))[:, 1]
        if not np.isfinite(probability).all() or not np.logical_and(
            probability >= 0.0, probability <= 1.0
        ).all():
            raise ValueError(f"Invalid probabilities for {model_name}, {item.subject}")
        score_rows.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "partition": "train_oof",
                    "fold": fold,
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
                "fold": fold,
                "supported_boundaries": len(sequences),
                "supported_hours": len(sequences) * EPOCH_SEC / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


# Section 6: event thresholds, metrics, uncertainty, and decisions

def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
    for (subject, pid), group in marked.groupby(["subject", "pid"], sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        starts = [0]
        starts.extend(
            (
                np.flatnonzero(
                    np.diff(group["candidate_time_sec"].to_numpy(dtype=float))
                    > EPOCH_SEC + 1e-6
                )
                + 1
            ).tolist()
        )
        stops = starts[1:] + [len(group)]
        for start, stop in zip(starts, stops):
            run = group.iloc[start:stop]
            maximum = float(run["probability"].max())
            best = run[np.isclose(run["probability"], maximum)].sort_values(
                "candidate_time_sec"
            ).iloc[0]
            rows.append(
                {
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(best.candidate_time_sec),
                    "probability": maximum,
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "subject",
            "pid",
            "event_time_sec",
            "probability",
            "threshold",
            "run_candidates",
        ],
    )


def threshold_curve(
    scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, ignored = local_event_inputs(references, "primary")
    curve_rows = []
    selected_rows = []
    for model_name in ["LR-OOF", "LC-1"]:
        local_scores = scores[scores["model"] == model_name]
        rows = []
        for threshold in THRESHOLDS:
            predictions = collapse_alarms(local_scores, float(threshold))
            _, _, _, summary = evaluate_events(
                eligible,
                predictions[["subject", "pid", "event_time_sec"]],
                ignored,
                support[["subject", "pid", "supported_hours"]],
                15.0,
            )
            row = {
                "model": model_name,
                "partition": "train_oof",
                "membership": "primary",
                "tolerance_sec": 15.0,
                "threshold": float(threshold),
                **summary,
            }
            rows.append(row)
            curve_rows.append(row)
        selected = pd.DataFrame(rows).sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        selected_rows.append(
            {
                **selected.to_dict(),
                "selection_rule": "max_f1_then_min_far_then_max_recall_then_max_threshold",
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(selected_rows)


def evaluate_selected(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    event_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for model_name in ["LR-OOF", "LC-1"]:
        threshold = float(selected[selected["model"] == model_name].iloc[0].threshold)
        predictions = collapse_alarms(scores[scores["model"] == model_name], threshold)
        local_events = predictions.copy()
        local_events.insert(0, "model", model_name)
        event_rows.append(local_events)
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
                configuration = {
                    "model": model_name,
                    "partition": "train_oof",
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": threshold,
                }
                metric_rows.append({**configuration, **summary})
                for frame, collection in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(configuration.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        collection.append(local)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "events": pd.concat(event_rows, ignore_index=True),
        "recordings": pd.concat(recording_rows, ignore_index=True),
        "participants": pd.concat(participant_rows, ignore_index=True),
        "matches": pd.concat(match_rows, ignore_index=True),
    }


def aggregate_participants(frame: pd.DataFrame) -> dict:
    return metric_values(
        int(frame["true_positive"].sum()),
        int(frame["false_positive"].sum()),
        int(frame["false_negative"].sum()),
        float(frame["supported_hours"].sum()),
    )


def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
    ]
    left = primary[primary["model"] == "LC-1"][columns]
    right = primary[primary["model"] == "LR-OOF"][columns]
    paired = left.merge(right, on="pid", suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != 64:
        raise ValueError("Incomplete paired train participant results")
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
    point_left = aggregate_participants(left)
    point_right = aggregate_participants(right)
    points = {
        "event_f1_difference": point_left["f1"] - point_right["f1"],
        "false_alarms_per_hour_difference": point_left["false_alarms_per_hour"]
        - point_right["false_alarms_per_hour"],
    }
    return pd.DataFrame(
        [
            {
                "comparison": "LC-1_minus_LR-OOF",
                "metric": metric,
                "point_difference": point,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BASE_SEED,
                "lower_95": float(sample_frame[metric].quantile(0.025)),
                "median": float(sample_frame[metric].quantile(0.5)),
                "upper_95": float(sample_frame[metric].quantile(0.975)),
            }
            for metric, point in points.items()
        ]
    )


def decision_table(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["membership"] == "primary")
        & (metrics["tolerance_sec"] == 15.0)
    ].set_index("model")
    f1_difference = float(primary.loc["LC-1", "f1"] - primary.loc["LR-OOF", "f1"])
    far_difference = float(
        primary.loc["LC-1", "false_alarms_per_hour"]
        - primary.loc["LR-OOF", "false_alarms_per_hour"]
    )
    point_pass = f1_difference >= MEANINGFUL_F1_GAIN and far_difference <= 0.0
    intervals = bootstrap.set_index("metric")
    participant_supported = bool(
        intervals.loc["event_f1_difference", "lower_95"] > 0.0
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-LC1_meaningful_temporal_advancement",
                "f1_difference": f1_difference,
                "required_f1_difference": MEANINGFUL_F1_GAIN,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": point_pass,
                "decision": "eligible_for_new_confirmation" if point_pass else "stop_lc1_v0.1",
            },
            {
                "hypothesis": "participant_interval_support",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.0,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": participant_supported,
                "decision": "participant_supported" if participant_supported else "point_result_only",
            },
        ]
    )


# Section 7: full experiment

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match the current checkout")

    synthetic = crf_synthetic_checks()
    if not synthetic["status"].eq("pass").all():
        raise ValueError("CRF synthetic checks failed")
    assignments = train_assignments()
    folds = frozen_fold_assignments(assignments)
    recordings, feature_schema = load_recording_sequences(assignments)
    candidates = labeled_candidates(assignments)
    sequences, tags, construction = build_labeled_sequences(candidates, recordings)
    retained = construction[construction["retained"]].reset_index(drop=True)
    labels = retained["label"].to_numpy(dtype=int)
    groups = retained["pid"].to_numpy(dtype=int)
    feature_names = feature_schema["feature_name"].tolist()

    score_frames = []
    support_frames = []
    fit_rows = []
    history_frames = []
    leakage_pass = True
    determinism_rows = []

    for fold in range(1, FOLDS + 1):
        heldout_pids = set(folds.loc[folds["fold"] == fold, "pid"].astype(int))
        fit_indices = np.flatnonzero(~np.isin(groups, list(heldout_pids)))
        heldout_indices = np.flatnonzero(np.isin(groups, list(heldout_pids)))
        fit_pids = set(groups[fit_indices])
        leakage_pass = leakage_pass and not bool(fit_pids & heldout_pids)
        heldout_assignments = assignments[assignments["pid"].isin(heldout_pids)]
        print(
            f"Fold {fold}/{FOLDS}: {len(fit_pids)} fit pid, "
            f"{len(heldout_pids)} held-out pid",
            flush=True,
        )

        seed = BASE_SEED + fold
        lstm, sequence_scaler, history = fit_lstm_crf(
            sequences[fit_indices],
            tags[fit_indices],
            labels[fit_indices],
            seed,
            f"LC-1_fold_{fold}",
        )
        history.insert(1, "fold", fold)
        history_frames.append(history)
        lstm_artifact = model_path("LC-1", fold)
        verify_or_create_json_gzip(
            lstm_artifact, lstm_payload(lstm, sequence_scaler, fold, feature_names)
        )
        lstm_scores, support = score_heldout_recordings(
            "LC-1",
            lstm,
            sequence_scaler,
            heldout_assignments,
            recordings,
            fold,
        )
        score_frames.append(lstm_scores)
        support_frames.append(support)

        logistic, warning_count, iterations = fit_logistic(
            sequences[fit_indices], labels[fit_indices]
        )
        logistic_artifact = model_path("LR-OOF", fold)
        verify_or_create_json_gzip(
            logistic_artifact, logistic_payload(logistic, fold, feature_names)
        )
        logistic_scores, _ = score_heldout_recordings(
            "LR-OOF", logistic, None, heldout_assignments, recordings, fold
        )
        score_frames.append(logistic_scores)

        common = {
            "fold": fold,
            "fit_pid": len(fit_pids),
            "heldout_pid": len(heldout_pids),
            "fit_candidates": len(fit_indices),
            "fit_positive": int(labels[fit_indices].sum()),
            "fit_negative": int((labels[fit_indices] == 0).sum()),
            "heldout_candidates": len(heldout_indices),
        }
        fit_rows.extend(
            [
                {
                    "model": "LC-1",
                    **common,
                    "seed": seed,
                    "training_epochs": TRAINING_EPOCHS,
                    "convergence_warning_count": 0,
                    "maximum_iterations_used": np.nan,
                    "model_sha256": sha256(lstm_artifact),
                },
                {
                    "model": "LR-OOF",
                    **common,
                    "seed": BASE_SEED,
                    "training_epochs": np.nan,
                    "convergence_warning_count": warning_count,
                    "maximum_iterations_used": iterations,
                    "model_sha256": sha256(logistic_artifact),
                },
            ]
        )

        if fold == 1:
            duplicate, duplicate_scaler, _ = fit_lstm_crf(
                sequences[fit_indices],
                tags[fit_indices],
                labels[fit_indices],
                seed,
                "LC-1_fold_1_duplicate",
            )
            original_probability = score_lstm(
                lstm, sequence_scaler, sequences[heldout_indices[:32]]
            )
            duplicate_probability = score_lstm(
                duplicate, duplicate_scaler, sequences[heldout_indices[:32]]
            )
            probability_difference = float(
                np.max(np.abs(original_probability - duplicate_probability))
            )
            state_equal = canonical_state_hash(lstm) == canonical_state_hash(duplicate)
            scaler_difference = float(
                max(
                    np.max(np.abs(sequence_scaler.mean_ - duplicate_scaler.mean_)),
                    np.max(np.abs(sequence_scaler.scale_ - duplicate_scaler.scale_)),
                )
            )
            determinism_rows.append(
                {
                    "fold": 1,
                    "state_hash_equal": state_equal,
                    "maximum_scaler_difference": scaler_difference,
                    "maximum_probability_difference": probability_difference,
                    "status": "pass"
                    if state_equal
                    and scaler_difference <= 1e-12
                    and probability_difference <= 1e-12
                    else "fail",
                }
            )

    scores = pd.concat(score_frames, ignore_index=True).sort_values(
        ["model", "subject", "candidate_time_sec"]
    )
    support = pd.concat(support_frames, ignore_index=True).sort_values("subject")
    verify_or_create_gzip_tsv(score_path(), scores)
    references = reference_events(assignments)
    curve, selected = threshold_curve(scores, support, references)
    outputs = evaluate_selected(scores, support, references, selected)
    bootstrap = paired_bootstrap(outputs["participants"])
    decisions = decision_table(outputs["metrics"], bootstrap)
    fit_record = pd.DataFrame(fit_rows)
    history = pd.concat(history_frames, ignore_index=True)
    determinism = pd.DataFrame(determinism_rows)

    external_paths = [
        ("source_train_feature", recording["path"])
        for recording in recordings.values()
    ]
    external_paths.extend(
        ("fold_model", model_path(model_name, fold))
        for model_name in ["LC-1", "LR-OOF"]
        for fold in range(1, FOLDS + 1)
    )
    external_paths.append(("train_oof_scores", score_path()))
    manifest = pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in external_paths
        ]
    ).sort_values(["artifact_role", "path_relative_to_data_parent"])

    folds_per_pid = folds.groupby("pid").size()
    model_subject_counts = scores.groupby("model")["subject"].nunique()
    model_row_counts = scores.groupby("model").size()
    paired_support = (
        scores.groupby(["model", "subject"]).size().unstack("model").nunique(axis=1)
    )
    checks = pd.DataFrame(
        [
            {
                "check": "train_membership",
                "status": "pass"
                if len(assignments) == 82 and assignments["pid"].nunique() == 64
                else "fail",
                "detail": "82 recordings; 64 pid groups",
            },
            {
                "check": "validation_and_test_closed",
                "status": "pass"
                if assignments["partition"].eq("train").all()
                and scores["partition"].eq("train_oof").all()
                and not manifest["path_relative_to_data_parent"]
                .str.contains(
                    "/validation/|/test/|(?:^|/)test_", case=False, regex=True
                )
                .any()
                else "fail",
                "detail": "train inputs and train_oof scores only",
            },
            {
                "check": "frozen_fold_reproduction",
                "status": "pass" if len(folds) == 64 else "fail",
                "detail": "reused Block 8 five-fold assignment",
            },
            {
                "check": "each_pid_held_out_once",
                "status": "pass" if folds_per_pid.eq(1).all() else "fail",
                "detail": "64/64 pid groups",
            },
            {
                "check": "no_participant_leakage",
                "status": "pass" if leakage_pass else "fail",
                "detail": "fit and held-out pid sets disjoint in five folds",
            },
            {
                "check": "feature_schema_and_values",
                "status": "pass"
                if len(feature_schema) == FEATURE_COUNT
                and np.isfinite(sequences).all()
                and sequences.shape[1:] == (SEQUENCE_LENGTH, FEATURE_COUNT)
                else "fail",
                "detail": f"shape={sequences.shape}",
            },
            {
                "check": "candidate_accounting",
                "status": "pass"
                if len(construction) == 2743
                and int(construction["retained"].sum()) == 2743
                and int(retained["label"].sum()) == 180
                else "fail",
                "detail": f"total={len(construction)}; retained={len(retained)}; positive={int(retained['label'].sum())}",
            },
            {
                "check": "crf_synthetic_math",
                "status": "pass" if synthetic["status"].eq("pass").all() else "fail",
                "detail": f"{synthetic['status'].eq('pass').sum()}/{len(synthetic)} checks",
            },
            {
                "check": "deterministic_duplicate_fit",
                "status": "pass" if determinism["status"].eq("pass").all() else "fail",
                "detail": f"maximum probability difference={determinism['maximum_probability_difference'].max():.12g}",
            },
            {
                "check": "complete_finite_oof_scores",
                "status": "pass"
                if model_subject_counts.eq(82).all()
                and model_row_counts.nunique() == 1
                and paired_support.eq(1).all()
                and np.isfinite(scores["probability"]).all()
                else "fail",
                "detail": f"rows per model={model_row_counts.to_dict()}",
            },
            {
                "check": "fixed_training_completion",
                "status": "pass"
                if len(history) == FOLDS * TRAINING_EPOCHS
                and history.groupby("fold")["epoch"].max().eq(TRAINING_EPOCHS).all()
                else "fail",
                "detail": f"{FOLDS} folds x {TRAINING_EPOCHS} epochs",
            },
            {
                "check": "threshold_selection",
                "status": "pass"
                if curve.groupby("model").size().eq(99).all() and len(selected) == 2
                else "fail",
                "detail": "99 thresholds per model",
            },
            {
                "check": "paired_participant_support",
                "status": "pass"
                if outputs["participants"]
                .query("membership == 'primary' and tolerance_sec == 15.0")
                .groupby("model")["pid"]
                .nunique()
                .eq(64)
                .all()
                else "fail",
                "detail": "64 paired train pid groups",
            },
            {
                "check": "external_artifact_manifest",
                "status": "pass"
                if len(manifest) == 93 and manifest["sha256"].str.len().eq(64).all()
                else "fail",
                "detail": f"{len(manifest)} source, model, and score artifacts",
            },
        ]
    )

    reviewed = {
        "feature_schema_v0.1.tsv": feature_schema,
        "candidate_construction_v0.1.tsv": construction,
        "train_oof_fold_assignments_v0.1.tsv": folds,
        "crf_synthetic_checks_v0.1.tsv": synthetic,
        "deterministic_fit_check_v0.1.tsv": determinism,
        "model_fit_record_v0.1.tsv": fit_record,
        "lstm_training_history_v0.1.tsv": history,
        "train_oof_support_v0.1.tsv": support,
        "train_oof_threshold_curve_v0.1.tsv": curve,
        "train_oof_threshold_selection_v0.1.tsv": selected,
        "train_oof_predicted_events_v0.1.tsv": outputs["events"],
        "train_oof_event_metrics_v0.1.tsv": outputs["metrics"],
        "train_oof_event_recordings_v0.1.tsv": outputs["recordings"],
        "train_oof_event_participants_v0.1.tsv": outputs["participants"],
        "train_oof_event_matches_v0.1.tsv": outputs["matches"],
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decisions,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "output_integrity_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)

    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
    }
    verify_or_create_text(
        output_dir() / "software_versions_v0.1.json",
        json.dumps(software, indent=2, sort_keys=True) + "\n",
    )

    primary = outputs["metrics"][
        (outputs["metrics"]["membership"] == "primary")
        & (outputs["metrics"]["tolerance_sec"] == 15.0)
    ]
    metric_lines = [
        f"| {item.model} | {item.threshold:.2f} | {item.precision:.4f} | "
        f"{item.recall:.4f} | {item.f1:.4f} | {item.false_alarms_per_hour:.4f} |"
        for item in primary.itertuples(index=False)
    ]
    decision = decisions.iloc[0]
    readme = "\n".join(
        [
            "# Train-Only LSTM-CRF Temporal Representation v0.1",
            "",
            "**Work date:** 2026-09-19",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Dataset:** BOAS OpenNeuro `ds005555`, snapshot `1.1.1`",
            "**Evaluation role:** Train-only participant-grouped developmental screen",
            "**Validation accessed:** No",
            "**Test accessed:** No",
            "",
            "## Primary Train-OOF Result",
            "",
            "| Model | Threshold | Precision | Recall | F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|---:|",
            *metric_lines,
            "",
            "## Frozen Decision",
            "",
            f"LC-1 minus LR-OOF event F1: `{decision.f1_difference:+.4f}` "
            f"(required at least `+{MEANINGFUL_F1_GAIN:.2f}`).",
            f"LC-1 minus LR-OOF false alarms/hour: `{decision.false_alarms_per_hour_difference:+.4f}` "
            "(required no increase).",
            f"Decision: **{str(decision.decision).replace('_', ' ')}**.",
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. "
            "Thresholds were selected on pooled out-of-fold predictions from the same 64 train participants. "
            "This result is developmental and is not independent confirmation.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", readme)

    print(primary[["model", "threshold", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(decisions.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one in-run check failed")


# Section 8: command entry point

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
