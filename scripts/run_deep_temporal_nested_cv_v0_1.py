"""Run the frozen nested train-only deep temporal architecture comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from reviewed_output import verify_or_create_tsv
from run_lstm_crf_train_oof_v0_1 import (
    BACKGROUND_PATH,
    BATCH_SIZE,
    ENDPOINT_PATH,
    FEATURE_COUNT,
    GRADIENT_CLIP,
    HIDDEN_SIZE,
    LEARNING_RATE,
    SEQUENCE_LENGTH,
    STATE_COUNT,
    TRAINING_EPOCHS,
    WEIGHT_DECAY,
    LinearChainCRF,
    build_labeled_sequences,
    canonical_state_hash,
    configure_torch,
    data_parent,
    deterministic_gzip_bytes,
    frozen_fold_assignments,
    load_recording_sequences,
    repo_root,
    train_assignments,
    verify_or_create_gzip_tsv,
    verify_or_create_json_gzip,
    verify_or_create_text,
)
from run_block7_transfer_validation_v0_1 import (
    labeled_candidates,
    local_event_inputs,
    reference_events,
    sha256,
)
from stage_first_event_evaluation_v0_1 import (
    evaluate_events,
    metric_values,
    optimal_matches,
)


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-22_deep_temporal_nested_cv_v0.1"
DERIVED_DIR = "deep_temporal_nested_cv_v0.1"
PROTOCOL_COMMIT = "533cad3"
BASE_SEED = 20260922
OUTER_FOLDS = 5
INNER_FOLDS = 4
BOOTSTRAP_RESAMPLES = 2000
EPOCH_SEC = 30.0
MEANINGFUL_F1_GAIN = 0.05
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
CANDIDATES = ["BLSTM-CRF", "BGRU-CRF", "TCN-CRF", "BLSTM-2H"]


def threshold_grid() -> np.ndarray:
    coarse = np.arange(0.01, 0.951, 0.05)
    logits = np.arange(3.0, 14.001, 0.25)
    tail = 1.0 / (1.0 + np.exp(-logits))
    return np.unique(np.concatenate([coarse, tail]))


THRESHOLDS = threshold_grid()


# Section 2: paths and small helpers

def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def result_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def model_path(candidate: str, outer: int, phase: str) -> Path:
    name = candidate.lower().replace("-", "_")
    return result_dir() / "models" / f"outer_{outer}" / f"{name}_{phase}_v0.1.json.gz"


def score_path(candidate: str, outer: int, phase: str) -> Path:
    name = candidate.lower().replace("-", "_")
    return result_dir() / "scores" / f"outer_{outer}" / f"{name}_{phase}_v0.1.tsv.gz"


def pooled_outer_score_path(pipeline: str) -> Path:
    name = pipeline.lower().replace("-", "_")
    return result_dir() / "scores" / f"{name}_outer_scores_v0.1.tsv.gz"


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root(), text=True
    ).strip()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def parameter_count(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def model_payload(
    model: nn.Module,
    scaler: StandardScaler,
    candidate: str,
    outer: int,
    phase: str,
    seed: int,
    feature_names: list[str],
) -> dict:
    return {
        "candidate": candidate,
        "outer_fold": outer,
        "phase": phase,
        "seed": seed,
        "parameter_count": parameter_count(model),
        "training_epochs": TRAINING_EPOCHS,
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "state_dict": {
            key: value.detach().cpu().numpy().tolist()
            for key, value in sorted(model.state_dict().items())
        },
    }


# Section 3: fixed candidate architectures

class RecurrentCrf(nn.Module):
    def __init__(self, recurrent: str):
        super().__init__()
        layer = nn.LSTM if recurrent == "lstm" else nn.GRU
        self.encoder = layer(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.emission = nn.Linear(HIDDEN_SIZE * 2, STATE_COUNT)
        self.crf = LinearChainCRF(STATE_COUNT)

    def emissions(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values)
        return self.emission(encoded)

    def losses(self, values: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return self.crf.negative_log_likelihood(self.emissions(values), tags)

    def probabilities(self, values: torch.Tensor) -> torch.Tensor:
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


class ResidualTemporalBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.convolution = nn.Conv1d(
            32, 32, kernel_size=3, dilation=dilation, padding=dilation
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.10)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.dropout(self.activation(self.convolution(values)))


class TemporalConvCrf(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Conv1d(FEATURE_COUNT, 32, kernel_size=1)
        self.block_1 = ResidualTemporalBlock(1)
        self.block_2 = ResidualTemporalBlock(2)
        self.block_3 = ResidualTemporalBlock(4)
        self.emission = nn.Linear(32, STATE_COUNT)
        self.crf = LinearChainCRF(STATE_COUNT)

    def emissions(self, values: torch.Tensor) -> torch.Tensor:
        encoded = values.transpose(1, 2)
        encoded = self.input_projection(encoded)
        encoded = self.block_1(encoded)
        encoded = self.block_2(encoded)
        encoded = self.block_3(encoded)
        return self.emission(encoded.transpose(1, 2))

    def losses(self, values: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return self.crf.negative_log_likelihood(self.emissions(values), tags)

    def probabilities(self, values: torch.Tensor) -> torch.Tensor:
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


class RecurrentTwoHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=FEATURE_COUNT,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.pre_head = nn.Linear(HIDDEN_SIZE * 2, 1)
        self.post_head = nn.Linear(HIDDEN_SIZE * 2, 1)

    def logits(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(values)
        return (
            self.pre_head(encoded[:, 3]).squeeze(1),
            self.post_head(encoded[:, 4]).squeeze(1),
        )

    def probabilities(self, values: torch.Tensor) -> torch.Tensor:
        pre, post = self.logits(values)
        return torch.sigmoid(pre) * torch.sigmoid(post)


def create_model(candidate: str) -> nn.Module:
    if candidate == "BLSTM-CRF":
        return RecurrentCrf("lstm")
    if candidate == "BGRU-CRF":
        return RecurrentCrf("gru")
    if candidate == "TCN-CRF":
        return TemporalConvCrf()
    if candidate == "BLSTM-2H":
        return RecurrentTwoHead()
    raise ValueError(f"Unknown candidate: {candidate}")


def candidate_configuration() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        configure_torch(BASE_SEED)
        model = create_model(candidate)
        rows.append(
            {
                "candidate": candidate,
                "encoder": {
                    "BLSTM-CRF": "bidirectional_lstm",
                    "BGRU-CRF": "bidirectional_gru",
                    "TCN-CRF": "residual_dilated_temporal_convolution",
                    "BLSTM-2H": "bidirectional_lstm",
                }[candidate],
                "output": "two_endpoint_heads"
                if candidate == "BLSTM-2H"
                else "three_state_linear_chain_crf",
                "trainable_parameters": parameter_count(model),
                "epochs": TRAINING_EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "batch_size": BATCH_SIZE,
            }
        )
    return pd.DataFrame(rows)


# Section 4: model fitting and full-night scoring

def fit_candidate(
    candidate: str,
    sequences: np.ndarray,
    tags: np.ndarray,
    labels: np.ndarray,
    seed: int,
    fit_name: str,
) -> tuple[nn.Module, StandardScaler, dict]:
    configure_torch(seed)
    scaler = StandardScaler().fit(sequences.reshape(-1, FEATURE_COUNT))
    scaled = scaler.transform(sequences.reshape(-1, FEATURE_COUNT)).reshape(
        sequences.shape
    )
    negative_count = int((labels == 0).sum())
    positive_count = int((labels == 1).sum())
    positive_weight = negative_count / positive_count
    sample_weights = np.where(labels == 1, positive_weight, 1.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(scaled.astype(np.float32)),
        torch.from_numpy(tags.astype(np.int64)),
        torch.from_numpy(labels.astype(np.float32)),
        torch.from_numpy(sample_weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = create_model(candidate)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    first_loss = math.nan
    final_loss = math.nan
    maximum_gradient = 0.0
    for epoch in range(1, TRAINING_EPOCHS + 1):
        model.train()
        epoch_losses = []
        for values, batch_tags, batch_labels, weights in loader:
            optimizer.zero_grad(set_to_none=True)
            if candidate == "BLSTM-2H":
                pre, post = model.logits(values)
                pre_loss = nn.functional.binary_cross_entropy_with_logits(
                    pre, batch_labels, reduction="none"
                )
                post_loss = nn.functional.binary_cross_entropy_with_logits(
                    post, batch_labels, reduction="none"
                )
                per_sequence = pre_loss + post_loss
            else:
                per_sequence = model.losses(values, batch_tags)
            loss = torch.sum(per_sequence * weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {fit_name}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))
            maximum_gradient = max(maximum_gradient, float(gradient.detach().item()))
        mean_loss = float(np.mean(epoch_losses))
        if epoch == 1:
            first_loss = mean_loss
        final_loss = mean_loss
    if not all(torch.isfinite(value).all() for value in model.parameters()):
        raise ValueError(f"Non-finite parameter: {fit_name}")
    summary = {
        "fit": fit_name,
        "candidate": candidate,
        "seed": seed,
        "fit_rows": len(labels),
        "fit_positive": int(labels.sum()),
        "fit_negative": int((labels == 0).sum()),
        "trainable_parameters": parameter_count(model),
        "epochs_completed": TRAINING_EPOCHS,
        "first_epoch_loss": first_loss,
        "final_epoch_loss": final_loss,
        "maximum_preclip_gradient_norm": maximum_gradient,
    }
    return model, scaler, summary


def score_model(model: nn.Module, scaler: StandardScaler, sequences: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(sequences.reshape(-1, FEATURE_COUNT)).reshape(
        sequences.shape
    )
    values = torch.from_numpy(scaled.astype(np.float32))
    rows = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            rows.append(model.probabilities(values[start : start + 1024]))
    result = torch.cat(rows).cpu().numpy().astype(float)
    if not np.isfinite(result).all() or not np.logical_and(result >= 0, result <= 1).all():
        raise ValueError("Invalid model probabilities")
    return result


def score_assignments(
    candidate: str,
    model: nn.Module,
    scaler: StandardScaler,
    assignments: pd.DataFrame,
    recordings: dict[str, dict],
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        recording = recordings[item.subject]
        probabilities = score_model(model, scaler, recording["sequences"])
        score_rows.append(
            pd.DataFrame(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["centers"],
                    "probability": probabilities,
                }
            )
        )
        support_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(probabilities),
                "supported_hours": len(probabilities) * EPOCH_SEC / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


# Section 5: efficient frozen threshold selection

def collapsed_times(group: pd.DataFrame, threshold: float) -> np.ndarray:
    times = group["candidate_time_sec"].to_numpy(dtype=float)
    probabilities = group["probability"].to_numpy(dtype=float)
    selected = np.flatnonzero(probabilities >= threshold)
    if len(selected) == 0:
        return np.asarray([], dtype=float)
    splits = np.flatnonzero(np.diff(times[selected]) > EPOCH_SEC + 1e-6) + 1
    runs = np.split(selected, splits)
    result = []
    for run in runs:
        local = probabilities[run]
        maximum = local.max()
        best = run[np.flatnonzero(np.isclose(local, maximum))[0]]
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
        for subject, group in scores.groupby("subject", sort=False)
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
        unmatched_indices = [
            index for index in range(len(predictions)) if index not in matched_predictions
        ]
        ignored_matches = optimal_matches(
            ignored_refs, predictions[unmatched_indices], 15.0
        )
        tp += len(matches)
        fn += len(refs) - len(matches)
        fp += len(unmatched_indices) - len(ignored_matches)
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


def select_threshold(
    candidate: str,
    outer: int,
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    for threshold in THRESHOLDS:
        rows.append(
            {
                "outer_fold": outer,
                "candidate": candidate,
                "threshold": float(threshold),
                **fast_primary_summary(scores, support, references, float(threshold)),
            }
        )
    curve = pd.DataFrame(rows)
    selected = curve.sort_values(
        ["f1", "false_alarms_per_hour", "recall", "threshold"],
        ascending=[False, True, False, False],
        kind="stable",
    ).iloc[0].to_dict()
    return curve, selected


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
            maximum = local.max()
            best = run[np.flatnonzero(np.isclose(local, maximum))[0]]
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


# Section 6: outer evaluation and uncertainty

def evaluate_pipelines(
    events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for pipeline in ["NESTED-BLSTM-CRF", "NESTED-SELECTED"]:
        predictions = events[events["pipeline"] == pipeline]
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
                for frame, collection in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        collection.append(local)
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
    columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
    ]
    left = primary[primary["pipeline"] == "NESTED-SELECTED"][columns]
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
    samples = pd.DataFrame(samples)
    point = {}
    for side, frame in [("left", left), ("right", right)]:
        point[side] = metric_values(
            int(frame["true_positive"].sum()),
            int(frame["false_positive"].sum()),
            int(frame["false_negative"].sum()),
            float(frame["supported_hours"].sum()),
        )
    points = {
        "event_f1_difference": point["left"]["f1"] - point["right"]["f1"],
        "false_alarms_per_hour_difference": point["left"]["false_alarms_per_hour"]
        - point["right"]["false_alarms_per_hour"],
    }
    return pd.DataFrame(
        [
            {
                "comparison": "NESTED-SELECTED_minus_NESTED-BLSTM-CRF",
                "metric": metric,
                "point_difference": value,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BASE_SEED,
                "lower_95": float(samples[metric].quantile(0.025)),
                "median": float(samples[metric].quantile(0.5)),
                "upper_95": float(samples[metric].quantile(0.975)),
            }
            for metric, value in points.items()
        ]
    )


def decisions(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["membership"] == "primary")
        & (metrics["tolerance_sec"] == 15.0)
    ].set_index("pipeline")
    f1_difference = float(
        primary.loc["NESTED-SELECTED", "f1"]
        - primary.loc["NESTED-BLSTM-CRF", "f1"]
    )
    far_difference = float(
        primary.loc["NESTED-SELECTED", "false_alarms_per_hour"]
        - primary.loc["NESTED-BLSTM-CRF", "false_alarms_per_hour"]
    )
    point_pass = f1_difference >= MEANINGFUL_F1_GAIN and far_difference <= 0.0
    intervals = bootstrap.set_index("metric")
    direction_supported = bool(
        intervals.loc["event_f1_difference", "lower_95"] > 0.0
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
    )
    material_supported = bool(
        intervals.loc["event_f1_difference", "lower_95"] >= MEANINGFUL_F1_GAIN
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-DT1_nested_architecture_advancement",
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
                "supported": direction_supported,
                "decision": "supported" if direction_supported else "not_supported",
            },
            {
                "hypothesis": "participant_material_support",
                "f1_difference": f1_difference,
                "required_f1_difference": MEANINGFUL_F1_GAIN,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": material_supported,
                "decision": "supported" if material_supported else "not_supported",
            },
        ]
    )


# Section 7: synthetic and deterministic controls

def synthetic_checks() -> pd.DataFrame:
    configure_torch(BASE_SEED)
    values = torch.randn(5, SEQUENCE_LENGTH, FEATURE_COUNT)
    rows = []
    for candidate in CANDIDATES:
        model = create_model(candidate)
        probability = model.probabilities(values)
        rows.append(
            {
                "check": f"{candidate}_shape_and_probability",
                "value": len(probability),
                "expected": len(values),
                "status": "pass"
                if probability.shape == (len(values),)
                and torch.isfinite(probability).all()
                and torch.logical_and(probability >= 0, probability <= 1).all()
                else "fail",
            }
        )
    tcn = create_model("TCN-CRF")
    impulse = torch.zeros(1, SEQUENCE_LENGTH, FEATURE_COUNT)
    impulse[:, 0, 0] = 1.0
    with torch.no_grad():
        encoded = tcn.block_3(
            tcn.block_2(tcn.block_1(tcn.input_projection(impulse.transpose(1, 2))))
        )
    rows.append(
        {
            "check": "TCN_complete_sequence_length_preserved",
            "value": encoded.shape[-1],
            "expected": SEQUENCE_LENGTH,
            "status": "pass" if encoded.shape[-1] == SEQUENCE_LENGTH else "fail",
        }
    )
    return pd.DataFrame(rows)


# Section 8: complete nested experiment

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    checks_synthetic = synthetic_checks()
    if not checks_synthetic["status"].eq("pass").all():
        raise ValueError("Synthetic candidate checks failed")
    assignments = train_assignments()
    folds = frozen_fold_assignments(assignments)
    recordings, feature_schema = load_recording_sequences(assignments)
    candidates = labeled_candidates(assignments)
    sequences, tags, construction = build_labeled_sequences(candidates, recordings)
    retained = construction[truth(construction["retained"])].reset_index(drop=True)
    labels = retained["label"].to_numpy(dtype=int)
    groups = retained["pid"].to_numpy(dtype=int)
    feature_names = feature_schema["feature_name"].tolist()
    configuration = candidate_configuration()
    parameter_map = configuration.set_index("candidate")["trainable_parameters"].to_dict()

    fit_rows = []
    determinism_rows = []
    inner_curve_rows = []
    inner_selection_rows = []
    architecture_rows = []
    outer_score_rows = {"NESTED-BLSTM-CRF": [], "NESTED-SELECTED": []}
    outer_event_rows = []
    outer_support_rows = []
    external_paths = []
    leakage_pass = True

    for outer in range(1, OUTER_FOLDS + 1):
        print(f"Outer fold {outer}/{OUTER_FOLDS}", flush=True)
        outer_heldout_pids = set(folds.loc[folds["fold"] == outer, "pid"].astype(int))
        outer_fit_pids = set(folds.loc[folds["fold"] != outer, "pid"].astype(int))
        leakage_pass = leakage_pass and not bool(outer_heldout_pids & outer_fit_pids)
        candidate_inner_scores = {}
        candidate_inner_support = None
        candidate_thresholds = {}

        for candidate in CANDIDATES:
            score_frames = []
            support_frames = []
            for inner_fold_id in sorted(set(range(1, OUTER_FOLDS + 1)) - {outer}):
                inner_heldout_pids = set(
                    folds.loc[folds["fold"] == inner_fold_id, "pid"].astype(int)
                )
                inner_fit_pids = outer_fit_pids - inner_heldout_pids
                leakage_pass = leakage_pass and not bool(inner_fit_pids & inner_heldout_pids)
                fit_indices = np.flatnonzero(np.isin(groups, list(inner_fit_pids)))
                heldout_assignments = assignments[
                    assignments["pid"].isin(inner_heldout_pids)
                ]
                seed = BASE_SEED + 100 * outer + 10 * inner_fold_id
                fit_name = f"outer_{outer}_inner_{inner_fold_id}"
                model, scaler, summary = fit_candidate(
                    candidate,
                    sequences[fit_indices],
                    tags[fit_indices],
                    labels[fit_indices],
                    seed,
                    fit_name,
                )
                summary.update(
                    {
                        "outer_fold": outer,
                        "inner_fold": inner_fold_id,
                        "phase": "inner",
                        "fit_pid": len(inner_fit_pids),
                        "heldout_pid": len(inner_heldout_pids),
                    }
                )
                artifact = model_path(candidate, outer, f"inner_{inner_fold_id}")
                verify_or_create_json_gzip(
                    artifact,
                    model_payload(
                        model,
                        scaler,
                        candidate,
                        outer,
                        f"inner_{inner_fold_id}",
                        seed,
                        feature_names,
                    ),
                )
                summary["model_sha256"] = sha256(artifact)
                fit_rows.append(summary)
                external_paths.append(("inner_model", artifact))
                scores, support = score_assignments(
                    candidate,
                    model,
                    scaler,
                    heldout_assignments,
                    recordings,
                    outer,
                    f"inner_{inner_fold_id}",
                )
                score_frames.append(scores)
                support_frames.append(support)

                if outer == 1 and inner_fold_id == 2:
                    duplicate, duplicate_scaler, _ = fit_candidate(
                        candidate,
                        sequences[fit_indices],
                        tags[fit_indices],
                        labels[fit_indices],
                        seed,
                        fit_name + "_duplicate",
                    )
                    probe = sequences[
                        np.flatnonzero(np.isin(groups, list(inner_heldout_pids)))[:32]
                    ]
                    original_probability = score_model(model, scaler, probe)
                    duplicate_probability = score_model(duplicate, duplicate_scaler, probe)
                    determinism_rows.append(
                        {
                            "candidate": candidate,
                            "state_hash_equal": canonical_state_hash(model)
                            == canonical_state_hash(duplicate),
                            "maximum_scaler_difference": float(
                                max(
                                    np.max(np.abs(scaler.mean_ - duplicate_scaler.mean_)),
                                    np.max(np.abs(scaler.scale_ - duplicate_scaler.scale_)),
                                )
                            ),
                            "maximum_probability_difference": float(
                                np.max(
                                    np.abs(original_probability - duplicate_probability)
                                )
                            ),
                        }
                    )

            inner_scores = pd.concat(score_frames, ignore_index=True)
            inner_support = pd.concat(support_frames, ignore_index=True).sort_values("subject")
            inner_score_artifact = score_path(candidate, outer, "inner_oof")
            verify_or_create_gzip_tsv(inner_score_artifact, inner_scores)
            external_paths.append(("inner_oof_scores", inner_score_artifact))
            outer_fit_assignments = assignments[assignments["pid"].isin(outer_fit_pids)]
            curve, selected = select_threshold(
                candidate,
                outer,
                inner_scores,
                inner_support,
                reference_events(outer_fit_assignments),
            )
            inner_curve_rows.append(curve)
            inner_selection_rows.append(selected)
            candidate_inner_scores[candidate] = inner_scores
            candidate_inner_support = inner_support
            candidate_thresholds[candidate] = float(selected["threshold"])

        local_selections = pd.DataFrame(
            [row for row in inner_selection_rows if int(row["outer_fold"]) == outer]
        )
        local_selections["trainable_parameters"] = local_selections["candidate"].map(
            parameter_map
        )
        local_selections = local_selections.sort_values(
            [
                "f1",
                "false_alarms_per_hour",
                "recall",
                "trainable_parameters",
                "candidate",
            ],
            ascending=[False, True, False, True, True],
            kind="stable",
        ).reset_index(drop=True)
        local_selections["inner_rank"] = np.arange(1, len(local_selections) + 1)
        for row in local_selections.to_dict("records"):
            architecture_rows.append(row)
        selected_candidate = str(local_selections.iloc[0].candidate)
        selected_threshold = float(local_selections.iloc[0].threshold)
        baseline_threshold = candidate_thresholds["BLSTM-CRF"]

        outer_fit_indices = np.flatnonzero(np.isin(groups, list(outer_fit_pids)))
        outer_heldout_assignments = assignments[
            assignments["pid"].isin(outer_heldout_pids)
        ]
        final_candidates = ["BLSTM-CRF"]
        if selected_candidate != "BLSTM-CRF":
            final_candidates.append(selected_candidate)
        final_models = {}
        for candidate in final_candidates:
            seed = BASE_SEED + 100 * outer + 90
            model, scaler, summary = fit_candidate(
                candidate,
                sequences[outer_fit_indices],
                tags[outer_fit_indices],
                labels[outer_fit_indices],
                seed,
                f"outer_{outer}_final",
            )
            summary.update(
                {
                    "outer_fold": outer,
                    "inner_fold": 9,
                    "phase": "outer_final",
                    "fit_pid": len(outer_fit_pids),
                    "heldout_pid": len(outer_heldout_pids),
                }
            )
            artifact = model_path(candidate, outer, "outer_final")
            verify_or_create_json_gzip(
                artifact,
                model_payload(
                    model,
                    scaler,
                    candidate,
                    outer,
                    "outer_final",
                    seed,
                    feature_names,
                ),
            )
            summary["model_sha256"] = sha256(artifact)
            fit_rows.append(summary)
            external_paths.append(("outer_model", artifact))
            scores, support = score_assignments(
                candidate,
                model,
                scaler,
                outer_heldout_assignments,
                recordings,
                outer,
                "outer_heldout",
            )
            final_models[candidate] = scores
            score_artifact = score_path(candidate, outer, "outer_heldout")
            verify_or_create_gzip_tsv(score_artifact, scores)
            external_paths.append(("outer_scores", score_artifact))
            if candidate == "BLSTM-CRF":
                outer_support_rows.append(support)

        baseline_scores = final_models["BLSTM-CRF"].copy()
        selected_scores = final_models[selected_candidate].copy()
        baseline_scores.insert(0, "pipeline", "NESTED-BLSTM-CRF")
        selected_scores.insert(0, "pipeline", "NESTED-SELECTED")
        selected_scores.insert(1, "selected_candidate", selected_candidate)
        baseline_scores.insert(1, "selected_candidate", "BLSTM-CRF")
        outer_score_rows["NESTED-BLSTM-CRF"].append(baseline_scores)
        outer_score_rows["NESTED-SELECTED"].append(selected_scores)
        outer_event_rows.append(
            collapse_events(baseline_scores, baseline_threshold, "NESTED-BLSTM-CRF")
        )
        outer_event_rows.append(
            collapse_events(selected_scores, selected_threshold, "NESTED-SELECTED")
        )

    fit_record = pd.DataFrame(fit_rows)
    determinism = pd.DataFrame(determinism_rows)
    determinism["status"] = np.where(
        determinism["state_hash_equal"].astype(bool)
        & determinism["maximum_scaler_difference"].le(1e-12)
        & determinism["maximum_probability_difference"].le(1e-12),
        "pass",
        "fail",
    )
    inner_curves = pd.concat(inner_curve_rows, ignore_index=True)
    inner_selections = pd.DataFrame(inner_selection_rows)
    candidate_order = {name: index for index, name in enumerate(CANDIDATES)}
    inner_selections["candidate_order"] = inner_selections["candidate"].map(
        candidate_order
    )
    inner_selections = inner_selections.sort_values(
        ["outer_fold", "candidate_order"]
    ).drop(columns="candidate_order").reset_index(drop=True)
    architecture = pd.DataFrame(architecture_rows)
    recommendation = (
        architecture.groupby("candidate", as_index=False)
        .agg(
            mean_inner_rank=("inner_rank", "mean"),
            mean_inner_f1=("f1", "mean"),
            mean_inner_far=("false_alarms_per_hour", "mean"),
            selected_outer_folds=("inner_rank", lambda value: int((value == 1).sum())),
            trainable_parameters=("trainable_parameters", "first"),
        )
        .sort_values(
            [
                "mean_inner_rank",
                "mean_inner_f1",
                "mean_inner_far",
                "trainable_parameters",
                "candidate",
            ],
            ascending=[True, False, True, True, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    recommendation["recommendation_rank"] = np.arange(1, len(recommendation) + 1)
    recommendation["recommended_for_new_confirmation"] = recommendation[
        "recommendation_rank"
    ].eq(1)

    pooled_scores = {}
    for pipeline, frames in outer_score_rows.items():
        pooled_scores[pipeline] = pd.concat(frames, ignore_index=True)
        artifact = pooled_outer_score_path(pipeline)
        verify_or_create_gzip_tsv(artifact, pooled_scores[pipeline])
        external_paths.append(("pooled_outer_scores", artifact))
    events = pd.concat(outer_event_rows, ignore_index=True)
    support = pd.concat(outer_support_rows, ignore_index=True).sort_values("subject")
    outputs = evaluate_pipelines(events, support, reference_events(assignments))
    bootstrap = paired_bootstrap(outputs["participants"])
    decision = decisions(outputs["metrics"], bootstrap)

    source_paths = [("source_train_feature", value["path"]) for value in recordings.values()]
    external_paths = source_paths + external_paths
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
    ).drop_duplicates("path_relative_to_data_parent").sort_values(
        ["artifact_role", "path_relative_to_data_parent"]
    )

    outer_counts = folds.groupby("pid").size()
    inner_complete = architecture.groupby("outer_fold")["candidate"].nunique()
    selection_counts = architecture[architecture["inner_rank"] == 1].groupby(
        "outer_fold"
    ).size()
    prediction_counts = events.groupby("pipeline")["pid"].nunique()
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
                if not manifest["path_relative_to_data_parent"]
                .str.contains("/validation/|/test/|(?:^|/)test_", case=False, regex=True)
                .any()
                else "fail",
                "detail": "train artifacts only",
            },
            {
                "check": "five_outer_folds",
                "status": "pass"
                if folds["fold"].nunique() == OUTER_FOLDS and outer_counts.eq(1).all()
                else "fail",
                "detail": "each pid held out once",
            },
            {
                "check": "complete_inner_architecture_selection",
                "status": "pass"
                if inner_complete.eq(len(CANDIDATES)).all() and selection_counts.eq(1).all()
                else "fail",
                "detail": "four candidates and one selection per outer fold",
            },
            {
                "check": "no_participant_leakage",
                "status": "pass" if leakage_pass else "fail",
                "detail": "all fit and heldout pid sets disjoint",
            },
            {
                "check": "common_candidate_and_feature_input",
                "status": "pass"
                if len(retained) == 2743
                and int(retained["label"].sum()) == 180
                and sequences.shape == (2743, SEQUENCE_LENGTH, FEATURE_COUNT)
                else "fail",
                "detail": f"shape={sequences.shape}; positive={int(retained['label'].sum())}",
            },
            {
                "check": "four_frozen_candidates",
                "status": "pass"
                if configuration["candidate"].tolist() == CANDIDATES
                else "fail",
                "detail": ";".join(CANDIDATES),
            },
            {
                "check": "synthetic_model_checks",
                "status": "pass"
                if checks_synthetic["status"].eq("pass").all()
                else "fail",
                "detail": f"{checks_synthetic['status'].eq('pass').sum()}/{len(checks_synthetic)} passed",
            },
            {
                "check": "fixed_training_completion",
                "status": "pass"
                if fit_record["epochs_completed"].eq(TRAINING_EPOCHS).all()
                and np.isfinite(fit_record["final_epoch_loss"]).all()
                else "fail",
                "detail": f"fits={len(fit_record)}; epochs={TRAINING_EPOCHS}",
            },
            {
                "check": "deterministic_duplicate_fits",
                "status": "pass"
                if len(determinism) == len(CANDIDATES)
                and determinism["status"].eq("pass").all()
                else "fail",
                "detail": f"{determinism['status'].eq('pass').sum()}/{len(determinism)} passed",
            },
            {
                "check": "fixed_threshold_grid",
                "status": "pass"
                if inner_curves.groupby(["outer_fold", "candidate"]).size().eq(len(THRESHOLDS)).all()
                else "fail",
                "detail": f"{len(THRESHOLDS)} thresholds x 20 candidate-outer pairs",
            },
            {
                "check": "complete_outer_scores",
                "status": "pass"
                if all(
                    frame["subject"].nunique() == 82
                    and frame["pid"].nunique() == 64
                    and np.isfinite(frame["probability"]).all()
                    for frame in pooled_scores.values()
                )
                else "fail",
                "detail": "two pipelines; 82 recordings; 64 pid",
            },
            {
                "check": "paired_participant_results",
                "status": "pass"
                if outputs["participants"]
                .query("membership == 'primary' and tolerance_sec == 15.0")
                .groupby("pipeline")["pid"]
                .nunique()
                .eq(64)
                .all()
                else "fail",
                "detail": "64 pid per pipeline",
            },
            {
                "check": "external_artifact_manifest",
                "status": "pass"
                if manifest["sha256"].str.len().eq(64).all()
                else "fail",
                "detail": f"{len(manifest)} hashed artifacts",
            },
        ]
    )

    reviewed = {
        "candidate_configuration_v0.1.tsv": configuration,
        "synthetic_model_checks_v0.1.tsv": checks_synthetic,
        "deterministic_fit_checks_v0.1.tsv": determinism,
        "model_fit_summary_v0.1.tsv": fit_record,
        "inner_threshold_curves_v0.1.tsv": inner_curves,
        "inner_threshold_selections_v0.1.tsv": inner_selections,
        "outer_architecture_rankings_v0.1.tsv": architecture,
        "architecture_recommendation_v0.1.tsv": recommendation,
        "outer_support_v0.1.tsv": support,
        "outer_predicted_events_v0.1.tsv": events,
        "outer_event_metrics_v0.1.tsv": outputs["metrics"],
        "outer_event_recordings_v0.1.tsv": outputs["recordings"],
        "outer_event_participants_v0.1.tsv": outputs["participants"],
        "outer_event_matches_v0.1.tsv": outputs["matches"],
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decision,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "output_integrity_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)

    software = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
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
        f"| {item.pipeline} | {item.precision:.4f} | {item.recall:.4f} | "
        f"{item.f1:.4f} | {item.false_alarms_per_hour:.4f} |"
        for item in primary.itertuples(index=False)
    ]
    selections = architecture[architecture["inner_rank"] == 1][
        ["outer_fold", "candidate", "threshold", "f1", "false_alarms_per_hour"]
    ]
    selection_lines = [
        f"| {int(item.outer_fold)} | {item.candidate} | {item.threshold:.8f} | "
        f"{item.f1:.4f} | {item.false_alarms_per_hour:.4f} |"
        for item in selections.itertuples(index=False)
    ]
    primary_decision = decision.iloc[0]
    recommended = recommendation.iloc[0].candidate
    readme = "\n".join(
        [
            "# Nested Train-Only Deep Temporal Exploration v0.1",
            "",
            "**Work date:** 2026-09-22",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Validation accessed:** No",
            "**Test accessed:** No",
            "",
            "## Inner Selections",
            "",
            "| Outer fold | Candidate | Threshold | Inner F1 | Inner FAR/hour |",
            "|---:|---|---:|---:|---:|",
            *selection_lines,
            "",
            f"Inner-ranked candidate for future confirmation: **{recommended}**.",
            "",
            "## Primary Nested Outer Result",
            "",
            "| Pipeline | Precision | Recall | F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|",
            *metric_lines,
            "",
            "## Frozen Decision",
            "",
            f"Selected minus baseline F1: `{primary_decision.f1_difference:+.4f}`.",
            f"Selected minus baseline false alarms/hour: `{primary_decision.false_alarms_per_hour_difference:+.4f}`.",
            f"Decision: **{str(primary_decision.decision).replace('_', ' ')}**.",
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. "
            "This is nested development on the reused train cohort, not independent confirmation.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", readme)

    print(selections.to_string(index=False))
    print(primary.to_string(index=False))
    print(decision.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one in-run check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
