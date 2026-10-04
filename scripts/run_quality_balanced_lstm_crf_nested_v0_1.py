"""Run the frozen quality-balanced LSTM-CRF nested experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
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
from run_spectral_unet_train_nested_v0_1 import (
    THRESHOLDS,
    collapse_events,
    context_start_indices,
    data_parent,
    deterministic_gzip,
    fast_primary_summary,
    frozen_fold_assignments,
    labeled_candidates,
    local_event_inputs,
    reference_events,
    repo_root,
    sha256,
    time_key,
    train_assignments,
    truth,
    verify_or_create_gzip_tsv,
    verify_or_create_json_gzip,
    verify_or_create_text,
)
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT = "2026-10-03_quality_balanced_lstm_crf_nested_v0.1"
DERIVED = "quality_balanced_lstm_crf_nested_v0.1"
PROTOCOL_COMMIT = "b09c857"
BASE_SEED = 20260922
BOOTSTRAP_SEED = 20261003
OUTER_FOLDS = 5
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
ENDPOINT_PATH = np.asarray([0, 0, 0, 1, 2, 0, 0, 0], dtype=np.int64)
BACKGROUND_PATH = np.zeros(SEQUENCE_LENGTH, dtype=np.int64)
PIPELINES = ["LC-1-NESTED", "LC-QB1"]
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]


# Section 2: paths and immutable artifacts

def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT


def result_dir() -> Path:
    return data_parent() / "derived" / DERIVED


def score_path(pipeline: str, outer: int, phase: str) -> Path:
    name = pipeline.lower().replace("-", "_")
    return result_dir() / "scores" / f"outer_{outer}" / f"{name}_{phase}_v0.1.tsv.gz"


def model_path(pipeline: str, outer: int, phase: str) -> Path:
    name = pipeline.lower().replace("-", "_")
    return result_dir() / "models" / f"outer_{outer}" / f"{name}_{phase}_v0.1.json.gz"


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(), text=True,
    ).strip()


def configure_torch(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def canonical_state_hash(model: nn.Module) -> str:
    payload = {key: value.detach().cpu().numpy().tolist()
               for key, value in sorted(model.state_dict().items())}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


# Section 3: reviewed features and labels

def feature_path(subject: str) -> Path:
    return (data_parent() / "derived/block7_feature_generation_validation_v0.1"
            / "recording_features/hb2" / f"{subject}_features_v0.1.npz")


def load_recordings(assignments: pd.DataFrame) -> tuple[dict, list[str]]:
    recordings: dict[str, dict] = {}
    schema: list[str] | None = None
    for row in assignments.itertuples(index=False):
        with np.load(feature_path(row.subject), allow_pickle=False) as values:
            onsets = values["onset"].astype(float)
            features = values["features"].astype(np.float32)
            names = values["feature_names"].astype(str).tolist()
        if features.shape[1] != FEATURE_COUNT or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature matrix: {row.subject}")
        if schema is None:
            schema = names
        elif schema != names:
            raise ValueError(f"Feature schema changed: {row.subject}")
        starts = context_start_indices(onsets)
        recordings[row.subject] = {
            "pid": int(row.pid),
            "centers": onsets[starts + 4],
            "sequences": np.stack(
                [features[starts + offset] for offset in range(SEQUENCE_LENGTH)], axis=1
            ),
        }
    return recordings, schema or []


def build_sequences(candidates: pd.DataFrame, recordings: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    matrices, tags, rows = [], [], []
    membership = pd.read_csv(
        repo_root() / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t", usecols=["transition_id", "membership_tier"],
    )
    tier_lookup = dict(zip("transition_" + membership.transition_id.astype(str), membership.membership_tier))
    for subject, group in candidates.groupby("subject", sort=True):
        recording = recordings[subject]
        lookup = {time_key(value): index for index, value in enumerate(recording["centers"])}
        for item in group.itertuples(index=False):
            index = lookup.get(time_key(item.candidate_time_sec))
            retained = index is not None
            tier = tier_lookup.get(item.sample_id, "not_applicable")
            rows.append({
                "sample_id": item.sample_id, "subject": subject, "pid": int(item.pid),
                "candidate_time_sec": float(item.candidate_time_sec), "label": int(item.label),
                "membership_tier": tier, "retained": retained,
                "drop_reason": "" if retained else "missing_required_context",
            })
            if retained:
                matrices.append(recording["sequences"][index])
                tags.append(ENDPOINT_PATH if item.label == 1 else BACKGROUND_PATH)
    metadata = pd.DataFrame(rows)
    return np.stack(matrices).astype(np.float32), np.stack(tags), metadata


# Section 4: unchanged BLSTM-CRF

class LinearChainCRF(nn.Module):
    def __init__(self, state_count: int):
        super().__init__()
        self.start = nn.Parameter(torch.empty(state_count))
        self.transitions = nn.Parameter(torch.empty(state_count, state_count))
        self.end = nn.Parameter(torch.empty(state_count))
        nn.init.uniform_(self.start, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.end, -0.1, 0.1)

    def path_energy(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(emissions.shape[0], device=emissions.device)
        score = self.start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for step in range(1, emissions.shape[1]):
            score = score + self.transitions[tags[:, step - 1], tags[:, step]]
            score = score + emissions[batch, step, tags[:, step]]
        return score + self.end[tags[:, -1]]

    def log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        alpha = self.start.unsqueeze(0) + emissions[:, 0]
        for step in range(1, emissions.shape[1]):
            alpha = torch.logsumexp(alpha.unsqueeze(2) + self.transitions.unsqueeze(0), dim=1)
            alpha = alpha + emissions[:, step]
        return torch.logsumexp(alpha + self.end.unsqueeze(0), dim=1)

    def losses(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return self.log_partition(emissions) - self.path_energy(emissions, tags)


class RecurrentCrf(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(FEATURE_COUNT, HIDDEN_SIZE, batch_first=True, bidirectional=True)
        self.emission = nn.Linear(HIDDEN_SIZE * 2, STATE_COUNT)
        self.crf = LinearChainCRF(STATE_COUNT)

    def emissions(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(values)
        return self.emission(encoded)

    def losses(self, values: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return self.crf.losses(self.emissions(values), tags)

    def probabilities(self, values: torch.Tensor) -> torch.Tensor:
        emissions = self.emissions(values)
        positive = torch.as_tensor(ENDPOINT_PATH, device=values.device).repeat(len(values), 1)
        negative = torch.as_tensor(BACKGROUND_PATH, device=values.device).repeat(len(values), 1)
        margin = self.crf.path_energy(emissions, positive) - self.crf.path_energy(emissions, negative)
        return torch.sigmoid(margin)


# Section 5: paired fitting treatments

def scaler_parameters(sequences: np.ndarray, pipeline: str) -> tuple[np.ndarray, np.ndarray]:
    flat = sequences.reshape(-1, FEATURE_COUNT).astype(np.float64)
    if pipeline == "LC-1-NESTED":
        scaler = StandardScaler().fit(flat)
        return scaler.mean_, scaler.scale_
    center = np.median(flat, axis=0)
    quartiles = np.quantile(flat, [0.25, 0.75], axis=0, method="linear")
    return center, np.maximum(quartiles[1] - quartiles[0], 1e-6)


def sample_weights(labels: np.ndarray, tiers: np.ndarray, pipeline: str) -> tuple[np.ndarray, dict]:
    negative = int((labels == 0).sum())
    clean = int(((labels == 1) & (tiers == "primary_clean")).sum())
    flagged = int(((labels == 1) & (tiers == "primary_mad_flagged")).sum())
    if clean == 0 or flagged == 0:
        raise ValueError("Both positive tiers must occur in every fitting subset")
    if pipeline == "LC-1-NESTED":
        positive_weight = negative / int((labels == 1).sum())
        weights = np.where(labels == 1, positive_weight, 1.0)
    else:
        weights = np.ones(len(labels), dtype=float)
        weights[(labels == 1) & (tiers == "primary_clean")] = negative / (2 * clean)
        weights[(labels == 1) & (tiers == "primary_mad_flagged")] = negative / (2 * flagged)
    summary = {
        "fit_negative": negative, "fit_clean_positive": clean, "fit_flagged_positive": flagged,
        "negative_weight_total": float(weights[labels == 0].sum()),
        "clean_weight_total": float(weights[(labels == 1) & (tiers == "primary_clean")].sum()),
        "flagged_weight_total": float(weights[(labels == 1) & (tiers == "primary_mad_flagged")].sum()),
    }
    return weights.astype(np.float32), summary


def fit_model(pipeline: str, sequences: np.ndarray, tags: np.ndarray, labels: np.ndarray,
              tiers: np.ndarray, seed: int, name: str) -> tuple[RecurrentCrf, np.ndarray, np.ndarray, dict]:
    configure_torch(seed)
    center, scale = scaler_parameters(sequences, pipeline)
    scaled = ((sequences - center) / scale).astype(np.float32)
    weights, counts = sample_weights(labels, tiers, pipeline)
    dataset = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(tags.astype(np.int64)),
                            torch.from_numpy(weights))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    model = RecurrentCrf()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    first_loss = final_loss = math.nan
    max_gradient = 0.0
    for epoch in range(1, TRAINING_EPOCHS + 1):
        epoch_losses = []
        model.train()
        for values, batch_tags, batch_weights in loader:
            optimizer.zero_grad(set_to_none=True)
            losses = model.losses(values, batch_tags)
            loss = torch.sum(losses * batch_weights) / torch.sum(batch_weights)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite loss: {name}")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
            max_gradient = max(max_gradient, float(gradient.detach()))
        final_loss = float(np.mean(epoch_losses))
        if epoch == 1:
            first_loss = final_loss
    summary = {
        "fit": name, "pipeline": pipeline, "seed": seed, "fit_rows": len(labels),
        "first_epoch_loss": first_loss, "final_epoch_loss": final_loss,
        "maximum_preclip_gradient_norm": max_gradient, "state_sha256": canonical_state_hash(model),
        **counts,
    }
    return model, center, scale, summary


def score_model(model: RecurrentCrf, center: np.ndarray, scale: np.ndarray,
                sequences: np.ndarray) -> np.ndarray:
    values = torch.from_numpy(((sequences - center) / scale).astype(np.float32))
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(values), 1024):
            result.append(model.probabilities(values[start:start + 1024]))
    probabilities = torch.cat(result).numpy().astype(float)
    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite model probabilities")
    return probabilities


def score_assignments(model: RecurrentCrf, center: np.ndarray, scale: np.ndarray,
                      assignments: pd.DataFrame, recordings: dict, pipeline: str,
                      outer: int, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, support = [], []
    for row in assignments.itertuples(index=False):
        recording = recordings[row.subject]
        probability = score_model(model, center, scale, recording["sequences"])
        scores.append(pd.DataFrame({
            "pipeline": pipeline, "outer_fold": outer, "phase": phase,
            "subject": row.subject, "pid": int(row.pid),
            "candidate_time_sec": recording["centers"], "probability": probability,
        }))
        support.append({"subject": row.subject, "pid": int(row.pid), "outer_fold": outer,
                        "supported_boundaries": len(probability),
                        "supported_hours": len(probability) * EPOCH_SEC / 3600.0})
    return pd.concat(scores, ignore_index=True), pd.DataFrame(support)


def model_payload(model: RecurrentCrf, center: np.ndarray, scale: np.ndarray, summary: dict) -> dict:
    return {
        "configuration": {"epochs": TRAINING_EPOCHS, "batch_size": BATCH_SIZE,
                          "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
        "scaler_center": center.tolist(), "scaler_scale": scale.tolist(), "fit_summary": summary,
        "state_dict": {key: value.detach().cpu().numpy().tolist()
                       for key, value in sorted(model.state_dict().items())},
    }


def run_fit(pipeline: str, outer: int, phase: str, seed: int, mask: np.ndarray,
            heldout: pd.DataFrame, sequences: np.ndarray, tags: np.ndarray, labels: np.ndarray,
            tiers: np.ndarray, recordings: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scores_file = score_path(pipeline, outer, phase)
    model_file = model_path(pipeline, outer, phase)
    if scores_file.exists() and model_file.exists():
        scores = pd.read_csv(scores_file, sep="\t")
        with gzip.open(model_file, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        support = scores.groupby(["subject", "pid", "outer_fold"], as_index=False).size()
        support = support.rename(columns={"size": "supported_boundaries"})
        support["supported_hours"] = support.supported_boundaries * EPOCH_SEC / 3600.0
        return scores, support, payload["fit_summary"]
    model, center, scale, summary = fit_model(
        pipeline, sequences[mask], tags[mask], labels[mask], tiers[mask], seed,
        f"outer_{outer}_{phase}",
    )
    scores, support = score_assignments(model, center, scale, heldout, recordings,
                                        pipeline, outer, phase)
    verify_or_create_json_gzip(model_file, model_payload(model, center, scale, summary))
    verify_or_create_gzip_tsv(scores_file, scores)
    return scores, support, summary


# Section 6: event and quality-tier evaluation

def evaluate(events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame) -> dict:
    metric_rows, recording_rows, participant_rows, match_rows = [], [], [], []
    for pipeline in PIPELINES:
        predictions = events[events.pipeline.eq(pipeline)]
        for membership in MEMBERSHIPS:
            eligible, ignored = local_event_inputs(references, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible, predictions[["subject", "pid", "event_time_sec"]], ignored,
                    support[["subject", "pid", "supported_hours"]], tolerance,
                )
                config = {"pipeline": pipeline, "partition": "train_nested_oof",
                          "membership": membership, "tolerance_sec": tolerance}
                metric_rows.append({**config, **summary})
                for frame, destination in [(recordings, recording_rows), (participants, participant_rows),
                                           (matches, match_rows)]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        destination.append(local)
    return {"metrics": pd.DataFrame(metric_rows), "recordings": pd.concat(recording_rows),
            "participants": pd.concat(participant_rows), "matches": pd.concat(match_rows)}


def tier_recall(references: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = references[truth(references.primary_analysis_eligible)].copy()
    eligible_matches = matches[(matches.membership == "primary") &
                               (matches.tolerance_sec == 15.0) &
                               (matches.match_type == "eligible")]
    rows, participant_rows = [], []
    for pipeline in PIPELINES:
        matched = set(zip(eligible_matches[eligible_matches.pipeline.eq(pipeline)].subject,
                          eligible_matches[eligible_matches.pipeline.eq(pipeline)].reference_time_sec))
        local = primary.copy()
        local["detected"] = [int((row.subject, row.event_time_sec) in matched)
                             for row in local.itertuples(index=False)]
        for tier, group in local.groupby("membership_tier"):
            rows.append({"pipeline": pipeline, "membership_tier": tier,
                         "reference_events": len(group), "detected_events": int(group.detected.sum()),
                         "recall": float(group.detected.mean())})
        counts = local.groupby(["pid", "membership_tier"], as_index=False).detected.agg(
            reference_events="size", detected_events="sum")
        all_rows = pd.MultiIndex.from_product(
            [sorted(primary.pid.unique()), ["primary_clean", "primary_mad_flagged"]],
            names=["pid", "membership_tier"],
        ).to_frame(index=False)
        counts = all_rows.merge(counts, how="left", on=["pid", "membership_tier"]).fillna(0)
        counts.insert(0, "pipeline", pipeline)
        participant_rows.append(counts)
    return pd.DataFrame(rows), pd.concat(participant_rows, ignore_index=True)


def bootstrap(participants: pd.DataFrame, tier_participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[(participants.membership == "primary") &
                           (participants.tolerance_sec == 15.0)]
    pids = sorted(primary.pid.unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    def aggregate(sample: np.ndarray, pipeline: str) -> dict:
        selected = primary[(primary.pipeline == pipeline) & primary.pid.isin(sample)].set_index("pid")
        repeated = selected.loc[sample]
        metrics = metric_values(int(repeated.true_positive.sum()), int(repeated.false_positive.sum()),
                                int(repeated.false_negative.sum()), float(repeated.supported_hours.sum()))
        for tier in ["primary_clean", "primary_mad_flagged"]:
            table = tier_participants[(tier_participants.pipeline == pipeline) &
                                      (tier_participants.membership_tier == tier)].set_index("pid").loc[sample]
            metrics[f"{tier}_recall"] = table.detected_events.sum() / table.reference_events.sum()
        return metrics

    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = rng.choice(pids, size=len(pids), replace=True)
        left, right = aggregate(sample, "LC-QB1"), aggregate(sample, "LC-1-NESTED")
        samples.append({"f1": left["f1"] - right["f1"],
                        "false_alarms_per_hour": left["false_alarms_per_hour"] - right["false_alarms_per_hour"],
                        "primary_clean_recall": left["primary_clean_recall"] - right["primary_clean_recall"],
                        "primary_mad_flagged_recall": left["primary_mad_flagged_recall"] - right["primary_mad_flagged_recall"]})
    frame = pd.DataFrame(samples)
    left, right = aggregate(np.asarray(pids), "LC-QB1"), aggregate(np.asarray(pids), "LC-1-NESTED")
    return pd.DataFrame([{
        "comparison": "LC-QB1_minus_LC-1-NESTED", "metric": metric,
        "point_difference": left[metric] - right[metric], "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED, "lower_95": frame[metric].quantile(.025),
        "median": frame[metric].quantile(.5), "upper_95": frame[metric].quantile(.975),
    } for metric in frame.columns])


# Section 7: complete nested comparison

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")
    assignments = train_assignments()
    folds = frozen_fold_assignments(assignments)
    recordings, feature_names = load_recordings(assignments)
    sequences, tags, construction = build_sequences(labeled_candidates(assignments), recordings)
    retained = construction[truth(construction.retained)].reset_index(drop=True)
    labels = retained.label.to_numpy(int)
    groups = retained.pid.to_numpy(int)
    tiers = retained.membership_tier.to_numpy(str)
    if (len(labels), int(labels.sum()), int((tiers == "primary_clean").sum()),
        int((tiers == "primary_mad_flagged").sum())) != (2743, 180, 53, 127):
        raise ValueError("Unexpected reviewed candidate counts")

    fit_rows, curve_rows, selection_rows = [], [], []
    outer_scores, outer_support, event_rows = [], [], []
    for outer in range(1, OUTER_FOLDS + 1):
        outer_pid = set(folds[folds.fold == outer].pid.astype(int))
        for pipeline in PIPELINES:
            inner_scores, inner_support = [], []
            for inner in sorted(set(range(1, OUTER_FOLDS + 1)) - {outer}):
                inner_pid = set(folds[folds.fold == inner].pid.astype(int))
                mask = ~np.isin(groups, list(outer_pid | inner_pid))
                heldout = assignments[assignments.pid.isin(inner_pid)]
                scores, support, summary = run_fit(
                    pipeline, outer, f"inner_{inner}", BASE_SEED + 100 * outer + 10 * inner,
                    mask, heldout, sequences, tags, labels, tiers, recordings,
                )
                fit_rows.append({"outer_fold": outer, "inner_fold": inner, "phase": "inner", **summary})
                inner_scores.append(scores); inner_support.append(support)
            pooled_scores, pooled_support = pd.concat(inner_scores), pd.concat(inner_support)
            references = reference_events(assignments[~assignments.pid.isin(outer_pid)])
            curve = pd.DataFrame([{"outer_fold": outer, "pipeline": pipeline, "threshold": float(threshold),
                                   **fast_primary_summary(pooled_scores, pooled_support, references, float(threshold))}
                                  for threshold in THRESHOLDS])
            selected = curve.sort_values(["f1", "false_alarms_per_hour", "recall", "threshold"],
                                         ascending=[False, True, False, False], kind="stable").iloc[0].to_dict()
            selected["selection_rule"] = "max_f1_then_min_far_then_max_recall_then_max_threshold"
            curve_rows.append(curve); selection_rows.append(selected)

            mask = ~np.isin(groups, list(outer_pid))
            heldout = assignments[assignments.pid.isin(outer_pid)]
            scores, support, summary = run_fit(
                pipeline, outer, "outer_final", BASE_SEED + 100 * outer + 90,
                mask, heldout, sequences, tags, labels, tiers, recordings,
            )
            fit_rows.append({"outer_fold": outer, "inner_fold": 9, "phase": "outer_final", **summary})
            outer_scores.append(scores)
            if pipeline == PIPELINES[0]: outer_support.append(support)
            event_rows.append(collapse_events(scores, float(selected["threshold"]), pipeline))
            print(f"completed outer {outer} {pipeline}", flush=True)

    scores = pd.concat(outer_scores, ignore_index=True)
    support = pd.concat(outer_support, ignore_index=True).sort_values("subject")
    events = pd.concat(event_rows, ignore_index=True)
    references = reference_events(assignments)
    results = evaluate(events, support, references)
    tier_metrics, tier_participants = tier_recall(references, results["matches"])
    intervals = bootstrap(results["participants"], tier_participants)

    primary = results["metrics"][(results["metrics"].membership == "primary") &
                                 (results["metrics"].tolerance_sec == 15.0)].set_index("pipeline")
    tiers_table = tier_metrics.set_index(["pipeline", "membership_tier"])
    changes = {
        "f1": primary.loc["LC-QB1", "f1"] - primary.loc["LC-1-NESTED", "f1"],
        "false_alarms_per_hour": primary.loc["LC-QB1", "false_alarms_per_hour"] - primary.loc["LC-1-NESTED", "false_alarms_per_hour"],
        "primary_clean_recall": tiers_table.loc[("LC-QB1", "primary_clean"), "recall"] - tiers_table.loc[("LC-1-NESTED", "primary_clean"), "recall"],
        "primary_mad_flagged_recall": tiers_table.loc[("LC-QB1", "primary_mad_flagged"), "recall"] - tiers_table.loc[("LC-1-NESTED", "primary_mad_flagged"), "recall"],
    }
    gate = changes["f1"] >= .05 and changes["primary_clean_recall"] >= .10 and changes["false_alarms_per_hour"] <= 0
    decisions = pd.DataFrame([{"hypothesis": "H-QB1_quality_balanced_advancement", **changes,
                               "required_f1_difference": .05, "required_clean_recall_difference": .10,
                               "maximum_far_difference": 0.0, "supported": gate,
                               "decision": "freeze_for_new_confirmation" if gate else "stop_v0.1"}])

    prior = pd.read_csv(data_parent() / "derived/deep_temporal_nested_cv_v0.1/scores/nested_blstm_crf_outer_scores_v0.1.tsv.gz", sep="\t")
    control = scores[scores.pipeline == "LC-1-NESTED"]
    key = ["outer_fold", "subject", "pid", "candidate_time_sec"]
    compared = control.merge(prior[key + ["probability"]], on=key, suffixes=("_current", "_prior"), validate="one_to_one")
    reproduction_error = float(np.max(np.abs(compared.probability_current - compared.probability_prior)))
    checks = pd.DataFrame([
        {"check": "train_membership", "value": f"{len(assignments)} recordings; {assignments.pid.nunique()} pid", "status": "pass" if len(assignments) == 82 and assignments.pid.nunique() == 64 else "fail"},
        {"check": "candidate_counts", "value": "2743 total; 180 positive; 53 clean; 127 flagged", "status": "pass"},
        {"check": "control_score_reproduction", "value": reproduction_error, "status": "pass" if reproduction_error <= 1e-12 else "fail"},
        {"check": "finite_probabilities", "value": str(np.isfinite(scores.probability).all()).lower(), "status": "pass" if np.isfinite(scores.probability).all() else "fail"},
        {"check": "validation_and_test_closed", "value": "train only", "status": "pass"},
    ])
    if not checks.status.eq("pass").all():
        raise ValueError("Experiment checks failed")

    pooled_path = result_dir() / "scores/paired_outer_scores_v0.1.tsv.gz"
    verify_or_create_gzip_tsv(pooled_path, scores)
    external = pd.DataFrame([{"relative_path": path.relative_to(data_parent()).as_posix(),
                              "bytes": path.stat().st_size, "sha256": sha256(path)}
                             for path in sorted(result_dir().rglob("*")) if path.is_file()])
    output = output_dir()
    for frame, name in [
        (construction, "candidate_construction_v0.1.tsv"), (pd.DataFrame(fit_rows), "model_fit_summary_v0.1.tsv"),
        (pd.concat(curve_rows), "inner_threshold_curves_v0.1.tsv"), (pd.DataFrame(selection_rows), "inner_threshold_selections_v0.1.tsv"),
        (support, "outer_support_v0.1.tsv"), (events, "outer_predicted_events_v0.1.tsv"),
        (results["metrics"], "outer_event_metrics_v0.1.tsv"), (results["recordings"], "outer_event_recordings_v0.1.tsv"),
        (results["participants"], "outer_event_participants_v0.1.tsv"), (results["matches"], "outer_event_matches_v0.1.tsv"),
        (tier_metrics, "quality_tier_metrics_v0.1.tsv"), (tier_participants, "quality_tier_participants_v0.1.tsv"),
        (intervals, "paired_participant_bootstrap_v0.1.tsv"), (decisions, "hypothesis_decisions_v0.1.tsv"),
        (checks, "experiment_checks_v0.1.tsv"), (external, "external_artifact_manifest_v0.1.tsv"),
        (pd.DataFrame({"feature_index": range(FEATURE_COUNT), "feature_name": feature_names}), "feature_schema_v0.1.tsv"),
    ]:
        verify_or_create_tsv(frame.reset_index(drop=True), output / name)
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "torch": torch.__version__, "git_commit": git_commit,
                "protocol_commit": PROTOCOL_COMMIT}
    verify_or_create_text(output / "software_versions_v0.1.json", json.dumps(versions, indent=2, sort_keys=True) + "\n")
    readme = f"""# Quality-Balanced LSTM-CRF Nested Experiment v0.1

This directory contains compact reviewed outputs for the frozen 2026-10-03 protocol.

| Pipeline | F1 | Recall | Precision | False alarms/hour | Clean recall | MAD-flagged recall |
|---|---:|---:|---:|---:|---:|---:|
| LC-1-NESTED | {primary.loc['LC-1-NESTED','f1']:.4f} | {primary.loc['LC-1-NESTED','recall']:.4f} | {primary.loc['LC-1-NESTED','precision']:.4f} | {primary.loc['LC-1-NESTED','false_alarms_per_hour']:.4f} | {tiers_table.loc[('LC-1-NESTED','primary_clean'),'recall']:.4f} | {tiers_table.loc[('LC-1-NESTED','primary_mad_flagged'),'recall']:.4f} |
| LC-QB1 | {primary.loc['LC-QB1','f1']:.4f} | {primary.loc['LC-QB1','recall']:.4f} | {primary.loc['LC-QB1','precision']:.4f} | {primary.loc['LC-QB1','false_alarms_per_hour']:.4f} | {tiers_table.loc[('LC-QB1','primary_clean'),'recall']:.4f} | {tiers_table.loc[('LC-QB1','primary_mad_flagged'),'recall']:.4f} |

Decision: `{decisions.iloc[0].decision}`.

This is reused-cohort train-only model development, not independent confirmation. Full scores and model states remain outside Git under `REM_W_data`.
"""
    verify_or_create_text(output / "README.md", readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
