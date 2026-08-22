"""Fit and evaluate the frozen simple direct REM-to-Wake baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage_first_event_evaluation_v0_1 import (
    evaluate_events,
    metric_values,
    participant_bootstrap,
)


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-08-22_direct_event_baseline_v0.1"
DERIVED_DIR = "direct_event_baseline_v0.1"
FEATURE_SOURCE_DIR = "stage_first_feature_baseline_v0.1"
FREEZE_MARKER = "DIRECT_MODELS_AND_THRESHOLDS_FROZEN_FOR_SINGLE_TEST_EVALUATION"
EPOCH_SEC = 30.0
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260822
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
MODEL_OFFSETS = {
    "DE-A": np.asarray([-30.0, 0.0]),
    "DE-B": np.arange(-120.0, 120.0, EPOCH_SEC),
}
MODEL_ROLES = {
    "DE-A": "direct_boundary_pair_ablation",
    "DE-B": "direct_eight_epoch_context_primary",
}
BASE_FEATURE_NAMES = [
    f"{channel}_{band}_log10_mean_psd"
    for channel in ["HB_1", "HB_2"]
    for band in ["delta", "theta", "alpha", "sigma", "beta"]
]


# Section 2: paths and frozen inputs

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def feature_path(subject: str) -> Path:
    return (
        data_parent()
        / "derived"
        / FEATURE_SOURCE_DIR
        / "recording_features"
        / f"{subject}_features_v0.1.npz"
    )


def model_path(comparator: str) -> Path:
    name = comparator.lower().replace("-", "_")
    return derived_dir() / "models" / f"{name}_model_v0.1.joblib"


def score_path(phase: str) -> Path:
    return derived_dir() / "candidate_scores" / f"{phase}_candidate_scores_v0.1.tsv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def subject_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    rows = []
    for item in split.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append(
                {"subject": subject, "pid": int(item.pid), "partition": item.partition}
            )
    result = pd.DataFrame(rows)
    if len(result) != 128 or result["subject"].nunique() != 128:
        raise ValueError("Expected 128 uniquely assigned BOAS recordings")
    overlap = result.groupby("pid")["partition"].nunique()
    if int((overlap > 1).sum()) != 0:
        raise ValueError("Participant leakage detected in the frozen split")
    return result.sort_values("subject", key=lambda value: value.map(subject_number))


def load_recording_features(subject: str) -> tuple[np.ndarray, np.ndarray]:
    path = feature_path(subject)
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen feature array: {path}")
    with np.load(path, allow_pickle=False) as values:
        onset = values["onset"].astype(float)
        features = values["features"].astype(float)
        names = values["feature_names"].astype(str).tolist()
    if names != BASE_FEATURE_NAMES:
        raise ValueError(f"Unexpected feature schema for {subject}")
    if features.shape != (len(onset), len(BASE_FEATURE_NAMES)):
        raise ValueError(f"Unexpected feature dimensions for {subject}")
    if len(np.unique(onset)) != len(onset) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature array for {subject}")
    return onset, features


# Section 3: direct labels and reviewed background rows

def reference_events() -> pd.DataFrame:
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
    rows = membership.merge(quality, on="transition_id", validate="one_to_one")
    rows = rows[
        truth(rows["is_primary_label"])
        & rows["transition_type"].eq("REM_to_Wake")
    ].copy()
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    return rows


def labeled_candidates() -> pd.DataFrame:
    positive = reference_events()
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
    negative = membership.merge(detail, on="background_review_id", validate="one_to_one")
    negative = negative[truth(negative["primary_analysis_eligible"])].copy()
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
    rows = pd.concat([positive_rows, negative_rows], ignore_index=True)
    if rows["sample_id"].duplicated().any():
        raise ValueError("Direct labeled candidate identifiers are not unique")
    return rows.sort_values(["partition", "subject", "candidate_time_sec", "label"])


# Section 4: deterministic feature construction

def time_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


def build_labeled_matrix(
    rows: pd.DataFrame, comparator: str
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    offsets = MODEL_OFFSETS[comparator]
    matrices = []
    retained = []
    dropped = []
    for subject, group in rows.groupby("subject", sort=True):
        onset, features = load_recording_features(subject)
        lookup = {time_key(value): index for index, value in enumerate(onset)}
        for item in group.itertuples(index=False):
            indices = [
                lookup.get(time_key(item.candidate_time_sec + offset)) for offset in offsets
            ]
            if any(index is None for index in indices):
                dropped.append(
                    {
                        **item._asdict(),
                        "comparator": comparator,
                        "drop_reason": "missing_required_context",
                    }
                )
                continue
            matrices.append(features[np.asarray(indices, dtype=int)].reshape(-1))
            retained.append({**item._asdict(), "comparator": comparator})
    columns = list(rows.columns) + ["comparator"]
    retained_frame = pd.DataFrame(retained, columns=columns)
    dropped_frame = pd.DataFrame(
        dropped, columns=columns + ["drop_reason"]
    )
    if not matrices:
        raise ValueError(f"No labeled rows retained for {comparator}")
    return np.vstack(matrices), retained_frame, dropped_frame


def construction_summary(
    requested: pd.DataFrame,
    retained: pd.DataFrame,
    dropped: pd.DataFrame,
    comparator: str,
) -> pd.DataFrame:
    keys = ["partition", "label", "source_tier"]
    request_counts = requested.groupby(keys, as_index=False).size().rename(
        columns={"size": "requested_rows"}
    )
    retained_counts = retained.groupby(keys, as_index=False).size().rename(
        columns={"size": "retained_rows"}
    )
    dropped_counts = dropped.groupby(keys, as_index=False).size().rename(
        columns={"size": "dropped_rows"}
    )
    summary = request_counts.merge(retained_counts, on=keys, how="left").merge(
        dropped_counts, on=keys, how="left"
    )
    for column in ["retained_rows", "dropped_rows"]:
        summary[column] = summary[column].fillna(0).astype(int)
    summary.insert(0, "comparator", comparator)
    return summary


def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            random_state=BOOTSTRAP_SEED,
        ),
    )


def fit_models(
    candidates: pd.DataFrame,
) -> tuple[
    dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    models = {}
    fit_rows = []
    metric_rows = []
    construction_rows = []
    score_rows = []
    for comparator in MODEL_OFFSETS:
        train_requested = candidates[candidates["partition"] == "train"].copy()
        validation_requested = candidates[candidates["partition"] == "validation"].copy()
        train_x, train_meta, train_dropped = build_labeled_matrix(
            train_requested, comparator
        )
        validation_x, validation_meta, validation_dropped = build_labeled_matrix(
            validation_requested, comparator
        )
        construction_rows.extend(
            [
                construction_summary(
                    train_requested, train_meta, train_dropped, comparator
                ),
                construction_summary(
                    validation_requested,
                    validation_meta,
                    validation_dropped,
                    comparator,
                ),
            ]
        )
        train_y = train_meta["label"].to_numpy(dtype=int)
        model = build_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        convergence = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        model_path(comparator).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path(comparator))
        models[comparator] = model
        iterations = int(model.named_steps["logisticregression"].n_iter_.max())
        fit_rows.append(
            {
                "comparator": comparator,
                "model_version": VERSION,
                "model_role": MODEL_ROLES[comparator],
                "train_rows": len(train_meta),
                "train_positive": int(train_y.sum()),
                "train_negative": int((train_y == 0).sum()),
                "input_features": train_x.shape[1],
                "maximum_iterations_used": iterations,
                "convergence_warning_count": len(convergence),
                "fit_decision": "retain_frozen_fit",
            }
        )
        for partition, matrix, metadata in [
            ("train", train_x, train_meta),
            ("validation", validation_x, validation_meta),
        ]:
            probability = model.predict_proba(matrix)[:, 1]
            labels = metadata["label"].to_numpy(dtype=int)
            metric_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": partition,
                    "rows": len(metadata),
                    "positive_rows": int(labels.sum()),
                    "negative_rows": int((labels == 0).sum()),
                    "average_precision": average_precision_score(labels, probability),
                    "roc_auc": roc_auc_score(labels, probability),
                }
            )
            scored = metadata.copy()
            scored["probability"] = probability
            score_rows.append(scored)
    return (
        models,
        pd.DataFrame(fit_rows),
        pd.DataFrame(metric_rows),
        pd.concat(construction_rows, ignore_index=True),
        pd.concat(score_rows, ignore_index=True),
    )


# Section 5: continuous candidate scoring and alarm consolidation

def recording_candidate_matrix(
    subject: str, comparator: str
) -> tuple[np.ndarray, np.ndarray]:
    onset, features = load_recording_features(subject)
    lookup = {time_key(value): index for index, value in enumerate(onset)}
    candidate_times = []
    matrices = []
    for candidate_time in onset:
        indices = [
            lookup.get(time_key(candidate_time + offset))
            for offset in MODEL_OFFSETS[comparator]
        ]
        if any(index is None for index in indices):
            continue
        candidate_times.append(candidate_time)
        matrices.append(features[np.asarray(indices, dtype=int)].reshape(-1))
    if not matrices:
        raise ValueError(f"No continuous candidates for {subject}, {comparator}")
    return np.asarray(candidate_times, dtype=float), np.vstack(matrices)


def score_recordings(
    assignments: pd.DataFrame, models: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for comparator, model in models.items():
        for item in assignments.itertuples(index=False):
            times, matrix = recording_candidate_matrix(item.subject, comparator)
            probability = model.predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_version": VERSION,
                        "partition": item.partition,
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": times,
                        "probability": probability,
                    }
                )
            )
            support_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": item.partition,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "supported_boundaries": len(times),
                    "supported_hours": len(times) * EPOCH_SEC / 3600.0,
                }
            )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    alarm_rows = []
    marked = scores[scores["probability"] >= threshold].copy()
    for (comparator, partition, subject, pid), group in marked.groupby(
        ["comparator", "partition", "subject", "pid"], sort=True
    ):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        if len(group) == 0:
            continue
        run_start = 0
        gaps = np.flatnonzero(
            np.diff(group["candidate_time_sec"].to_numpy(dtype=float)) > EPOCH_SEC + 1e-6
        )
        run_stops = list(gaps + 1) + [len(group)]
        for run_stop in run_stops:
            run = group.iloc[run_start:run_stop]
            highest = float(run["probability"].max())
            best = run[np.isclose(run["probability"], highest)].sort_values(
                "candidate_time_sec"
            ).iloc[0]
            alarm_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": partition,
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(best["candidate_time_sec"]),
                    "probability": float(best["probability"]),
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
            run_start = run_stop
    return pd.DataFrame(
        alarm_rows,
        columns=[
            "comparator",
            "model_version",
            "partition",
            "subject",
            "pid",
            "event_time_sec",
            "probability",
            "threshold",
            "run_candidates",
        ],
    )


# Section 6: validation threshold selection and event evaluation

def local_event_inputs(
    references: pd.DataFrame,
    partition: str,
    membership: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local = references[references["partition"] == partition]
    eligible_column = (
        "primary_analysis_eligible"
        if membership == "primary"
        else "expanded_quality_analysis_eligible"
    )
    eligibility = truth(local[eligible_column])
    columns = ["subject", "pid", "event_time_sec"]
    return local.loc[eligibility, columns], local.loc[~eligibility, columns]


def select_validation_thresholds(
    scores: pd.DataFrame, support: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = reference_events()
    curve_rows = []
    selected_rows = []
    for comparator in MODEL_OFFSETS:
        local_scores = scores[
            (scores["comparator"] == comparator)
            & (scores["partition"] == "validation")
        ]
        local_support = support[
            (support["comparator"] == comparator)
            & (support["partition"] == "validation")
        ][["subject", "pid", "supported_hours"]]
        eligible, ignored = local_event_inputs(references, "validation", "primary")
        for threshold in THRESHOLDS:
            alarms = collapse_alarms(local_scores, threshold)
            _, _, _, summary = evaluate_events(
                eligible,
                alarms[["subject", "pid", "event_time_sec"]],
                ignored,
                local_support,
                15.0,
            )
            curve_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": "validation",
                    "membership": "primary",
                    "tolerance_sec": 15.0,
                    "threshold": threshold,
                    **summary,
                }
            )
        local_curve = pd.DataFrame(
            [row for row in curve_rows if row["comparator"] == comparator]
        )
        selected = local_curve.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        selected_rows.append(
            {
                **selected.to_dict(),
                "model_sha256": sha256(model_path(comparator)),
                "selection_rule": "max_f1_then_min_far_then_max_recall_then_max_threshold",
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(selected_rows)


def evaluate_selected_scores(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    selected: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    references = reference_events()
    threshold_map = selected.set_index("comparator")["threshold"].to_dict()
    alarm_frames = []
    for (comparator, partition), group in scores.groupby(["comparator", "partition"]):
        alarm_frames.append(collapse_alarms(group, float(threshold_map[comparator])))
    alarms = pd.concat(alarm_frames, ignore_index=True)

    summary_rows = []
    bootstrap_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for comparator, partition in support[["comparator", "partition"]].drop_duplicates().itertuples(index=False):
        local_support = support[
            (support["comparator"] == comparator)
            & (support["partition"] == partition)
        ][["subject", "pid", "supported_hours"]]
        local_predictions = alarms[
            (alarms["comparator"] == comparator)
            & (alarms["partition"] == partition)
        ][["subject", "pid", "event_time_sec"]]
        for membership in MEMBERSHIPS:
            eligible, ignored = local_event_inputs(references, partition, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible,
                    local_predictions,
                    ignored,
                    local_support,
                    tolerance,
                )
                config = {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "model_role": MODEL_ROLES[comparator],
                    "partition": partition,
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": float(threshold_map[comparator]),
                }
                summary_rows.append({**config, **summary})
                bootstrap = participant_bootstrap(
                    participants,
                    resamples=BOOTSTRAP_RESAMPLES,
                    seed=BOOTSTRAP_SEED,
                )
                for key, value in reversed(list(config.items())):
                    bootstrap.insert(0, key, value)
                bootstrap_rows.append(bootstrap)
                for frame, collection in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    if len(frame):
                        frame = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in frame.columns:
                                frame.insert(0, key, value)
                        collection.append(frame)
    return {
        "predicted_events": alarms,
        "event_metrics": pd.DataFrame(summary_rows),
        "event_bootstrap": pd.concat(bootstrap_rows, ignore_index=True),
        "event_recordings": pd.concat(recording_rows, ignore_index=True),
        "event_participants": pd.concat(participant_rows, ignore_index=True),
        "event_matches": pd.concat(match_rows, ignore_index=True)
        if match_rows
        else pd.DataFrame(),
    }


# Section 7: records, comparisons, and interpretation

def save_candidate_scores(scores: pd.DataFrame, phase: str) -> None:
    path = score_path(phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(
        path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


def write_event_outputs(prefix: str, outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        frame.to_csv(output_dir() / f"{prefix}_{name}_v0.1.tsv", sep="\t", index=False)


def external_manifest(assignments: pd.DataFrame) -> pd.DataFrame:
    paths = []
    for subject in assignments["subject"]:
        paths.append(("frozen_recording_features", feature_path(subject)))
    for comparator in MODEL_OFFSETS:
        if model_path(comparator).exists():
            paths.append(("fitted_direct_model", model_path(comparator)))
    for phase in ["train_validation", "test"]:
        if score_path(phase).exists():
            paths.append(("continuous_candidate_scores", score_path(phase)))
    rows = []
    for role, path in paths:
        rows.append(
            {
                "artifact_role": role,
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows).drop_duplicates().sort_values(
        ["artifact_role", "path_relative_to_data_parent"]
    )


def write_validation_freeze(selected: pd.DataFrame, window_metrics: pd.DataFrame) -> None:
    validation_window = window_metrics[window_metrics["partition"] == "validation"].set_index(
        "comparator"
    )
    selected_index = selected.set_index("comparator")
    rows = []
    for comparator in MODEL_OFFSETS:
        threshold = selected_index.loc[comparator]
        window = validation_window.loc[comparator]
        rows.append(
            f"| {comparator} | {window.average_precision:.6f} | {window.roc_auc:.6f} | "
            f"{threshold.threshold:.2f} | {threshold.precision:.6f} | {threshold.recall:.6f} | "
            f"{threshold.f1:.6f} | {threshold.false_alarms_per_hour:.6f} |"
        )
    context_improved = float(selected_index.loc["DE-B", "f1"]) > float(
        selected_index.loc["DE-A", "f1"]
    )
    text = "\n".join(
        [
            "# Direct Baseline Validation Decision and Test Freeze v0.1",
            "",
            "**Created:** 2026-08-22",
            f"**Marker:** `{FREEZE_MARKER}`",
            "",
            "The train/validation phase completed under the predeclared protocol. Test feature arrays were not loaded by this phase.",
            "",
            "| Model | Window AP | Window ROC AUC | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"DE-B improved validation event F1 over DE-A: **{context_improved}**. This records H6.2 without changing DE-B's primary role.",
            "",
            "## Frozen Test Decision",
            "",
            "Apply both fitted models and their recorded validation thresholds once to test feature arrays. No model, feature, alarm, threshold, membership, tolerance, or selection rule may be changed after this file. Retain the outcome whether positive, negative, or inconclusive.",
            "",
        ]
    )
    output_dir().joinpath("validation_decision_and_test_freeze_v0.1.md").write_text(
        text, encoding="utf-8"
    )


def paired_direct_vs_sf_c(
    direct_participants: pd.DataFrame,
) -> pd.DataFrame:
    direct = direct_participants[
        (direct_participants["comparator"] == "DE-B")
        & (direct_participants["membership"] == "primary")
        & (direct_participants["tolerance_sec"] == 15.0)
    ].copy()
    stage = pd.read_csv(
        repo_root()
        / "experiments/2026-08-15_stage_first_feature_baseline_v0.1/test_event_participants_v0.1.tsv",
        sep="\t",
    )
    stage = stage[
        (stage["comparator"] == "SF-C")
        & (stage["membership"] == "primary")
        & (stage["tolerance_sec"] == 15.0)
    ].copy()
    columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
    ]
    paired = direct[columns].merge(
        stage[columns], on="pid", suffixes=("_direct", "_stage"), validate="one_to_one"
    )
    if len(paired) != 20:
        raise ValueError("Expected 20 paired test participants")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_rows = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        metrics = {}
        for suffix in ["direct", "stage"]:
            metrics[suffix] = metric_values(
                int(sampled[f"true_positive_{suffix}"].sum()),
                int(sampled[f"false_positive_{suffix}"].sum()),
                int(sampled[f"false_negative_{suffix}"].sum()),
                float(sampled[f"supported_hours_{suffix}"].sum()),
            )
        sample_rows.append(
            {
                "event_f1_difference": metrics["direct"]["f1"] - metrics["stage"]["f1"],
                "false_alarms_per_hour_difference": metrics["direct"]["false_alarms_per_hour"]
                - metrics["stage"]["false_alarms_per_hour"],
            }
        )
    samples = pd.DataFrame(sample_rows)
    direct_point = metric_values(
        int(paired["true_positive_direct"].sum()),
        int(paired["false_positive_direct"].sum()),
        int(paired["false_negative_direct"].sum()),
        float(paired["supported_hours_direct"].sum()),
    )
    stage_point = metric_values(
        int(paired["true_positive_stage"].sum()),
        int(paired["false_positive_stage"].sum()),
        int(paired["false_negative_stage"].sum()),
        float(paired["supported_hours_stage"].sum()),
    )
    result = []
    point_values = {
        "event_f1_difference": direct_point["f1"] - stage_point["f1"],
        "false_alarms_per_hour_difference": direct_point["false_alarms_per_hour"]
        - stage_point["false_alarms_per_hour"],
    }
    for metric, point in point_values.items():
        result.append(
            {
                "comparison": "DE-B_minus_SF-C",
                "metric": metric,
                "point_difference": point,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "lower_95": float(samples[metric].quantile(0.025)),
                "median": float(samples[metric].quantile(0.5)),
                "upper_95": float(samples[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(result)


def comparator_summary(direct_metrics: pd.DataFrame) -> pd.DataFrame:
    direct = direct_metrics[
        (direct_metrics["partition"] == "test")
        & (direct_metrics["membership"] == "primary")
        & (direct_metrics["tolerance_sec"] == 15.0)
    ].copy()
    stage = pd.read_csv(
        repo_root()
        / "experiments/2026-08-15_stage_first_comparison_v0.1/stage_first_event_metrics_v0.1.tsv",
        sep="\t",
    )
    stage = stage[
        (stage["partition"] == "test")
        & (stage["membership"] == "primary")
        & (stage["tolerance_sec"] == 15.0)
        & stage["comparator"].isin(["SF-A", "SF-C"])
    ].copy()
    columns = [
        "comparator",
        "reference_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
    ]
    result = pd.concat([stage[columns], direct[columns]], ignore_index=True)
    result.insert(
        1,
        "interpretation_role",
        result["comparator"].map(
            {
                "SF-A": "fixed_descriptive_unknown_training_provenance",
                "SF-C": "transparent_stage_first_primary",
                "DE-A": MODEL_ROLES["DE-A"],
                "DE-B": MODEL_ROLES["DE-B"],
            }
        ),
    )
    return result


def write_readme() -> None:
    selected = pd.read_csv(output_dir() / "frozen_model_thresholds_v0.1.tsv", sep="\t")
    window = pd.read_csv(output_dir() / "train_validation_window_metrics_v0.1.tsv", sep="\t")
    validation_events = pd.read_csv(
        output_dir() / "train_validation_event_metrics_v0.1.tsv", sep="\t"
    )
    sections = [
        "# Direct REM-to-Wake Feature Baseline v0.1",
        "",
        "**Created:** 2026-08-22",
        "**Protocol:** `docs/evaluation/direct_event_baseline_protocol_v0.1.md`",
        "**Models:** DE-A boundary-pair logistic ablation; DE-B eight-epoch-context logistic primary",
        "",
        "## Validation Result",
        "",
        "| Model | Window AP | Window ROC AUC | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparator in MODEL_OFFSETS:
        window_row = window[
            (window["comparator"] == comparator)
            & (window["partition"] == "validation")
        ].iloc[0]
        event_row = validation_events[
            (validation_events["comparator"] == comparator)
            & (validation_events["partition"] == "validation")
            & (validation_events["membership"] == "primary")
            & (validation_events["tolerance_sec"] == 15.0)
        ].iloc[0]
        threshold = float(selected[selected["comparator"] == comparator]["threshold"].iloc[0])
        sections.append(
            f"| {comparator} | {window_row.average_precision:.4f} | {window_row.roc_auc:.4f} | {threshold:.2f} | "
            f"{event_row.precision:.4f} | {event_row.recall:.4f} | {event_row.f1:.4f} | {event_row.false_alarms_per_hour:.4f} |"
        )
    test_path = output_dir() / "test_event_metrics_v0.1.tsv"
    if test_path.exists():
        comparison = pd.read_csv(output_dir() / "test_comparator_summary_v0.1.tsv", sep="\t")
        sections.extend(
            [
                "",
                "## Frozen Test Result",
                "",
                "| Model | Role | Precision | Recall | Event F1 | False alarms/hour |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for item in comparison.itertuples(index=False):
            sections.append(
                f"| {item.comparator} | {item.interpretation_role} | {item.precision:.4f} | "
                f"{item.recall:.4f} | {item.f1:.4f} | {item.false_alarms_per_hour:.4f} |"
            )
        index = comparison.set_index("comparator")
        supported = (
            float(index.loc["DE-B", "f1"]) > float(index.loc["SF-C", "f1"])
            and float(index.loc["DE-B", "false_alarms_per_hour"])
            < float(index.loc["SF-C", "false_alarms_per_hour"])
        )
        sections.extend(
            [
                "",
                "## Decision",
                "",
                f"The prespecified directional direct-value criterion was met: **{supported}**. The criterion requires DE-B to have both higher event F1 and lower false alarms per supported hour than SF-C.",
                "",
                "This test partition was previously used for the planned stage-first comparator. The direct configuration and thresholds were frozen before direct test-feature access, but external confirmation remains necessary.",
            ]
        )
    sections.extend(
        [
            "",
            "## Artifact Boundary",
            "",
            "Fitted models and continuous candidate scores remain outside Git. Git retains the protocol, fit and threshold records, event outputs, comparison, and SHA-256 manifest.",
            "",
        ]
    )
    output_dir().joinpath("README.md").write_text("\n".join(sections), encoding="utf-8")


# Section 8: separated train/validation and test execution

def train_validation_phase() -> None:
    assignments = subject_assignments()
    active = assignments[assignments["partition"].isin(["train", "validation"])].copy()
    output_dir().mkdir(parents=True, exist_ok=True)
    derived_dir().mkdir(parents=True, exist_ok=True)
    candidates = labeled_candidates()
    active_candidates = candidates[candidates["partition"].isin(["train", "validation"])]
    models, fit, window_metrics, construction, labeled_scores = fit_models(active_candidates)
    continuous_scores, support = score_recordings(active, models)
    save_candidate_scores(continuous_scores, "train_validation")
    curve, selected = select_validation_thresholds(continuous_scores, support)
    outputs = evaluate_selected_scores(continuous_scores, support, selected)

    fit.to_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t", index=False)
    window_metrics.to_csv(
        output_dir() / "train_validation_window_metrics_v0.1.tsv", sep="\t", index=False
    )
    construction.to_csv(
        output_dir() / "train_validation_construction_summary_v0.1.tsv",
        sep="\t",
        index=False,
    )
    labeled_scores.to_csv(
        output_dir() / "train_validation_labeled_window_scores_v0.1.tsv",
        sep="\t",
        index=False,
    )
    support.to_csv(
        output_dir() / "train_validation_event_support_v0.1.tsv", sep="\t", index=False
    )
    curve.to_csv(
        output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t", index=False
    )
    selected.to_csv(
        output_dir() / "frozen_model_thresholds_v0.1.tsv", sep="\t", index=False
    )
    write_event_outputs("train_validation", outputs)
    write_validation_freeze(selected, window_metrics)
    external_manifest(active).to_csv(
        output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    environment = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": __import__("sklearn").__version__,
        "joblib": joblib.__version__,
    }
    output_dir().joinpath("software_versions_v0.1.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    write_readme()
    print(selected[["comparator", "threshold", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))


def test_phase() -> None:
    freeze = output_dir() / "validation_decision_and_test_freeze_v0.1.md"
    if not freeze.exists() or FREEZE_MARKER not in freeze.read_text(encoding="utf-8"):
        raise RuntimeError("Test phase blocked: validation freeze marker is missing")
    selected = pd.read_csv(output_dir() / "frozen_model_thresholds_v0.1.tsv", sep="\t")
    models = {}
    for item in selected.itertuples(index=False):
        path = model_path(item.comparator)
        if sha256(path) != item.model_sha256:
            raise RuntimeError(f"Frozen model hash mismatch for {item.comparator}")
        models[item.comparator] = joblib.load(path)
    assignments = subject_assignments()
    active = assignments[assignments["partition"] == "test"].copy()
    continuous_scores, support = score_recordings(active, models)
    save_candidate_scores(continuous_scores, "test")
    outputs = evaluate_selected_scores(continuous_scores, support, selected)
    support.to_csv(output_dir() / "test_event_support_v0.1.tsv", sep="\t", index=False)
    write_event_outputs("test", outputs)
    comparison = comparator_summary(outputs["event_metrics"])
    comparison.to_csv(
        output_dir() / "test_comparator_summary_v0.1.tsv", sep="\t", index=False
    )
    paired_direct_vs_sf_c(outputs["event_participants"]).to_csv(
        output_dir() / "test_de_b_vs_sf_c_paired_bootstrap_v0.1.tsv",
        sep="\t",
        index=False,
    )
    external_manifest(assignments).to_csv(
        output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    output_dir().joinpath("test_access_record_v0.1.md").write_text(
        "# Direct Baseline Test Access Record v0.1\n\n"
        "**First direct-model test-feature access:** 2026-08-22\n"
        "**Access count under direct protocol:** 1\n\n"
        "The broader test partition had already been used for the stage-first comparator. "
        "For this direct experiment, model files and validation-selected thresholds were hash-locked "
        "before loading test feature arrays. Both direct results were retained without refitting or adjustment.\n",
        encoding="utf-8",
    )
    write_readme()
    primary = outputs["event_metrics"]
    primary = primary[
        (primary["membership"] == "primary")
        & (primary["tolerance_sec"] == 15.0)
    ]
    print(primary[["comparator", "threshold", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["train-validation", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "train-validation":
        train_validation_phase()
    else:
        test_phase()


if __name__ == "__main__":
    main()
