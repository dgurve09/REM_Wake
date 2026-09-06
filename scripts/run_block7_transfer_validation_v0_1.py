"""Run the frozen Block 7 train/validation paired-transfer comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values
from validate_block7_feature_generation_v0_1 import (
    HB2,
    PSG2,
    PSG6,
    epoch_features,
    filter_resample,
    filter_sos,
    read_uv,
    valid_events,
    verify_or_create_npz,
)


# Section 1: frozen execution configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-06_block7_transfer_validation_v0.1"
DERIVED_DIR = "block7_transfer_validation_v0.1"
FEATURE_GATE_DIR = "block7_feature_generation_validation_v0.1"
PLAN_COMMIT = "a10dd71"
FEATURE_GATE_RESULT_COMMIT = "1d914ab"
AUTHORIZED_PARTITIONS = {"train", "validation"}
EPOCH_SEC = 30.0
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, EPOCH_SEC)
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906
ROBUST_SCALE_FACTOR = 1.4826
PSG_OVERLAP_TOLERANCE = 1e-10
ZERO_SHOT_F1_DEFICIT = 0.03
ZERO_SHOT_FAR_EXCESS = 0.50
SHIFT_DIMENSION_THRESHOLD = 0.50
SHIFT_DIMENSION_FRACTION = 0.20

DIRECT_MODALITIES = {
    "P6-D": "PSG-6",
    "P2-D": "PSG-2",
    "H2-D": "HB-2",
}
COMPARATOR_ROLES = {
    "P6-D": "direct_six_channel_psg",
    "P2-D": "direct_two_channel_psg",
    "H2-D": "direct_two_channel_wearable",
    "P2-H2-Z": "strict_psg_to_wearable_zero_shot",
    "P2-H2-A": "conditional_train_only_robust_alignment",
}
MODALITY_FOLDERS = {
    "PSG-6": "psg6",
    "PSG-2": "psg2",
    "HB-2": "hb2",
    "HB-2-PSGscale": "hb2_psgscale",
}


# Section 2: paths, identifiers, and immutable external files

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def feature_gate_path(subject: str, modality: str) -> Path:
    return (
        data_parent()
        / "derived"
        / FEATURE_GATE_DIR
        / "recording_features"
        / MODALITY_FOLDERS[modality]
        / f"{subject}_features_v0.1.npz"
    )


def generated_feature_path(subject: str, partition: str, modality: str) -> Path:
    return (
        derived_dir()
        / "recording_features"
        / partition
        / MODALITY_FOLDERS[modality]
        / f"{subject}_features_v0.1.npz"
    )


def feature_path(subject: str, partition: str, modality: str) -> Path:
    if partition == "train" and modality in {"PSG-6", "PSG-2", "HB-2"}:
        return feature_gate_path(subject, modality)
    return generated_feature_path(subject, partition, modality)


def model_path(comparator: str) -> Path:
    name = comparator.lower().replace("-", "_")
    return derived_dir() / "models" / f"{name}_model_v0.1.joblib"


def score_path(kind: str) -> Path:
    return derived_dir() / "candidate_scores" / f"{kind}_scores_v0.1.tsv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def verify_or_create_gzip_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as stream:
            frame.to_csv(stream, sep="\t", index=False, lineterminator="\n")
    if path.exists():
        if sha256(path) != sha256(temporary):
            temporary.unlink()
            raise RuntimeError(f"External score artifact changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


# Section 3: frozen train and validation membership

def subject_assignments() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    source = source[source["partition"].isin(AUTHORIZED_PARTITIONS)].copy()
    rows = []
    for item in source.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append(
                {"subject": subject, "pid": int(item.pid), "partition": item.partition}
            )
    result = pd.DataFrame(rows)
    counts = result.groupby("partition")["subject"].count().to_dict()
    pid_counts = result.groupby("partition")["pid"].nunique().to_dict()
    if counts != {"train": 82, "validation": 20}:
        raise ValueError(f"Unexpected authorized recording counts: {counts}")
    if pid_counts != {"train": 64, "validation": 16}:
        raise ValueError(f"Unexpected authorized pid counts: {pid_counts}")
    if result["subject"].duplicated().any():
        raise ValueError("A recording appears more than once in the authorized assignments")
    return result.sort_values("subject", key=lambda value: value.map(subject_number))


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


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
    authorized = set(assignments["subject"])
    rows = membership[
        membership["subject"].isin(authorized)
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    if not set(rows["partition"]).issubset(AUTHORIZED_PARTITIONS):
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
    authorized = set(assignments["subject"])
    negative = membership[
        membership["subject"].isin(authorized)
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
    rows = pd.concat([positive_rows, negative_rows], ignore_index=True)
    if rows["sample_id"].duplicated().any():
        raise ValueError("Labeled candidate identifiers are not unique")
    if not set(rows["partition"]).issubset(AUTHORIZED_PARTITIONS):
        raise ValueError("Unauthorized labeled-candidate partition")
    return rows.sort_values(["partition", "subject", "candidate_time_sec", "label"])


# Section 4: validation and zero-shot feature generation

def scaler_maps() -> tuple[dict, dict, dict]:
    rows = pd.read_csv(
        repo_root()
        / "experiments/2026-09-06_block7_feature_generation_validation_v0.1/train_robust_scalers_v0.1.tsv",
        sep="\t",
    )
    psg = rows[rows["scaler_owner"] == "PSG-6"].set_index("channel")
    hb = rows[rows["scaler_owner"] == "HB-2"].set_index("channel")

    def selected(table: pd.DataFrame, channels: list[str]) -> dict:
        return {
            channel: {
                "median_uv": float(table.loc[channel, "median_uv"]),
                "robust_scale_uv": float(table.loc[channel, "robust_scale_uv"]),
            }
            for channel in channels
        }

    psg_map = selected(psg, PSG6)
    hb_map = selected(hb, HB2)
    zero_map = {
        "HB_1": psg_map["PSG_F3"],
        "HB_2": psg_map["PSG_F4"],
    }
    return psg_map, hb_map, zero_map


def normalize_signal(
    signal: np.ndarray, channels: list[str], scaler: dict
) -> np.ndarray:
    result = signal.copy()
    for index, channel in enumerate(channels):
        center = float(scaler[channel]["median_uv"])
        scale = float(scaler[channel]["robust_scale_uv"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid scaler for {channel}")
        result[index] = (result[index] - center) / scale
    return result


def feature_artifact_row(
    subject: str,
    pid: int,
    partition: str,
    modality: str,
    path: Path,
    onsets: np.ndarray,
    features: np.ndarray,
) -> dict:
    centers, _ = context_matrix(onsets, features)
    return {
        "subject": subject,
        "pid": int(pid),
        "partition": partition,
        "modality": modality,
        "epochs": len(onsets),
        "context_rows": len(centers),
        "base_feature_dimensions": features.shape[1],
        "all_features_finite": bool(np.isfinite(features).all()),
        "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def save_feature(
    subject: str,
    pid: int,
    partition: str,
    modality: str,
    onsets: np.ndarray,
    stages: np.ndarray,
    features: np.ndarray,
    names: list[str],
) -> dict:
    path = generated_feature_path(subject, partition, modality)
    verify_or_create_npz(path, onsets, stages, features, names)
    return feature_artifact_row(
        subject, pid, partition, modality, path, onsets, features
    )


def generate_required_features(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    psg_scaler, hb_scaler, zero_scaler = scaler_maps()
    sos = filter_sos()
    artifact_rows = []
    parity_rows = []
    total = len(assignments)
    for index, item in enumerate(assignments.itertuples(index=False), start=1):
        print(f"Feature generation {index}/{total}: {item.subject} ({item.partition})", flush=True)
        events = valid_events(item.subject)

        if item.partition == "validation":
            psg_raw = read_uv(item.subject, "psg", PSG6)
            psg6_signal = normalize_signal(
                filter_resample(psg_raw, sos), PSG6, psg_scaler
            )
            psg2_signal = normalize_signal(
                filter_resample(psg_raw[:2], sos), PSG2, psg_scaler
            )
            psg6 = epoch_features(psg6_signal, PSG6, events)
            psg2 = epoch_features(psg2_signal, PSG2, events)
            artifact_rows.append(
                save_feature(item.subject, item.pid, item.partition, "PSG-6", *psg6)
            )
            artifact_rows.append(
                save_feature(item.subject, item.pid, item.partition, "PSG-2", *psg2)
            )
        else:
            psg6 = load_feature(item.subject, item.partition, "PSG-6", include_stage=True)
            psg2 = load_feature(item.subject, item.partition, "PSG-2", include_stage=True)

        hb_raw = read_uv(item.subject, "headband", HB2)
        hb_filtered = filter_resample(hb_raw, sos)
        zero_signal = normalize_signal(hb_filtered, HB2, zero_scaler)
        hb_zero = epoch_features(zero_signal, HB2, events)
        artifact_rows.append(
            save_feature(
                item.subject,
                item.pid,
                item.partition,
                "HB-2-PSGscale",
                *hb_zero,
            )
        )

        if item.partition == "validation":
            hb_signal = normalize_signal(hb_filtered, HB2, hb_scaler)
            hb_direct = epoch_features(hb_signal, HB2, events)
            artifact_rows.append(
                save_feature(
                    item.subject, item.pid, item.partition, "HB-2", *hb_direct
                )
            )
        else:
            hb_direct = load_feature(
                item.subject, item.partition, "HB-2", include_stage=True
            )

        psg_overlap = float(
            np.max(np.abs(psg6[2][:, : len(psg2[3])] - psg2[2]))
        )
        timing_pass = all(
            np.array_equal(psg6[0], values[0]) and np.array_equal(psg6[1], values[1])
            for values in [psg2, hb_direct, hb_zero]
        )
        context_centers = [context_matrix(values[0], values[2])[0] for values in [psg6, psg2, hb_direct, hb_zero]]
        context_pass = all(
            np.array_equal(context_centers[0], centers) for centers in context_centers[1:]
        )
        parity_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": item.partition,
                "epoch_timing_and_stage_parity": timing_pass,
                "context_center_parity": context_pass,
                "psg_overlap_max_abs_difference": psg_overlap,
                "psg_overlap_pass": psg_overlap <= PSG_OVERLAP_TOLERANCE,
            }
        )
    return pd.DataFrame(artifact_rows), pd.DataFrame(parity_rows)


# Section 5: context matrices and labeled direct fits

def load_feature(
    subject: str,
    partition: str,
    modality: str,
    include_stage: bool = False,
):
    path = feature_path(subject, partition, modality)
    with np.load(path, allow_pickle=False) as values:
        onset = values["onset"].astype(np.float64)
        stage = values["stage"].astype(np.int8)
        features = values["features"].astype(np.float32)
        names = values["feature_names"].astype(str).tolist()
    if len(onset) != len(stage) or features.shape[0] != len(onset):
        raise ValueError(f"Invalid feature dimensions for {subject}, {modality}")
    if len(np.unique(onset)) != len(onset) or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature values for {subject}, {modality}")
    if include_stage:
        return onset, stage, features, names
    return onset, features, names


def context_matrix(
    onsets: np.ndarray, features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    indices = context_start_indices(onsets)
    if len(indices) == 0:
        raise ValueError("No complete eight-epoch contexts")
    matrix = np.concatenate(
        [features[indices + offset] for offset in range(len(CONTEXT_OFFSETS))],
        axis=1,
    )
    return onsets[indices + 4], matrix


def context_start_indices(onsets: np.ndarray) -> np.ndarray:
    required_gaps = len(CONTEXT_OFFSETS) - 1
    if len(onsets) < len(CONTEXT_OFFSETS):
        return np.asarray([], dtype=int)
    contiguous = np.isclose(
        np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0
    ).astype(np.int8)
    counts = np.convolve(
        contiguous, np.ones(required_gaps, dtype=np.int8), mode="valid"
    )
    return np.flatnonzero(counts == required_gaps)


def time_key(value: float) -> int:
    return int(round(float(value) * 1000.0))


def build_labeled_matrix(
    rows: pd.DataFrame, comparator: str
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    modality = DIRECT_MODALITIES[comparator]
    matrices = []
    retained = []
    dropped = []
    for subject, group in rows.groupby("subject", sort=True):
        partition = str(group["partition"].iloc[0])
        onsets, features, _ = load_feature(subject, partition, modality)
        centers, contexts = context_matrix(onsets, features)
        lookup = {time_key(value): index for index, value in enumerate(centers)}
        for item in group.itertuples(index=False):
            row = {**item._asdict(), "comparator": comparator}
            index = lookup.get(time_key(item.candidate_time_sec))
            if index is None:
                row["drop_reason"] = "missing_required_context"
                dropped.append(row)
            else:
                matrices.append(contexts[index])
                retained.append(row)
    columns = list(rows.columns) + ["comparator"]
    if not matrices:
        raise ValueError(f"No labeled rows retained for {comparator}")
    return (
        np.vstack(matrices),
        pd.DataFrame(retained, columns=columns),
        pd.DataFrame(dropped, columns=columns + ["drop_reason"]),
    )


def construction_summary(
    requested: pd.DataFrame,
    retained: pd.DataFrame,
    dropped: pd.DataFrame,
    comparator: str,
) -> pd.DataFrame:
    keys = ["partition", "label", "source_tier"]
    request = requested.groupby(keys, as_index=False).size().rename(
        columns={"size": "requested_rows"}
    )
    keep = retained.groupby(keys, as_index=False).size().rename(
        columns={"size": "retained_rows"}
    )
    drop = dropped.groupby(keys, as_index=False).size().rename(
        columns={"size": "dropped_rows"}
    )
    result = request.merge(keep, on=keys, how="left").merge(drop, on=keys, how="left")
    result[["retained_rows", "dropped_rows"]] = result[
        ["retained_rows", "dropped_rows"]
    ].fillna(0).astype(int)
    result.insert(0, "comparator", comparator)
    return result


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


def model_arrays(model) -> list[np.ndarray]:
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    return [
        scaler.mean_,
        scaler.scale_,
        scaler.var_,
        logistic.coef_,
        logistic.intercept_,
        logistic.classes_,
    ]


def verify_or_create_model(model, path: Path) -> None:
    if path.exists():
        existing = joblib.load(path)
        checks = [
            np.array_equal(left, right)
            for left, right in zip(model_arrays(model), model_arrays(existing))
        ]
        if not all(checks):
            raise RuntimeError(f"Fitted model changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def fit_direct_models(candidates: pd.DataFrame):
    models = {}
    fit_rows = []
    metric_rows = []
    construction_rows = []
    labeled_score_rows = []
    for comparator in DIRECT_MODALITIES:
        matrices = {}
        metadata = {}
        for partition in ["train", "validation"]:
            requested = candidates[candidates["partition"] == partition].copy()
            matrix, retained, dropped = build_labeled_matrix(requested, comparator)
            matrices[partition] = matrix
            metadata[partition] = retained
            construction_rows.append(
                construction_summary(requested, retained, dropped, comparator)
            )

        train_y = metadata["train"]["label"].to_numpy(dtype=int)
        model = build_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(matrices["train"], train_y)
        convergence = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        verify_or_create_model(model, model_path(comparator))
        models[comparator] = model
        logistic = model.named_steps["logisticregression"]
        fit_rows.append(
            {
                "comparator": comparator,
                "model_role": COMPARATOR_ROLES[comparator],
                "train_rows": len(train_y),
                "train_positive": int(train_y.sum()),
                "train_negative": int((train_y == 0).sum()),
                "input_features": matrices["train"].shape[1],
                "maximum_iterations_used": int(logistic.n_iter_.max()),
                "convergence_warning_count": len(convergence),
                "model_sha256": sha256(model_path(comparator)),
            }
        )
        for partition in ["train", "validation"]:
            labels = metadata[partition]["label"].to_numpy(dtype=int)
            probability = model.predict_proba(matrices[partition])[:, 1]
            metric_rows.append(
                {
                    "comparator": comparator,
                    "partition": partition,
                    "rows": len(labels),
                    "positive_rows": int(labels.sum()),
                    "negative_rows": int((labels == 0).sum()),
                    "average_precision": average_precision_score(labels, probability),
                    "roc_auc": roc_auc_score(labels, probability),
                }
            )
            scored = metadata[partition].copy()
            scored["probability"] = probability
            labeled_score_rows.append(scored)
    return (
        models,
        pd.DataFrame(fit_rows),
        pd.DataFrame(metric_rows),
        pd.concat(construction_rows, ignore_index=True),
        pd.concat(labeled_score_rows, ignore_index=True),
    )


# Section 6: validation full-night scores and direct thresholds

def validation_scores(assignments: pd.DataFrame, models: dict):
    validation = assignments[assignments["partition"] == "validation"]
    score_rows = []
    support_rows = []
    specifications = [
        ("P6-D", "P6-D", "PSG-6"),
        ("P2-D", "P2-D", "PSG-2"),
        ("H2-D", "H2-D", "HB-2"),
        ("P2-H2-Z", "P2-D", "HB-2-PSGscale"),
    ]
    for comparator, model_name, modality in specifications:
        for item in validation.itertuples(index=False):
            onsets, features, _ = load_feature(item.subject, item.partition, modality)
            centers, matrix = context_matrix(onsets, features)
            probability = models[model_name].predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": model_name,
                        "partition": "validation",
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": centers,
                        "probability": probability,
                    }
                )
            )
            support_rows.append(
                {
                    "comparator": comparator,
                    "partition": "validation",
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "supported_boundaries": len(centers),
                    "supported_hours": len(centers) * EPOCH_SEC / 3600.0,
                }
            )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
    for (comparator, subject, pid), group in marked.groupby(
        ["comparator", "subject", "pid"], sort=True
    ):
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
                    "comparator": comparator,
                    "partition": "validation",
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(best.candidate_time_sec),
                    "probability": float(best.probability),
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "comparator",
            "partition",
            "subject",
            "pid",
            "event_time_sec",
            "probability",
            "threshold",
            "run_candidates",
        ],
    )


def local_event_inputs(
    references: pd.DataFrame, membership: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_column = (
        "primary_analysis_eligible"
        if membership == "primary"
        else "expanded_quality_analysis_eligible"
    )
    eligible = truth(references[eligible_column])
    columns = ["subject", "pid", "event_time_sec"]
    return references.loc[eligible, columns], references.loc[~eligible, columns]


def direct_thresholds(
    scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows = []
    selected_rows = []
    eligible, ignored = local_event_inputs(references, "primary")
    for comparator in DIRECT_MODALITIES:
        local_scores = scores[scores["comparator"] == comparator]
        local_support = support[support["comparator"] == comparator][
            ["subject", "pid", "supported_hours"]
        ]
        rows = []
        for threshold in THRESHOLDS:
            alarms = collapse_alarms(local_scores, float(threshold))
            _, _, _, summary = evaluate_events(
                eligible,
                alarms[["subject", "pid", "event_time_sec"]],
                ignored,
                local_support,
                15.0,
            )
            row = {
                "comparator": comparator,
                "partition": "validation",
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
                "model_sha256": sha256(model_path(comparator)),
                "selection_rule": "max_f1_then_min_far_then_max_recall_then_max_threshold",
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(selected_rows)


def primary_metric(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    comparator: str,
    threshold: float,
) -> dict:
    eligible, ignored = local_event_inputs(references, "primary")
    local_scores = scores[scores["comparator"] == comparator]
    local_support = support[support["comparator"] == comparator][
        ["subject", "pid", "supported_hours"]
    ]
    alarms = collapse_alarms(local_scores, threshold)
    _, _, _, summary = evaluate_events(
        eligible,
        alarms[["subject", "pid", "event_time_sec"]],
        ignored,
        local_support,
        15.0,
    )
    return summary


# Section 7: train-only feature-shift gate and conditional alignment

def feature_shift(assignments: pd.DataFrame) -> pd.DataFrame:
    train = assignments[assignments["partition"] == "train"]
    cache = {}
    for item in train.itertuples(index=False):
        psg_onset, psg_feature, psg_names = load_feature(
            item.subject, "train", "PSG-2"
        )
        hb_onset, hb_feature, hb_names = load_feature(
            item.subject, "train", "HB-2-PSGscale"
        )
        psg_starts = context_start_indices(psg_onset)
        hb_starts = context_start_indices(hb_onset)
        if not np.array_equal(psg_onset[psg_starts + 4], hb_onset[hb_starts + 4]):
            raise ValueError(f"Train common support differs for {item.subject}")
        cache[item.subject] = (
            psg_feature,
            hb_feature,
            psg_starts,
            hb_starts,
            psg_names,
            hb_names,
        )

    rows = []
    for offset_index, offset_sec in enumerate(CONTEXT_OFFSETS):
        source = np.vstack(
            [values[0][values[2] + offset_index] for values in cache.values()]
        )
        target = np.vstack(
            [values[1][values[3] + offset_index] for values in cache.values()]
        )
        for feature_index in range(10):
            source_values = source[:, feature_index].astype(np.float64)
            target_values = target[:, feature_index].astype(np.float64)
            source_median = float(np.median(source_values))
            target_median = float(np.median(target_values))
            source_mad = float(np.median(np.abs(source_values - source_median)))
            target_mad = float(np.median(np.abs(target_values - target_median)))
            source_raw_scale = ROBUST_SCALE_FACTOR * source_mad
            target_raw_scale = ROBUST_SCALE_FACTOR * target_mad
            pooled = np.concatenate([source_values, target_values])
            pooled_median = float(np.median(pooled))
            pooled_mad = float(np.median(np.abs(pooled - pooled_median)))
            pooled_raw_scale = ROBUST_SCALE_FACTOR * pooled_mad
            source_scale = source_raw_scale if source_raw_scale > 0 else 1.0
            target_scale = target_raw_scale if target_raw_scale > 0 else 1.0
            pooled_scale = pooled_raw_scale if pooled_raw_scale > 0 else 1.0
            difference = abs(source_median - target_median) / pooled_scale
            first = next(iter(cache.values()))
            rows.append(
                {
                    "dimension_index": offset_index * 10 + feature_index,
                    "context_offset_sec": float(offset_sec),
                    "source_feature_name": first[4][feature_index],
                    "target_feature_name": first[5][feature_index],
                    "common_train_boundaries": len(source_values),
                    "source_median": source_median,
                    "source_mad": source_mad,
                    "source_robust_scale": source_scale,
                    "source_zero_scale_replaced": source_raw_scale == 0,
                    "target_median": target_median,
                    "target_mad": target_mad,
                    "target_robust_scale": target_scale,
                    "target_zero_scale_replaced": target_raw_scale == 0,
                    "pooled_median": pooled_median,
                    "pooled_mad": pooled_mad,
                    "pooled_robust_scale": pooled_scale,
                    "pooled_zero_scale_replaced": pooled_raw_scale == 0,
                    "absolute_median_difference_pooled_scale": difference,
                    "exceeds_0_50_pooled_scale": difference > SHIFT_DIMENSION_THRESHOLD,
                }
            )
    return pd.DataFrame(rows).sort_values("dimension_index")


def adaptation_gate(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    selected: pd.DataFrame,
    shift: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = selected.set_index("comparator")["threshold"].to_dict()
    h2 = primary_metric(scores, support, references, "H2-D", thresholds["H2-D"])
    zero = primary_metric(scores, support, references, "P2-H2-Z", thresholds["P2-D"])
    f1_deficit = float(h2["f1"] - zero["f1"])
    far_excess = float(zero["false_alarms_per_hour"] - h2["false_alarms_per_hour"])
    performance_open = (
        f1_deficit >= ZERO_SHOT_F1_DEFICIT or far_excess >= ZERO_SHOT_FAR_EXCESS
    )
    shifted_dimensions = int(truth(shift["exceeds_0_50_pooled_scale"]).sum())
    shifted_fraction = shifted_dimensions / len(shift)
    distribution_open = shifted_fraction >= SHIFT_DIMENSION_FRACTION
    gate_open = performance_open and distribution_open
    return pd.DataFrame(
        [
            {
                "h2_direct_validation_f1": h2["f1"],
                "zero_shot_validation_f1": zero["f1"],
                "zero_shot_f1_deficit": f1_deficit,
                "required_f1_deficit": ZERO_SHOT_F1_DEFICIT,
                "h2_direct_false_alarms_per_hour": h2["false_alarms_per_hour"],
                "zero_shot_false_alarms_per_hour": zero["false_alarms_per_hour"],
                "zero_shot_far_excess": far_excess,
                "required_far_excess": ZERO_SHOT_FAR_EXCESS,
                "performance_condition_open": performance_open,
                "shifted_dimensions": shifted_dimensions,
                "total_dimensions": len(shift),
                "shifted_dimension_fraction": shifted_fraction,
                "required_shifted_fraction": SHIFT_DIMENSION_FRACTION,
                "distribution_condition_open": distribution_open,
                "adaptation_gate_open": gate_open,
                "adaptation_action": "execute_P2-H2-A_once" if gate_open else "skip_P2-H2-A",
            }
        ]
    )


def add_aligned_scores(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    assignments: pd.DataFrame,
    p2_model,
    shift: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_median = shift["source_median"].to_numpy(dtype=float)
    source_scale = shift["source_robust_scale"].to_numpy(dtype=float)
    target_median = shift["target_median"].to_numpy(dtype=float)
    target_scale = shift["target_robust_scale"].to_numpy(dtype=float)
    score_rows = []
    support_rows = []
    validation = assignments[assignments["partition"] == "validation"]
    for item in validation.itertuples(index=False):
        onsets, features, _ = load_feature(
            item.subject, "validation", "HB-2-PSGscale"
        )
        centers, matrix = context_matrix(onsets, features)
        aligned = ((matrix - target_median) / target_scale) * source_scale + source_median
        probability = p2_model.predict_proba(aligned)[:, 1]
        score_rows.append(
            pd.DataFrame(
                {
                    "comparator": "P2-H2-A",
                    "model_source": "P2-D",
                    "partition": "validation",
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": centers,
                    "probability": probability,
                }
            )
        )
        support_rows.append(
            {
                "comparator": "P2-H2-A",
                "partition": "validation",
                "subject": item.subject,
                "pid": int(item.pid),
                "supported_boundaries": len(centers),
                "supported_hours": len(centers) * EPOCH_SEC / 3600.0,
            }
        )
    return (
        pd.concat([scores, *score_rows], ignore_index=True),
        pd.concat([support, pd.DataFrame(support_rows)], ignore_index=True),
    )


# Section 8: complete event evaluation and paired participant contrasts

def evaluate_all(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    thresholds: dict[str, float],
) -> dict[str, pd.DataFrame]:
    alarm_frames = []
    for comparator, group in scores.groupby("comparator", sort=True):
        alarm_frames.append(collapse_alarms(group, thresholds[comparator]))
    alarms = pd.concat(alarm_frames, ignore_index=True)

    summaries = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for comparator in sorted(scores["comparator"].unique()):
        local_support = support[support["comparator"] == comparator][
            ["subject", "pid", "supported_hours"]
        ]
        predictions = alarms[alarms["comparator"] == comparator][
            ["subject", "pid", "event_time_sec"]
        ]
        for membership in MEMBERSHIPS:
            eligible, ignored = local_event_inputs(references, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible, predictions, ignored, local_support, tolerance
                )
                config = {
                    "comparator": comparator,
                    "model_role": COMPARATOR_ROLES[comparator],
                    "partition": "validation",
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": thresholds[comparator],
                }
                summaries.append({**config, **summary})
                for frame, collection in [
                    (recordings, recordings_all),
                    (participants, participants_all),
                    (matches, matches_all),
                ]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        collection.append(local)
    return {
        "predicted_events": alarms,
        "event_metrics": pd.DataFrame(summaries),
        "event_recordings": pd.concat(recordings_all, ignore_index=True),
        "event_participants": pd.concat(participants_all, ignore_index=True),
        "event_matches": pd.concat(matches_all, ignore_index=True),
    }


def paired_participant_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    comparisons = [
        ("P6-D", "P2-D"),
        ("P2-D", "H2-D"),
        ("P2-H2-Z", "H2-D"),
    ]
    if "P2-H2-A" in set(primary["comparator"]):
        comparisons.append(("P2-H2-A", "P2-H2-Z"))

    rows = []
    count_columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
    ]
    for left_name, right_name in comparisons:
        left = primary[primary["comparator"] == left_name][count_columns]
        right = primary[primary["comparator"] == right_name][count_columns]
        paired = left.merge(
            right, on="pid", suffixes=("_left", "_right"), validate="one_to_one"
        )
        if len(paired) != 16:
            raise ValueError(f"Expected 16 paired validation participants: {left_name}")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
            metrics = {}
            for side in ["left", "right"]:
                metrics[side] = metric_values(
                    int(sample[f"true_positive_{side}"].sum()),
                    int(sample[f"false_positive_{side}"].sum()),
                    int(sample[f"false_negative_{side}"].sum()),
                    float(sample[f"supported_hours_{side}"].sum()),
                )
            samples.append(
                {
                    "event_f1_difference": metrics["left"]["f1"]
                    - metrics["right"]["f1"],
                    "false_alarms_per_hour_difference": metrics["left"][
                        "false_alarms_per_hour"
                    ]
                    - metrics["right"]["false_alarms_per_hour"],
                }
            )
        sample_frame = pd.DataFrame(samples)
        point = {}
        for side in ["left", "right"]:
            point[side] = metric_values(
                int(paired[f"true_positive_{side}"].sum()),
                int(paired[f"false_positive_{side}"].sum()),
                int(paired[f"false_negative_{side}"].sum()),
                float(paired[f"supported_hours_{side}"].sum()),
            )
        point_values = {
            "event_f1_difference": point["left"]["f1"] - point["right"]["f1"],
            "false_alarms_per_hour_difference": point["left"][
                "false_alarms_per_hour"
            ]
            - point["right"]["false_alarms_per_hour"],
        }
        for metric, value in point_values.items():
            rows.append(
                {
                    "comparison": f"{left_name}_minus_{right_name}",
                    "metric": metric,
                    "point_difference": value,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "lower_95": float(sample_frame[metric].quantile(0.025)),
                    "median": float(sample_frame[metric].quantile(0.5)),
                    "upper_95": float(sample_frame[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


# Section 9: manifests, checks, and reviewed interpretation

def external_manifest(
    assignments: pd.DataFrame, generated: pd.DataFrame
) -> pd.DataFrame:
    paths = []
    train = assignments[assignments["partition"] == "train"]
    for item in train.itertuples(index=False):
        for modality in ["PSG-6", "PSG-2", "HB-2"]:
            paths.append(("passed_train_feature", feature_path(item.subject, "train", modality)))
    for item in generated.itertuples(index=False):
        paths.append(("generated_transfer_feature", data_parent() / item.path_relative_to_data_parent))
    for comparator in DIRECT_MODALITIES:
        paths.append(("fitted_direct_model", model_path(comparator)))
    paths.append(("validation_continuous_scores", score_path("validation_continuous")))
    paths.append(("train_validation_labeled_scores", score_path("train_validation_labeled")))

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


def run_checks(
    assignments: pd.DataFrame,
    generated: pd.DataFrame,
    parity: pd.DataFrame,
    fits: pd.DataFrame,
    construction: pd.DataFrame,
    curve: pd.DataFrame,
    selected: pd.DataFrame,
    scores: pd.DataFrame,
    support: pd.DataFrame,
    shift: pd.DataFrame,
    gate: pd.DataFrame,
    event_metrics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    expected_comparators = 5 if bool(gate.iloc[0].adaptation_gate_open) else 4
    retained = construction.groupby(["comparator", "partition"])["retained_rows"].sum()
    selected_thresholds = selected.set_index("comparator")["threshold"]
    inherited = event_metrics[
        event_metrics["comparator"].isin(["P2-H2-Z", "P2-H2-A"])
    ]["threshold"]
    rows = [
        ("authorized_membership", len(assignments) == 102 and assignments["pid"].nunique() == 80, "82 train and 20 validation recordings"),
        ("generated_feature_artifacts", len(generated) == 162 and truth(generated["all_features_finite"]).all(), "82 train zero-shot plus 80 validation arrays"),
        ("cross_modality_epoch_parity", len(parity) == 102 and truth(parity["epoch_timing_and_stage_parity"]).all(), "all authorized recordings"),
        ("cross_modality_context_parity", truth(parity["context_center_parity"]).all(), "all authorized recordings"),
        ("psg_overlap_parity", float(parity["psg_overlap_max_abs_difference"].max()) <= PSG_OVERLAP_TOLERANCE, f"maximum={parity['psg_overlap_max_abs_difference'].max():.12g}"),
        ("three_direct_models_fitted", len(fits) == 3 and set(fits["comparator"]) == set(DIRECT_MODALITIES), "P6-D, P2-D, H2-D"),
        ("direct_input_dimensions", fits.set_index("comparator")["input_features"].to_dict() == {"P6-D": 240, "P2-D": 80, "H2-D": 80}, "240/80/80 context features"),
        ("no_convergence_warning", fits["convergence_warning_count"].eq(0).all(), f"total={fits['convergence_warning_count'].sum()}"),
        ("labeled_construction_equal", retained.groupby("partition").nunique().eq(1).all(), "same retained candidate count for all direct inputs"),
        ("threshold_grid_complete", len(curve) == 297 and curve.groupby("comparator")["threshold"].nunique().eq(99).all(), "three complete 99-threshold grids"),
        ("direct_thresholds_frozen", len(selected) == 3, "one validation threshold per direct model"),
        ("validation_support_complete", support.groupby("comparator")["subject"].nunique().eq(20).all() and support["comparator"].nunique() == expected_comparators, f"comparators={expected_comparators}"),
        ("zero_shot_threshold_inheritance", len(inherited) > 0 and np.isclose(inherited, float(selected_thresholds.loc["P2-D"]), atol=1e-12, rtol=0.0).all(), "zero-shot/alignment use P2-D threshold"),
        ("feature_shift_complete", len(shift) == 80 and shift["dimension_index"].nunique() == 80 and shift["common_train_boundaries"].nunique() == 1, f"common_boundaries={int(shift['common_train_boundaries'].iloc[0])}"),
        ("adaptation_action_followed", (bool(gate.iloc[0].adaptation_gate_open) and "P2-H2-A" in set(scores["comparator"])) or (not bool(gate.iloc[0].adaptation_gate_open) and "P2-H2-A" not in set(scores["comparator"])), str(gate.iloc[0].adaptation_action)),
        ("external_hashes_recorded", len(manifest) == 413 and manifest["sha256"].str.len().eq(64).all(), f"artifacts={len(manifest)}"),
        ("test_partition_closed", not scores["partition"].eq("test").any() and not support["partition"].eq("test").any() and not generated["partition"].eq("test").any(), "no test feature, score, support, or result row"),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in rows
        ]
    )


def write_readme(
    result_code_commit: str,
    window_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    gate: pd.DataFrame,
    paired: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    primary = event_metrics[
        (event_metrics["membership"] == "primary")
        & (event_metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    direct_window = window_metrics[
        window_metrics["partition"] == "validation"
    ].set_index("comparator")
    threshold_map = selected.set_index("comparator")["threshold"].to_dict()
    table_rows = []
    for comparator in primary.index:
        window_ap = (
            f"{direct_window.loc[comparator, 'average_precision']:.4f}"
            if comparator in direct_window.index
            else "not applicable"
        )
        row = primary.loc[comparator]
        table_rows.append(
            f"| {comparator} | {window_ap} | {row.threshold:.2f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        )
    gate_row = gate.iloc[0]
    paired_rows = []
    for item in paired.itertuples(index=False):
        paired_rows.append(
            f"| {item.comparison} | {item.metric} | {item.point_difference:+.4f} | "
            f"{item.lower_95:+.4f} to {item.upper_95:+.4f} |"
        )
    text = "\n".join(
        [
            "# Block 7 Paired-Transfer Validation v0.1",
            "",
            "**Work date:** 2026-09-06",
            f"**Execution-plan commit:** `{PLAN_COMMIT}`",
            f"**Feature-gate result commit:** `{FEATURE_GATE_RESULT_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Partitions accessed:** Train and validation only",
            "**Test data accessed:** No",
            "",
            "## Primary Validation Result",
            "",
            "| Comparator | Labeled-window AP | Threshold | Event precision | Event recall | Event F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *table_rows,
            "",
            "The direct comparators selected their own validation thresholds. Strict zero-shot and conditional alignment, when executed, inherited the P2-D threshold.",
            "",
            "## Adaptation Gate",
            "",
            f"- Zero-shot F1 deficit relative to H2-D: `{gate_row.zero_shot_f1_deficit:.4f}`.",
            f"- Zero-shot false-alarm excess per hour: `{gate_row.zero_shot_far_excess:.4f}`.",
            f"- Shifted dimensions: `{int(gate_row.shifted_dimensions)}/{int(gate_row.total_dimensions)}`.",
            f"- Performance condition open: **{bool(gate_row.performance_condition_open)}**.",
            f"- Distribution condition open: **{bool(gate_row.distribution_condition_open)}**.",
            f"- Adaptation action: **{gate_row.adaptation_action}**.",
            "",
            "## Paired Participant Contrasts",
            "",
            "| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |",
            "|---|---|---:|---:|",
            *paired_rows,
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. This is validation-only evidence. It does not authorize method revision from the later test result and does not provide independent confirmation.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)


# Section 10: execute the frozen train/validation phase

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")

    output_dir().mkdir(parents=True, exist_ok=True)
    assignments = subject_assignments()
    generated, parity = generate_required_features(assignments)
    candidates = labeled_candidates(assignments)
    models, fits, window_metrics, construction, labeled_scores = fit_direct_models(
        candidates
    )
    scores, support = validation_scores(assignments, models)
    references = reference_events(
        assignments[assignments["partition"] == "validation"]
    )
    curve, selected = direct_thresholds(scores, support, references)
    shift = feature_shift(assignments)
    gate = adaptation_gate(scores, support, references, selected, shift)
    if bool(gate.iloc[0].adaptation_gate_open):
        scores, support = add_aligned_scores(
            scores, support, assignments, models["P2-D"], shift
        )

    threshold_map = selected.set_index("comparator")["threshold"].to_dict()
    thresholds = {**threshold_map, "P2-H2-Z": threshold_map["P2-D"]}
    if bool(gate.iloc[0].adaptation_gate_open):
        thresholds["P2-H2-A"] = threshold_map["P2-D"]
    outputs = evaluate_all(scores, support, references, thresholds)
    paired = paired_participant_bootstrap(outputs["event_participants"])

    verify_or_create_gzip_tsv(labeled_scores, score_path("train_validation_labeled"))
    verify_or_create_gzip_tsv(scores, score_path("validation_continuous"))
    manifest = external_manifest(assignments, generated)
    checks = run_checks(
        assignments,
        generated,
        parity,
        fits,
        construction,
        curve,
        selected,
        scores,
        support,
        shift,
        gate,
        outputs["event_metrics"],
        manifest,
    )

    reviewed = {
        "generated_feature_artifacts_v0.1.tsv": generated,
        "feature_parity_checks_v0.1.tsv": parity,
        "model_fit_summary_v0.1.tsv": fits,
        "labeled_candidate_construction_v0.1.tsv": construction,
        "labeled_window_metrics_v0.1.tsv": window_metrics,
        "direct_validation_threshold_curve_v0.1.tsv": curve,
        "selected_direct_thresholds_v0.1.tsv": selected,
        "validation_support_v0.1.tsv": support,
        "train_feature_shift_v0.1.tsv": shift,
        "adaptation_gate_v0.1.tsv": gate,
        "validation_predicted_events_v0.1.tsv": outputs["predicted_events"],
        "validation_event_metrics_v0.1.tsv": outputs["event_metrics"],
        "validation_event_recordings_v0.1.tsv": outputs["event_recordings"],
        "validation_event_participants_v0.1.tsv": outputs["event_participants"],
        "validation_event_matches_v0.1.tsv": outputs["event_matches"],
        "paired_participant_bootstrap_v0.1.tsv": paired,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
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
    }
    verify_or_create_text(
        output_dir() / "software_versions_v0.1.json",
        json.dumps(software, indent=2, sort_keys=True) + "\n",
    )
    write_readme(
        args.result_code_commit,
        window_metrics,
        outputs["event_metrics"],
        selected,
        gate,
        paired,
        checks,
    )
    print(selected[["comparator", "threshold", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(gate.to_string(index=False))
    primary = outputs["event_metrics"]
    primary = primary[(primary["membership"] == "primary") & (primary["tolerance_sec"] == 15.0)]
    print(primary[["comparator", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 7 validation check failed")


if __name__ == "__main__":
    main()
