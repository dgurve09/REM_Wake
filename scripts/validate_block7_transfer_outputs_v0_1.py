"""Independently validate Block 7 train/validation transfer outputs."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: fixed validation configuration

EXPERIMENT_DIR = "2026-09-06_block7_transfer_validation_v0.1"
DERIVED_DIR = "block7_transfer_validation_v0.1"
FEATURE_GATE_DIR = "block7_feature_generation_validation_v0.1"
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, 30.0)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906
ROBUST_SCALE_FACTOR = 1.4826
EXPECTED_MODALITIES = {
    "PSG-6": ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"],
    "PSG-2": ["PSG_F3", "PSG_F4"],
    "HB-2": ["HB_1", "HB_2"],
    "HB-2-PSGscale": ["HB_1", "HB_2"],
}
EXPECTED_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
DIRECT_MODALITY = {"P6-D": "PSG-6", "P2-D": "PSG-2", "H2-D": "HB-2"}
MODEL_SOURCE = {
    "P6-D": "P6-D",
    "P2-D": "P2-D",
    "H2-D": "H2-D",
    "P2-H2-Z": "P2-D",
}
SCORE_MODALITY = {
    "P6-D": "PSG-6",
    "P2-D": "PSG-2",
    "H2-D": "HB-2",
    "P2-H2-Z": "HB-2-PSGscale",
}


# Section 2: paths and small helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def model_path(comparator: str) -> Path:
    return derived_dir() / "models" / f"{comparator.lower().replace('-', '_')}_model_v0.1.joblib"


def score_path(kind: str) -> Path:
    return derived_dir() / "candidate_scores" / f"{kind}_scores_v0.1.tsv.gz"


def feature_path(subject: str, partition: str, modality: str) -> Path:
    folder = modality.lower().replace("-", "").replace("psgscale", "_psgscale")
    if partition == "train" and modality != "HB-2-PSGscale":
        return (
            data_parent()
            / "derived"
            / FEATURE_GATE_DIR
            / "recording_features"
            / folder
            / f"{subject}_features_v0.1.npz"
        )
    return (
        derived_dir()
        / "recording_features"
        / partition
        / folder
        / f"{subject}_features_v0.1.npz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def record(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append(
        {"check": name, "status": "pass" if passed else "fail", "detail": detail}
    )


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed validation output changed: {path}")
        return
    path.write_text(expected, encoding="utf-8")


def frames_match(left: pd.DataFrame, right: pd.DataFrame, sort_by: list[str]) -> bool:
    if set(left.columns) != set(right.columns) or len(left) != len(right):
        return False
    columns = sorted(left.columns)
    left = left.sort_values(sort_by, kind="stable").reset_index(drop=True)[columns]
    right = right.sort_values(sort_by, kind="stable").reset_index(drop=True)[columns]
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            ):
                return False
        else:
            left_values = left[column].astype(str).str.lower().replace("nan", "")
            right_values = right[column].astype(str).str.lower().replace("nan", "")
            if not left_values.equals(right_values):
                return False
    return True


# Section 3: authorized assignments, references, and feature arrays

def assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    split = split[split["partition"].isin(["train", "validation"])]
    rows = []
    for item in split.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append(
                {"subject": subject, "pid": int(item.pid), "partition": item.partition}
            )
    return pd.DataFrame(rows).sort_values(["partition", "subject"])


def references(local_assignments: pd.DataFrame) -> pd.DataFrame:
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
    allowed = set(local_assignments["subject"])
    result = membership[
        membership["subject"].isin(allowed)
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    result["event_time_sec"] = result["nominal_boundary_sec"].astype(float)
    return result


def load_feature(subject: str, partition: str, modality: str) -> dict[str, np.ndarray]:
    path = feature_path(subject, partition, modality)
    with np.load(path, allow_pickle=False) as values:
        if set(values.files) != {"onset", "stage", "features", "feature_names"}:
            raise ValueError(f"Unexpected arrays in {path}")
        result = {name: values[name].copy() for name in values.files}
    channels = EXPECTED_MODALITIES[modality]
    expected_names = np.asarray(
        [
            f"{channel}_{band}_log10_mean_psd"
            for channel in channels
            for band in EXPECTED_BANDS
        ]
    )
    if not np.array_equal(result["feature_names"], expected_names):
        raise ValueError(f"Feature schema differs for {subject}, {modality}")
    if result["features"].shape != (len(result["onset"]), len(expected_names)):
        raise ValueError(f"Feature dimensions differ for {subject}, {modality}")
    if not np.isfinite(result["features"]).all():
        raise ValueError(f"Nonfinite feature values for {subject}, {modality}")
    return result


def context_indices(onsets: np.ndarray) -> np.ndarray:
    required = len(CONTEXT_OFFSETS) - 1
    contiguous = np.isclose(np.diff(onsets), 30.0, atol=1e-9, rtol=0.0).astype(
        np.int8
    )
    if len(contiguous) < required:
        return np.asarray([], dtype=int)
    counts = np.convolve(contiguous, np.ones(required, dtype=np.int8), mode="valid")
    return np.flatnonzero(counts == required)


def context_matrix(values: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    indices = context_indices(values["onset"])
    matrix = np.concatenate(
        [values["features"][indices + offset] for offset in range(8)], axis=1
    )
    return values["onset"][indices + 4].astype(float), matrix


def validate_feature_artifacts(rows: list[dict], local_assignments: pd.DataFrame) -> None:
    generated = pd.read_csv(output_dir() / "generated_feature_artifacts_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    expected_counts = {
        ("train", "HB-2-PSGscale"): 82,
        ("validation", "PSG-6"): 20,
        ("validation", "PSG-2"): 20,
        ("validation", "HB-2"): 20,
        ("validation", "HB-2-PSGscale"): 20,
    }
    counts = generated.groupby(["partition", "modality"]).size().to_dict()
    record(rows, "generated_feature_cardinality", counts == expected_counts, str(counts))

    hashes_pass = True
    arrays_pass = True
    for item in generated.itertuples(index=False):
        path = (data_parent() / item.path_relative_to_data_parent).resolve()
        hashes_pass &= path.exists() and path.stat().st_size == int(item.bytes)
        if path.exists():
            hashes_pass &= sha256(path) == item.sha256
            values = load_feature(item.subject, item.partition, item.modality)
            arrays_pass &= (
                len(values["onset"]) == int(item.epochs)
                and len(context_indices(values["onset"])) == int(item.context_rows)
            )
    record(rows, "generated_feature_hashes", hashes_pass, "162 size and SHA-256 checks")
    record(rows, "generated_feature_schemas", arrays_pass, "162 arrays reopened")

    parity_pass = True
    psg_maximum = 0.0
    for item in local_assignments.itertuples(index=False):
        modalities = ["PSG-6", "PSG-2", "HB-2", "HB-2-PSGscale"]
        values = {
            modality: load_feature(item.subject, item.partition, modality)
            for modality in modalities
        }
        anchor = values["PSG-6"]
        parity_pass &= all(
            np.array_equal(anchor["onset"], value["onset"])
            and np.array_equal(anchor["stage"], value["stage"])
            and np.array_equal(
                context_indices(anchor["onset"]), context_indices(value["onset"])
            )
            for value in values.values()
        )
        difference = float(
            np.max(
                np.abs(
                    values["PSG-6"]["features"][:, :10]
                    - values["PSG-2"]["features"]
                )
            )
        )
        psg_maximum = max(psg_maximum, difference)
    record(rows, "cross_modality_timing_parity", parity_pass, "102 authorized recordings")
    record(rows, "psg_overlap_recomputed", psg_maximum <= 1e-10, f"maximum={psg_maximum:.12g}")

    generated_paths = set(generated["path_relative_to_data_parent"])
    manifest_generated = set(
        manifest[manifest["artifact_role"] == "generated_transfer_feature"][
            "path_relative_to_data_parent"
        ]
    )
    record(rows, "generated_manifest_linkage", generated_paths == manifest_generated, "162 linked paths")


# Section 4: external manifest, models, and score reproduction

def validate_external_manifest(rows: list[dict]) -> pd.DataFrame:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    root = data_parent().resolve()
    passed = len(manifest) == 413 and manifest["path_relative_to_data_parent"].is_unique
    for item in manifest.itertuples(index=False):
        path = (root / item.path_relative_to_data_parent).resolve()
        passed &= (
            path.is_relative_to(root)
            and path.exists()
            and path.stat().st_size == int(item.bytes)
            and sha256(path) == item.sha256
        )
    record(rows, "external_manifest_rehashed", passed, f"artifacts={len(manifest)}")
    return manifest


def validate_models(rows: list[dict]) -> dict[str, object]:
    fits = pd.read_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t")
    expected_dimensions = {"P6-D": 240, "P2-D": 80, "H2-D": 80}
    models = {}
    passed = len(fits) == 3 and fits["convergence_warning_count"].eq(0).all()
    for item in fits.itertuples(index=False):
        path = model_path(item.comparator)
        model = joblib.load(path)
        dimensions = int(model.named_steps["logisticregression"].n_features_in_)
        passed &= (
            dimensions == expected_dimensions[item.comparator]
            and sha256(path) == item.model_sha256
        )
        models[item.comparator] = model
    record(rows, "models_reopened", passed, "three hashes and 240/80/80 dimensions")
    return models


def probability_reproduction(
    rows: list[dict], models: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled = pd.read_csv(score_path("train_validation_labeled"), sep="\t")
    continuous = pd.read_csv(score_path("validation_continuous"), sep="\t")
    score_scope_pass = (
        set(labeled["partition"]) == {"train", "validation"}
        and set(continuous["partition"]) == {"validation"}
        and set(continuous["comparator"])
        == {"P6-D", "P2-D", "H2-D", "P2-H2-Z"}
    )
    record(rows, "score_partition_scope", score_scope_pass, "train/validation only; four validation comparators")
    labeled_maximum = 0.0
    for (comparator, subject, partition), group in labeled.groupby(
        ["comparator", "subject", "partition"], sort=True
    ):
        values = load_feature(subject, partition, DIRECT_MODALITY[comparator])
        centers, matrix = context_matrix(values)
        lookup = {int(round(value * 1000)): index for index, value in enumerate(centers)}
        indices = [lookup[int(round(value * 1000))] for value in group["candidate_time_sec"]]
        expected = models[comparator].predict_proba(matrix[indices])[:, 1]
        difference = float(np.max(np.abs(expected - group["probability"].to_numpy())))
        labeled_maximum = max(labeled_maximum, difference)
    record(rows, "labeled_probabilities_reproduced", labeled_maximum <= 1e-12, f"maximum={labeled_maximum:.12g}")

    continuous_maximum = 0.0
    for (comparator, subject), group in continuous.groupby(["comparator", "subject"], sort=True):
        values = load_feature(subject, "validation", SCORE_MODALITY[comparator])
        centers, matrix = context_matrix(values)
        group = group.sort_values("candidate_time_sec")
        if not np.array_equal(centers, group["candidate_time_sec"].to_numpy(dtype=float)):
            continuous_maximum = np.inf
            continue
        expected = models[MODEL_SOURCE[comparator]].predict_proba(matrix)[:, 1]
        difference = float(np.max(np.abs(expected - group["probability"].to_numpy())))
        continuous_maximum = max(continuous_maximum, difference)
    record(rows, "continuous_probabilities_reproduced", continuous_maximum <= 1e-12, f"maximum={continuous_maximum:.12g}")

    metrics = []
    for (comparator, partition), group in labeled.groupby(["comparator", "partition"], sort=True):
        metrics.append(
            {
                "comparator": comparator,
                "partition": partition,
                "rows": len(group),
                "positive_rows": int(group["label"].sum()),
                "negative_rows": int((group["label"] == 0).sum()),
                "average_precision": average_precision_score(group["label"], group["probability"]),
                "roc_auc": roc_auc_score(group["label"], group["probability"]),
            }
        )
    saved_metrics = pd.read_csv(output_dir() / "labeled_window_metrics_v0.1.tsv", sep="\t")
    metric_pass = frames_match(saved_metrics, pd.DataFrame(metrics), ["comparator", "partition"])
    record(rows, "window_metrics_recomputed", metric_pass, "six train/validation rows")
    return labeled, continuous


# Section 5: event thresholds and complete metric recomputation

def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = []
    marked = scores[scores["probability"] >= threshold]
    for (comparator, subject, pid), group in marked.groupby(
        ["comparator", "subject", "pid"], sort=True
    ):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        starts = [0] + (
            np.flatnonzero(
                np.diff(group["candidate_time_sec"].to_numpy(dtype=float)) > 30.000001
            )
            + 1
        ).tolist()
        stops = starts[1:] + [len(group)]
        for start, stop in zip(starts, stops):
            run = group.iloc[start:stop]
            maximum = float(run["probability"].max())
            best = run[np.isclose(run["probability"], maximum)].sort_values(
                "candidate_time_sec"
            ).iloc[0]
            result.append(
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
        result,
        columns=["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"],
    )


def event_inputs(reference: pd.DataFrame, membership: str):
    column = "primary_analysis_eligible" if membership == "primary" else "expanded_quality_analysis_eligible"
    eligible = truth(reference[column])
    columns = ["subject", "pid", "event_time_sec"]
    return reference.loc[eligible, columns], reference.loc[~eligible, columns]


def recompute_thresholds(
    rows: list[dict], scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    eligible, ignored = event_inputs(reference, "primary")
    curve_rows = []
    selected_rows = []
    fits = pd.read_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t").set_index("comparator")
    for comparator in DIRECT_MODALITY:
        local_scores = scores[scores["comparator"] == comparator]
        local_support = support[support["comparator"] == comparator][["subject", "pid", "supported_hours"]]
        comparator_rows = []
        for threshold in THRESHOLDS:
            alarms = collapse_alarms(local_scores, float(threshold))
            _, _, _, summary = evaluate_events(
                eligible,
                alarms[["subject", "pid", "event_time_sec"]],
                ignored,
                local_support,
                15.0,
            )
            item = {
                "comparator": comparator,
                "partition": "validation",
                "membership": "primary",
                "tolerance_sec": 15.0,
                "threshold": float(threshold),
                **summary,
            }
            curve_rows.append(item)
            comparator_rows.append(item)
        selected = pd.DataFrame(comparator_rows).sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        selected_rows.append(
            {
                **selected.to_dict(),
                "model_sha256": fits.loc[comparator, "model_sha256"],
                "selection_rule": "max_f1_then_min_far_then_max_recall_then_max_threshold",
            }
        )
    curve = pd.DataFrame(curve_rows)
    selected = pd.DataFrame(selected_rows)
    saved_curve = pd.read_csv(output_dir() / "direct_validation_threshold_curve_v0.1.tsv", sep="\t")
    saved_selected = pd.read_csv(output_dir() / "selected_direct_thresholds_v0.1.tsv", sep="\t")
    record(rows, "threshold_curve_recomputed", frames_match(saved_curve, curve, ["comparator", "threshold"]), "297 rows")
    record(rows, "selected_thresholds_recomputed", frames_match(saved_selected, selected, ["comparator"]), "P6-D=0.99, P2-D=0.99, H2-D=0.96")
    return selected


def recompute_event_outputs(
    rows: list[dict], scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    threshold_map = selected.set_index("comparator")["threshold"].to_dict()
    threshold_map["P2-H2-Z"] = threshold_map["P2-D"]
    alarms = pd.concat(
        [collapse_alarms(group, threshold_map[comparator]) for comparator, group in scores.groupby("comparator", sort=True)],
        ignore_index=True,
    )
    summaries = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    roles = {
        "P6-D": "direct_six_channel_psg",
        "P2-D": "direct_two_channel_psg",
        "H2-D": "direct_two_channel_wearable",
        "P2-H2-Z": "strict_psg_to_wearable_zero_shot",
    }
    for comparator in sorted(scores["comparator"].unique()):
        local_support = support[support["comparator"] == comparator][["subject", "pid", "supported_hours"]]
        predictions = alarms[alarms["comparator"] == comparator][["subject", "pid", "event_time_sec"]]
        for membership in ["primary", "expanded"]:
            eligible, ignored = event_inputs(reference, membership)
            for tolerance in [15.0, 45.0]:
                recordings, participants, matches, summary = evaluate_events(
                    eligible, predictions, ignored, local_support, tolerance
                )
                config = {
                    "comparator": comparator,
                    "model_role": roles[comparator],
                    "partition": "validation",
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": threshold_map[comparator],
                }
                summaries.append({**config, **summary})
                for frame, collection in [(recordings, recording_rows), (participants, participant_rows), (matches, match_rows)]:
                    if len(frame):
                        frame = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in frame.columns:
                                frame.insert(0, key, value)
                        collection.append(frame)

    expected = {
        "validation_predicted_events_v0.1.tsv": (alarms, ["comparator", "subject", "event_time_sec"]),
        "validation_event_metrics_v0.1.tsv": (pd.DataFrame(summaries), ["comparator", "membership", "tolerance_sec"]),
        "validation_event_recordings_v0.1.tsv": (pd.concat(recording_rows, ignore_index=True), ["comparator", "membership", "tolerance_sec", "subject"]),
        "validation_event_participants_v0.1.tsv": (pd.concat(participant_rows, ignore_index=True), ["comparator", "membership", "tolerance_sec", "pid"]),
        "validation_event_matches_v0.1.tsv": (pd.concat(match_rows, ignore_index=True), ["comparator", "membership", "tolerance_sec", "subject", "prediction_time_sec"]),
    }
    for name, (frame, keys) in expected.items():
        saved = pd.read_csv(output_dir() / name, sep="\t")
        record(rows, name.replace("_v0.1.tsv", "_recomputed"), frames_match(saved, frame, keys), f"rows={len(frame)}")
    return pd.concat(participant_rows, ignore_index=True)


def validate_paired_bootstrap(rows: list[dict], participants: pd.DataFrame) -> None:
    primary = participants[
        (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    comparisons = [
        ("P6-D", "P2-D"),
        ("P2-D", "H2-D"),
        ("P2-H2-Z", "H2-D"),
    ]
    result = []
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    for left_name, right_name in comparisons:
        left = primary[primary["comparator"] == left_name][columns]
        right = primary[primary["comparator"] == right_name][columns]
        paired = left.merge(
            right, on="pid", suffixes=("_left", "_right"), validate="one_to_one"
        )
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired)))
        samples = {}
        points = {}
        for side in ["left", "right"]:
            tp_values = paired[f"true_positive_{side}"].to_numpy(dtype=int)
            fp_values = paired[f"false_positive_{side}"].to_numpy(dtype=int)
            fn_values = paired[f"false_negative_{side}"].to_numpy(dtype=int)
            hour_values = paired[f"supported_hours_{side}"].to_numpy(dtype=float)
            tp = tp_values[indices].sum(axis=1)
            fp = fp_values[indices].sum(axis=1)
            fn = fn_values[indices].sum(axis=1)
            hours = hour_values[indices].sum(axis=1)
            samples[side] = {
                "event_f1_difference": 2 * tp / (2 * tp + fp + fn),
                "false_alarms_per_hour_difference": fp / hours,
            }
            points[side] = metric_values(
                int(tp_values.sum()),
                int(fp_values.sum()),
                int(fn_values.sum()),
                float(hour_values.sum()),
            )
        for metric in ["event_f1_difference", "false_alarms_per_hour_difference"]:
            values = samples["left"][metric] - samples["right"][metric]
            point_name = "f1" if metric == "event_f1_difference" else "false_alarms_per_hour"
            result.append(
                {
                    "comparison": f"{left_name}_minus_{right_name}",
                    "metric": metric,
                    "point_difference": points["left"][point_name]
                    - points["right"][point_name],
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "lower_95": float(np.quantile(values, 0.025)),
                    "median": float(np.quantile(values, 0.5)),
                    "upper_95": float(np.quantile(values, 0.975)),
                }
            )
    recomputed = pd.DataFrame(result)
    saved = pd.read_csv(output_dir() / "paired_participant_bootstrap_v0.1.tsv", sep="\t")
    record(
        rows,
        "paired_bootstrap_recomputed",
        frames_match(saved, recomputed, ["comparison", "metric"]),
        "three paired contrasts and 2,000 resamples",
    )


# Section 6: feature-shift and gate arithmetic

def recompute_shift(rows: list[dict], local_assignments: pd.DataFrame) -> pd.DataFrame:
    cache = []
    train = local_assignments[local_assignments["partition"] == "train"]
    for item in train.itertuples(index=False):
        psg = load_feature(item.subject, "train", "PSG-2")
        hb = load_feature(item.subject, "train", "HB-2-PSGscale")
        psg_indices = context_indices(psg["onset"])
        hb_indices = context_indices(hb["onset"])
        if not np.array_equal(psg["onset"][psg_indices + 4], hb["onset"][hb_indices + 4]):
            raise ValueError(f"Common train support differs for {item.subject}")
        cache.append((psg, hb, psg_indices, hb_indices))

    result = []
    for offset_index, offset_sec in enumerate(CONTEXT_OFFSETS):
        source = np.vstack([item[0]["features"][item[2] + offset_index] for item in cache])
        target = np.vstack([item[1]["features"][item[3] + offset_index] for item in cache])
        for feature_index in range(10):
            source_values = source[:, feature_index].astype(float)
            target_values = target[:, feature_index].astype(float)
            source_median = float(np.median(source_values))
            target_median = float(np.median(target_values))
            source_mad = float(np.median(np.abs(source_values - source_median)))
            target_mad = float(np.median(np.abs(target_values - target_median)))
            pooled = np.concatenate([source_values, target_values])
            pooled_median = float(np.median(pooled))
            pooled_mad = float(np.median(np.abs(pooled - pooled_median)))
            source_raw = ROBUST_SCALE_FACTOR * source_mad
            target_raw = ROBUST_SCALE_FACTOR * target_mad
            pooled_raw = ROBUST_SCALE_FACTOR * pooled_mad
            source_scale = source_raw if source_raw > 0 else 1.0
            target_scale = target_raw if target_raw > 0 else 1.0
            pooled_scale = pooled_raw if pooled_raw > 0 else 1.0
            difference = abs(source_median - target_median) / pooled_scale
            result.append(
                {
                    "dimension_index": offset_index * 10 + feature_index,
                    "context_offset_sec": float(offset_sec),
                    "source_feature_name": cache[0][0]["feature_names"][feature_index],
                    "target_feature_name": cache[0][1]["feature_names"][feature_index],
                    "common_train_boundaries": len(source_values),
                    "source_median": source_median,
                    "source_mad": source_mad,
                    "source_robust_scale": source_scale,
                    "source_zero_scale_replaced": source_raw == 0,
                    "target_median": target_median,
                    "target_mad": target_mad,
                    "target_robust_scale": target_scale,
                    "target_zero_scale_replaced": target_raw == 0,
                    "pooled_median": pooled_median,
                    "pooled_mad": pooled_mad,
                    "pooled_robust_scale": pooled_scale,
                    "pooled_zero_scale_replaced": pooled_raw == 0,
                    "absolute_median_difference_pooled_scale": difference,
                    "exceeds_0_50_pooled_scale": difference > 0.50,
                }
            )
    recomputed = pd.DataFrame(result)
    saved = pd.read_csv(output_dir() / "train_feature_shift_v0.1.tsv", sep="\t")
    record(rows, "feature_shift_recomputed", frames_match(saved, recomputed, ["dimension_index"]), "80 dimensions across 75,539 boundaries")
    return recomputed


def validate_gate(rows: list[dict], shift: pd.DataFrame) -> None:
    metrics = pd.read_csv(output_dir() / "validation_event_metrics_v0.1.tsv", sep="\t")
    metrics = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index("comparator")
    f1_deficit = float(metrics.loc["H2-D", "f1"] - metrics.loc["P2-H2-Z", "f1"])
    far_excess = float(metrics.loc["P2-H2-Z", "false_alarms_per_hour"] - metrics.loc["H2-D", "false_alarms_per_hour"])
    performance = f1_deficit >= 0.03 or far_excess >= 0.50
    shifted = int(truth(shift["exceeds_0_50_pooled_scale"]).sum())
    distribution = shifted / 80 >= 0.20
    gate = pd.read_csv(output_dir() / "adaptation_gate_v0.1.tsv", sep="\t").iloc[0]
    gate_performance = str(gate.performance_condition_open).lower() == "true"
    gate_distribution = str(gate.distribution_condition_open).lower() == "true"
    gate_open = str(gate.adaptation_gate_open).lower() == "true"
    passed = (
        np.isclose(gate.zero_shot_f1_deficit, f1_deficit)
        and np.isclose(gate.zero_shot_far_excess, far_excess)
        and gate_performance == performance
        and int(gate.shifted_dimensions) == shifted
        and gate_distribution == distribution
        and gate_open == (performance and distribution)
        and gate.adaptation_action == "skip_P2-H2-A"
    )
    record(rows, "adaptation_gate_recomputed", passed, f"performance={performance}, shifted={shifted}/80, distribution={distribution}")


# Section 7: execute independent validation

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator-code-commit", required=True)
    args = parser.parse_args()
    if len(args.validator_code_commit) < 7:
        raise ValueError("A committed validator code hash is required")

    rows = []
    local_assignments = assignments()
    record(rows, "authorized_assignments", len(local_assignments) == 102 and local_assignments["pid"].nunique() == 80, "82 train and 20 validation recordings")
    validate_external_manifest(rows)
    validate_feature_artifacts(rows, local_assignments)
    models = validate_models(rows)
    _, continuous = probability_reproduction(rows, models)
    support = pd.read_csv(output_dir() / "validation_support_v0.1.tsv", sep="\t")
    reference = references(local_assignments[local_assignments["partition"] == "validation"])
    selected = recompute_thresholds(rows, continuous, support, reference)
    participants = recompute_event_outputs(rows, continuous, support, reference, selected)
    validate_paired_bootstrap(rows, participants)
    shift = recompute_shift(rows, local_assignments)
    validate_gate(rows, shift)

    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")
    passed = int(checks["status"].eq("pass").sum())
    note = f"""# Block 7 Transfer Output Validation v0.1

**Work date:** 2026-09-06
**Validator code commit:** `{args.validator_code_commit}`
**Scope:** Stored train/validation features, models, scores, and reviewed tables
**Raw EDF access:** No
**Test data accessed:** No

The independent validator passed **{passed}/{len(checks)}** checks. It rehashed all 413 external artifacts, reopened the stored feature arrays and models, reproduced labeled and continuous probabilities, recomputed the complete direct-threshold grid and event outputs, and independently verified the feature-shift and adaptation-gate arithmetic.
"""
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", note)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 7 output-integrity check failed")
    print(f"Passed {passed}/{len(checks)} independent output-integrity checks")


if __name__ == "__main__":
    main()
