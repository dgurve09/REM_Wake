"""Run the frozen one-time Block 7 descriptive test comparison."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from reviewed_output import verify_or_create_tsv
from run_block7_transfer_validation_v0_1 import (
    COMPARATOR_ROLES,
    CONTEXT_OFFSETS,
    EPOCH_SEC,
    HB2,
    MEMBERSHIPS,
    PSG2,
    PSG6,
    PSG_OVERLAP_TOLERANCE,
    TOLERANCES,
    context_matrix,
    epoch_features,
    filter_resample,
    filter_sos,
    normalize_signal,
    read_uv,
    scaler_maps,
    valid_events,
    verify_or_create_npz,
)
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen test configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-06_block7_descriptive_test_v0.1"
DERIVED_DIR = "block7_descriptive_test_v0.1"
TRANSFER_DERIVED_DIR = "block7_transfer_validation_v0.1"
FREEZE_COMMIT = "4a17fbd"
FREEZE_MARKER = "BLOCK7_VALIDATION_MODELS_THRESHOLDS_AND_ADAPTATION_DECISION_FROZEN"
PARTITION = "test"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906
FIXED_SENSITIVITY_EXCLUSIONS = {"sub-32", "sub-50"}

MODEL_SHA256 = {
    "P6-D": "243ab382909b30f8288bc24dc1e22b205fcdcea77c2011c2a3c2a452f7969082",
    "P2-D": "a3176329a142c4569a36813b08107b11a0acc4634ef47eb7e983b85fc51f7e51",
    "H2-D": "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066",
}
THRESHOLDS = {
    "P6-D": 0.99,
    "P2-D": 0.99,
    "H2-D": 0.96,
    "P2-H2-Z": 0.99,
}
SCORE_SPECS = {
    "P6-D": ("P6-D", "PSG-6"),
    "P2-D": ("P2-D", "PSG-2"),
    "H2-D": ("H2-D", "HB-2"),
    "P2-H2-Z": ("P2-D", "HB-2-PSGscale"),
}
MODALITY_FOLDERS = {
    "PSG-6": "psg6",
    "PSG-2": "psg2",
    "HB-2": "hb2",
    "HB-2-PSGscale": "hb2_psgscale",
}


# Section 2: paths and immutable file helpers

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


def feature_path(subject: str, modality: str) -> Path:
    return (
        derived_dir()
        / "recording_features"
        / MODALITY_FOLDERS[modality]
        / f"{subject}_features_v0.1.npz"
    )


def model_path(comparator: str) -> Path:
    name = comparator.lower().replace("-", "_")
    return (
        data_parent()
        / "derived"
        / TRANSFER_DERIVED_DIR
        / "models"
        / f"{name}_model_v0.1.joblib"
    )


def score_path() -> Path:
    return derived_dir() / "candidate_scores" / "test_continuous_scores_v0.1.tsv.gz"


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


# Section 3: frozen membership and references

def test_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in split[split["partition"] == PARTITION].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": PARTITION})
    result = pd.DataFrame(rows)
    if len(result) != 26 or result["pid"].nunique() != 20:
        raise ValueError("Frozen test assignment must contain 26 recordings and 20 pid groups")
    if result["subject"].duplicated().any():
        raise ValueError("A test recording appears more than once")
    return result.sort_values("subject", key=lambda values: values.map(subject_number))


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
    rows = membership[
        membership["subject"].isin(set(assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
        & membership["partition"].eq(PARTITION)
    ].merge(quality, on="transition_id", validate="one_to_one")
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    if set(rows["partition"]) != {PARTITION}:
        raise ValueError("Reference events escaped the frozen test partition")
    return rows


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


# Section 4: test feature generation

def save_feature(
    subject: str,
    pid: int,
    modality: str,
    values: tuple[np.ndarray, np.ndarray, np.ndarray, list[str]],
) -> dict:
    onsets, stages, features, names = values
    path = feature_path(subject, modality)
    verify_or_create_npz(path, onsets, stages, features, names)
    centers, _ = context_matrix(onsets, features)
    return {
        "subject": subject,
        "pid": int(pid),
        "partition": PARTITION,
        "modality": modality,
        "epochs": len(onsets),
        "context_rows": len(centers),
        "base_feature_dimensions": features.shape[1],
        "all_features_finite": bool(np.isfinite(features).all()),
        "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def generate_test_features(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    psg_scaler, hb_scaler, zero_scaler = scaler_maps()
    sos = filter_sos()
    artifacts = []
    parity = []
    for index, item in enumerate(assignments.itertuples(index=False), start=1):
        print(f"Test feature generation {index}/{len(assignments)}: {item.subject}", flush=True)
        events = valid_events(item.subject)

        psg_raw = read_uv(item.subject, "psg", PSG6)
        psg6 = epoch_features(
            normalize_signal(filter_resample(psg_raw, sos), PSG6, psg_scaler),
            PSG6,
            events,
        )
        psg2 = epoch_features(
            normalize_signal(filter_resample(psg_raw[:2], sos), PSG2, psg_scaler),
            PSG2,
            events,
        )

        hb_raw = read_uv(item.subject, "headband", HB2)
        hb_filtered = filter_resample(hb_raw, sos)
        hb2 = epoch_features(
            normalize_signal(hb_filtered, HB2, hb_scaler), HB2, events
        )
        hb2_psg = epoch_features(
            normalize_signal(hb_filtered, HB2, zero_scaler), HB2, events
        )

        values = {
            "PSG-6": psg6,
            "PSG-2": psg2,
            "HB-2": hb2,
            "HB-2-PSGscale": hb2_psg,
        }
        for modality, feature_values in values.items():
            artifacts.append(save_feature(item.subject, item.pid, modality, feature_values))

        timing_pass = all(
            np.array_equal(psg6[0], feature_values[0])
            and np.array_equal(psg6[1], feature_values[1])
            for feature_values in [psg2, hb2, hb2_psg]
        )
        centers = [context_matrix(feature_values[0], feature_values[2])[0] for feature_values in values.values()]
        context_pass = all(np.array_equal(centers[0], item_centers) for item_centers in centers[1:])
        psg_overlap = float(np.max(np.abs(psg6[2][:, : psg2[2].shape[1]] - psg2[2])))
        parity.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": PARTITION,
                "epoch_timing_and_stage_parity": timing_pass,
                "context_center_parity": context_pass,
                "psg_overlap_max_abs_difference": psg_overlap,
                "psg_overlap_pass": psg_overlap <= PSG_OVERLAP_TOLERANCE,
            }
        )
    return pd.DataFrame(artifacts), pd.DataFrame(parity)


def load_feature(subject: str, modality: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(feature_path(subject, modality), allow_pickle=False) as values:
        onsets = values["onset"].astype(np.float64)
        features = values["features"].astype(np.float32)
    if len(np.unique(onsets)) != len(onsets) or not np.isfinite(features).all():
        raise ValueError(f"Invalid test feature artifact: {subject}, {modality}")
    return onsets, features


# Section 5: frozen models and full-night scores

def verify_model_freeze() -> pd.DataFrame:
    freeze_text = (
        repo_root() / "docs/evaluation/block7_validation_freeze_and_test_entry_v0.1.md"
    ).read_text(encoding="utf-8")
    if FREEZE_MARKER not in freeze_text:
        raise ValueError("Block 7 freeze marker is absent")
    rows = []
    for comparator, expected in MODEL_SHA256.items():
        path = model_path(comparator)
        observed = sha256(path)
        rows.append(
            {
                "comparator": comparator,
                "expected_model_sha256": expected,
                "observed_model_sha256": observed,
                "hash_match": observed == expected,
                "threshold": THRESHOLDS[comparator],
            }
        )
    result = pd.DataFrame(rows)
    if not result["hash_match"].all():
        raise ValueError("A frozen Block 7 model hash changed")
    return result


def score_test(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = {name: joblib.load(model_path(name)) for name in MODEL_SHA256}
    score_rows = []
    support_rows = []
    for comparator, (model_name, modality) in SCORE_SPECS.items():
        for item in assignments.itertuples(index=False):
            onsets, features = load_feature(item.subject, modality)
            centers, matrix = context_matrix(onsets, features)
            probability = models[model_name].predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": model_name,
                        "partition": PARTITION,
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
                    "partition": PARTITION,
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
                    "partition": PARTITION,
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


# Section 6: fixed test evaluation and paired contrasts

def evaluate_test(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    alarms = pd.concat(
        [collapse_alarms(group, THRESHOLDS[name]) for name, group in scores.groupby("comparator", sort=True)],
        ignore_index=True,
    )
    subsets = {
        "all_test": set(),
        "exclude_sub32_sub50": FIXED_SENSITIVITY_EXCLUSIONS,
    }
    summaries = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for subset, exclusions in subsets.items():
        subset_references = references[~references["subject"].isin(exclusions)]
        for comparator in sorted(scores["comparator"].unique()):
            local_support = support[
                (support["comparator"] == comparator)
                & ~support["subject"].isin(exclusions)
            ][["subject", "pid", "supported_hours"]]
            predictions = alarms[
                (alarms["comparator"] == comparator)
                & ~alarms["subject"].isin(exclusions)
            ][["subject", "pid", "event_time_sec"]]
            for membership in MEMBERSHIPS:
                eligible, ignored = local_event_inputs(subset_references, membership)
                for tolerance in TOLERANCES:
                    recordings, participants, matches, summary = evaluate_events(
                        eligible, predictions, ignored, local_support, tolerance
                    )
                    config = {
                        "analysis_subset": subset,
                        "comparator": comparator,
                        "model_role": COMPARATOR_ROLES[comparator],
                        "partition": PARTITION,
                        "membership": membership,
                        "tolerance_sec": tolerance,
                        "threshold": THRESHOLDS[comparator],
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
        (participants["analysis_subset"] == "all_test")
        & (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    comparisons = [
        ("P6-D", "P2-D"),
        ("P2-D", "H2-D"),
        ("P2-D", "P2-H2-Z"),
        ("P2-H2-Z", "H2-D"),
    ]
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
        paired = left.merge(right, on="pid", suffixes=("_left", "_right"), validate="one_to_one")
        if len(paired) != 20:
            raise ValueError(f"Expected 20 paired test participants: {left_name}")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
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
        for side in ["left", "right"]:
            point[side] = metric_values(
                int(paired[f"true_positive_{side}"].sum()),
                int(paired[f"false_positive_{side}"].sum()),
                int(paired[f"false_negative_{side}"].sum()),
                float(paired[f"supported_hours_{side}"].sum()),
            )
        point_values = {
            "event_f1_difference": point["left"]["f1"] - point["right"]["f1"],
            "false_alarms_per_hour_difference": point["left"]["false_alarms_per_hour"]
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


def validation_pattern_summary(metrics: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["analysis_subset"] == "all_test")
        & (metrics["membership"] == "primary")
        & (metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    intervals = paired.set_index(["comparison", "metric"])
    specifications = [
        (
            "H7.1_channel_reduction",
            "P6-D_minus_P2-D",
            "validation favored P2-D F1 and FAR",
            primary.loc["P6-D", "f1"] - primary.loc["P2-D", "f1"],
            primary.loc["P6-D", "false_alarms_per_hour"]
            - primary.loc["P2-D", "false_alarms_per_hour"],
        ),
        (
            "H7.2_source_to_zero_shot",
            "P2-D_minus_P2-H2-Z",
            "validation showed a P2-D F1 advantage; FAR difference inconclusive",
            primary.loc["P2-D", "f1"] - primary.loc["P2-H2-Z", "f1"],
            primary.loc["P2-D", "false_alarms_per_hour"]
            - primary.loc["P2-H2-Z", "false_alarms_per_hour"],
        ),
        (
            "H7.2_zero_shot_to_direct_wearable",
            "P2-H2-Z_minus_H2-D",
            "validation showed lower zero-shot F1 but fewer false alarms",
            primary.loc["P2-H2-Z", "f1"] - primary.loc["H2-D", "f1"],
            primary.loc["P2-H2-Z", "false_alarms_per_hour"]
            - primary.loc["H2-D", "false_alarms_per_hour"],
        ),
    ]
    rows = []
    for question, comparison, validation_result, f1_difference, far_difference in specifications:
        f1_interval = intervals.loc[(comparison, "event_f1_difference")]
        far_interval = intervals.loc[(comparison, "false_alarms_per_hour_difference")]
        rows.append(
            {
                "question": question,
                "comparison": comparison,
                "validation_result_frozen_before_test": validation_result,
                "test_f1_difference": f1_difference,
                "test_f1_lower_95": f1_interval.lower_95,
                "test_f1_upper_95": f1_interval.upper_95,
                "test_far_difference": far_difference,
                "test_far_lower_95": far_interval.lower_95,
                "test_far_upper_95": far_interval.upper_95,
            }
        )
    return pd.DataFrame(rows)


# Section 7: manifests, checks, and reviewed summary

def external_manifest(features: pd.DataFrame) -> pd.DataFrame:
    paths = [
        ("generated_test_feature", data_parent() / item.path_relative_to_data_parent)
        for item in features.itertuples(index=False)
    ]
    paths.extend(("frozen_direct_model", model_path(name)) for name in MODEL_SHA256)
    paths.append(("test_continuous_scores", score_path()))
    rows = [
        {
            "artifact_role": role,
            "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for role, path in paths
    ]
    return pd.DataFrame(rows).sort_values(["artifact_role", "path_relative_to_data_parent"])


def run_checks(
    assignments: pd.DataFrame,
    features: pd.DataFrame,
    parity: pd.DataFrame,
    model_freeze: pd.DataFrame,
    scores: pd.DataFrame,
    support: pd.DataFrame,
    metrics: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    expected_comparators = set(THRESHOLDS)
    threshold_map = metrics.groupby("comparator")["threshold"].unique().to_dict()
    rows = [
        ("frozen_test_membership", len(assignments) == 26 and assignments["pid"].nunique() == 20, "26 recordings; 20 pid groups"),
        ("four_features_per_recording", len(features) == 104 and features.groupby("subject")["modality"].nunique().eq(4).all(), "104 feature artifacts"),
        ("finite_test_features", truth(features["all_features_finite"]).all(), "all generated features finite"),
        ("epoch_and_stage_parity", len(parity) == 26 and truth(parity["epoch_timing_and_stage_parity"]).all(), "all test recordings"),
        ("context_center_parity", truth(parity["context_center_parity"]).all(), "all test recordings"),
        ("psg_overlap_parity", float(parity["psg_overlap_max_abs_difference"].max()) <= PSG_OVERLAP_TOLERANCE, f"maximum={parity['psg_overlap_max_abs_difference'].max():.12g}"),
        ("frozen_model_hashes", len(model_freeze) == 3 and truth(model_freeze["hash_match"]).all(), "three exact model hashes"),
        ("four_frozen_comparators", set(scores["comparator"]) == expected_comparators and "P2-H2-A" not in set(scores["comparator"]), "alignment comparator remains excluded"),
        ("test_partition_only", set(scores["partition"]) == {PARTITION} and set(support["partition"]) == {PARTITION}, "test rows only"),
        ("complete_test_support", support.groupby("comparator")["subject"].nunique().eq(26).all() and support["comparator"].nunique() == 4, "26 recordings for each comparator"),
        ("frozen_thresholds_applied", all(len(threshold_map[name]) == 1 and np.isclose(threshold_map[name][0], THRESHOLDS[name]) for name in THRESHOLDS), str(THRESHOLDS)),
        ("complete_fixed_evaluations", len(metrics) == 32, "4 comparators x 2 subsets x 2 memberships x 2 tolerances"),
        ("external_hashes_recorded", len(manifest) == 108 and manifest["sha256"].str.len().eq(64).all(), "104 features, 3 models, 1 score artifact"),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in rows]
    )


def write_readme(
    result_code_commit: str,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    patterns: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    primary = metrics[
        (metrics["analysis_subset"] == "all_test")
        & (metrics["membership"] == "primary")
        & (metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    metric_rows = []
    for comparator in sorted(primary.index):
        row = primary.loc[comparator]
        metric_rows.append(
            f"| {comparator} | {row.threshold:.2f} | {int(row.reference_events)} | {int(row.predicted_events)} | "
            f"{row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        )
    paired_rows = [
        f"| {item.comparison} | {item.metric} | {item.point_difference:+.4f} | {item.lower_95:+.4f} to {item.upper_95:+.4f} |"
        for item in paired.itertuples(index=False)
    ]
    pattern_rows = [
        f"| {item.question} | {item.comparison} | {item.test_f1_difference:+.4f} "
        f"[{item.test_f1_lower_95:+.4f}, {item.test_f1_upper_95:+.4f}] | "
        f"{item.test_far_difference:+.4f} [{item.test_far_lower_95:+.4f}, {item.test_far_upper_95:+.4f}] |"
        for item in patterns.itertuples(index=False)
    ]
    text = "\n".join(
        [
            "# Block 7 Frozen Descriptive Test v0.1",
            "",
            "**Work date:** 2026-09-06",
            f"**Validation freeze commit:** `{FREEZE_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Partition accessed:** Frozen test only",
            "**Interpretation:** Descriptive; the same test participants were used in earlier project blocks",
            "",
            "## Primary Descriptive Result",
            "",
            "| Comparator | Threshold | References | Alarms | Precision | Recall | F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            *metric_rows,
            "",
            "All models and thresholds were frozen before this test execution. No model was fitted, recalibrated, selected, or revised after test access. The predeclared feature-alignment branch remained closed and was not evaluated.",
            "",
            "## Paired Participant Contrasts",
            "",
            "| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |",
            "|---|---|---:|---:|",
            *paired_rows,
            "",
            "## Validation-Pattern Check",
            "",
            "| Question | Comparison | Test F1 difference [95% interval] | Test false-alarm difference [95% interval] |",
            "|---|---|---:|---:|",
            *pattern_rows,
            "",
            "The table compares the frozen validation interpretation with the descriptive test direction; it is not an independent confirmatory test. The fixed `sub-32`/`sub-50` exclusion is retained separately in the full metric table and does not replace the all-test result.",
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. These results close the fixed Block 7 comparison but cannot authorize post-test tuning. Any later performance claim requires a newly locked or external cohort.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)


# Section 8: execute once under the committed freeze

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")

    output_dir().mkdir(parents=True, exist_ok=True)
    assignments = test_assignments()
    features, parity = generate_test_features(assignments)
    model_freeze = verify_model_freeze()
    scores, support = score_test(assignments)
    verify_or_create_gzip_tsv(scores, score_path())
    references = reference_events(assignments)
    outputs = evaluate_test(scores, support, references)
    paired = paired_participant_bootstrap(outputs["event_participants"])
    patterns = validation_pattern_summary(outputs["event_metrics"], paired)
    manifest = external_manifest(features)
    checks = run_checks(
        assignments,
        features,
        parity,
        model_freeze,
        scores,
        support,
        outputs["event_metrics"],
        manifest,
    )

    reviewed = {
        "generated_test_feature_artifacts_v0.1.tsv": features,
        "test_feature_parity_checks_v0.1.tsv": parity,
        "frozen_model_verification_v0.1.tsv": model_freeze,
        "test_support_v0.1.tsv": support,
        "test_predicted_events_v0.1.tsv": outputs["predicted_events"],
        "test_event_metrics_v0.1.tsv": outputs["event_metrics"],
        "test_event_recordings_v0.1.tsv": outputs["event_recordings"],
        "test_event_participants_v0.1.tsv": outputs["event_participants"],
        "test_event_matches_v0.1.tsv": outputs["event_matches"],
        "test_paired_participant_bootstrap_v0.1.tsv": paired,
        "validation_pattern_summary_v0.1.tsv": patterns,
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
    write_readme(args.result_code_commit, outputs["event_metrics"], paired, patterns, checks)

    primary = outputs["event_metrics"]
    primary = primary[
        (primary["analysis_subset"] == "all_test")
        & (primary["membership"] == "primary")
        & (primary["tolerance_sec"] == 15.0)
    ]
    print(primary[["comparator", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(patterns.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 7 test check failed")


if __name__ == "__main__":
    main()
