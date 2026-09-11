"""Run the frozen validation-only Block 8 single-channel robustness test."""

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
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-11_block8_single_channel_robustness_v0.1"
TRANSFER_DERIVED_DIR = "block7_transfer_validation_v0.1"
ROBUSTNESS_DERIVED_DIR = "block8_single_channel_robustness_v0.1"
PROTOCOL_COMMIT = "b55849f"
PARTITION = "validation"
MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
THRESHOLD = 0.96
EPOCH_SEC = 30.0
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, EPOCH_SEC)
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
MATERIAL_F1_DROP = 0.03
MATERIAL_FAR_INCREASE = 0.50

COMPARATORS = {
    "H2-INTACT": None,
    "H2-NO-HB1": "HB_1",
    "H2-NO-HB2": "HB_2",
}
MODEL_ROLES = {
    "H2-INTACT": "frozen_two_channel_wearable",
    "H2-NO-HB1": "neutralized_hb1_feature_contribution",
    "H2-NO-HB2": "neutralized_hb2_feature_contribution",
}


# Section 2: paths and immutable helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def source_derived_dir() -> Path:
    return data_parent() / "derived" / TRANSFER_DERIVED_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / ROBUSTNESS_DERIVED_DIR


def feature_path(subject: str) -> Path:
    return (
        source_derived_dir()
        / "recording_features"
        / PARTITION
        / "hb2"
        / f"{subject}_features_v0.1.npz"
    )


def model_path() -> Path:
    return source_derived_dir() / "models" / "h2_d_model_v0.1.joblib"


def source_score_path() -> Path:
    return source_derived_dir() / "candidate_scores" / "validation_continuous_scores_v0.1.tsv.gz"


def score_path() -> Path:
    return derived_dir() / "candidate_scores" / "validation_ablation_scores_v0.1.tsv.gz"


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


# Section 3: validation membership, references, and feature schema

def validation_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in split[split["partition"] == PARTITION].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": PARTITION})
    result = pd.DataFrame(rows).sort_values("subject")
    if len(result) != 20 or result["pid"].nunique() != 16:
        raise ValueError("Frozen validation assignment must contain 20 recordings and 16 pid groups")
    if set(result["partition"]) != {PARTITION} or result["subject"].duplicated().any():
        raise ValueError("Invalid validation-only assignment")
    return result


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def reference_events(assignments: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(
        repo_root() / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        repo_root() / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    result = membership[
        membership["subject"].isin(set(assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
        & membership["partition"].eq(PARTITION)
    ].merge(quality, on="transition_id", validate="one_to_one")
    result["event_time_sec"] = result["nominal_boundary_sec"].astype(float)
    if set(result["partition"]) != {PARTITION}:
        raise ValueError("Reference events escaped validation")
    return result


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


def load_feature(subject: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(feature_path(subject), allow_pickle=False) as values:
        onsets = values["onset"].astype(np.float64)
        features = values["features"].astype(np.float32)
        names = values["feature_names"].astype(str).tolist()
    if len(onsets) != len(features) or len(np.unique(onsets)) != len(onsets):
        raise ValueError(f"Invalid validation feature dimensions: {subject}")
    if features.shape[1] != 10 or not np.isfinite(features).all():
        raise ValueError(f"Invalid validation wearable features: {subject}")
    return onsets, features, names


def context_matrix(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    required_gaps = len(CONTEXT_OFFSETS) - 1
    contiguous = np.isclose(np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(required_gaps, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(counts == required_gaps)
    matrix = np.concatenate(
        [features[indices + offset] for offset in range(len(CONTEXT_OFFSETS))],
        axis=1,
    )
    return onsets[indices + 4], matrix


def ablation_map(model, names: list[str]) -> pd.DataFrame:
    if names[:5] != [
        "HB_1_delta_log10_mean_psd",
        "HB_1_theta_log10_mean_psd",
        "HB_1_alpha_log10_mean_psd",
        "HB_1_sigma_log10_mean_psd",
        "HB_1_beta_log10_mean_psd",
    ] or not all(name.startswith("HB_2_") for name in names[5:]):
        raise ValueError("Unexpected frozen wearable feature order")
    means = model.named_steps["standardscaler"].mean_
    if len(means) != 80:
        raise ValueError("Frozen H2-D model does not have 80 context dimensions")
    rows = []
    for context_index, offset in enumerate(CONTEXT_OFFSETS):
        for base_index, name in enumerate(names):
            dimension = context_index * len(names) + base_index
            channel = "HB_1" if name.startswith("HB_1_") else "HB_2"
            rows.append(
                {
                    "dimension_index": dimension,
                    "context_offset_sec": float(offset),
                    "base_feature_index": base_index,
                    "feature_name": name,
                    "channel": channel,
                    "neutral_value": float(means[dimension]),
                    "neutralized_in_comparator": "H2-NO-HB1" if channel == "HB_1" else "H2-NO-HB2",
                }
            )
    return pd.DataFrame(rows)


# Section 4: fixed scoring and baseline reproduction

def verify_model() -> tuple[object, pd.DataFrame]:
    observed = sha256(model_path())
    if observed != MODEL_SHA256:
        raise ValueError("Frozen H2-D model hash changed")
    model = joblib.load(model_path())
    record = pd.DataFrame(
        [
            {
                "model": "H2-D",
                "expected_sha256": MODEL_SHA256,
                "observed_sha256": observed,
                "hash_match": True,
                "threshold": THRESHOLD,
            }
        ]
    )
    return model, record


def score_comparators(
    assignments: pd.DataFrame, model, mapping: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    indices = {
        comparator: mapping.loc[
            mapping["neutralized_in_comparator"] == comparator, "dimension_index"
        ].to_numpy(dtype=int)
        for comparator in ["H2-NO-HB1", "H2-NO-HB2"]
    }
    means = model.named_steps["standardscaler"].mean_
    score_rows = []
    support_rows = []
    feature_rows = []
    for item in assignments.itertuples(index=False):
        onsets, features, names = load_feature(item.subject)
        centers, intact = context_matrix(onsets, features)
        feature_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": PARTITION,
                "epochs": len(onsets),
                "context_rows": len(centers),
                "feature_dimensions": intact.shape[1],
                "ordered_schema_sha256": hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest(),
                "path_relative_to_data_parent": feature_path(item.subject).relative_to(data_parent()).as_posix(),
                "bytes": feature_path(item.subject).stat().st_size,
                "sha256": sha256(feature_path(item.subject)),
            }
        )
        for comparator in COMPARATORS:
            matrix = intact.copy()
            if comparator != "H2-INTACT":
                selected = indices[comparator]
                matrix[:, selected] = means[selected]
            probability = model.predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": "H2-D",
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
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows), pd.DataFrame(feature_rows)


def baseline_probability_reproduction(scores: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(source_score_path(), sep="\t")
    source = source[source["comparator"] == "H2-D"][
        ["subject", "pid", "candidate_time_sec", "probability"]
    ].sort_values(["subject", "candidate_time_sec"]).reset_index(drop=True)
    intact = scores[scores["comparator"] == "H2-INTACT"][
        ["subject", "pid", "candidate_time_sec", "probability"]
    ].sort_values(["subject", "candidate_time_sec"]).reset_index(drop=True)
    if not source.drop(columns="probability").equals(intact.drop(columns="probability")):
        raise ValueError("Intact and stored Block 7 validation score rows differ")
    difference = np.abs(source["probability"].to_numpy() - intact["probability"].to_numpy())
    result = pd.DataFrame(
        [
            {
                "source_comparator": "H2-D",
                "reproduced_comparator": "H2-INTACT",
                "rows": len(source),
                "maximum_absolute_probability_difference": float(difference.max()),
                "probability_tolerance": 1e-12,
                "reproduction_pass": bool(difference.max() <= 1e-12),
            }
        ]
    )
    if not bool(result.iloc[0].reproduction_pass):
        raise ValueError("Frozen intact probabilities did not reproduce")
    return result


# Section 5: event evaluation

def collapse_alarms(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= THRESHOLD]
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
            best = run[np.isclose(run["probability"], maximum)].sort_values("candidate_time_sec").iloc[0]
            rows.append(
                {
                    "comparator": comparator,
                    "partition": PARTITION,
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(best.candidate_time_sec),
                    "probability": float(best.probability),
                    "threshold": THRESHOLD,
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"],
    )


def evaluate_all(
    scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    alarms = collapse_alarms(scores)
    summaries = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for comparator in COMPARATORS:
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
                    "model_role": MODEL_ROLES[comparator],
                    "partition": PARTITION,
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": THRESHOLD,
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


def baseline_event_reproduction(metrics: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "experiments/2026-09-06_block7_transfer_validation_v0.1/validation_event_metrics_v0.1.tsv",
        sep="\t",
    )
    source = source[source["comparator"] == "H2-D"].sort_values(["membership", "tolerance_sec"])
    intact = metrics[metrics["comparator"] == "H2-INTACT"].sort_values(["membership", "tolerance_sec"])
    rows = []
    for source_row, intact_row in zip(source.itertuples(index=False), intact.itertuples(index=False)):
        differences = {
            field: abs(float(getattr(source_row, field)) - float(getattr(intact_row, field)))
            for field in ["precision", "recall", "f1", "false_alarms_per_hour"]
        }
        rows.append(
            {
                "membership": intact_row.membership,
                "tolerance_sec": intact_row.tolerance_sec,
                "maximum_absolute_metric_difference": max(differences.values()),
                "metric_tolerance": 1e-12,
                "reproduction_pass": max(differences.values()) <= 1e-12,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 4 or not result["reproduction_pass"].all():
        raise ValueError("Frozen intact event metrics did not reproduce")
    return result


# Section 6: material effects and leave-one-participant-out influence

def primary_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[
        (metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    intact = primary.loc["H2-INTACT"]
    rows = []
    for comparator in ["H2-NO-HB1", "H2-NO-HB2"]:
        ablated = primary.loc[comparator]
        f1_difference = float(ablated.f1 - intact.f1)
        far_difference = float(ablated.false_alarms_per_hour - intact.false_alarms_per_hour)
        materially_adverse = (
            f1_difference <= -MATERIAL_F1_DROP
            or far_difference >= MATERIAL_FAR_INCREASE
        )
        rows.append(
            {
                "comparison": f"{comparator}_minus_H2-INTACT",
                "ablated_channel": COMPARATORS[comparator],
                "f1_difference": f1_difference,
                "false_alarms_per_hour_difference": far_difference,
                "material_f1_drop_bound": MATERIAL_F1_DROP,
                "material_far_increase_bound": MATERIAL_FAR_INCREASE,
                "materially_adverse": materially_adverse,
                "apparent_metric_improvement": f1_difference > 0 or far_difference < 0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_participants(frame: pd.DataFrame) -> dict:
    return metric_values(
        int(frame["true_positive"].sum()),
        int(frame["false_positive"].sum()),
        int(frame["false_negative"].sum()),
        float(frame["supported_hours"].sum()),
    )


def sign_matches(value: float, reference: float) -> bool:
    return int(np.sign(value)) == int(np.sign(reference))


def leave_one_pid_out(
    participants: pd.DataFrame, comparisons: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = participants[
        (participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)
    ]
    full = comparisons.set_index("comparison")
    rows = []
    for comparator in ["H2-NO-HB1", "H2-NO-HB2"]:
        comparison = f"{comparator}_minus_H2-INTACT"
        for omitted_pid in sorted(primary["pid"].unique()):
            local = primary[primary["pid"] != omitted_pid]
            intact = aggregate_participants(local[local["comparator"] == "H2-INTACT"])
            ablated = aggregate_participants(local[local["comparator"] == comparator])
            f1_difference = ablated["f1"] - intact["f1"]
            far_difference = ablated["false_alarms_per_hour"] - intact["false_alarms_per_hour"]
            rows.append(
                {
                    "comparison": comparison,
                    "ablated_channel": COMPARATORS[comparator],
                    "omitted_pid": int(omitted_pid),
                    "remaining_pid": local["pid"].nunique(),
                    "f1_difference": f1_difference,
                    "false_alarms_per_hour_difference": far_difference,
                    "f1_direction_matches_full": sign_matches(f1_difference, full.loc[comparison, "f1_difference"]),
                    "far_direction_matches_full": sign_matches(far_difference, full.loc[comparison, "false_alarms_per_hour_difference"]),
                }
            )
    detail = pd.DataFrame(rows)
    summary_rows = []
    for comparison, group in detail.groupby("comparison", sort=True):
        summary_rows.append(
            {
                "comparison": comparison,
                "ablated_channel": group["ablated_channel"].iloc[0],
                "folds": len(group),
                "f1_direction_match_folds": int(group["f1_direction_matches_full"].sum()),
                "far_direction_match_folds": int(group["far_direction_matches_full"].sum()),
                "f1_difference_min": group["f1_difference"].min(),
                "f1_difference_max": group["f1_difference"].max(),
                "far_difference_min": group["false_alarms_per_hour_difference"].min(),
                "far_difference_max": group["false_alarms_per_hour_difference"].max(),
                "participant_stable": bool(
                    group["f1_direction_matches_full"].all()
                    and group["far_direction_matches_full"].all()
                ),
            }
        )
    return detail, pd.DataFrame(summary_rows)


def participant_concentration(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)
    ]
    rows = []
    for comparator, group in primary.groupby("comparator", sort=True):
        false_positives = int(group["false_positive"].sum())
        top_four = int(group.nlargest(4, "false_positive")["false_positive"].sum())
        rows.append(
            {
                "comparator": comparator,
                "pid": group["pid"].nunique(),
                "pid_with_false_positive": int((group["false_positive"] > 0).sum()),
                "total_false_positive": false_positives,
                "top_four_pid_false_positive": top_four,
                "top_four_pid_false_positive_share": top_four / false_positives if false_positives else np.nan,
            }
        )
    return pd.DataFrame(rows)


# Section 7: manifests, checks, and result summary

def external_manifest(features: pd.DataFrame) -> pd.DataFrame:
    paths = [("source_validation_feature", data_parent() / item.path_relative_to_data_parent) for item in features.itertuples(index=False)]
    paths.extend(
        [
            ("frozen_h2_model", model_path()),
            ("source_validation_scores", source_score_path()),
            ("ablation_validation_scores", score_path()),
        ]
    )
    return pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in paths
        ]
    ).sort_values(["artifact_role", "path_relative_to_data_parent"])


def run_checks(
    assignments: pd.DataFrame,
    model_record: pd.DataFrame,
    mapping: pd.DataFrame,
    scores: pd.DataFrame,
    support: pd.DataFrame,
    features: pd.DataFrame,
    probability_reproduction: pd.DataFrame,
    event_reproduction: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    comparisons: pd.DataFrame,
    lopo: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    support_counts = support.groupby("comparator")["subject"].nunique()
    rows = [
        ("validation_assignment_only", len(assignments) == 20 and assignments["pid"].nunique() == 16 and set(assignments["partition"]) == {PARTITION}, "20 recordings; 16 pid groups"),
        ("frozen_model_hash", bool(model_record.iloc[0].hash_match), MODEL_SHA256),
        ("ablation_dimensions", len(mapping) == 80 and mapping.groupby("channel").size().eq(40).all(), "40 dimensions for each channel"),
        ("validation_features_reopened", len(features) == 20 and features["feature_dimensions"].eq(80).all(), "20 hashed feature arrays"),
        ("three_fixed_comparators", set(scores["comparator"]) == set(COMPARATORS), "intact, no-HB1, no-HB2"),
        ("validation_scores_only", set(scores["partition"]) == {PARTITION}, "no non-validation score rows"),
        ("intact_probability_reproduction", bool(probability_reproduction.iloc[0].reproduction_pass), f"maximum={probability_reproduction.iloc[0].maximum_absolute_probability_difference:.12g}"),
        ("intact_event_reproduction", len(event_reproduction) == 4 and event_reproduction["reproduction_pass"].all(), "four membership/tolerance rows"),
        ("identical_support", len(support) == 60 and support_counts.eq(20).all() and support.groupby("subject")["supported_boundaries"].nunique().eq(1).all(), "same 20 recordings and boundaries"),
        ("fixed_threshold_only", outputs["event_metrics"]["threshold"].eq(THRESHOLD).all(), f"threshold={THRESHOLD}"),
        ("complete_event_outputs", len(outputs["event_metrics"]) == 12, "3 comparators x 2 memberships x 2 tolerances"),
        ("two_material_comparisons", len(comparisons) == 2, "one comparison per neutralized channel"),
        ("complete_leave_one_pid_out", len(lopo) == 32 and lopo.groupby("comparison")["omitted_pid"].nunique().eq(16).all(), "two comparisons x 16 omitted pid groups"),
        ("external_manifest", len(manifest) == 23 and manifest["sha256"].str.len().eq(64).all(), "20 features, model, source scores, ablation scores"),
        ("test_artifacts_closed", not manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any(), "no test path in external manifest"),
    ]
    return pd.DataFrame([{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in rows])


def write_readme(
    result_code_commit: str,
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    lopo_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index("comparator")
    metric_rows = [
        f"| {comparator} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        for comparator, row in primary.iterrows()
    ]
    comparison_rows = [
        f"| {item.comparison} | {item.f1_difference:+.4f} | {item.false_alarms_per_hour_difference:+.4f} | {bool(item.materially_adverse)} | {bool(item.apparent_metric_improvement)} |"
        for item in comparisons.itertuples(index=False)
    ]
    lopo_rows = [
        f"| {item.comparison} | {int(item.f1_direction_match_folds)}/16 | {int(item.far_direction_match_folds)}/16 | {item.f1_difference_min:+.4f} to {item.f1_difference_max:+.4f} | {item.far_difference_min:+.4f} to {item.far_difference_max:+.4f} | {bool(item.participant_stable)} |"
        for item in lopo_summary.itertuples(index=False)
    ]
    concentration_rows = [
        f"| {item.comparator} | {int(item.pid_with_false_positive)}/16 | {item.top_four_pid_false_positive_share:.2%} |"
        for item in concentration.itertuples(index=False)
    ]
    passed = not comparisons["materially_adverse"].any()
    unexpected = comparisons["apparent_metric_improvement"].any()
    decision = "pass_no_material_adverse_change" if passed else "fail_material_single_channel_effect"
    if passed and unexpected:
        decision = "pass_bounds_but_investigate_unexpected_ablation_improvement"
    text = "\n".join(
        [
            "# Block 8 Single-Channel Robustness v0.1",
            "",
            "**Work date:** 2026-09-11",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Partition accessed:** Validation only",
            "**Test or raw-signal artifacts accessed:** No",
            f"**Screen decision:** `{decision}`",
            "",
            "## Primary Result",
            "",
            "| Comparator | Precision | Recall | F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|",
            *metric_rows,
            "",
            "## Ablation Effects",
            "",
            "| Comparison | F1 difference | False-alarm difference/hour | Materially adverse | Any apparent improvement |",
            "|---|---:|---:|---:|---:|",
            *comparison_rows,
            "",
            "A materially adverse change was predefined as an F1 decrease of at least 0.03 or a false-alarm increase of at least 0.50/hour. An apparent improvement after neutralization is treated as channel-contribution or calibration instability, not as evidence that losing a channel is beneficial.",
            "",
            "## Leave-One-Participant-Out Influence",
            "",
            "| Comparison | F1 direction matches | FAR direction matches | F1 difference range | FAR difference range | Participant-stable |",
            "|---|---:|---:|---:|---:|---:|",
            *lopo_rows,
            "",
            "## False-Alarm Concentration",
            "",
            "| Comparator | Participants with false alarms | Top-four participant share |",
            "|---|---:|---:|",
            *concentration_rows,
            "",
            "## Interpretation Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. This validation-only feature-contribution ablation does not simulate physical electrode failure and cannot support post-test model revision. A raw-signal degradation experiment requires a separate protocol.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)


# Section 8: execute the frozen validation-only experiment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")

    output_dir().mkdir(parents=True, exist_ok=True)
    assignments = validation_assignments()
    model, model_record = verify_model()
    first_subject = assignments.iloc[0].subject
    _, _, names = load_feature(first_subject)
    mapping = ablation_map(model, names)
    scores, support, feature_manifest = score_comparators(assignments, model, mapping)
    probability_reproduction = baseline_probability_reproduction(scores)
    verify_or_create_gzip_tsv(scores, score_path())
    outputs = evaluate_all(scores, support, reference_events(assignments))
    event_reproduction = baseline_event_reproduction(outputs["event_metrics"])
    comparisons = primary_comparisons(outputs["event_metrics"])
    lopo, lopo_summary = leave_one_pid_out(outputs["event_participants"], comparisons)
    concentration = participant_concentration(outputs["event_participants"])
    manifest = external_manifest(feature_manifest)
    checks = run_checks(
        assignments,
        model_record,
        mapping,
        scores,
        support,
        feature_manifest,
        probability_reproduction,
        event_reproduction,
        outputs,
        comparisons,
        lopo,
        manifest,
    )

    reviewed = {
        "frozen_model_verification_v0.1.tsv": model_record,
        "feature_ablation_map_v0.1.tsv": mapping,
        "source_validation_feature_manifest_v0.1.tsv": feature_manifest,
        "intact_probability_reproduction_v0.1.tsv": probability_reproduction,
        "intact_event_reproduction_v0.1.tsv": event_reproduction,
        "validation_support_v0.1.tsv": support,
        "validation_predicted_events_v0.1.tsv": outputs["predicted_events"],
        "validation_event_metrics_v0.1.tsv": outputs["event_metrics"],
        "validation_event_recordings_v0.1.tsv": outputs["event_recordings"],
        "validation_event_participants_v0.1.tsv": outputs["event_participants"],
        "validation_event_matches_v0.1.tsv": outputs["event_matches"],
        "primary_ablation_comparisons_v0.1.tsv": comparisons,
        "leave_one_pid_out_influence_v0.1.tsv": lopo,
        "leave_one_pid_out_summary_v0.1.tsv": lopo_summary,
        "participant_false_alarm_concentration_v0.1.tsv": concentration,
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
        outputs["event_metrics"],
        comparisons,
        lopo_summary,
        concentration,
        checks,
    )

    primary = outputs["event_metrics"]
    primary = primary[(primary["membership"] == "primary") & (primary["tolerance_sec"] == 15.0)]
    print(primary[["comparator", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(comparisons.to_string(index=False))
    print(lopo_summary.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 8 in-run check failed")


if __name__ == "__main__":
    main()
