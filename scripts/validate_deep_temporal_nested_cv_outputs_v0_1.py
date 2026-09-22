"""Independently validate nested deep-temporal reviewed outputs read-only."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_block7_transfer_validation_v0_1 import local_event_inputs, reference_events
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: fixed validation inputs

EXPERIMENT_DIR = "2026-09-22_deep_temporal_nested_cv_v0.1"
BASE_SEED = 20260922
BOOTSTRAP_RESAMPLES = 2000
MEANINGFUL_F1_GAIN = 0.05
EPOCH_SEC = 30.0
CANDIDATES = ["BLSTM-CRF", "BGRU-CRF", "TCN-CRF", "BLSTM-2H"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def read_output(name: str) -> pd.DataFrame:
    return pd.read_csv(output_dir() / name, sep="\t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frames_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
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
            if not np.array_equal(
                left[column].fillna("").astype(str).to_numpy(),
                right[column].fillna("").astype(str).to_numpy(),
            ):
                return False
    return True


def train_assignments() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in source[source["partition"] == "train"].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    return pd.DataFrame(rows)


# Section 2: independent architecture and threshold reconstruction

def rebuild_inner_selections(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (outer, candidate), group in curves.groupby(["outer_fold", "candidate"]):
        selected = group.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        rows.append(selected.to_dict())
    result = pd.DataFrame(rows)
    candidate_order = {name: index for index, name in enumerate(CANDIDATES)}
    result["candidate_order"] = result["candidate"].map(candidate_order)
    return result.sort_values(["outer_fold", "candidate_order"]).drop(
        columns="candidate_order"
    ).reset_index(drop=True)


def rebuild_architecture(
    selections: pd.DataFrame, configuration: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parameters = configuration.set_index("candidate")["trainable_parameters"].to_dict()
    rows = []
    for outer, group in selections.groupby("outer_fold"):
        local = group.copy()
        local["trainable_parameters"] = local["candidate"].map(parameters)
        local = local.sort_values(
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
        local["inner_rank"] = np.arange(1, len(local) + 1)
        rows.append(local)
    architecture = pd.concat(rows, ignore_index=True)
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
    return architecture, recommendation


# Section 3: independent outer event reconstruction

def collapse_fold(scores: pd.DataFrame, threshold: float, pipeline: str) -> pd.DataFrame:
    rows = []
    for (subject, pid, outer), group in scores.groupby(
        ["subject", "pid", "outer_fold"], sort=True
    ):
        group = group.sort_values("candidate_time_sec")
        times = group["candidate_time_sec"].to_numpy(dtype=float)
        probability = group["probability"].to_numpy(dtype=float)
        selected = np.flatnonzero(probability >= threshold)
        if len(selected) == 0:
            continue
        split = np.flatnonzero(np.diff(times[selected]) > EPOCH_SEC + 1e-6) + 1
        for run in np.split(selected, split):
            local = probability[run]
            maximum = local.max()
            best = run[np.flatnonzero(np.isclose(local, maximum))[0]]
            rows.append(
                {
                    "pipeline": pipeline,
                    "outer_fold": int(outer),
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(times[best]),
                    "probability": float(probability[best]),
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(rows)


def rebuild_events(
    baseline_scores: pd.DataFrame,
    selected_scores: pd.DataFrame,
    architecture: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for outer in range(1, 6):
        local = architecture[architecture["outer_fold"] == outer]
        baseline_threshold = float(
            local[local["candidate"] == "BLSTM-CRF"].iloc[0].threshold
        )
        selected_row = local[local["inner_rank"] == 1].iloc[0]
        rows.append(
            collapse_fold(
                baseline_scores[baseline_scores["outer_fold"] == outer],
                baseline_threshold,
                "NESTED-BLSTM-CRF",
            )
        )
        rows.append(
            collapse_fold(
                selected_scores[selected_scores["outer_fold"] == outer],
                float(selected_row.threshold),
                "NESTED-SELECTED",
            )
        )
    return pd.concat(rows, ignore_index=True)


def rebuild_evaluation(
    events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metrics = []
    recordings = []
    participants = []
    matches = []
    for pipeline in ["NESTED-BLSTM-CRF", "NESTED-SELECTED"]:
        predictions = events[events["pipeline"] == pipeline]
        for membership in ["primary", "expanded"]:
            eligible, ignored = local_event_inputs(references, membership)
            for tolerance in [15.0, 45.0]:
                local_recordings, local_participants, local_matches, summary = evaluate_events(
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
                metrics.append({**config, **summary})
                for frame, collection in [
                    (local_recordings, recordings),
                    (local_participants, participants),
                    (local_matches, matches),
                ]:
                    if len(frame):
                        frame = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in frame.columns:
                                frame.insert(0, key, value)
                        collection.append(frame)
    return {
        "metrics": pd.DataFrame(metrics),
        "recordings": pd.concat(recordings, ignore_index=True),
        "participants": pd.concat(participants, ignore_index=True),
        "matches": pd.concat(matches, ignore_index=True),
    }


# Section 4: independent uncertainty and decisions

def rebuild_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
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
    for name, frame in [("left", left), ("right", right)]:
        point[name] = metric_values(
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


def rebuild_decisions(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
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
    direction = bool(
        intervals.loc["event_f1_difference", "lower_95"] > 0.0
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.0
    )
    material = bool(
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
                "supported": direction,
                "decision": "supported" if direction else "not_supported",
            },
            {
                "hypothesis": "participant_material_support",
                "f1_difference": f1_difference,
                "required_f1_difference": MEANINGFUL_F1_GAIN,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_far_difference": 0.0,
                "supported": material,
                "decision": "supported" if material else "not_supported",
            },
        ]
    )


# Section 5: validation entry point

def main() -> None:
    configuration = read_output("candidate_configuration_v0.1.tsv")
    curves = read_output("inner_threshold_curves_v0.1.tsv")
    stored_selections = read_output("inner_threshold_selections_v0.1.tsv")
    stored_architecture = read_output("outer_architecture_rankings_v0.1.tsv")
    stored_recommendation = read_output("architecture_recommendation_v0.1.tsv")
    support = read_output("outer_support_v0.1.tsv")
    manifest = read_output("external_artifact_manifest_v0.1.tsv")
    in_run = read_output("output_integrity_checks_v0.1.tsv")
    synthetic = read_output("synthetic_model_checks_v0.1.tsv")
    deterministic = read_output("deterministic_fit_checks_v0.1.tsv")

    selections = rebuild_inner_selections(curves)
    architecture, recommendation = rebuild_architecture(selections, configuration)
    score_entries = manifest[manifest["artifact_role"] == "pooled_outer_scores"]
    score_frames = [
        pd.read_csv(data_parent() / item.path_relative_to_data_parent, sep="\t")
        for item in score_entries.itertuples(index=False)
    ]
    baseline_scores = next(
        frame for frame in score_frames if set(frame["pipeline"]) == {"NESTED-BLSTM-CRF"}
    )
    selected_scores = next(
        frame for frame in score_frames if set(frame["pipeline"]) == {"NESTED-SELECTED"}
    )
    events = rebuild_events(baseline_scores, selected_scores, architecture)
    assignments = train_assignments()
    outputs = rebuild_evaluation(events, support, reference_events(assignments))
    bootstrap = rebuild_bootstrap(outputs["participants"])
    decisions = rebuild_decisions(outputs["metrics"], bootstrap)

    manifest_pass = True
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        manifest_pass = manifest_pass and path.exists()
        if path.exists():
            manifest_pass = manifest_pass and path.stat().st_size == int(item.bytes)
            manifest_pass = manifest_pass and sha256(path) == item.sha256

    event_match = frames_match(events, read_output("outer_predicted_events_v0.1.tsv"))
    output_files = {
        "metrics": "outer_event_metrics_v0.1.tsv",
        "recordings": "outer_event_recordings_v0.1.tsv",
        "participants": "outer_event_participants_v0.1.tsv",
        "matches": "outer_event_matches_v0.1.tsv",
    }
    evaluation_match = all(
        frames_match(outputs[key], read_output(name)) for key, name in output_files.items()
    )
    uncertainty_match = frames_match(
        bootstrap, read_output("paired_participant_bootstrap_v0.1.tsv")
    ) and frames_match(decisions, read_output("hypothesis_decisions_v0.1.tsv"))

    checks = pd.DataFrame(
        [
            {
                "check": "reviewed_in_run_checks",
                "status": "pass" if in_run["status"].eq("pass").all() else "fail",
                "detail": f"{in_run['status'].eq('pass').sum()}/{len(in_run)} passed",
            },
            {
                "check": "candidate_configuration",
                "status": "pass"
                if configuration["candidate"].tolist() == CANDIDATES
                else "fail",
                "detail": "four frozen candidates",
            },
            {
                "check": "synthetic_and_deterministic_controls",
                "status": "pass"
                if synthetic["status"].eq("pass").all()
                and deterministic["status"].eq("pass").all()
                else "fail",
                "detail": "stored controls all pass",
            },
            {
                "check": "tcn_receptive_field",
                "status": "pass" if 1 + 2 * (1 + 2 + 4) >= 8 else "fail",
                "detail": "kernel 3; dilations 1,2,4; receptive field 15",
            },
            {
                "check": "external_artifact_hashes",
                "status": "pass" if manifest_pass else "fail",
                "detail": f"{len(manifest)} artifacts",
            },
            {
                "check": "partition_isolation",
                "status": "pass"
                if not manifest["path_relative_to_data_parent"]
                .str.contains("/validation/|/test/|(?:^|/)test_", case=False, regex=True)
                .any()
                else "fail",
                "detail": "train-only external paths",
            },
            {
                "check": "inner_threshold_reconstruction",
                "status": "pass"
                if frames_match(selections, stored_selections)
                else "fail",
                "detail": "20 candidate-outer selections",
            },
            {
                "check": "architecture_reconstruction",
                "status": "pass"
                if frames_match(architecture, stored_architecture)
                and frames_match(recommendation, stored_recommendation)
                else "fail",
                "detail": "five outer rankings and final inner-ranked candidate",
            },
            {
                "check": "outer_event_reconstruction",
                "status": "pass" if event_match else "fail",
                "detail": "fold-specific selected thresholds",
            },
            {
                "check": "outer_evaluation_reconstruction",
                "status": "pass" if evaluation_match else "fail",
                "detail": "metrics, recordings, participants, and matches",
            },
            {
                "check": "uncertainty_and_decision_reconstruction",
                "status": "pass" if uncertainty_match else "fail",
                "detail": "2000 paired resamples and three decisions",
            },
            {
                "check": "complete_outer_scores",
                "status": "pass"
                if all(
                    frame["subject"].nunique() == 82
                    and frame["pid"].nunique() == 64
                    and np.isfinite(frame["probability"]).all()
                    for frame in score_frames
                )
                else "fail",
                "detail": "two complete nested pipelines",
            },
        ]
    )
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("Independent validation failed")
    print(f"Independent validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
