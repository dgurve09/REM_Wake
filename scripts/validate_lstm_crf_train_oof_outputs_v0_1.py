"""Independently validate the reviewed train-only LSTM-CRF outputs read-only."""

from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_block7_transfer_validation_v0_1 import (
    local_event_inputs,
    reference_events,
)
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: fixed validation configuration and paths

EXPERIMENT_DIR = "2026-09-19_lstm_crf_train_oof_v0.1"
BASE_SEED = 20260919
EPOCH_SEC = 30.0
BOOTSTRAP_RESAMPLES = 2000
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
MEANINGFUL_F1_GAIN = 0.05


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
            left_values = left[column].fillna("").astype(str).to_numpy()
            right_values = right[column].fillna("").astype(str).to_numpy()
            if not np.array_equal(left_values, right_values):
                return False
    return True


# Section 2: independent train membership and event calculations

def train_assignments() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    source = source[source["partition"] == "train"]
    rows = []
    for item in source.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    return pd.DataFrame(rows)


def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
    for (subject, pid), group in marked.groupby(["subject", "pid"], sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        boundaries = np.flatnonzero(
            np.diff(group["candidate_time_sec"].to_numpy(dtype=float))
            > EPOCH_SEC + 1e-6
        ) + 1
        starts = [0] + boundaries.tolist()
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


def rebuild_thresholds(
    scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, ignored = local_event_inputs(references, "primary")
    curve_rows = []
    selected_rows = []
    for model_name in ["LR-OOF", "LC-1"]:
        rows = []
        local_scores = scores[scores["model"] == model_name]
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


def rebuild_events(
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
        events = predictions.copy()
        events.insert(0, "model", model_name)
        event_rows.append(events)
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


# Section 3: independent uncertainty and decision reconstruction

def aggregate(frame: pd.DataFrame) -> dict:
    return metric_values(
        int(frame["true_positive"].sum()),
        int(frame["false_positive"].sum()),
        int(frame["false_negative"].sum()),
        float(frame["supported_hours"].sum()),
    )


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
    left = primary[primary["model"] == "LC-1"][columns]
    right = primary[primary["model"] == "LR-OOF"][columns]
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
    point_left = aggregate(left)
    point_right = aggregate(right)
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
                "lower_95": float(samples[metric].quantile(0.025)),
                "median": float(samples[metric].quantile(0.5)),
                "upper_95": float(samples[metric].quantile(0.975)),
            }
            for metric, point in points.items()
        ]
    )


def rebuild_decisions(metrics: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
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


def independent_crf_math_check() -> bool:
    start = np.asarray([0.1, -0.2, 0.3])
    transition = np.asarray(
        [[0.2, -0.1, 0.0], [-0.3, 0.4, 0.1], [0.0, -0.2, 0.2]]
    )
    end = np.asarray([-0.1, 0.2, 0.0])
    emissions = np.asarray(
        [[0.2, -0.2, 0.1], [0.0, 0.3, -0.1], [-0.2, 0.1, 0.4]]
    )

    def energy(path: tuple[int, ...]) -> float:
        value = start[path[0]] + emissions[0, path[0]]
        for step in range(1, len(path)):
            value += transition[path[step - 1], path[step]] + emissions[step, path[step]]
        return float(value + end[path[-1]])

    energies = np.asarray(
        [energy(path) for path in itertools.product(range(3), repeat=3)]
    )
    maximum = energies.max()
    exhaustive = maximum + np.log(np.exp(energies - maximum).sum())
    alpha = start + emissions[0]
    for step in range(1, len(emissions)):
        values = alpha[:, None] + transition
        maxima = values.max(axis=0)
        alpha = maxima + np.log(np.exp(values - maxima).sum(axis=0)) + emissions[step]
    values = alpha + end
    forward = values.max() + np.log(np.exp(values - values.max()).sum())
    return bool(abs(exhaustive - forward) <= 1e-12)


# Section 4: read-only validation entry point

def main() -> None:
    assignments = train_assignments()
    references = reference_events(assignments)
    construction = read_output("candidate_construction_v0.1.tsv")
    folds = read_output("train_oof_fold_assignments_v0.1.tsv")
    support = read_output("train_oof_support_v0.1.tsv")
    stored_curve = read_output("train_oof_threshold_curve_v0.1.tsv")
    stored_selected = read_output("train_oof_threshold_selection_v0.1.tsv")
    manifest = read_output("external_artifact_manifest_v0.1.tsv")
    in_run_checks = read_output("output_integrity_checks_v0.1.tsv")

    score_entry = manifest[manifest["artifact_role"] == "train_oof_scores"]
    if len(score_entry) != 1:
        raise ValueError("Expected one external score artifact")
    score_file = data_parent() / score_entry.iloc[0].path_relative_to_data_parent
    scores = pd.read_csv(score_file, sep="\t")

    rebuilt_curve, rebuilt_selected = rebuild_thresholds(scores, support, references)
    rebuilt_outputs = rebuild_events(scores, support, references, rebuilt_selected)
    rebuilt_bootstrap_frame = rebuild_bootstrap(rebuilt_outputs["participants"])
    rebuilt_decision_frame = rebuild_decisions(
        rebuilt_outputs["metrics"], rebuilt_bootstrap_frame
    )

    source_folds = pd.read_csv(
        repo_root()
        / "experiments/2026-09-12_block8_noise_augmented_training_v0.1"
        / "train_oof_fold_assignments_v0.1.tsv",
        sep="\t",
    ).sort_values(["fold", "pid"]).reset_index(drop=True)
    fold_match = frames_match(source_folds, folds)
    manifest_valid = True
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        manifest_valid = manifest_valid and path.exists()
        if path.exists():
            manifest_valid = manifest_valid and path.stat().st_size == int(item.bytes)
            manifest_valid = manifest_valid and sha256(path) == item.sha256

    model_counts = scores.groupby("model").size()
    participant_counts = scores.groupby("model")["pid"].nunique()
    support_match = (
        scores.groupby(["model", "subject"]).size().unstack("model").nunique(axis=1).eq(1).all()
    )
    partition_closed = bool(
        scores["partition"].eq("train_oof").all()
        and not manifest["path_relative_to_data_parent"]
        .str.contains(
            "/validation/|/test/|(?:^|/)test_", case=False, regex=True
        )
        .any()
    )
    candidate_valid = bool(
        len(construction) == 2743
        and construction["retained"].astype(str).str.lower().eq("true").sum() == 2743
        and construction.loc[
            construction["retained"].astype(str).str.lower().eq("true"), "label"
        ].sum()
        == 180
    )

    event_files = {
        "metrics": "train_oof_event_metrics_v0.1.tsv",
        "events": "train_oof_predicted_events_v0.1.tsv",
        "recordings": "train_oof_event_recordings_v0.1.tsv",
        "participants": "train_oof_event_participants_v0.1.tsv",
        "matches": "train_oof_event_matches_v0.1.tsv",
    }
    event_match = all(
        frames_match(rebuilt_outputs[key], read_output(filename))
        for key, filename in event_files.items()
    )
    uncertainty_match = frames_match(
        rebuilt_bootstrap_frame, read_output("paired_participant_bootstrap_v0.1.tsv")
    ) and frames_match(
        rebuilt_decision_frame, read_output("hypothesis_decisions_v0.1.tsv")
    )

    checks = pd.DataFrame(
        [
            {
                "check": "reviewed_in_run_checks",
                "status": "pass" if in_run_checks["status"].eq("pass").all() else "fail",
                "detail": f"{in_run_checks['status'].eq('pass').sum()}/{len(in_run_checks)} passed",
            },
            {
                "check": "train_partition_isolation",
                "status": "pass" if partition_closed else "fail",
                "detail": "external scores and manifest contain train-only artifacts",
            },
            {
                "check": "frozen_fold_identity",
                "status": "pass" if fold_match else "fail",
                "detail": "exact Block 8 pid-fold assignment",
            },
            {
                "check": "candidate_reconstruction",
                "status": "pass" if candidate_valid else "fail",
                "detail": "2743 retained candidates; 180 positive",
            },
            {
                "check": "independent_crf_math",
                "status": "pass" if independent_crf_math_check() else "fail",
                "detail": "forward partition equals exhaustive enumeration",
            },
            {
                "check": "external_artifact_hashes",
                "status": "pass" if manifest_valid else "fail",
                "detail": f"{len(manifest)} artifacts checked",
            },
            {
                "check": "external_score_completeness",
                "status": "pass"
                if model_counts.nunique() == 1
                and participant_counts.eq(64).all()
                and support_match
                and np.isfinite(scores["probability"]).all()
                and scores["probability"].between(0.0, 1.0).all()
                else "fail",
                "detail": f"rows per model={model_counts.to_dict()}",
            },
            {
                "check": "threshold_reconstruction",
                "status": "pass"
                if frames_match(rebuilt_curve, stored_curve)
                and frames_match(rebuilt_selected, stored_selected)
                else "fail",
                "detail": "198 curve rows and two selected thresholds",
            },
            {
                "check": "event_and_participant_reconstruction",
                "status": "pass" if event_match else "fail",
                "detail": "events, metrics, recordings, participants, and matches",
            },
            {
                "check": "uncertainty_and_decision_reconstruction",
                "status": "pass" if uncertainty_match else "fail",
                "detail": "2000 paired resamples and two decisions",
            },
        ]
    )
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("Independent validation failed")
    print(f"Independent validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
