"""Diagnose frozen train-only deep temporal error modes without fitting a model."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import (
    evaluate_events,
    metric_values,
    optimal_matches,
)


# Section 1: fixed paths and configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-23_deep_temporal_error_diagnostic_v0.1"
SOURCE_EXPERIMENT = "2026-09-22_deep_temporal_nested_cv_v0.1"
SOURCE_DERIVED = "deep_temporal_nested_cv_v0.1"
TOLERANCES = [15.0, 45.0, 75.0, 105.0, 135.0]
BOUNDARY_WINDOWS = [0.0, 30.0, 60.0, 90.0]
EPOCH_SEC = 30.0


def threshold_grid() -> np.ndarray:
    coarse = np.arange(0.01, 0.951, 0.05)
    logits = np.arange(3.0, 14.001, 0.25)
    tail = 1.0 / (1.0 + np.exp(-logits))
    return np.unique(np.concatenate([coarse, tail]))


THRESHOLDS = threshold_grid()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def train_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    train = split[split["partition"].eq("train")].copy()
    rows = []
    for item in train.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    result = pd.DataFrame(rows)
    if len(result) != 82 or result["pid"].nunique() != 64:
        raise ValueError("Unexpected frozen train membership")
    if result["subject"].duplicated().any() or not result["partition"].eq("train").all():
        raise ValueError("Invalid frozen train assignment")
    return result.sort_values(
        "subject", key=lambda values: values.map(subject_number)
    ).reset_index(drop=True)


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
    ].merge(quality, on="transition_id", validate="one_to_one")
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    if set(rows["partition"]) != {"train"}:
        raise ValueError("Unauthorized reference-event partition")
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


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def source_result_dir() -> Path:
    return repo_root() / "experiments" / SOURCE_EXPERIMENT


def prior_result_dir() -> Path:
    return repo_root() / "experiments" / "2026-09-19_lstm_crf_train_oof_v0.1"


def source_score_path() -> Path:
    return (
        data_parent()
        / "derived"
        / SOURCE_DERIVED
        / "scores"
        / "nested_selected_outer_scores_v0.1.tsv.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Section 2: source loading and frozen-reference reconstruction

def load_sources() -> dict[str, pd.DataFrame]:
    assignments = train_assignments()
    references = reference_events(assignments)
    scores = pd.read_csv(source_score_path(), sep="\t")
    support = pd.read_csv(source_result_dir() / "outer_support_v0.1.tsv", sep="\t")
    predictions = pd.read_csv(
        source_result_dir() / "outer_predicted_events_v0.1.tsv", sep="\t"
    )
    metrics = pd.read_csv(
        source_result_dir() / "outer_event_metrics_v0.1.tsv", sep="\t"
    )
    rankings = pd.read_csv(
        source_result_dir() / "outer_architecture_rankings_v0.1.tsv", sep="\t"
    )
    membership = pd.read_csv(
        repo_root()
        / "labels"
        / "quality_analysis_membership_v0.1"
        / "transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    transitions = pd.read_csv(
        repo_root()
        / "labels"
        / "transition_labels_v0.1"
        / "transition_labels_v0.1.tsv",
        sep="\t",
    )
    return {
        "assignments": assignments,
        "references": references,
        "scores": scores,
        "support": support,
        "predictions": predictions,
        "metrics": metrics,
        "rankings": rankings,
        "membership": membership,
        "transitions": transitions,
    }


def primary_inputs(references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return local_event_inputs(references, "primary")


def evaluate(
    predictions: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    tolerance: float,
) -> tuple[pd.DataFrame, dict]:
    eligible, ignored = primary_inputs(references)
    _, _, matches, summary = evaluate_events(
        eligible,
        predictions[["subject", "pid", "event_time_sec"]],
        ignored,
        support[["subject", "pid", "supported_hours"]],
        tolerance,
    )
    return matches, summary


def collapsed_times(group: pd.DataFrame, threshold: float) -> np.ndarray:
    times = group["candidate_time_sec"].to_numpy(dtype=float)
    probabilities = group["probability"].to_numpy(dtype=float)
    selected = np.flatnonzero(probabilities >= threshold)
    if len(selected) == 0:
        return np.asarray([], dtype=float)
    splits = np.flatnonzero(np.diff(times[selected]) > EPOCH_SEC + 1e-6) + 1
    result = []
    for run in np.split(selected, splits):
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
    tp = fp = fn = ignored_count = predicted = reference_count = 0
    for item in support.itertuples(index=False):
        predictions = collapsed_times(score_groups[item.subject], threshold)
        refs = eligible_groups.get(item.subject, np.asarray([], dtype=float))
        ignored_refs = ignored_groups.get(item.subject, np.asarray([], dtype=float))
        matches = optimal_matches(refs, predictions, 15.0)
        matched_predictions = {value[1] for value in matches}
        unmatched = [
            index for index in range(len(predictions)) if index not in matched_predictions
        ]
        ignored_matches = optimal_matches(ignored_refs, predictions[unmatched], 15.0)
        tp += len(matches)
        fn += len(refs) - len(matches)
        fp += len(unmatched) - len(ignored_matches)
        ignored_count += len(ignored_matches)
        predicted += len(predictions)
        reference_count += len(refs)
    hours = float(support["supported_hours"].sum())
    return {
        "recordings": len(support),
        "pid": support["pid"].nunique(),
        "reference_events": reference_count,
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


# Section 3: boundary timing and optimistic threshold diagnostics

def tolerance_curve(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    predictions = data["predictions"].query("pipeline == 'NESTED-SELECTED'")
    rows = []
    for tolerance in TOLERANCES:
        _, summary = evaluate(
            predictions, data["support"], data["references"], tolerance
        )
        rows.append(
            {
                "analysis": "frozen_predictions_timing_diagnostic",
                "tolerance_sec": tolerance,
                **summary,
            }
        )
    return pd.DataFrame(rows)


def threshold_diagnostics(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = data["scores"]
    support = data["support"]
    references = data["references"]
    rankings = data["rankings"]

    frozen = data["metrics"].query(
        "pipeline == 'NESTED-SELECTED' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    rows = [
        {
            "threshold_analysis": "frozen_inner_selected",
            "label_use": "inner_only",
            **frozen[
                [
                    "predicted_events",
                    "true_positive",
                    "false_positive",
                    "false_negative",
                    "ignored_predictions",
                    "supported_hours",
                    "precision",
                    "recall",
                    "f1",
                    "false_alarms_per_hour",
                ]
            ].to_dict(),
        }
    ]

    _, pooled = select_threshold(
        "POOLED-OUTER-ORACLE", 0, scores, support, references
    )
    rows.append(
        {
            "threshold_analysis": "pooled_outer_label_oracle",
            "label_use": "outer_labels_optimistic_not_deployable",
            **{
                key: pooled[key]
                for key in [
                    "threshold",
                    "predicted_events",
                    "true_positive",
                    "false_positive",
                    "false_negative",
                    "ignored_predictions",
                    "supported_hours",
                    "precision",
                    "recall",
                    "f1",
                    "false_alarms_per_hour",
                ]
            },
        }
    )

    selected = rankings.query("inner_rank == 1").set_index("outer_fold")
    fold_rows = []
    oracle_events = []
    for fold in range(1, 6):
        fold_scores = scores[scores["outer_fold"] == fold]
        fold_support = support[support["outer_fold"] == fold]
        frozen_threshold = float(selected.loc[fold, "threshold"])
        frozen_summary = fast_primary_summary(
            fold_scores, fold_support, references, frozen_threshold
        )
        _, oracle = select_threshold(
            "OUTER-FOLD-ORACLE", fold, fold_scores, fold_support, references
        )
        oracle_events.append(
            collapse_events(
                fold_scores, float(oracle["threshold"]), "OUTER-FOLD-ORACLE"
            )
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "selected_candidate": selected.loc[fold, "candidate"],
                "frozen_threshold": frozen_threshold,
                "frozen_f1": frozen_summary["f1"],
                "frozen_false_alarms_per_hour": frozen_summary[
                    "false_alarms_per_hour"
                ],
                "outer_label_oracle_threshold": oracle["threshold"],
                "outer_label_oracle_f1": oracle["f1"],
                "outer_label_oracle_false_alarms_per_hour": oracle[
                    "false_alarms_per_hour"
                ],
                "f1_optimistic_gap": oracle["f1"] - frozen_summary["f1"],
            }
        )

    oracle_predictions = pd.concat(oracle_events, ignore_index=True)
    _, oracle_summary = evaluate(
        oracle_predictions, support, references, tolerance=15.0
    )
    rows.append(
        {
            "threshold_analysis": "outer_fold_label_oracle",
            "label_use": "each_outer_fold_labels_optimistic_not_deployable",
            **oracle_summary,
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows), oracle_predictions


# Section 4: event-level score and quality diagnostics

def matched_reference_keys(matches: pd.DataFrame) -> set[tuple[str, float]]:
    eligible = matches[matches["match_type"] == "eligible"]
    return set(
        zip(
            eligible["subject"].astype(str),
            eligible["reference_time_sec"].astype(float),
        )
    )


def boundary_scores(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, _ = primary_inputs(data["references"])
    predictions = data["predictions"].query("pipeline == 'NESTED-SELECTED'")
    matches_15, _ = evaluate(
        predictions, data["support"], data["references"], tolerance=15.0
    )
    matches_45, _ = evaluate(
        predictions, data["support"], data["references"], tolerance=45.0
    )
    keys_15 = matched_reference_keys(matches_15)
    keys_45 = matched_reference_keys(matches_45)
    selected = data["rankings"].query("inner_rank == 1").set_index("outer_fold")
    score_groups = {
        subject: group.sort_values("candidate_time_sec")
        for subject, group in data["scores"].groupby("subject")
    }
    subject_fold = data["support"].set_index("subject")["outer_fold"].to_dict()
    event_meta = data["references"].copy()
    event_meta = event_meta[event_meta["primary_analysis_eligible"]].copy()
    rows = []
    for event in event_meta.itertuples(index=False):
        group = score_groups[event.subject]
        times = group["candidate_time_sec"].to_numpy(dtype=float)
        probabilities = group["probability"].to_numpy(dtype=float)
        fold = int(subject_fold[event.subject])
        row = {
            "transition_id": int(event.transition_id),
            "subject": event.subject,
            "pid": int(event.pid),
            "outer_fold": fold,
            "event_time_sec": float(event.event_time_sec),
            "membership_tier": event.membership_tier,
            "preprocessing_decision": event.preprocessing_decision,
            "selected_candidate": selected.loc[fold, "candidate"],
            "frozen_threshold": float(selected.loc[fold, "threshold"]),
            "detected_within_15_sec": (event.subject, float(event.event_time_sec))
            in keys_15,
            "detected_within_45_sec": (event.subject, float(event.event_time_sec))
            in keys_45,
        }
        for window in BOUNDARY_WINDOWS:
            mask = np.abs(times - float(event.event_time_sec)) <= window + 1e-6
            name = "exact" if window == 0 else f"plus_minus_{int(window)}sec"
            row[f"maximum_probability_{name}"] = (
                float(probabilities[mask].max()) if mask.any() else np.nan
            )
        rows.append(row)
    events = pd.DataFrame(rows)
    summary_rows = []
    for column in [
        "maximum_probability_exact",
        "maximum_probability_plus_minus_30sec",
        "maximum_probability_plus_minus_60sec",
        "maximum_probability_plus_minus_90sec",
    ]:
        values = events[column]
        above = values >= events["frozen_threshold"]
        summary_rows.append(
            {
                "score_window": column.replace("maximum_probability_", ""),
                "events": len(events),
                "above_frozen_threshold": int(above.sum()),
                "fraction_above_frozen_threshold": float(above.mean()),
                "median_probability": float(values.median()),
                "q25_probability": float(values.quantile(0.25)),
                "q75_probability": float(values.quantile(0.75)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    return events, summary


def quality_recall(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tier, group in events.groupby("membership_tier", sort=True):
        for tolerance in [15, 45]:
            detected = group[f"detected_within_{tolerance}_sec"]
            rows.append(
                {
                    "model": "NESTED-SELECTED",
                    "membership_tier": tier,
                    "tolerance_sec": tolerance,
                    "reference_events": len(group),
                    "detected_events": int(detected.sum()),
                    "recall": float(detected.mean()),
                }
            )
    return pd.DataFrame(rows)


def historical_quality_recall(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    predictions = pd.read_csv(
        prior_result_dir() / "train_oof_predicted_events_v0.1.tsv", sep="\t"
    )
    support = pd.read_csv(
        prior_result_dir() / "train_oof_support_v0.1.tsv", sep="\t"
    )
    event_meta = data["references"][
        data["references"]["primary_analysis_eligible"]
    ][["subject", "pid", "event_time_sec", "membership_tier"]].copy()
    rows = []
    for model in ["LR-OOF", "LC-1"]:
        model_predictions = predictions[predictions["model"] == model]
        for tolerance in [15.0, 45.0]:
            matches, _ = evaluate(
                model_predictions, support, data["references"], tolerance
            )
            keys = matched_reference_keys(matches)
            local = event_meta.copy()
            local["detected"] = [
                (subject, float(event_time)) in keys
                for subject, event_time in zip(
                    local["subject"], local["event_time_sec"]
                )
            ]
            for tier, group in local.groupby("membership_tier", sort=True):
                rows.append(
                    {
                        "model": model,
                        "membership_tier": tier,
                        "tolerance_sec": int(tolerance),
                        "reference_events": len(group),
                        "detected_events": int(group["detected"].sum()),
                        "recall": float(group["detected"].mean()),
                    }
                )
    return pd.DataFrame(rows)


# Section 5: false-alarm context

def nearest_distance(time: float, candidates: np.ndarray) -> float:
    if len(candidates) == 0:
        return np.nan
    return float(np.min(np.abs(candidates - time)))


def false_alarm_context(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = data["predictions"].query("pipeline == 'NESTED-SELECTED'").copy()
    matches, summary = evaluate(
        predictions, data["support"], data["references"], tolerance=15.0
    )
    matched_prediction_keys = set(
        zip(
            matches["subject"].astype(str),
            matches["prediction_time_sec"].astype(float),
        )
    )
    false_alarms = predictions[
        ~predictions.apply(
            lambda row: (str(row["subject"]), float(row["event_time_sec"]))
            in matched_prediction_keys,
            axis=1,
        )
    ].copy()

    train_subjects = set(data["assignments"]["subject"])
    transitions = data["transitions"][
        data["transitions"]["subject"].isin(train_subjects)
    ].copy()
    primary, _ = primary_inputs(data["references"])
    quality_only = data["references"].query(
        "expanded_quality_analysis_eligible == True and primary_analysis_eligible == False"
    )
    wake_to_rem = transitions.query("transition_type == 'Wake_to_REM'")
    primary_groups = {
        subject: group["event_time_sec"].to_numpy(dtype=float)
        for subject, group in primary.groupby("subject")
    }
    quality_groups = {
        subject: group["event_time_sec"].to_numpy(dtype=float)
        for subject, group in quality_only.groupby("subject")
    }
    reverse_groups = {
        subject: group["nominal_boundary_sec"].to_numpy(dtype=float)
        for subject, group in wake_to_rem.groupby("subject")
    }
    false_alarms["nearest_primary_r2w_sec"] = false_alarms.apply(
        lambda row: nearest_distance(
            float(row.event_time_sec),
            primary_groups.get(row.subject, np.asarray([], dtype=float)),
        ),
        axis=1,
    )
    false_alarms["nearest_quality_only_r2w_sec"] = false_alarms.apply(
        lambda row: nearest_distance(
            float(row.event_time_sec),
            quality_groups.get(row.subject, np.asarray([], dtype=float)),
        ),
        axis=1,
    )
    false_alarms["nearest_wake_to_rem_sec"] = false_alarms.apply(
        lambda row: nearest_distance(
            float(row.event_time_sec),
            reverse_groups.get(row.subject, np.asarray([], dtype=float)),
        ),
        axis=1,
    )

    def category(row: pd.Series) -> str:
        if pd.notna(row.nearest_primary_r2w_sec) and row.nearest_primary_r2w_sec <= 45:
            return "primary_r2w_16_to_45sec"
        if (
            pd.notna(row.nearest_quality_only_r2w_sec)
            and row.nearest_quality_only_r2w_sec <= 45
        ):
            return "quality_only_r2w_within_45sec"
        if pd.notna(row.nearest_wake_to_rem_sec) and row.nearest_wake_to_rem_sec <= 45:
            return "wake_to_rem_within_45sec"
        return "not_near_labeled_transition"

    false_alarms["context_category"] = false_alarms.apply(category, axis=1)
    context = (
        false_alarms.groupby("context_category", as_index=False)
        .size()
        .rename(columns={"size": "false_alarms"})
    )
    context["fraction"] = context["false_alarms"] / len(false_alarms)
    if len(false_alarms) != int(summary["false_positive"]):
        raise ValueError("False-alarm reconstruction did not match the frozen result")
    return false_alarms, context


# Section 6: integrity, report, and output

def source_manifest() -> pd.DataFrame:
    paths = [
        source_score_path(),
        source_result_dir() / "outer_support_v0.1.tsv",
        source_result_dir() / "outer_predicted_events_v0.1.tsv",
        source_result_dir() / "outer_event_metrics_v0.1.tsv",
        source_result_dir() / "outer_architecture_rankings_v0.1.tsv",
        prior_result_dir() / "train_oof_predicted_events_v0.1.tsv",
        prior_result_dir() / "train_oof_support_v0.1.tsv",
        repo_root()
        / "labels"
        / "quality_analysis_membership_v0.1"
        / "transition_analysis_membership_v0.1.tsv",
        repo_root()
        / "labels"
        / "transition_labels_v0.1"
        / "transition_labels_v0.1.tsv",
    ]
    rows = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(repo_root().resolve())
            scope = "repository"
        except ValueError:
            relative = resolved.relative_to(data_parent().resolve())
            scope = "data_root"
        rows.append(
            {
                "path_scope": scope,
                "relative_path": relative.as_posix(),
                "bytes": resolved.stat().st_size,
                "sha256": sha256(resolved),
            }
        )
    return pd.DataFrame(rows)


def checks(
    data: dict[str, pd.DataFrame],
    threshold_summary: pd.DataFrame,
    boundary_events: pd.DataFrame,
    false_alarms: pd.DataFrame,
) -> pd.DataFrame:
    primary, _ = primary_inputs(data["references"])
    frozen = threshold_summary.query(
        "threshold_analysis == 'frozen_inner_selected'"
    ).iloc[0]
    rows = [
        ("train_recordings", len(data["support"]) == 82, str(len(data["support"]))),
        (
            "train_participants",
            data["support"]["pid"].nunique() == 64,
            str(data["support"]["pid"].nunique()),
        ),
        ("primary_events", len(primary) == 180, str(len(primary))),
        (
            "outer_scores_only",
            set(data["scores"]["phase"]) == {"outer_heldout"},
            ";".join(sorted(set(data["scores"]["phase"]))),
        ),
        (
            "five_frozen_thresholds",
            data["rankings"].query("inner_rank == 1")["outer_fold"].nunique() == 5,
            "one per outer fold",
        ),
        (
            "event_score_completeness",
            len(boundary_events) == 180,
            str(len(boundary_events)),
        ),
        (
            "frozen_metric_reconstruction",
            abs(float(frozen.f1) - 0.16452442159383032) <= 1e-12,
            f"f1={float(frozen.f1):.12f}",
        ),
        (
            "false_alarm_reconstruction",
            len(false_alarms) == 177,
            str(len(false_alarms)),
        ),
        (
            "threshold_grid_unchanged",
            len(THRESHOLDS) == 64,
            str(len(THRESHOLDS)),
        ),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in rows
        ]
    )


def build_readme(
    tolerance: pd.DataFrame,
    threshold: pd.DataFrame,
    quality: pd.DataFrame,
    context: pd.DataFrame,
    integrity: pd.DataFrame,
) -> str:
    frozen_15 = tolerance.query("tolerance_sec == 15").iloc[0]
    frozen_45 = tolerance.query("tolerance_sec == 45").iloc[0]
    pooled = threshold.query(
        "threshold_analysis == 'pooled_outer_label_oracle'"
    ).iloc[0]
    fold_oracle = threshold.query(
        "threshold_analysis == 'outer_fold_label_oracle'"
    ).iloc[0]
    quality_lines = [
        f"| {row.model} | {row.membership_tier} | {int(row.tolerance_sec)} | "
        f"{int(row.detected_events)}/{int(row.reference_events)} | {row.recall:.4f} |"
        for row in quality.itertuples(index=False)
    ]
    context_lines = [
        f"| {row.context_category} | {int(row.false_alarms)} | {row.fraction:.4f} |"
        for row in context.itertuples(index=False)
    ]
    return "\n".join(
        [
            "# Deep Temporal Error Diagnostic v0.1",
            "",
            "**Work date:** 2026-09-23",
            "**Source:** frozen nested train outer-fold results",
            "**Model fitting:** No",
            "**Validation accessed:** No",
            "**Test accessed:** No",
            "",
            "## Main Findings",
            "",
            f"Widening tolerance from +/-15 to +/-45 seconds increased true detections "
            f"from {int(frozen_15.true_positive)} to {int(frozen_45.true_positive)} and F1 "
            f"from {frozen_15.f1:.4f} to {frozen_45.f1:.4f}. Most missed events therefore "
            "cannot be explained by one adjacent 30-second boundary bin.",
            "",
            f"A pooled outer-label oracle threshold reached F1 {pooled.f1:.4f}; separate "
            f"outer-fold label oracles reached pooled F1 {fold_oracle.f1:.4f}. These are "
            "optimistic post-result bounds, not usable estimates or replacement thresholds. "
            "Even the stronger bound remained below 0.25.",
            "",
            "## Quality-Tier Recall",
            "",
            "| Model | Membership tier | Tolerance (s) | Detected | Recall |",
            "|---|---|---:|---:|---:|",
            *quality_lines,
            "",
            "## Primary False-Alarm Context",
            "",
            "| Context | False alarms | Fraction |",
            "|---|---:|---:|",
            *context_lines,
            "",
            "## Decision",
            "",
            "Threshold transfer contributes to the low frozen F1, but cannot explain most "
            "missed events. Timing uncertainty contributes a smaller component. The next "
            "model experiment should change the information or supervision mechanism rather "
            "than repeat architecture search on the same bandpower inputs. Candidate mechanisms "
            "are raw-waveform epoch embeddings and interval-aware boundary supervision. Any such "
            "experiment requires a separately committed protocol and a new evaluation boundary.",
            "",
            "This retrospective diagnostic cannot authorize model or threshold replacement.",
            "",
            "## Verification",
            "",
            f"All {int((integrity.status == 'pass').sum())}/{len(integrity)} integrity checks passed.",
            "",
        ]
    )


def main() -> None:
    data = load_sources()
    tolerance = tolerance_curve(data)
    threshold, fold_thresholds, _ = threshold_diagnostics(data)
    boundary_events, boundary_summary = boundary_scores(data)
    quality = pd.concat(
        [quality_recall(boundary_events), historical_quality_recall(data)],
        ignore_index=True,
    )
    false_alarms, context = false_alarm_context(data)
    manifest = source_manifest()
    integrity = checks(data, threshold, boundary_events, false_alarms)
    if not integrity["status"].eq("pass").all():
        print(integrity.to_string(index=False))
        raise SystemExit("Integrity check failed")

    threshold_output = threshold.astype(object).where(
        threshold.notna(), "not_applicable"
    )
    outputs = {
        "frozen_tolerance_curve_v0.1.tsv": tolerance,
        "threshold_upper_bounds_v0.1.tsv": threshold_output,
        "outer_threshold_gap_v0.1.tsv": fold_thresholds,
        "primary_event_boundary_scores_v0.1.tsv": boundary_events,
        "boundary_score_summary_v0.1.tsv": boundary_summary,
        "quality_tier_recall_v0.1.tsv": quality,
        "false_alarm_event_context_v0.1.tsv": false_alarms,
        "false_alarm_context_summary_v0.1.tsv": context,
        "source_manifest_v0.1.tsv": manifest,
        "output_integrity_checks_v0.1.tsv": integrity,
    }
    output_dir().mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    readme = build_readme(tolerance, threshold, quality, context, integrity)
    verify_or_create_text(output_dir() / "README.md", readme)

    print(tolerance.to_string(index=False))
    print(threshold.to_string(index=False))
    print(quality.to_string(index=False))
    print(context.to_string(index=False))
    print(integrity.to_string(index=False))


if __name__ == "__main__":
    main()
