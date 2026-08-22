"""Run validation-only factorized endpoint experiment DE-D."""

from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, roc_auc_score

from analyze_direct_event_failure_modes_v0_1 import add_human_stage_pair
from run_direct_event_baseline_v0_1 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MEMBERSHIPS,
    THRESHOLDS,
    TOLERANCES,
    build_labeled_matrix,
    build_model,
    collapse_alarms,
    data_parent,
    evaluate_events,
    feature_path,
    labeled_candidates,
    local_event_inputs,
    participant_bootstrap,
    recording_candidate_matrix,
    reference_events,
    repo_root,
    sha256,
    subject_assignments,
)


# Section 1: frozen configuration and paths

VERSION = "v0.1"
COMPARATOR = "DE-D"
EXPERIMENT_DIR = "2026-08-22_direct_endpoint_factorization_v0.1"
DERIVED_DIR = "direct_endpoint_factorization_v0.1"
HEADS = {
    "rem_before": "human stage at t-30 seconds is REM",
    "wake_after": "human stage at t is Wake",
}


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def model_path(head: str) -> Path:
    return derived_dir() / "models" / f"de_d_{head}_model_v0.1.joblib"


def candidate_score_path() -> Path:
    return derived_dir() / "candidate_scores" / "validation_candidate_scores_v0.1.tsv.gz"


def stage_lookup(subject: str) -> dict[int, int]:
    with np.load(feature_path(subject), allow_pickle=False) as values:
        onset = values["onset"].astype(float)
        stage = values["stage"].astype(int)
    return {
        int(round(time * 1000.0)): int(code) for time, code in zip(onset, stage)
    }


# Section 2: endpoint target construction

def add_endpoint_targets(rows: pd.DataFrame) -> pd.DataFrame:
    target_rows = []
    for subject, group in rows.groupby("subject", sort=True):
        lookup = stage_lookup(subject)
        for item in group.itertuples(index=False):
            key = int(round(float(item.candidate_time_sec) * 1000.0))
            previous = lookup.get(key - 30000)
            current = lookup.get(key)
            if previous is None or current is None:
                raise ValueError(
                    f"Missing endpoint stages for {subject} at {item.candidate_time_sec}"
                )
            target_rows.append(
                {
                    **item._asdict(),
                    "human_stage_from": previous,
                    "human_stage_to": current,
                    "rem_before": int(previous == 4),
                    "wake_after": int(current == 0),
                }
            )
    result = pd.DataFrame(target_rows)
    reconstructed = result["rem_before"] & result["wake_after"]
    if not np.array_equal(reconstructed.astype(int), result["label"].astype(int)):
        raise ValueError("Endpoint conjunction does not reproduce the direct event label")
    return result


# Section 3: fit two frozen endpoint heads

def fit_endpoint_heads(
    rows: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_rows = rows[rows["partition"] == "train"].copy()
    validation_rows = rows[rows["partition"] == "validation"].copy()
    train_x, train_meta, train_dropped = build_labeled_matrix(train_rows, "DE-B")
    validation_x, validation_meta, validation_dropped = build_labeled_matrix(
        validation_rows, "DE-B"
    )
    if len(train_dropped) or len(validation_dropped):
        raise ValueError(
            f"Unexpected context drops: train={len(train_dropped)}, validation={len(validation_dropped)}"
        )

    models = {}
    fit_rows = []
    metric_rows = []
    scored_rows = []
    for head in HEADS:
        train_y = train_meta[head].to_numpy(dtype=int)
        model = build_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_x, train_y)
        convergence = [
            item for item in caught if issubclass(item.category, ConvergenceWarning)
        ]
        model_path(head).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path(head))
        models[head] = model
        fit_rows.append(
            {
                "comparator": COMPARATOR,
                "model_version": VERSION,
                "head": head,
                "target_definition": HEADS[head],
                "train_rows": len(train_meta),
                "train_positive": int(train_y.sum()),
                "train_negative": int((train_y == 0).sum()),
                "input_features": train_x.shape[1],
                "maximum_iterations_used": int(
                    model.named_steps["logisticregression"].n_iter_.max()
                ),
                "convergence_warning_count": len(convergence),
                "fit_decision": "retain_frozen_fit",
            }
        )
        for partition, matrix, metadata in [
            ("train", train_x, train_meta),
            ("validation", validation_x, validation_meta),
        ]:
            probability = model.predict_proba(matrix)[:, 1]
            target = metadata[head].to_numpy(dtype=int)
            metric_rows.append(
                {
                    "comparator": COMPARATOR,
                    "model_version": VERSION,
                    "head": head,
                    "partition": partition,
                    "rows": len(metadata),
                    "positive_rows": int(target.sum()),
                    "negative_rows": int((target == 0).sum()),
                    "average_precision": average_precision_score(target, probability),
                    "roc_auc": roc_auc_score(target, probability),
                }
            )
            local = metadata[
                [
                    "sample_id",
                    "subject",
                    "pid",
                    "partition",
                    "candidate_time_sec",
                    "label",
                    "source_tier",
                    "human_stage_from",
                    "human_stage_to",
                    "rem_before",
                    "wake_after",
                ]
            ].copy()
            local["head"] = head
            local["head_probability"] = probability
            scored_rows.append(local)
    return (
        models,
        pd.DataFrame(fit_rows),
        pd.DataFrame(metric_rows),
        pd.concat(scored_rows, ignore_index=True),
    )


# Section 4: score validation nights without test access

def score_validation(
    assignments: pd.DataFrame, models: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        times, matrix = recording_candidate_matrix(item.subject, "DE-B")
        rem_probability = models["rem_before"].predict_proba(matrix)[:, 1]
        wake_probability = models["wake_after"].predict_proba(matrix)[:, 1]
        probability = rem_probability * wake_probability
        score_rows.append(
            pd.DataFrame(
                {
                    "comparator": COMPARATOR,
                    "model_version": VERSION,
                    "partition": item.partition,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": times,
                    "probability_rem_before": rem_probability,
                    "probability_wake_after": wake_probability,
                    "probability": probability,
                }
            )
        )
        support_rows.append(
            {
                "comparator": COMPARATOR,
                "model_version": VERSION,
                "partition": item.partition,
                "subject": item.subject,
                "pid": int(item.pid),
                "supported_boundaries": len(times),
                "supported_hours": len(times) * 30.0 / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def save_candidate_scores(scores: pd.DataFrame) -> None:
    path = candidate_score_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(
        path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )


# Section 5: threshold selection and frozen validation evaluation

def threshold_curve(
    scores: pd.DataFrame, support: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = reference_events()
    eligible, ignored = local_event_inputs(references, "validation", "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    rows = []
    for threshold in THRESHOLDS:
        alarms = collapse_alarms(scores, threshold)
        _, _, _, summary = evaluate_events(
            eligible,
            alarms[["subject", "pid", "event_time_sec"]],
            ignored,
            local_support,
            15.0,
        )
        rows.append(
            {
                "comparator": COMPARATOR,
                "model_version": VERSION,
                "partition": "validation",
                "membership": "primary",
                "tolerance_sec": 15.0,
                "threshold": threshold,
                **summary,
            }
        )
    curve = pd.DataFrame(rows)
    selected = curve.sort_values(
        ["f1", "false_alarms_per_hour", "recall", "threshold"],
        ascending=[False, True, False, False],
        kind="stable",
    ).iloc[[0]].copy()
    selected["rem_before_model_sha256"] = sha256(model_path("rem_before"))
    selected["wake_after_model_sha256"] = sha256(model_path("wake_after"))
    selected[
        "selection_rule"
    ] = "max_f1_then_min_far_then_max_recall_then_max_threshold"
    return curve, selected


def evaluate_selected(
    scores: pd.DataFrame, support: pd.DataFrame, selected: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    threshold = float(selected.iloc[0]["threshold"])
    alarms = collapse_alarms(scores, threshold)
    references = reference_events()
    local_support = support[["subject", "pid", "supported_hours"]]
    summaries = []
    bootstraps = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for membership in MEMBERSHIPS:
        eligible, ignored = local_event_inputs(references, "validation", membership)
        for tolerance in TOLERANCES:
            recordings, participants, matches, summary = evaluate_events(
                eligible,
                alarms[["subject", "pid", "event_time_sec"]],
                ignored,
                local_support,
                tolerance,
            )
            config = {
                "comparator": COMPARATOR,
                "model_version": VERSION,
                "partition": "validation",
                "membership": membership,
                "tolerance_sec": tolerance,
                "threshold": threshold,
            }
            summaries.append({**config, **summary})
            bootstrap = participant_bootstrap(
                participants,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            for key, value in reversed(list(config.items())):
                bootstrap.insert(0, key, value)
            bootstraps.append(bootstrap)
            for frame, collection in [
                (recordings, recordings_all),
                (participants, participants_all),
                (matches, matches_all),
            ]:
                if len(frame):
                    frame = frame.copy()
                    for key, value in reversed(list(config.items())):
                        if key not in frame.columns:
                            frame.insert(0, key, value)
                    collection.append(frame)
    return {
        "predicted_events": alarms,
        "event_metrics": pd.DataFrame(summaries),
        "event_bootstrap": pd.concat(bootstraps, ignore_index=True),
        "event_recordings": pd.concat(recordings_all, ignore_index=True),
        "event_participants": pd.concat(participants_all, ignore_index=True),
        "event_matches": pd.concat(matches_all, ignore_index=True),
    }


# Section 6: false-positive categories and baseline comparison

def false_positive_rows(
    alarms: pd.DataFrame, matches: pd.DataFrame, expected: int
) -> pd.DataFrame:
    local_matches = matches[
        (matches["membership"] == "primary")
        & (matches["tolerance_sec"] == 15.0)
    ]
    matched = {
        (item.subject, round(float(item.prediction_time_sec), 6))
        for item in local_matches.itertuples(index=False)
    }
    rows = alarms[
        [
            (item.subject, round(float(item.event_time_sec), 6)) not in matched
            for item in alarms.itertuples(index=False)
        ]
    ].copy()
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} false positives, found {len(rows)}")
    return add_human_stage_pair(rows)


def category_summary(rows: pd.DataFrame, comparator: str) -> pd.DataFrame:
    summary = (
        rows.groupby("human_stage_pair_category")
        .size()
        .rename("false_positive_alarms")
        .reset_index()
    )
    summary["false_positive_share"] = summary["false_positive_alarms"] / len(rows)
    summary.insert(0, "comparator", comparator)
    return summary


def compare_with_de_b(
    de_d_outputs: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dir = (
        repo_root() / "experiments/2026-08-22_direct_event_baseline_v0.1"
    )
    baseline_metrics = pd.read_csv(
        baseline_dir / "train_validation_event_metrics_v0.1.tsv", sep="\t"
    )
    de_b = baseline_metrics[
        (baseline_metrics["comparator"] == "DE-B")
        & (baseline_metrics["partition"] == "validation")
        & (baseline_metrics["membership"] == "primary")
        & (baseline_metrics["tolerance_sec"] == 15.0)
    ].iloc[0]
    de_d = de_d_outputs["event_metrics"]
    de_d = de_d[
        (de_d["membership"] == "primary") & (de_d["tolerance_sec"] == 15.0)
    ].iloc[0]
    comparison = pd.DataFrame(
        [
            {
                "comparator": "DE-B",
                "role": "frozen_binary_direct_baseline",
                "threshold": de_b.threshold,
                "true_positive": de_b.true_positive,
                "false_positive": de_b.false_positive,
                "false_negative": de_b.false_negative,
                "precision": de_b.precision,
                "recall": de_b.recall,
                "f1": de_b.f1,
                "false_alarms_per_hour": de_b.false_alarms_per_hour,
            },
            {
                "comparator": COMPARATOR,
                "role": "factorized_endpoint_validation_only",
                "threshold": de_d.threshold,
                "true_positive": de_d.true_positive,
                "false_positive": de_d.false_positive,
                "false_negative": de_d.false_negative,
                "precision": de_d.precision,
                "recall": de_d.recall,
                "f1": de_d.f1,
                "false_alarms_per_hour": de_d.false_alarms_per_hour,
            },
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "comparison": "DE-D_vs_DE-B_validation",
                "event_f1_difference": float(de_d.f1 - de_b.f1),
                "false_alarms_per_hour_difference": float(
                    de_d.false_alarms_per_hour - de_b.false_alarms_per_hour
                ),
                "f1_improved": bool(de_d.f1 > de_b.f1),
                "false_alarm_rate_reduced": bool(
                    de_d.false_alarms_per_hour < de_b.false_alarms_per_hour
                ),
                "prespecified_success_rule_met": bool(
                    (de_d.f1 > de_b.f1)
                    and (de_d.false_alarms_per_hour < de_b.false_alarms_per_hour)
                ),
            }
        ]
    )
    return comparison, decision


def false_positive_comparison(
    de_d_outputs: dict[str, pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    de_d_metrics = de_d_outputs["event_metrics"]
    de_d_metrics = de_d_metrics[
        (de_d_metrics["membership"] == "primary")
        & (de_d_metrics["tolerance_sec"] == 15.0)
    ].iloc[0]
    de_d_rows = false_positive_rows(
        de_d_outputs["predicted_events"],
        de_d_outputs["event_matches"],
        int(de_d_metrics.false_positive),
    )

    baseline_dir = (
        repo_root() / "experiments/2026-08-22_direct_event_baseline_v0.1"
    )
    de_b_alarms = pd.read_csv(
        baseline_dir / "train_validation_predicted_events_v0.1.tsv", sep="\t"
    )
    de_b_alarms = de_b_alarms[
        (de_b_alarms["comparator"] == "DE-B")
        & (de_b_alarms["partition"] == "validation")
    ].copy()
    de_b_matches = pd.read_csv(
        baseline_dir / "train_validation_event_matches_v0.1.tsv", sep="\t"
    )
    de_b_matches = de_b_matches[
        (de_b_matches["comparator"] == "DE-B")
        & (de_b_matches["partition"] == "validation")
    ].copy()
    baseline_metrics = pd.read_csv(
        baseline_dir / "train_validation_event_metrics_v0.1.tsv", sep="\t"
    )
    expected_de_b = int(
        baseline_metrics[
            (baseline_metrics["comparator"] == "DE-B")
            & (baseline_metrics["partition"] == "validation")
            & (baseline_metrics["membership"] == "primary")
            & (baseline_metrics["tolerance_sec"] == 15.0)
        ]["false_positive"].iloc[0]
    )
    de_b_rows = false_positive_rows(de_b_alarms, de_b_matches, expected_de_b)
    summary = pd.concat(
        [
            category_summary(de_b_rows, "DE-B"),
            category_summary(de_d_rows, COMPARATOR),
        ],
        ignore_index=True,
    )
    return de_d_rows, summary


# Section 7: artifact records and result narrative

def artifact_manifest(assignments: pd.DataFrame) -> pd.DataFrame:
    paths = [("frozen_recording_features", feature_path(subject)) for subject in assignments["subject"]]
    paths.extend(("fitted_endpoint_model", model_path(head)) for head in HEADS)
    paths.append(("validation_candidate_scores", candidate_score_path()))
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
    return pd.DataFrame(rows).sort_values(
        ["artifact_role", "path_relative_to_data_parent"]
    )


def input_manifest() -> pd.DataFrame:
    paths = [
        repo_root() / "docs/evaluation/direct_endpoint_factorization_protocol_v0.1.md",
        repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        repo_root()
        / "labels/quality_analysis_membership_v0.1/background_analysis_membership_v0.1.tsv",
        repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        repo_root()
        / "labels/background_windows_v0.1/background_review_windows_v0.1.tsv",
        repo_root()
        / "experiments/2026-08-22_direct_event_baseline_v0.1/train_validation_event_metrics_v0.1.tsv",
        repo_root()
        / "experiments/2026-08-22_direct_event_baseline_v0.1/train_validation_predicted_events_v0.1.tsv",
        repo_root()
        / "experiments/2026-08-22_direct_event_baseline_v0.1/train_validation_event_matches_v0.1.tsv",
    ]
    return pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
    )


def write_result(
    fit: pd.DataFrame,
    metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    decision: pd.DataFrame,
    fp_summary: pd.DataFrame,
) -> None:
    threshold = selected.iloc[0]
    decision_row = decision.iloc[0]
    de_d = comparison[comparison["comparator"] == COMPARATOR].iloc[0]
    de_b = comparison[comparison["comparator"] == "DE-B"].iloc[0]
    partial_categories = ["human_REM_to_other", "human_other_to_Wake"]
    partial = fp_summary[fp_summary["human_stage_pair_category"].isin(partial_categories)]
    partial_by_model = partial.groupby("comparator")["false_positive_alarms"].sum()
    fit_lines = []
    for item in fit.itertuples(index=False):
        fit_lines.append(
            f"| {item.head} | {item.train_positive} | {item.train_negative} | "
            f"{item.maximum_iterations_used} | {item.convergence_warning_count} |"
        )
    metric_lines = []
    for item in metrics[metrics["partition"] == "validation"].itertuples(index=False):
        metric_lines.append(
            f"| {item.head} | {item.average_precision:.4f} | {item.roc_auc:.4f} |"
        )
    text = "\n".join(
        [
            "# Direct Endpoint Factorization v0.1",
            "",
            "**Created:** 2026-08-22",
            "**Status:** Sequential exploratory validation-only experiment",
            "**Protocol:** `docs/evaluation/direct_endpoint_factorization_protocol_v0.1.md`",
            "**Test access:** None",
            "",
            "## Fit Record",
            "",
            "| Head | Train positive | Train negative | Iterations | Convergence warnings |",
            "|---|---:|---:|---:|---:|",
            *fit_lines,
            "",
            "## Endpoint Discrimination",
            "",
            "| Head | Validation average precision | Validation ROC AUC |",
            "|---|---:|---:|",
            *metric_lines,
            "",
            "## Validation Event Comparison",
            "",
            "| Model | Threshold | Precision | Recall | Event F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|---:|",
            f"| DE-B | {de_b.threshold:.2f} | {de_b.precision:.4f} | {de_b.recall:.4f} | {de_b.f1:.4f} | {de_b.false_alarms_per_hour:.4f} |",
            f"| DE-D | {de_d.threshold:.2f} | {de_d.precision:.4f} | {de_d.recall:.4f} | {de_d.f1:.4f} | {de_d.false_alarms_per_hour:.4f} |",
            "",
            f"DE-D minus DE-B event F1 was {decision_row.event_f1_difference:+.4f}; false alarms per hour changed by {decision_row.false_alarms_per_hour_difference:+.4f}. The prespecified two-part success rule was met: **{decision_row.prespecified_success_rule_met}**.",
            "",
            "## Partial-Endpoint Errors",
            "",
            f"DE-B produced {int(partial_by_model.get('DE-B', 0))} validation false positives at REM-to-other or other-to-Wake boundaries. DE-D produced {int(partial_by_model.get(COMPARATOR, 0))}. This comparison is descriptive because the method was designed after the earlier test failure was known.",
            "",
            "## Decision Boundary",
            "",
            "This experiment does not alter the frozen Block 6 test result. Even if validation improves, DE-D remains a candidate for a new locked or external evaluation; it must not be applied to the current test partition for iterative selection.",
            "",
            "Fitted endpoint models and continuous validation scores remain outside Git. Their hashes are recorded in `external_artifact_manifest_v0.1.tsv`.",
            "",
        ]
    )
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


# Section 8: execute validation-only experiment

def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    derived_dir().mkdir(parents=True, exist_ok=True)
    assignments = subject_assignments()
    active = assignments[assignments["partition"].isin(["train", "validation"])].copy()
    rows = labeled_candidates()
    rows = rows[rows["partition"].isin(["train", "validation"])].copy()
    rows = add_endpoint_targets(rows)

    models, fit, endpoint_metrics, labeled_scores = fit_endpoint_heads(rows)
    validation_assignments = active[active["partition"] == "validation"].copy()
    continuous_scores, support = score_validation(validation_assignments, models)
    save_candidate_scores(continuous_scores)
    curve, selected = threshold_curve(continuous_scores, support)
    outputs = evaluate_selected(continuous_scores, support, selected)
    comparison, decision = compare_with_de_b(outputs)
    false_positive, fp_summary = false_positive_comparison(outputs)

    fit.to_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t", index=False)
    endpoint_metrics.to_csv(
        output_dir() / "endpoint_head_metrics_v0.1.tsv", sep="\t", index=False
    )
    labeled_scores.to_csv(
        output_dir() / "train_validation_labeled_endpoint_scores_v0.1.tsv",
        sep="\t",
        index=False,
    )
    support.to_csv(output_dir() / "validation_event_support_v0.1.tsv", sep="\t", index=False)
    curve.to_csv(output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t", index=False)
    selected.to_csv(output_dir() / "selected_threshold_v0.1.tsv", sep="\t", index=False)
    for name, frame in outputs.items():
        frame.to_csv(output_dir() / f"validation_{name}_v0.1.tsv", sep="\t", index=False)
    comparison.to_csv(
        output_dir() / "validation_comparison_with_de_b_v0.1.tsv", sep="\t", index=False
    )
    decision.to_csv(output_dir() / "validation_decision_v0.1.tsv", sep="\t", index=False)
    false_positive.to_csv(
        output_dir() / "validation_false_positive_context_v0.1.tsv", sep="\t", index=False
    )
    fp_summary.to_csv(
        output_dir() / "validation_false_positive_category_comparison_v0.1.tsv",
        sep="\t",
        index=False,
    )
    artifact_manifest(active).to_csv(
        output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    input_manifest().to_csv(
        output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t", index=False
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
    write_result(fit, endpoint_metrics, selected, comparison, decision, fp_summary)
    print(comparison.to_string(index=False))
    print(decision.to_string(index=False))
    print(fp_summary.to_string(index=False))


if __name__ == "__main__":
    main()
