"""Validate DE-D threshold robustness and LOPO calibration outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv

from analyze_direct_endpoint_threshold_robustness_v0_1 import (
    SELECTED_THRESHOLD,
    apply_lopo_thresholds,
    candidate_score_path,
    data_parent,
    factorization_dir,
    input_manifest,
    load_scores_and_support,
    make_decision,
    output_dir,
    paired_bootstrap_vs_de_b,
    repo_root,
    select_lopo_thresholds,
    sha256,
    threshold_success_intervals,
)
from run_direct_event_baseline_v0_1 import (
    collapse_alarms,
    evaluate_events,
    local_event_inputs,
    reference_events,
)


# Section 1: helpers

checks = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append(
        {"check": name, "status": "pass" if passed else "fail", "detail": detail}
    )
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def numeric_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    columns = left.select_dtypes(include=[np.number]).columns
    return bool(np.allclose(left[columns], right[columns], equal_nan=True))


def exact_alarm_match(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    keys = ["subject", "pid", "event_time_sec"]
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    if len(left) != len(right) or not left[keys].equals(right[keys]):
        return False
    numeric = ["probability", "calibration_threshold", "held_out_pid", "run_candidates"]
    return bool(np.allclose(left[numeric], right[numeric], equal_nan=True))


# Section 2: curve intervals and fold selection

def validate_threshold_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    saved_curve = pd.read_csv(
        output_dir() / "annotated_validation_threshold_curve_v0.1.tsv", sep="\t"
    )
    saved_intervals = pd.read_csv(
        output_dir() / "threshold_success_intervals_v0.1.tsv", sep="\t"
    )
    recomputed_curve, recomputed_intervals = threshold_success_intervals()
    record(
        "annotated_threshold_curve",
        len(saved_curve) == 99
        and list(saved_curve.columns) == list(recomputed_curve.columns)
        and numeric_equal(saved_curve, recomputed_curve)
        and saved_curve.select_dtypes(exclude=[np.number]).equals(
            recomputed_curve.select_dtypes(exclude=[np.number])
        ),
        f"rows={len(saved_curve)}",
    )
    record(
        "threshold_interval_recomputation",
        list(saved_intervals.columns) == list(recomputed_intervals.columns)
        and numeric_equal(saved_intervals, recomputed_intervals)
        and saved_intervals.select_dtypes(exclude=[np.number]).equals(
            recomputed_intervals.select_dtypes(exclude=[np.number])
        ),
        f"intervals={len(saved_intervals)}",
    )
    selected_interval = saved_intervals[saved_intervals["contains_selected_threshold"]]
    record(
        "selected_threshold_interval",
        len(selected_interval) == 1
        and float(selected_interval.iloc[0].threshold_start) <= SELECTED_THRESHOLD
        and float(selected_interval.iloc[0].threshold_end) >= SELECTED_THRESHOLD
        and int(selected_interval.iloc[0].threshold_count) >= 5,
        selected_interval.to_dict("records").__str__(),
    )

    participant_curve = pd.read_csv(
        output_dir() / "participant_threshold_metrics_v0.1.tsv", sep="\t"
    )
    record(
        "participant_threshold_grid",
        len(participant_curve) == 1584
        and participant_curve["threshold"].nunique() == 99
        and participant_curve["pid"].nunique() == 16
        and bool((participant_curve.groupby("threshold")["pid"].nunique() == 16).all()),
        f"rows={len(participant_curve)}",
    )
    selected = pd.read_csv(output_dir() / "lopo_selected_thresholds_v0.1.tsv", sep="\t")
    recomputed_selected = select_lopo_thresholds(participant_curve)
    record(
        "lopo_threshold_selection",
        len(selected) == 16
        and numeric_equal(selected, recomputed_selected)
        and list(selected.columns) == list(recomputed_selected.columns),
        f"folds={len(selected)}, unique_thresholds={selected['threshold'].nunique()}",
    )
    record(
        "lopo_calibration_excludes_one_pid",
        set(selected["calibration_pid"]) == {15},
        f"calibration_pid={sorted(selected['calibration_pid'].unique())}",
    )
    return saved_intervals, participant_curve, selected


# Section 3: held-out alarms and metrics

def validate_lopo_outputs(selected: pd.DataFrame) -> pd.DataFrame:
    scores, support = load_scores_and_support()
    recomputed_alarms = apply_lopo_thresholds(scores, selected)
    saved_alarms = pd.read_csv(output_dir() / "lopo_predicted_events_v0.1.tsv", sep="\t")
    record(
        "lopo_alarm_recomputation",
        exact_alarm_match(saved_alarms, recomputed_alarms),
        f"saved={len(saved_alarms)}, recomputed={len(recomputed_alarms)}",
    )

    metrics = pd.read_csv(output_dir() / "lopo_event_metrics_v0.1.tsv", sep="\t")
    references = reference_events()
    predictions = saved_alarms[["subject", "pid", "event_time_sec"]]
    local_support = support[["subject", "pid", "supported_hours"]]
    metric_columns = [
        "recordings",
        "pid",
        "reference_events",
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
        "median_absolute_error_sec",
        "maximum_absolute_error_sec",
    ]
    failures = []
    for item in metrics.itertuples(index=False):
        eligible, ignored = local_event_inputs(references, "validation", item.membership)
        _, _, _, summary = evaluate_events(
            eligible, predictions, ignored, local_support, float(item.tolerance_sec)
        )
        for column in metric_columns:
            left = getattr(item, column)
            right = summary[column]
            if pd.isna(left) and pd.isna(right):
                continue
            if not np.isclose(float(left), float(right), atol=1e-12, rtol=0.0):
                failures.append(f"{item.membership}/{item.tolerance_sec}/{column}")
    record(
        "lopo_event_metric_recomputation",
        not failures,
        f"configurations={len(metrics)}, failures={len(failures)}",
    )

    saved_participants = pd.read_csv(
        output_dir() / "lopo_event_participants_v0.1.tsv", sep="\t"
    )
    saved_paired = pd.read_csv(
        output_dir() / "lopo_vs_de_b_paired_bootstrap_v0.1.tsv", sep="\t"
    )
    recomputed_paired = paired_bootstrap_vs_de_b(saved_participants)
    record(
        "paired_bootstrap_recomputation",
        saved_paired[["comparison", "metric"]].equals(
            recomputed_paired[["comparison", "metric"]]
        )
        and numeric_equal(saved_paired, recomputed_paired),
        f"rows={len(saved_paired)}",
    )
    return metrics


# Section 4: decision and input integrity

def validate_decision(intervals: pd.DataFrame, metrics: pd.DataFrame) -> None:
    saved = pd.read_csv(
        output_dir() / "threshold_robustness_decision_v0.1.tsv", sep="\t"
    )
    recomputed = make_decision(intervals, metrics)
    record(
        "robustness_decision_recomputation",
        list(saved.columns) == list(recomputed.columns)
        and numeric_equal(saved, recomputed)
        and saved.select_dtypes(exclude=[np.number]).equals(
            recomputed.select_dtypes(exclude=[np.number])
        ),
        f"supported={bool(saved.iloc[0].threshold_robustness_supported)}",
    )


def validate_inputs() -> None:
    saved = pd.read_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t")
    expected = input_manifest()
    failures = []
    for item in saved.itertuples(index=False):
        root = repo_root() if item.path_base == "repo" else data_parent()
        path = root / item.relative_path
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(item.relative_path)
    record(
        "input_artifact_hashes",
        len(saved) == 8 and not failures and saved.equals(expected),
        f"verified={len(saved)}, failures={len(failures)}",
    )
    forbidden_roles = saved["artifact_role"].str.contains(
        "test|model|feature|train", case=False, regex=True
    )
    record(
        "no_test_model_feature_or_train_input",
        not bool(forbidden_roles.any()),
        f"forbidden_roles={int(forbidden_roles.sum())}",
    )
    forbidden_files = list(repo_root().rglob("*.joblib")) + list(repo_root().rglob("*.npz"))
    record(
        "no_model_or_array_files_in_repo",
        len(forbidden_files) == 0,
        f"forbidden_files={len(forbidden_files)}",
    )


# Section 5: execute

def main() -> None:
    intervals, _, selected = validate_threshold_outputs()
    metrics = validate_lopo_outputs(selected)
    validate_decision(intervals, metrics)
    validate_inputs()
    result = pd.DataFrame(checks)
    verify_or_create_tsv(result, output_dir() / "output_integrity_checks_v0.1.tsv")
    print(result.to_string(index=False))
    print(f"Passed {int((result['status'] == 'pass').sum())}/{len(result)} checks")


if __name__ == "__main__":
    main()
