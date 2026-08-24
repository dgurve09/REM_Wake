"""Validate the DE-D endpoint contribution analysis outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv

from run_direct_event_baseline_v0_1 import (
    collapse_alarms,
    data_parent,
    evaluate_events,
    local_event_inputs,
    reference_events,
    repo_root,
)


# Section 1: fixed paths and comparator scores

EXPERIMENT_DIR = "2026-08-22_direct_endpoint_contribution_analysis_v0.1"
FACTORIZATION_DIR = "2026-08-22_direct_endpoint_factorization_v0.1"
SCORE_COLUMNS = {
    "DE-D-rem-only": "probability_rem_before",
    "DE-D-wake-only": "probability_wake_after",
    "DE-D-product": "probability",
}


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def factorization_dir() -> Path:
    return repo_root() / "experiments" / FACTORIZATION_DIR


def candidate_score_path() -> Path:
    return (
        data_parent()
        / "derived/direct_endpoint_factorization_v0.1/candidate_scores"
        / "validation_candidate_scores_v0.1.tsv.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left: float, right: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=0))


# Section 2: independent checks

def validate_manifest(manifest: pd.DataFrame) -> tuple[bool, str]:
    failures = []
    for item in manifest.itertuples(index=False):
        base = repo_root() if item.scope == "repository" else data_parent()
        path = base / item.relative_path
        if not path.exists():
            failures.append(f"missing:{item.relative_path}")
        elif path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(f"mismatch:{item.relative_path}")
    return not failures, f"verified={len(manifest)}, failures={len(failures)}"


def recompute_selected_metrics(
    scores: pd.DataFrame, support: pd.DataFrame, selected: pd.DataFrame
) -> tuple[bool, str]:
    references = reference_events()
    eligible, ignored = local_event_inputs(references, "validation", "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    failures = []
    exact = [
        "recordings",
        "pid",
        "reference_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "false_negative",
        "ignored_predictions",
    ]
    numeric = [
        "supported_hours",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
        "median_absolute_error_sec",
    ]

    for item in selected.itertuples(index=False):
        local = scores[["partition", "subject", "pid", "candidate_time_sec"]].copy()
        local.insert(0, "comparator", item.comparator)
        local["probability"] = scores[SCORE_COLUMNS[item.comparator]].to_numpy(dtype=float)
        alarms = collapse_alarms(local, float(item.threshold))
        _, _, _, summary = evaluate_events(
            eligible,
            alarms[["subject", "pid", "event_time_sec"]],
            ignored,
            local_support,
            15.0,
        )
        for column in exact:
            if int(summary[column]) != int(getattr(item, column)):
                failures.append(f"{item.comparator}:{column}")
        for column in numeric:
            if not close(summary[column], getattr(item, column)):
                failures.append(f"{item.comparator}:{column}")
    return not failures, f"comparators={len(selected)}, failures={len(failures)}"


def validate_threshold_selection(
    curve: pd.DataFrame, selected: pd.DataFrame
) -> tuple[bool, str]:
    failures = []
    for comparator in SCORE_COLUMNS:
        local = curve[curve["comparator"] == comparator]
        expected = local.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        observed = selected[selected["comparator"] == comparator].iloc[0]
        for column in ["threshold", "f1", "false_alarms_per_hour", "recall"]:
            if not close(expected[column], observed[column]):
                failures.append(f"{comparator}:{column}")
    return not failures, f"comparators={len(SCORE_COLUMNS)}, failures={len(failures)}"


def validate_product_control(selected: pd.DataFrame) -> tuple[bool, str]:
    product = selected[selected["comparator"] == "DE-D-product"].iloc[0]
    saved = pd.read_csv(
        factorization_dir() / "selected_threshold_v0.1.tsv", sep="\t"
    ).iloc[0]
    columns = [
        "threshold",
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
    ]
    failures = [column for column in columns if not close(product[column], saved[column])]
    return not failures, f"checked={len(columns)}, failures={len(failures)}"


def validate_decision(
    selected: pd.DataFrame, categories: pd.DataFrame, decision: pd.DataFrame
) -> tuple[bool, str]:
    metrics = selected.set_index("comparator")
    product = metrics.loc["DE-D-product"]
    single = metrics.loc[["DE-D-rem-only", "DE-D-wake-only"]]
    higher_f1 = bool((float(product.f1) > single["f1"].astype(float)).all())
    lower_far = bool(
        (
            float(product.false_alarms_per_hour)
            < single["false_alarms_per_hour"].astype(float)
        ).all()
    )
    dominating = single[
        (single["f1"].astype(float) >= float(product.f1))
        & (
            single["false_alarms_per_hour"].astype(float)
            <= float(product.false_alarms_per_hour)
        )
    ].index.tolist()
    expected = (
        "both_endpoint_contribution_supported"
        if higher_f1 and lower_far
        else "explicit_conjunction_not_supported"
        if dominating
        else "endpoint_contribution_inconclusive"
    )
    observed = decision.iloc[0]
    category_counts = categories.groupby("comparator")["false_positive_alarms"].sum()
    category_ok = all(
        int(category_counts[comparator]) == int(metrics.loc[comparator, "false_positive"])
        for comparator in SCORE_COLUMNS
    )
    passed = (
        str(observed.decision) == expected
        and str(observed.product_higher_f1_than_both_heads).lower() == str(higher_f1).lower()
        and str(observed.product_lower_far_than_both_heads).lower() == str(lower_far).lower()
        and category_ok
    )
    return passed, f"decision={expected}, category_accounting={category_ok}"


# Section 3: run all checks

def main() -> None:
    curve = pd.read_csv(output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t")
    selected = pd.read_csv(output_dir() / "selected_event_metrics_v0.1.tsv", sep="\t")
    categories = pd.read_csv(output_dir() / "false_positive_category_summary_v0.1.tsv", sep="\t")
    decision = pd.read_csv(output_dir() / "mechanism_decision_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t")
    scores = pd.read_csv(candidate_score_path(), sep="\t", compression="gzip")
    support = pd.read_csv(factorization_dir() / "validation_event_support_v0.1.tsv", sep="\t")

    product = scores["probability_rem_before"].to_numpy(dtype=float) * scores[
        "probability_wake_after"
    ].to_numpy(dtype=float)
    manifest_ok, manifest_detail = validate_manifest(manifest)
    metric_ok, metric_detail = recompute_selected_metrics(scores, support, selected)
    selection_ok, selection_detail = validate_threshold_selection(curve, selected)
    control_ok, control_detail = validate_product_control(selected)
    decision_ok, decision_detail = validate_decision(selected, categories, decision)
    result_file = "independent_validation_checks_v0.1.tsv"
    expected_files = {
        "README.md",
        "access_check_wording_correction_v0.1.md",
        "false_positive_category_summary_v0.1.tsv",
        "input_artifact_manifest_v0.1.tsv",
        "mechanism_decision_v0.1.tsv",
        "output_integrity_checks_v0.1.tsv",
        "selected_event_metrics_v0.1.tsv",
        "validation_threshold_curve_v0.1.tsv",
    }
    actual_files = {path.name for path in output_dir().iterdir() if path.is_file()}
    output_set_ok = actual_files in [expected_files, expected_files | {result_file}]

    checks = [
        ("candidate_partition_is_validation", set(scores["partition"]) == {"validation"}, f"rows={len(scores)}"),
        ("saved_product_recomputed", bool(np.allclose(product, scores["probability"], atol=1e-12, rtol=0)), "two endpoint scores multiplied"),
        ("threshold_grid_complete", len(curve) == 3 * 99 and curve.groupby("comparator")["threshold"].nunique().eq(99).all(), f"rows={len(curve)}"),
        ("selected_threshold_rule", selection_ok, selection_detail),
        ("selected_metrics_recomputed", metric_ok, metric_detail),
        ("product_control_reproduced", control_ok, control_detail),
        ("mechanism_decision_recomputed", decision_ok, decision_detail),
        ("input_artifact_hashes", manifest_ok, manifest_detail),
        (
            "no_test_score_model_train_score_or_raw_input",
            not manifest["artifact_role"].str.contains(
                "test.*score|model|train.*score|raw", case=False, regex=True
            ).any(),
            f"input_rows={len(manifest)}; shared membership filtered to validation",
        ),
        ("output_file_set", output_set_ok, f"files_before_validation_record={len(actual_files)}"),
        ("no_model_or_array_output", not any(path.suffix.lower() in {".joblib", ".npz", ".npy", ".pkl", ".pt", ".pth"} for path in output_dir().iterdir()), "forbidden_files=0"),
    ]
    result = pd.DataFrame(checks, columns=["check", "status", "detail"])
    result["status"] = result["status"].map({True: "pass", False: "fail"})
    verify_or_create_tsv(result, output_dir() / result_file)
    print(result.to_string(index=False))
    failed = result[result["status"] != "pass"]
    if len(failed):
        raise SystemExit(f"Failed checks: {failed['check'].tolist()}")
    print(f"Passed {len(result)}/{len(result)} checks")


if __name__ == "__main__":
    main()
