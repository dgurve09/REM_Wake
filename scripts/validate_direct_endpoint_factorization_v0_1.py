"""Validate validation-only endpoint factorization experiment DE-D."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from run_direct_endpoint_factorization_v0_1 import (
    COMPARATOR,
    HEADS,
    candidate_score_path,
    data_parent,
    model_path,
    output_dir,
    repo_root,
    sha256,
    subject_assignments,
)
from run_direct_event_baseline_v0_1 import (
    collapse_alarms,
    evaluate_events,
    local_event_inputs,
    reference_events,
)


# Section 1: check helpers

checks = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append(
        {"check": name, "status": "pass" if passed else "fail", "detail": detail}
    )
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def close(left, right, tolerance: float = 1e-12) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    return bool(np.isclose(float(left), float(right), atol=tolerance, rtol=0.0))


# Section 2: protocol, endpoint labels, and head metrics

def validate_endpoint_rows() -> None:
    scores = pd.read_csv(
        output_dir() / "train_validation_labeled_endpoint_scores_v0.1.tsv", sep="\t"
    )
    record(
        "two_scores_per_labeled_row",
        len(scores) == 6800
        and scores["sample_id"].nunique() == 3400
        and bool((scores.groupby("sample_id")["head"].nunique() == 2).all()),
        f"score_rows={len(scores)}, samples={scores['sample_id'].nunique()}",
    )
    unique = scores.drop_duplicates("sample_id")
    conjunction = unique["rem_before"].astype(int) & unique["wake_after"].astype(int)
    record(
        "endpoint_conjunction_reconstructs_label",
        bool((conjunction == unique["label"].astype(int)).all()),
        f"rows={len(unique)}",
    )
    counts = unique.groupby("partition").size().to_dict()
    record(
        "labeled_partition_counts",
        counts == {"train": 2743, "validation": 657},
        str(counts),
    )

    saved_metrics = pd.read_csv(output_dir() / "endpoint_head_metrics_v0.1.tsv", sep="\t")
    failures = []
    for item in saved_metrics.itertuples(index=False):
        local = scores[
            (scores["head"] == item.head) & (scores["partition"] == item.partition)
        ]
        target = local[item.head].to_numpy(dtype=int)
        probability = local["head_probability"].to_numpy(dtype=float)
        if not close(average_precision_score(target, probability), item.average_precision):
            failures.append(f"{item.head}/{item.partition}/average_precision")
        if not close(roc_auc_score(target, probability), item.roc_auc):
            failures.append(f"{item.head}/{item.partition}/roc_auc")
    record(
        "endpoint_metric_recomputation",
        not failures,
        f"configurations={len(saved_metrics)}, failures={len(failures)}",
    )


# Section 3: frozen threshold, external hashes, and phase isolation

def validate_frozen_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected = pd.read_csv(output_dir() / "selected_threshold_v0.1.tsv", sep="\t")
    curve = pd.read_csv(output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t")
    record(
        "threshold_grid",
        len(curve) == 99
        and np.allclose(curve["threshold"].to_numpy(), np.arange(1, 100) / 100.0),
        f"rows={len(curve)}",
    )
    recomputed = curve.sort_values(
        ["f1", "false_alarms_per_hour", "recall", "threshold"],
        ascending=[False, True, False, False],
        kind="stable",
    ).iloc[0]
    record(
        "selected_threshold",
        len(selected) == 1 and close(selected.iloc[0].threshold, recomputed.threshold),
        f"threshold={selected.iloc[0].threshold}",
    )
    selected_row = selected.iloc[0]
    model_hashes_match = (
        sha256(model_path("rem_before")) == selected_row.rem_before_model_sha256
        and sha256(model_path("wake_after")) == selected_row.wake_after_model_sha256
    )
    record("endpoint_model_hashes", model_hashes_match, "two fitted heads verified")

    scores = pd.read_csv(candidate_score_path(), sep="\t", compression="gzip")
    assignments = subject_assignments()
    validation_subjects = set(
        assignments[assignments["partition"] == "validation"]["subject"]
    )
    record(
        "validation_only_candidate_scores",
        set(scores["partition"]) == {"validation"}
        and set(scores["subject"]) == validation_subjects,
        f"partitions={sorted(scores['partition'].unique())}, recordings={scores['subject'].nunique()}",
    )
    product = scores["probability_rem_before"] * scores["probability_wake_after"]
    record(
        "factorized_probability_product",
        bool(np.allclose(product, scores["probability"], atol=1e-15, rtol=0.0)),
        f"candidate_rows={len(scores)}",
    )

    support = pd.read_csv(output_dir() / "validation_event_support_v0.1.tsv", sep="\t")
    counts = (
        scores.groupby(["comparator", "partition", "subject", "pid"])
        .size()
        .rename("score_boundaries")
        .reset_index()
    )
    linked = support.merge(
        counts,
        on=["comparator", "partition", "subject", "pid"],
        validate="one_to_one",
    )
    support_ok = (
        len(linked) == 20
        and bool((linked["supported_boundaries"] == linked["score_boundaries"]).all())
        and bool(
            np.allclose(
                linked["supported_hours"],
                linked["supported_boundaries"] * 30.0 / 3600.0,
            )
        )
    )
    record(
        "validation_support_accounting",
        support_ok,
        f"support_rows={len(support)}, score_rows={len(scores)}",
    )
    return selected, scores, support


def validate_manifests() -> None:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    failures = []
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(item.path_relative_to_data_parent)
    record(
        "external_artifact_hashes",
        not failures and len(manifest) == 105,
        f"verified={len(manifest)}, failures={len(failures)}",
    )
    current = manifest[manifest["artifact_role"] == "frozen_recording_features"]
    prior = pd.read_csv(
        repo_root()
        / "experiments/2026-08-15_stage_first_feature_baseline_v0.1/external_artifact_manifest_v0.1.tsv",
        sep="\t",
    )
    prior = prior[prior["artifact_role"] == "recording_features"]
    joined = current.merge(
        prior,
        on="path_relative_to_data_parent",
        suffixes=("_factorized", "_stage_first"),
        validate="one_to_one",
    )
    same = (joined["bytes_factorized"] == joined["bytes_stage_first"]) & (
        joined["sha256_factorized"] == joined["sha256_stage_first"]
    )
    record(
        "feature_hashes_match_stage_first",
        len(joined) == 102 and bool(same.all()),
        f"matched={len(joined)}, mismatches={int((~same).sum())}",
    )

    inputs = pd.read_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t")
    input_failures = []
    for item in inputs.itertuples(index=False):
        path = repo_root() / item.relative_path
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            input_failures.append(item.relative_path)
    record(
        "repository_input_hashes",
        not input_failures and len(inputs) == 8,
        f"verified={len(inputs)}, failures={len(input_failures)}",
    )

    forbidden = list(repo_root().rglob("*.joblib")) + list(repo_root().rglob("*.npz"))
    record(
        "no_model_or_array_files_in_repo",
        len(forbidden) == 0,
        f"forbidden_files={len(forbidden)}",
    )


# Section 4: alarm, event, comparison, and error recomputation

def validate_event_outputs(
    selected: pd.DataFrame, scores: pd.DataFrame, support: pd.DataFrame
) -> None:
    threshold = float(selected.iloc[0].threshold)
    recomputed_alarms = collapse_alarms(scores, threshold)
    saved_alarms = pd.read_csv(output_dir() / "validation_predicted_events_v0.1.tsv", sep="\t")
    keys = ["comparator", "partition", "subject", "event_time_sec"]
    left = recomputed_alarms.sort_values(keys).reset_index(drop=True)
    right = saved_alarms.sort_values(keys).reset_index(drop=True)
    alarms_match = (
        len(left) == len(right)
        and left[keys].equals(right[keys])
        and bool(np.allclose(left["probability"], right["probability"], atol=1e-12, rtol=0.0))
        and bool((left["run_candidates"] == right["run_candidates"]).all())
    )
    record(
        "validation_alarm_recomputation",
        alarms_match,
        f"saved={len(right)}, recomputed={len(left)}",
    )

    metrics = pd.read_csv(output_dir() / "validation_event_metrics_v0.1.tsv", sep="\t")
    references = reference_events()
    failures = []
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
    local_support = support[["subject", "pid", "supported_hours"]]
    predictions = saved_alarms[["subject", "pid", "event_time_sec"]]
    for item in metrics.itertuples(index=False):
        eligible, ignored = local_event_inputs(references, "validation", item.membership)
        _, _, _, summary = evaluate_events(
            eligible, predictions, ignored, local_support, float(item.tolerance_sec)
        )
        for column in metric_columns:
            if not close(getattr(item, column), summary[column]):
                failures.append(f"{item.membership}/{item.tolerance_sec}/{column}")
    record(
        "validation_event_metric_recomputation",
        not failures,
        f"configurations={len(metrics)}, failures={len(failures)}",
    )

    comparison = pd.read_csv(
        output_dir() / "validation_comparison_with_de_b_v0.1.tsv", sep="\t"
    ).set_index("comparator")
    decision = pd.read_csv(output_dir() / "validation_decision_v0.1.tsv", sep="\t").iloc[0]
    de_b = comparison.loc["DE-B"]
    de_d = comparison.loc[COMPARATOR]
    decision_ok = (
        close(decision.event_f1_difference, de_d.f1 - de_b.f1)
        and close(
            decision.false_alarms_per_hour_difference,
            de_d.false_alarms_per_hour - de_b.false_alarms_per_hour,
        )
        and bool(decision.prespecified_success_rule_met)
        == bool(
            (de_d.f1 > de_b.f1)
            and (de_d.false_alarms_per_hour < de_b.false_alarms_per_hour)
        )
    )
    record(
        "validation_decision_recomputation",
        decision_ok,
        f"success={bool(decision.prespecified_success_rule_met)}",
    )

    false_positive = pd.read_csv(
        output_dir() / "validation_false_positive_context_v0.1.tsv", sep="\t"
    )
    categories = pd.read_csv(
        output_dir() / "validation_false_positive_category_comparison_v0.1.tsv",
        sep="\t",
    )
    de_d_expected = int(de_d.false_positive)
    de_d_category_sum = int(
        categories[categories["comparator"] == COMPARATOR]["false_positive_alarms"].sum()
    )
    de_b_category_sum = int(
        categories[categories["comparator"] == "DE-B"]["false_positive_alarms"].sum()
    )
    record(
        "false_positive_category_accounting",
        len(false_positive) == de_d_expected
        and de_d_category_sum == de_d_expected
        and de_b_category_sum == int(de_b.false_positive),
        f"DE-D={de_d_category_sum}, DE-B={de_b_category_sum}",
    )

    bootstrap = pd.read_csv(output_dir() / "validation_event_bootstrap_v0.1.tsv", sep="\t")
    record(
        "bootstrap_configuration",
        len(bootstrap) == 16
        and set(bootstrap["resamples"]) == {2000}
        and set(bootstrap["seed"]) == {20260822},
        f"rows={len(bootstrap)}",
    )


# Section 5: execute

def main() -> None:
    validate_endpoint_rows()
    selected, scores, support = validate_frozen_outputs()
    validate_manifests()
    validate_event_outputs(selected, scores, support)
    result = pd.DataFrame(checks)
    result.to_csv(output_dir() / "output_integrity_checks_v0.1.tsv", sep="\t", index=False)
    print(result.to_string(index=False))
    print(f"Passed {int((result['status'] == 'pass').sum())}/{len(result)} checks")


if __name__ == "__main__":
    main()
