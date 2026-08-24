"""Validate phase isolation and reported outputs for direct baseline v0.1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv

from run_direct_event_baseline_v0_1 import (
    MODEL_OFFSETS,
    TOLERANCES,
    collapse_alarms,
    data_parent,
    evaluate_events,
    local_event_inputs,
    model_path,
    output_dir,
    reference_events,
    repo_root,
    score_path,
    sha256,
    subject_assignments,
)


# Section 1: check recorder

checks: list[dict] = []


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


# Section 2: frozen models, thresholds, and construction

def validate_frozen_configuration() -> pd.DataFrame:
    selected = pd.read_csv(output_dir() / "frozen_model_thresholds_v0.1.tsv", sep="\t")
    curve = pd.read_csv(output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t")
    construction = pd.read_csv(
        output_dir() / "train_validation_construction_summary_v0.1.tsv", sep="\t"
    )
    fit = pd.read_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t")

    record(
        "selected_models_complete",
        set(selected["comparator"]) == set(MODEL_OFFSETS) and len(selected) == 2,
        f"selected_rows={len(selected)}",
    )
    for comparator in MODEL_OFFSETS:
        local = curve[curve["comparator"] == comparator]
        record(
            f"{comparator}_threshold_grid",
            len(local) == 99
            and np.allclose(local["threshold"].to_numpy(), np.arange(1, 100) / 100.0),
            f"threshold_rows={len(local)}",
        )
        recomputed = local.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        frozen = selected[selected["comparator"] == comparator].iloc[0]
        record(
            f"{comparator}_selected_threshold",
            close(recomputed["threshold"], frozen["threshold"]),
            f"threshold={frozen['threshold']}",
        )
        model_hash = sha256(model_path(comparator))
        record(
            f"{comparator}_model_hash",
            model_hash == frozen["model_sha256"],
            f"sha256={model_hash}",
        )

    required_columns = {
        "comparator",
        "partition",
        "label",
        "source_tier",
        "requested_rows",
        "retained_rows",
        "dropped_rows",
    }
    record(
        "construction_schema",
        required_columns.issubset(construction.columns),
        ",".join(construction.columns),
    )
    accounting = construction["requested_rows"] == (
        construction["retained_rows"] + construction["dropped_rows"]
    )
    record(
        "construction_accounting",
        bool(accounting.all()),
        f"requested={construction['requested_rows'].sum()}, retained={construction['retained_rows'].sum()}, dropped={construction['dropped_rows'].sum()}",
    )
    record(
        "fit_convergence_record",
        len(fit) == 2 and int(fit["convergence_warning_count"].sum()) == 0,
        f"fit_rows={len(fit)}, convergence_warnings={int(fit['convergence_warning_count'].sum())}",
    )
    return selected


# Section 3: external artifact integrity

def validate_external_artifacts() -> None:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    failures = []
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(item.path_relative_to_data_parent)
    record(
        "external_manifest_hashes",
        not failures,
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
        suffixes=("_direct", "_stage_first"),
        validate="one_to_one",
    )
    same = (joined["bytes_direct"] == joined["bytes_stage_first"]) & (
        joined["sha256_direct"] == joined["sha256_stage_first"]
    )
    record(
        "reused_feature_hashes_match_stage_first",
        len(joined) == 128 and bool(same.all()),
        f"matched={len(joined)}, mismatches={int((~same).sum())}",
    )

    forbidden = list(repo_root().rglob("*.joblib")) + list(repo_root().rglob("*.npz"))
    record(
        "no_model_or_array_files_in_git_tree",
        len(forbidden) == 0,
        f"forbidden_files={len(forbidden)}",
    )


# Section 4: phase isolation and continuous support

def load_phase(phase: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(score_path(phase), sep="\t", compression="gzip")
    support = pd.read_csv(output_dir() / f"{phase}_event_support_v0.1.tsv", sep="\t")
    alarms = pd.read_csv(output_dir() / f"{phase}_predicted_events_v0.1.tsv", sep="\t")
    return scores, support, alarms


def validate_phase_isolation() -> None:
    assignments = subject_assignments()[["subject", "pid", "partition"]]
    train_scores, train_support, _ = load_phase("train_validation")
    test_scores, test_support, _ = load_phase("test")
    record(
        "train_validation_phase_excludes_test",
        set(train_scores["partition"]) == {"train", "validation"}
        and set(train_support["partition"]) == {"train", "validation"},
        f"score_partitions={sorted(train_scores['partition'].unique())}",
    )
    record(
        "test_phase_contains_test_only",
        set(test_scores["partition"]) == {"test"}
        and set(test_support["partition"]) == {"test"},
        f"score_partitions={sorted(test_scores['partition'].unique())}",
    )
    all_scores = pd.concat([train_scores, test_scores], ignore_index=True)
    linkage = all_scores[["subject", "pid", "partition"]].drop_duplicates().merge(
        assignments,
        on=["subject", "pid"],
        suffixes=("_score", "_split"),
        validate="one_to_one",
    )
    record(
        "score_partition_assignment",
        len(linkage) == 128
        and bool((linkage["partition_score"] == linkage["partition_split"]).all()),
        f"recordings={len(linkage)}",
    )
    for phase, scores, support in [
        ("train_validation", train_scores, train_support),
        ("test", test_scores, test_support),
    ]:
        counts = (
            scores.groupby(["comparator", "partition", "subject", "pid"])
            .size()
            .rename("recomputed_boundaries")
            .reset_index()
        )
        linked = support.merge(
            counts,
            on=["comparator", "partition", "subject", "pid"],
            validate="one_to_one",
        )
        exact = linked["supported_boundaries"] == linked["recomputed_boundaries"]
        hours = np.isclose(
            linked["supported_hours"], linked["supported_boundaries"] * 30.0 / 3600.0
        )
        record(
            f"{phase}_support_accounting",
            len(linked) == len(support) and bool(exact.all()) and bool(hours.all()),
            f"support_rows={len(support)}, score_rows={len(scores)}",
        )


# Section 5: exact alarm and metric recomputation

def validate_alarm_table(
    phase: str, scores: pd.DataFrame, saved: pd.DataFrame, selected: pd.DataFrame
) -> None:
    frames = []
    threshold_map = selected.set_index("comparator")["threshold"].to_dict()
    for (comparator, partition), group in scores.groupby(["comparator", "partition"]):
        frames.append(collapse_alarms(group, float(threshold_map[comparator])))
    recomputed = pd.concat(frames, ignore_index=True)
    keys = ["comparator", "partition", "subject", "event_time_sec"]
    saved_ordered = saved.sort_values(keys).reset_index(drop=True)
    recomputed_ordered = recomputed.sort_values(keys).reset_index(drop=True)
    exact_keys = saved_ordered[keys].equals(recomputed_ordered[keys])
    numeric = all(
        np.allclose(saved_ordered[column], recomputed_ordered[column], atol=1e-12, rtol=0.0)
        for column in ["probability", "threshold", "run_candidates"]
    )
    record(
        f"{phase}_alarm_recomputation",
        len(saved_ordered) == len(recomputed_ordered) and exact_keys and numeric,
        f"saved={len(saved_ordered)}, recomputed={len(recomputed_ordered)}",
    )


def validate_event_metrics(phase: str) -> None:
    _, support, alarms = load_phase(phase)
    saved_metrics = pd.read_csv(output_dir() / f"{phase}_event_metrics_v0.1.tsv", sep="\t")
    references = reference_events()
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
    for item in saved_metrics.itertuples(index=False):
        local_support = support[
            (support["comparator"] == item.comparator)
            & (support["partition"] == item.partition)
        ][["subject", "pid", "supported_hours"]]
        local_predictions = alarms[
            (alarms["comparator"] == item.comparator)
            & (alarms["partition"] == item.partition)
        ][["subject", "pid", "event_time_sec"]]
        eligible, ignored = local_event_inputs(references, item.partition, item.membership)
        _, _, _, summary = evaluate_events(
            eligible, local_predictions, ignored, local_support, float(item.tolerance_sec)
        )
        for column in metric_columns:
            if not close(getattr(item, column), summary[column]):
                failures.append(
                    f"{item.comparator}/{item.partition}/{item.membership}/{item.tolerance_sec}/{column}"
                )
    record(
        f"{phase}_event_metric_recomputation",
        not failures,
        f"configurations={len(saved_metrics)}, failures={len(failures)}",
    )


# Section 6: execute and save

def main() -> None:
    selected = validate_frozen_configuration()
    validate_external_artifacts()
    validate_phase_isolation()
    for phase in ["train_validation", "test"]:
        scores, _, alarms = load_phase(phase)
        validate_alarm_table(phase, scores, alarms, selected)
        validate_event_metrics(phase)

    result = pd.DataFrame(checks)
    verify_or_create_tsv(result, output_dir() / "output_integrity_checks_v0.1.tsv")
    print(result.to_string(index=False))
    print(f"Passed {int((result['status'] == 'pass').sum())}/{len(result)} checks")


if __name__ == "__main__":
    main()
