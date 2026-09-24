"""Read-only validation of the deep temporal error diagnostic."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_deep_temporal_error_modes_v0_1 import (
    data_parent,
    local_event_inputs,
    reference_events,
    repo_root,
    train_assignments,
)
from stage_first_event_evaluation_v0_1 import evaluate_events


OUTPUT_DIR = (
    repo_root() / "experiments" / "2026-09-23_deep_temporal_error_diagnostic_v0.1"
)
SOURCE_DIR = (
    repo_root() / "experiments" / "2026-09-22_deep_temporal_nested_cv_v0.1"
)
PRIOR_DIR = repo_root() / "experiments" / "2026-09-19_lstm_crf_train_oof_v0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, sep="\t")


def historical_quality_recall(
    references: pd.DataFrame, model: str, tolerance: float
) -> pd.DataFrame:
    eligible, ignored = local_event_inputs(references, "primary")
    predictions = pd.read_csv(
        PRIOR_DIR / "train_oof_predicted_events_v0.1.tsv", sep="\t"
    )
    predictions = predictions[predictions["model"] == model]
    support = pd.read_csv(PRIOR_DIR / "train_oof_support_v0.1.tsv", sep="\t")
    _, _, matches, _ = evaluate_events(
        eligible,
        predictions[["subject", "pid", "event_time_sec"]],
        ignored,
        support[["subject", "pid", "supported_hours"]],
        tolerance,
    )
    matched = matches[matches["match_type"] == "eligible"]
    keys = set(zip(matched["subject"], matched["reference_time_sec"]))
    events = references[references["primary_analysis_eligible"]].copy()
    events["detected"] = [
        (subject, float(time)) in keys
        for subject, time in zip(events["subject"], events["event_time_sec"])
    ]
    return (
        events.groupby("membership_tier", as_index=False)["detected"]
        .agg(["sum", "count", "mean"])
        .reset_index()
    )


def main() -> None:
    checks = []

    manifest = read("source_manifest_v0.1.tsv")
    hash_ok = True
    for row in manifest.itertuples(index=False):
        base = repo_root() if row.path_scope == "repository" else data_parent()
        path = base / Path(row.relative_path)
        hash_ok &= (
            path.exists()
            and path.stat().st_size == int(row.bytes)
            and sha256(path) == row.sha256
        )
    checks.append(("source_hashes", hash_ok, f"{len(manifest)} files"))

    references = reference_events(train_assignments())
    eligible, _ = local_event_inputs(references, "primary")
    checks.append(("primary_membership", len(eligible) == 180, str(len(eligible))))

    tolerance = read("frozen_tolerance_curve_v0.1.tsv")
    source_metrics = pd.read_csv(SOURCE_DIR / "outer_event_metrics_v0.1.tsv", sep="\t")
    source_primary = source_metrics.query(
        "pipeline == 'NESTED-SELECTED' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    diagnostic_primary = tolerance[tolerance["tolerance_sec"] == 15].iloc[0]
    metric_columns = [
        "true_positive",
        "false_positive",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
    ]
    metric_ok = all(
        np.isclose(float(source_primary[column]), float(diagnostic_primary[column]))
        for column in metric_columns
    )
    checks.append(("frozen_metric_source_match", metric_ok, "primary +/-15 s"))

    event_scores = read("primary_event_boundary_scores_v0.1.tsv")
    quality = read("quality_tier_recall_v0.1.tsv")
    nested = quality[quality["model"] == "NESTED-SELECTED"]
    quality_ok = True
    for row in nested.itertuples(index=False):
        local = event_scores[event_scores["membership_tier"] == row.membership_tier]
        detected = local[f"detected_within_{int(row.tolerance_sec)}_sec"]
        quality_ok &= (
            len(local) == int(row.reference_events)
            and int(detected.sum()) == int(row.detected_events)
            and np.isclose(float(detected.mean()), float(row.recall))
        )
    checks.append(("nested_quality_reconstruction", quality_ok, "4 rows"))

    historical_ok = True
    for model in ["LR-OOF", "LC-1"]:
        for tolerance_value in [15.0, 45.0]:
            rebuilt = historical_quality_recall(references, model, tolerance_value)
            stored = quality[
                (quality["model"] == model)
                & (quality["tolerance_sec"] == tolerance_value)
            ]
            for row in stored.itertuples(index=False):
                source = rebuilt[rebuilt["membership_tier"] == row.membership_tier].iloc[0]
                historical_ok &= (
                    int(source["sum"]) == int(row.detected_events)
                    and int(source["count"]) == int(row.reference_events)
                    and np.isclose(float(source["mean"]), float(row.recall))
                )
    checks.append(("historical_quality_reconstruction", historical_ok, "8 rows"))

    false_alarm_events = read("false_alarm_event_context_v0.1.tsv")
    false_alarm_summary = read("false_alarm_context_summary_v0.1.tsv")
    context_counts = false_alarm_events["context_category"].value_counts()
    context_ok = len(false_alarm_events) == 177
    for row in false_alarm_summary.itertuples(index=False):
        context_ok &= context_counts.get(row.context_category, 0) == int(row.false_alarms)
    checks.append(("false_alarm_context_reconstruction", context_ok, "177 alarms"))

    thresholds = read("threshold_upper_bounds_v0.1.tsv")
    frozen = thresholds[thresholds["threshold_analysis"] == "frozen_inner_selected"].iloc[0]
    fold_oracle = thresholds[
        thresholds["threshold_analysis"] == "outer_fold_label_oracle"
    ].iloc[0]
    oracle_ok = (
        np.isclose(float(frozen.f1), 0.16452442159383032)
        and np.isclose(float(fold_oracle.f1), 0.2324455205811138)
        and float(fold_oracle.f1) < 0.25
    )
    checks.append(("threshold_bound_values", oracle_ok, "frozen and fold oracle"))

    readme = (OUTPUT_DIR / "README.md").read_text(encoding="utf-8")
    readme_ok = all(
        value in readme
        for value in ["32", "41", "0.2177", "0.2324", "not usable estimates"]
    )
    checks.append(("readme_key_results", readme_ok, "five values"))

    table = pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in checks
        ]
    )
    print(table.to_string(index=False))
    if not table["status"].eq("pass").all():
        sys.exit("Independent diagnostic validation failed")
    print(f"Independent diagnostic validation passed {len(table)}/{len(table)} checks.")


if __name__ == "__main__":
    main()
