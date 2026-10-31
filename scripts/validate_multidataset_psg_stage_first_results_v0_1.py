"""Independently validate the multi-dataset PSG stage-first result package."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: load retained tables

def table(name: str) -> pd.DataFrame:
    return pd.read_csv(common.output_dir() / name, sep="\t")


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=0.0))


def aggregate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    tp = int(frame.true_positive.sum())
    fp = int(frame.false_positive.sum())
    fn = int(frame.false_negative.sum())
    hours = float(frame.supported_hours.sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": fp / hours,
        "reference_events": tp + fn,
        "predicted_events": int(frame.predicted_events.sum()),
        "ignored_predictions": int(frame.ignored_predictions.sum()),
    }


# Section 2: independent consistency checks

def main() -> None:
    fits = table("stage_fit_summary_v0.1.tsv")
    stage = table("heldout_stage_metrics_v0.1.tsv")
    classes = table("heldout_stage_class_metrics_v0.1.tsv")
    confusion = table("heldout_stage_confusion_v0.1.tsv")
    support = table("outer_support_v0.1.tsv")
    alarms = table("derived_event_alarms_v0.1.tsv")
    event = table("heldout_event_metrics_v0.1.tsv")
    participant = table("heldout_event_participants_v0.1.tsv")
    comparison = table("direct_metric_comparison_v0.1.tsv")
    bootstrap = table("paired_participant_bootstrap_v0.1.tsv")
    decision = table("hypothesis_decision_v0.1.tsv").iloc[0]
    manifest = table("external_artifact_manifest_v0.1.tsv")
    in_run = table("in_run_checks_v0.1.tsv")

    checks = []

    def add(name: str, observed, expected, passed: bool) -> None:
        checks.append(
            {"check": name, "observed": observed, "expected": expected, "passed": bool(passed)}
        )

    add("three_heldout_fits", len(fits), 3, len(fits) == 3)
    add("all_fits_report_convergence", fits.converged.notna().all(), True, fits.converged.notna().all())
    add("nonconvergence_preserved", int((~fits.converged).sum()), 3, int((~fits.converged).sum()) == 3)
    add("all_probabilities_finite", fits.all_probabilities_finite.all(), True, fits.all_probabilities_finite.all())
    add(
        "probability_sums_valid",
        float(fits.probability_row_sums_max_error.max()),
        "<=1e-12",
        fits.probability_row_sums_max_error.le(1e-12).all(),
    )
    add("known_score_rows", int(stage.rows.sum()), 372361, int(stage.rows.sum()) == 372361)
    add(
        "support_matches_stage_rows",
        int(support.supported_boundaries.sum()),
        int(stage.rows.sum()),
        int(support.supported_boundaries.sum()) == int(stage.rows.sum()),
    )
    class_totals = classes.groupby("heldout_dataset").support.sum().sort_index()
    stage_totals = stage.set_index("heldout_dataset").rows.sort_index()
    add("class_support_matches_stage_rows", class_totals.equals(stage_totals), True, class_totals.equals(stage_totals))
    confusion_totals = confusion.groupby("heldout_dataset").rows.sum().sort_index()
    add("confusion_matches_stage_rows", confusion_totals.equals(stage_totals), True, confusion_totals.equals(stage_totals))

    artifact_ok = True
    artifact_rows = 0
    for item in manifest.itertuples(index=False):
        path = features.data_parent() / item.relative_path
        artifact_ok &= path.exists() and path.stat().st_size == int(item.bytes)
        artifact_ok &= path.exists() and features.sha256(path) == item.sha256
        if item.artifact_type == "model" and path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                payload = json.load(source)
            artifact_ok &= int(sum(payload["class_counts"])) == int(
                fits.loc[fits.heldout_dataset.eq(item.heldout_dataset), "training_rows"].iloc[0]
            )
        if item.artifact_type == "stage_scores" and path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as source:
                artifact_rows += sum(1 for _ in source) - 1
    add("six_external_artifacts_match_manifest", artifact_ok, True, artifact_ok and len(manifest) == 6)
    add("external_score_rows", artifact_rows, 372361, artifact_rows == 372361)

    metric_ok = True
    alarm_ok = True
    for item in event.itertuples(index=False):
        if item.heldout_dataset == "POOLED_EXTERNAL":
            local = participant[
                participant.heldout_dataset.isin(common.PRIMARY_EXTERNAL)
                & participant.candidate.eq(item.candidate)
            ]
            expected_alarms = len(
                alarms[
                    alarms.heldout_dataset.isin(common.PRIMARY_EXTERNAL)
                    & alarms.candidate.eq(item.candidate)
                ]
            )
        else:
            local = participant[
                participant.heldout_dataset.eq(item.heldout_dataset)
                & participant.candidate.eq(item.candidate)
            ]
            expected_alarms = len(
                alarms[
                    alarms.heldout_dataset.eq(item.heldout_dataset)
                    & alarms.candidate.eq(item.candidate)
                ]
            )
        aggregate = aggregate_metrics(local)
        for metric in ["precision", "recall", "f1", "false_alarms_per_hour"]:
            metric_ok &= close(float(getattr(item, metric)), float(aggregate[metric]))
        for metric in ["reference_events", "predicted_events", "ignored_predictions"]:
            metric_ok &= int(getattr(item, metric)) == int(aggregate[metric])
        alarm_ok &= int(item.predicted_events) == expected_alarms
    add("event_metrics_reaggregate", metric_ok, True, metric_ok)
    add("alarm_counts_match_metrics", alarm_ok, True, alarm_ok)

    direct = pd.read_csv(
        features.repo_root()
        / "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/heldout_event_metrics_v0.2.tsv",
        sep="\t",
    ).query("candidate == 'ADV3-CHANGE'")
    merged = comparison.merge(
        event,
        on=["heldout_dataset", "candidate"],
        validate="one_to_one",
    ).merge(
        direct[["heldout_dataset", "f1", "recall", "precision", "false_alarms_per_hour"]],
        on="heldout_dataset",
        suffixes=("_stage_table", "_direct_table"),
        validate="many_to_one",
    )
    comparison_ok = np.allclose(
        merged.f1_difference,
        merged.f1_stage_table - merged.f1_direct_table,
        atol=1e-12,
        rtol=0.0,
    )
    add("direct_comparison_recomputed", comparison_ok, True, comparison_ok)
    add("bootstrap_shape", len(bootstrap), 6, len(bootstrap) == 6)
    interval_ok = (bootstrap.lower_95 <= bootstrap["median"]).all() and (
        bootstrap["median"] <= bootstrap.upper_95
    ).all()
    add("bootstrap_intervals_ordered", interval_ok, True, interval_ok)

    primary = event[
        event.heldout_dataset.eq("POOLED_EXTERNAL")
        & event.candidate.eq(common.PRIMARY_CANDIDATE)
    ].iloc[0]
    direct_pool = direct[direct.heldout_dataset.eq("POOLED_EXTERNAL")].iloc[0]
    decision_ok = (
        close(decision.stage_first_f1, primary.f1)
        and close(decision.direct_f1, direct_pool.f1)
        and close(decision.f1_difference, primary.f1 - direct_pool.f1)
        and not bool(decision.point_gate_pass)
        and decision.decision == "stop_stage_first_v0.1"
    )
    add("decision_recomputed", decision_ok, True, decision_ok)
    add("in_run_checks_pass", in_run.passed.all(), True, in_run.passed.all())

    result = pd.DataFrame(checks)
    if not result.passed.all():
        print(result.to_string(index=False))
        raise ValueError("Independent stage-first result validation failed")
    verify_or_create_tsv(
        result,
        common.output_dir() / "independent_result_validation_v0.1.tsv",
    )
    print(f"Independent stage-first result validation passed {len(result)}/{len(result)} checks.")


if __name__ == "__main__":
    main()
