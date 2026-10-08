"""Read-only validation for full-night risk-set formulation v0.1."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

import run_full_night_risk_set_formulation_v0_1 as experiment
import run_paired_enriched_feature_pruning_v0_1 as prior


def read_table(name: str) -> pd.DataFrame:
    path = experiment.output_dir() / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=tolerance))


def main() -> None:
    versions = json.loads(
        (experiment.output_dir() / "software_versions_v0.1.json").read_text(encoding="utf-8")
    )
    require(versions["protocol_commit"].startswith(experiment.PROTOCOL_COMMIT), "Protocol commit mismatch")
    require(
        versions["protocol_amendment_commit"].startswith(experiment.PROTOCOL_AMENDMENT_COMMIT),
        "Protocol amendment commit mismatch",
    )

    construction = read_table("candidate_construction_summary_v0.1.tsv").set_index("metric")["value"]
    require(int(construction["eligible_backgrounds_before_context_intersection"]) == 73476, "Eligible background count changed")
    require(int(construction["retained_primary_events"]) == 180, "Positive count changed")
    retained_negatives = int(construction["retained_backgrounds"])
    risk_rows = int(construction["risk_set_rows"])
    require(risk_rows == retained_negatives + 180, "Risk-set row accounting failed")

    fits = read_table("model_fit_summary_v0.1.tsv")
    require(len(fits) == 75, "Expected 75 nested fit records")
    require(set(fits["candidate"]) == set(experiment.CANDIDATES), "Candidate set changed")
    require(fits.groupby("candidate").size().eq(25).all(), "Expected 25 fit records per candidate")
    new_fits = fits[fits["candidate"].isin(experiment.NEW_CANDIDATES)]
    expected_convergence = new_fits["iterations"].astype(int) < experiment.MAX_ITER
    require(
        np.array_equal(prior.truth(new_fits["converged"]).to_numpy(), expected_convergence.to_numpy()),
        "Convergence status does not match solver iteration count",
    )

    selections = read_table("inner_threshold_selections_v0.1.tsv")
    require(len(selections) == 60, "Expected 60 threshold selections")
    require(set(np.round(selections["alarm_budget_per_hour"], 2)) == set(experiment.ALARM_BUDGETS), "Alarm budgets changed")
    require(
        (selections["false_alarms_per_hour"] <= selections["alarm_budget_per_hour"] + 1e-12).all(),
        "An inner-selected threshold exceeds its alarm budget",
    )

    support = read_table("outer_support_v0.1.tsv")
    require(len(support) == 82 and support["pid"].nunique() == 64, "Outer support membership changed")
    metrics = read_table("outer_event_metrics_v0.1.tsv")
    require(len(metrics) == 12, "Expected 12 event-metric rows")
    participants = read_table("outer_event_participants_v0.1.tsv")
    require(len(participants) == 768, "Expected 768 participant event rows")

    for row in metrics.itertuples(index=False):
        local = participants[
            participants["candidate"].eq(row.candidate)
            & participants["membership"].eq(row.membership)
            & participants["tolerance_sec"].eq(row.tolerance_sec)
        ]
        values = prior.metric_values(
            int(local["true_positive"].sum()),
            int(local["false_positive"].sum()),
            int(local["false_negative"].sum()),
            float(local["supported_hours"].sum()),
        )
        for metric in ["precision", "recall", "f1", "false_alarms_per_hour"]:
            require(close(float(getattr(row, metric)), float(values[metric])), f"Event metric mismatch: {row.candidate}, {metric}")

    manifest = read_table("external_artifact_manifest_v0.1.tsv")
    for row in manifest.itertuples(index=False):
        path = experiment.data_parent() / row.relative_path
        require(path.exists(), f"Missing external artifact: {row.relative_path}")
        require(path.stat().st_size == int(row.bytes), f"Artifact size changed: {row.relative_path}")
        require(prior.sha256(path) == row.sha256, f"Artifact hash changed: {row.relative_path}")

    scores = pd.read_csv(experiment.risk_score_path(), sep="\t", compression="gzip")
    require(len(scores) == risk_rows * 3, "Labelled risk-score count changed")
    require(scores.groupby("candidate")["pid"].nunique().eq(64).all(), "Risk-score participant coverage changed")
    calibration = read_table("outer_risk_set_metrics_v0.1.tsv").set_index("candidate")
    for candidate in experiment.CANDIDATES:
        local = scores[scores["candidate"].eq(candidate)]
        labels = local["label"].to_numpy(dtype=int)
        probability = local["probability"].to_numpy(dtype=float)
        clipped = np.clip(probability, experiment.EPSILON, 1.0 - experiment.EPSILON)
        expected = {
            "average_precision": average_precision_score(labels, probability),
            "brier_score": brier_score_loss(labels, probability),
            "log_loss": log_loss(labels, clipped, labels=[0, 1]),
        }
        for metric, value in expected.items():
            require(close(float(calibration.loc[candidate, metric]), float(value)), f"Risk metric mismatch: {candidate}, {metric}")

    decisions = read_table("hypothesis_decisions_v0.1.tsv")
    require(set(decisions["hypothesis"]) == {"H-RISK", "H-PRIOR"}, "Decision set changed")
    failures = read_table("execution_failures_v0.1.tsv")
    require(len(failures) == 3, "Execution-failure record changed")
    archived = (
        experiment.data_parent()
        / "derived/full_night_risk_set_formulation_v0.1_aborted_threaded_warning_capture_20261007"
    )
    require(archived.exists(), "Aborted concurrent-run artifacts were not retained")
    checks = read_table("in_run_checks_v0.1.tsv")
    required_checks = checks.loc[~checks["check"].eq("convergence"), "passed"]
    require(prior.truth(required_checks).all(), "Required in-run check failed")
    print(
        f"Validated {risk_rows:,} risk rows, {len(fits)} fit records, "
        f"{len(manifest)} external artifacts, and all reviewed metrics."
    )


if __name__ == "__main__":
    main()
