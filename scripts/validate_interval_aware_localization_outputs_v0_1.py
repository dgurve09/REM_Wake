"""Independently validate interval-aware localization experiment v0.1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_interval_aware_localization_v0_1 as experiment
import run_paired_enriched_feature_pruning_v0_1 as prior


def read_table(name: str) -> pd.DataFrame:
    path = experiment.output_dir() / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, sep="\t")


def same_frame(left: pd.DataFrame, right: pd.DataFrame, sort_columns: list[str]) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.sort_values(sort_columns).reset_index(drop=True),
            right.sort_values(sort_columns).reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
        return True
    except AssertionError:
        return False


def validate(result_code_commit: str) -> pd.DataFrame:
    versions = json.loads(
        (experiment.output_dir() / "software_versions_v0.1.json").read_text(
            encoding="utf-8"
        )
    )
    feature_summary = read_table("feature_grid_summary_v0.1.tsv")
    targets = read_table("target_construction_summary_v0.1.tsv")
    fits = read_table("model_fit_summary_v0.1.tsv")
    selections = read_table("inner_threshold_selections_v0.1.tsv")
    support = read_table("outer_support_v0.1.tsv")
    events = read_table("outer_predicted_events_v0.1.tsv")
    stored_metrics = read_table("outer_event_metrics_v0.1.tsv")
    stored_participants = read_table("outer_event_participants_v0.1.tsv")
    stored_matches = read_table("outer_event_matches_v0.1.tsv")
    stored_bootstrap = read_table("paired_participant_bootstrap_v0.1.tsv")
    stored_localization = read_table("primary_localization_summary_v0.1.tsv")
    stored_decisions = read_table("hypothesis_decisions_v0.1.tsv")
    manifest = read_table("external_artifact_manifest_v0.1.tsv")
    in_run_checks = read_table("in_run_checks_v0.1.tsv")

    checks = []
    version_pass = (
        versions["git_commit"].startswith(result_code_commit)
        and versions["protocol_commit"].startswith(experiment.PROTOCOL_COMMIT)
        and versions["protocol_amendment_commit"].startswith(experiment.AMENDMENT_COMMIT)
    )
    checks.append(("frozen_commits", version_pass, versions["git_commit"][:12]))

    feature_pass = (
        len(feature_summary) == 82
        and feature_summary.pid.nunique() == 64
        and int(feature_summary.base_supported_boundaries.sum()) == 75539
        and int(feature_summary.shifted_supported_boundaries.sum()) == 75428
        and int(feature_summary.grid_supported_boundaries.sum()) == 150967
    )
    checks.append(("feature_grid", feature_pass, "150,967 supported 15-second boundaries"))

    target_index = targets.set_index("candidate")
    target_pass = (
        set(targets.candidate) == set(experiment.CANDIDATES)
        and targets.negative_rows.eq(147055).all()
        and int(target_index.loc["HARD15", "positive_rows"]) == 180
        and int(target_index.loc["INT15", "positive_rows"]) == 540
        and np.allclose(targets.positive_weight, 180.0)
    )
    checks.append(("target_accounting", target_pass, "equal positive mass and shared negatives"))

    expected_convergence = fits.iterations.astype(int) < experiment.MAX_ITER
    convergence_matches = np.array_equal(
        prior.truth(fits.converged).to_numpy(), expected_convergence.to_numpy()
    )
    shared_preprocessing = (
        fits.groupby(["outer_fold", "phase"]).preprocessing_signature.nunique().eq(1).all()
    )
    fit_pass = (
        len(fits) == 50
        and fits.groupby("candidate").size().eq(25).all()
        and convergence_matches
        and shared_preprocessing
    )
    checks.append(("fit_accounting", fit_pass, "50 fits with paired preprocessing"))

    selection_pass = (
        len(selections) == 40
        and set(np.round(selections.alarm_budget_per_hour, 2))
        == set(experiment.ALARM_BUDGETS)
        and (
            selections.false_alarms_per_hour
            <= selections.alarm_budget_per_hour + 1e-12
        ).all()
    )
    checks.append(("nested_thresholds", selection_pass, "40 budget-constrained selections"))

    support_pass = (
        len(support) == 82
        and support.pid.nunique() == 64
        and int(support.supported_boundaries.sum()) == 150967
    )
    checks.append(("outer_support", support_pass, "82 recordings; 64 participants"))

    assignments = prior.train_assignments()
    references = prior.reference_events(assignments)
    reconstructed = experiment.evaluate_all(events, support, references)
    metric_pass = same_frame(
        reconstructed["metrics"],
        stored_metrics,
        ["candidate", "membership", "tolerance_sec", "alarm_budget_per_hour"],
    )
    participant_pass = same_frame(
        reconstructed["participants"],
        stored_participants,
        ["candidate", "membership", "tolerance_sec", "alarm_budget_per_hour", "pid"],
    )
    match_pass = same_frame(
        reconstructed["matches"],
        stored_matches,
        ["candidate", "membership", "tolerance_sec", "alarm_budget_per_hour", "subject", "reference_time_sec"],
    )
    checks.append(
        ("event_reconstruction", metric_pass and participant_pass and match_pass, "metrics, participant rows, and matches")
    )

    reconstructed_bootstrap = experiment.event_bootstrap(reconstructed["participants"])
    bootstrap_pass = same_frame(
        reconstructed_bootstrap, stored_bootstrap, ["comparison", "metric"]
    )
    checks.append(("bootstrap_reconstruction", bootstrap_pass, "2,000 paired participant resamples"))

    reconstructed_localization = experiment.localization_summary(reconstructed["matches"])
    localization_pass = same_frame(
        reconstructed_localization, stored_localization, ["candidate"]
    )
    checks.append(("localization_reconstruction", localization_pass, "primary +/-15-second offsets"))

    reconstructed_decisions = experiment.decision_table(
        reconstructed["metrics"], reconstructed_bootstrap, fits
    )
    decision_pass = same_frame(
        reconstructed_decisions, stored_decisions, ["hypothesis"]
    )
    checks.append(("decision_reconstruction", decision_pass, "frozen H-INT gate"))

    hash_pass = True
    for row in manifest.itertuples(index=False):
        path = experiment.data_parent() / row.relative_path
        hash_pass &= path.exists()
        if path.exists():
            hash_pass &= path.stat().st_size == int(row.bytes)
            hash_pass &= prior.sha256(path) == row.sha256
    checks.append(("external_hashes", hash_pass, f"{len(manifest)} external artifacts"))

    required_in_run = in_run_checks.loc[
        ~in_run_checks.check.eq("convergence"), "passed"
    ]
    checks.append(
        (
            "reviewed_in_run_checks",
            prior.truth(required_in_run).all(),
            f"{prior.truth(required_in_run).sum()}/{len(required_in_run)} required controls passed",
        )
    )

    result = pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in checks
        ]
    )
    if not result.status.eq("pass").all():
        raise ValueError("Validation failed:\n" + result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.result_code_commit)
    print(result.to_string(index=False))
    if args.report:
        expected = result.to_csv(sep="\t", index=False, lineterminator="\n")
        if args.report.exists() and args.report.read_text(encoding="utf-8") != expected:
            raise RuntimeError("Validation report changed")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
