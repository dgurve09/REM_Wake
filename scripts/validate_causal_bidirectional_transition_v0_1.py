"""Validate the reviewed causal bidirectional transition outputs read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score

import run_causal_bidirectional_transition_v0_1 as experiment
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


def stage_b_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    actual = scores.transition_type.eq("REM_to_Wake").to_numpy(dtype=int)
    predicted = scores.predicted_transition_type.eq("REM_to_Wake").to_numpy(dtype=int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    recalls = recall_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    return pd.DataFrame(
        [
            {
                "partition": "train_outer_oof_oracle_events",
                "events": len(scores),
                "rem_to_wake_events": int(actual.sum()),
                "wake_to_rem_events": int((actual == 0).sum()),
                "macro_f1": float(f1_score(actual, predicted, average="macro")),
                "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
                "rem_to_wake_recall": float(recalls[1]),
                "wake_to_rem_recall": float(recalls[0]),
                "wake_to_rem_predicted_wake_to_rem": int(matrix[0, 0]),
                "wake_to_rem_predicted_rem_to_wake": int(matrix[0, 1]),
                "rem_to_wake_predicted_wake_to_rem": int(matrix[1, 0]),
                "rem_to_wake_predicted_rem_to_wake": int(matrix[1, 1]),
            }
        ]
    )


def validate(result_code_commit: str) -> pd.DataFrame:
    versions = json.loads(
        (experiment.output_dir() / "software_versions_v0.1.json").read_text(encoding="utf-8")
    )
    feature_summary = read_table("feature_grid_summary_v0.1.tsv")
    target_summary = read_table("target_construction_summary_v0.1.tsv")
    event_support = read_table("primary_event_causal_support_v0.1.tsv")
    support_audit = read_table("pre_result_support_audit_v0.1.tsv")
    in_run_checks = read_table("in_run_checks_v0.1.tsv")
    fits = read_table("model_fit_summary_v0.1.tsv")
    selections = read_table("inner_threshold_selections_v0.1.tsv")
    support = read_table("outer_support_v0.1.tsv")
    stored_alarms = read_table("outer_predicted_events_v0.1.tsv")
    stored_metrics = read_table("outer_event_metrics_v0.1.tsv")
    stored_participants = read_table("outer_event_participants_v0.1.tsv")
    stored_matches = read_table("outer_event_matches_v0.1.tsv")
    stored_stage_b_scores = read_table("stage_b_direction_scores_v0.1.tsv")
    stored_stage_b_metrics = read_table("stage_b_direction_metrics_v0.1.tsv")
    stored_bootstrap = read_table("paired_participant_bootstrap_v0.1.tsv")
    stored_decision = read_table("hypothesis_decisions_v0.1.tsv")
    manifest = read_table("external_artifact_manifest_v0.1.tsv")

    checks = []
    commit_pass = (
        versions["execution_commit"].startswith(result_code_commit)
        and versions["protocol_commit"].startswith(experiment.PROTOCOL_COMMIT)
    )
    checks.append(("frozen_commits", commit_pass, versions["execution_commit"][:12]))

    feature_pass = (
        len(feature_summary) == 82
        and feature_summary.pid.nunique() == 64
        and int(feature_summary.grid_decisions.sum()) == 150967
        and feature_summary.future_signal_sec.eq(0).all()
        and feature_summary.context_sec.eq(240).all()
    )
    checks.append(("causal_feature_grid", feature_pass, "150,967 decisions; zero future signal"))

    targets = target_summary.set_index("candidate")
    target_pass = (
        int(targets.loc["CAUSAL-RW", "positive_rows"]) == 180
        and int(targets.loc["CAUSAL-BI", "positive_rows"]) == 241
        and target_summary.negative_rows.eq(146987).all()
        and event_support.causal_context_supported.astype(bool).sum() == 241
        and event_support.loc[~event_support.causal_context_supported.astype(bool), "transition_id"].astype(int).tolist() == [406]
    )
    checks.append(("target_accounting", target_pass, "180 R-W; 61 W-R; one support exclusion"))

    audit_pass = len(support_audit) == 8 and prior.truth(support_audit.passed).all()
    checks.append(("pre_result_audit", audit_pass, "8/8 support controls"))

    in_run_pass = len(in_run_checks) == 15 and prior.truth(in_run_checks.passed).all()
    checks.append(("in_run_controls", in_run_pass, "15/15 execution controls"))

    stage_a = fits[fits.stage.eq("A")]
    stage_b = fits[fits.stage.eq("B")]
    fit_pass = (
        len(stage_a) == 50
        and stage_a.groupby("candidate").size().eq(25).all()
        and len(stage_b) == 5
        and prior.truth(fits.converged).all()
        and prior.truth(fits.all_values_finite).all()
        and fits.iterations.lt(experiment.MAX_ITER).all()
        and stage_a.exact_endpoint_fraction.lt(0.01).all()
        and stage_a.unique_probabilities.gt(100).all()
    )
    checks.append(("fit_and_numerical_gate", fit_pass, "55 fits; Stage A gate passed"))

    selection_pass = (
        len(selections) == 10
        and selections.groupby("candidate").size().eq(5).all()
        and selections.false_alarms_per_hour.le(experiment.PRIMARY_BUDGET + 1e-12).all()
    )
    checks.append(("nested_thresholds", selection_pass, "10 inner-selected thresholds"))

    support_pass = (
        len(support) == 82
        and support.pid.nunique() == 64
        and int(support.supported_boundaries.sum()) == 150967
    )
    checks.append(("outer_support", support_pass, "82 recordings; 64 participants"))

    score_frames = []
    for row in manifest[manifest.artifact_type.eq("outer_score")].itertuples(index=False):
        score_frames.append(
            pd.read_csv(experiment.data_parent() / row.relative_path, sep="\t", compression="gzip")
        )
    outer_scores = pd.concat(score_frames, ignore_index=True)
    alarm_frames = []
    for item in selections.itertuples(index=False):
        local_scores = outer_scores[
            outer_scores.candidate.eq(item.candidate)
            & outer_scores.outer_fold.eq(item.outer_fold)
        ]
        predictions = experiment.predictions_from_scores(local_scores, float(item.threshold))
        predictions.insert(0, "candidate", item.candidate)
        predictions.insert(1, "outer_fold", int(item.outer_fold))
        predictions["threshold"] = float(item.threshold)
        alarm_frames.append(predictions)
    reconstructed_alarms = pd.concat(alarm_frames, ignore_index=True)
    alarm_pass = same_frame(
        reconstructed_alarms,
        stored_alarms,
        ["candidate", "outer_fold", "subject", "event_time_sec"],
    )
    checks.append(("alarm_reconstruction", alarm_pass, "outer alarms from external scores"))

    assignments = prior.train_assignments()
    references = experiment.load_references(assignments)
    supported_ids = set(
        event_support.loc[event_support.causal_context_supported.astype(bool), "transition_id"].astype(int)
    )
    references["causal_primary_eligible"] = (
        prior.truth(references.primary_analysis_eligible)
        & references.transition_id.astype(int).isin(supported_ids)
    )
    reconstructed = experiment.evaluate_stage_a(
        {"alarms": reconstructed_alarms, "support": support}, references
    )
    metric_pass = same_frame(
        reconstructed["metrics"], stored_metrics, ["candidate", "scope"]
    )
    participant_pass = same_frame(
        reconstructed["participants"], stored_participants, ["candidate", "scope", "pid"]
    )
    match_pass = same_frame(
        reconstructed["matches"],
        stored_matches,
        ["candidate", "scope", "subject", "match_type", "reference_time_sec"],
    )
    checks.append(("event_reconstruction", metric_pass and participant_pass and match_pass, "metrics, participants, matches"))

    reconstructed_stage_b = stage_b_metrics(stored_stage_b_scores)
    stage_b_pass = same_frame(
        reconstructed_stage_b, stored_stage_b_metrics, ["partition"]
    )
    checks.append(("direction_reconstruction", stage_b_pass, "oracle-event direction metrics"))

    reconstructed_bootstrap = experiment.participant_bootstrap(reconstructed["participants"])
    bootstrap_pass = same_frame(
        reconstructed_bootstrap, stored_bootstrap, ["comparison", "metric"]
    )
    reconstructed_decision = experiment.decision_table(reconstructed["metrics"], fits)
    decision_pass = same_frame(reconstructed_decision, stored_decision, ["hypothesis"])
    checks.append(("bootstrap_and_decision", bootstrap_pass and decision_pass, "2,000 resamples and H-BI gate"))

    hash_pass = True
    for row in manifest.itertuples(index=False):
        path = experiment.data_parent() / row.relative_path
        hash_pass &= path.exists()
        if path.exists():
            hash_pass &= path.stat().st_size == int(row.bytes)
            hash_pass &= prior.sha256(path) == row.sha256
    checks.append(("external_hashes", hash_pass, f"{len(manifest)} external artifacts"))

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
        if args.report.exists() and args.report.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
            raise RuntimeError("Validation report changed")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
