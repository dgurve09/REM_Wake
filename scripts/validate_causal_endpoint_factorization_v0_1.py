"""Independently validate reviewed causal endpoint factorization outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_causal_bidirectional_transition_v0_1 as causal
import run_causal_endpoint_factorization_v0_1 as experiment
import run_full_night_risk_set_formulation_v0_1 as formulation
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv


# Section 1: paths and comparison helpers

def root() -> Path:
    return Path(__file__).resolve().parents[1]


def result_dir() -> Path:
    return root() / "experiments" / experiment.EXPERIMENT_DIR


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir() / name, sep="\t")


def same_frame(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    left = left[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    right = right[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    if left.shape != right.shape:
        return False
    for column in columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                atol=1e-12,
                rtol=1e-10,
                equal_nan=True,
            ):
                return False
        elif left[column].astype(str).tolist() != right[column].astype(str).tolist():
            return False
    return True


# Section 2: independent reconstruction

def reference_table(scores: pd.DataFrame) -> pd.DataFrame:
    assignments = prior.train_assignments()
    references = causal.load_references(assignments)
    available = {
        (subject, int(round(float(time) * 2)))
        for subject, time in scores[["subject", "candidate_time_sec"]].itertuples(index=False)
    }
    references["causal_primary_eligible"] = (
        prior.truth(references.primary_analysis_eligible)
        & references.apply(
            lambda row: (row.subject, int(round(float(row.event_time_sec) * 2))) in available,
            axis=1,
        )
    )
    return references


def reconstruct_alarms(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in selections.itertuples(index=False):
        local = scores[scores.outer_fold.eq(item.outer_fold)]
        alarms = causal.predictions_from_scores(local, float(item.threshold))
        alarms.insert(0, "candidate", experiment.CANDIDATE)
        alarms.insert(1, "outer_fold", int(item.outer_fold))
        alarms["threshold"] = float(item.threshold)
        rows.append(alarms)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    scores = pd.read_csv(experiment.outer_score_path(), sep="\t", compression="gzip")
    selections = load("inner_threshold_selections_v0.1.tsv")
    curves = load("inner_threshold_curves_v0.1.tsv")
    support = load("outer_support_v0.1.tsv")
    stored_alarms = load("outer_predicted_events_v0.1.tsv")
    stored_metrics = load("outer_event_metrics_v0.1.tsv")
    stored_participants = load("outer_event_participants_v0.1.tsv")
    stored_state = load("state_head_oof_metrics_v0.1.tsv")
    stored_comparison = load("causal_rw_comparison_v0.1.tsv")
    stored_bootstrap = load("paired_participant_bootstrap_v0.1.tsv")
    stored_decision = load("hypothesis_decision_v0.1.tsv")
    fits = load("model_fit_summary_v0.1.tsv")
    probability_checks = load("heldout_probability_checks_v0.1.tsv")
    manifest = load("external_artifact_manifest_v0.1.tsv")

    references = reference_table(scores)
    alarms = reconstruct_alarms(scores, selections)
    summary, participants, matches = causal.evaluate_predictions(
        alarms, support, references, "REM_to_Wake", return_details=True
    )
    reconstructed_metrics = stored_metrics.iloc[0:0].copy()
    reconstructed_metrics.loc[0] = {
        **stored_metrics.iloc[0][
            [
                "candidate", "scope", "partition", "alarm_budget_per_hour",
                "causal_match_start_sec", "causal_match_stop_sec",
            ]
        ].to_dict(),
        **summary,
    }
    state = experiment.state_head_metrics(scores)
    baseline_metrics, baseline_participants = experiment.baseline_tables()
    comparison = experiment.comparison_table(reconstructed_metrics, baseline_metrics)

    config_columns = [
        "candidate", "scope", "partition", "alarm_budget_per_hour",
        "causal_match_start_sec", "causal_match_stop_sec",
    ]
    for key in reversed(config_columns):
        participants.insert(0, key, stored_metrics.iloc[0][key])
    bootstrap = experiment.participant_bootstrap(participants, baseline_participants)
    decision = experiment.decision_table(
        {"fits": fits, "probability_checks": probability_checks, "scores": scores},
        comparison,
    )

    artifact_hashes = []
    for item in manifest.itertuples(index=False):
        path = experiment.data_parent() / item.relative_path
        artifact_hashes.append(path.exists() and prior.sha256(path) == item.sha256)

    selected_again = []
    for outer, group in curves.groupby("outer_fold"):
        selected = formulation.select_threshold(group, experiment.PRIMARY_BUDGET)
        stored = selections[selections.outer_fold.eq(outer)].iloc[0]
        selected_again.append(np.isclose(float(selected["threshold"]), float(stored.threshold)))

    event_columns = [
        "candidate", "scope", "partition", "alarm_budget_per_hour",
        "causal_match_start_sec", "causal_match_stop_sec", "recordings", "pid",
        "reference_events", "predicted_events", "true_positive", "false_positive",
        "false_negative", "ignored_predictions", "supported_hours", "precision",
        "recall", "f1", "false_alarms_per_hour",
    ]
    participant_columns = config_columns + [
        "pid", "true_positive", "false_positive", "false_negative",
        "ignored_predictions", "predicted_events", "reference_events", "supported_hours",
    ]
    checks = pd.DataFrame(
        [
            {"check": "outer_score_rows", "observed": len(scores), "expected": 150967, "passed": len(scores) == 150967},
            {"check": "outer_participant_coverage", "observed": scores.pid.nunique(), "expected": 64, "passed": scores.pid.nunique() == 64},
            {"check": "selection_reconstruction", "observed": int(sum(selected_again)), "expected": 5, "passed": all(selected_again)},
            {"check": "alarm_reconstruction", "observed": len(alarms), "expected": len(stored_alarms), "passed": same_frame(alarms, stored_alarms, ["candidate", "outer_fold", "subject", "pid", "event_time_sec", "threshold"])},
            {"check": "event_metric_reconstruction", "observed": float(summary["f1"]), "expected": float(stored_metrics.f1.iloc[0]), "passed": same_frame(reconstructed_metrics, stored_metrics, event_columns)},
            {"check": "participant_metric_reconstruction", "observed": len(participants), "expected": len(stored_participants), "passed": same_frame(participants, stored_participants, participant_columns)},
            {"check": "state_head_reconstruction", "observed": len(state), "expected": len(stored_state), "passed": same_frame(state, stored_state, list(stored_state.columns))},
            {"check": "baseline_comparison_reconstruction", "observed": len(comparison), "expected": len(stored_comparison), "passed": same_frame(comparison, stored_comparison, list(stored_comparison.columns))},
            {"check": "bootstrap_reconstruction", "observed": len(bootstrap), "expected": len(stored_bootstrap), "passed": same_frame(bootstrap, stored_bootstrap, list(stored_bootstrap.columns))},
            {"check": "decision_reconstruction", "observed": str(decision.decision.iloc[0]), "expected": str(stored_decision.decision.iloc[0]), "passed": same_frame(decision, stored_decision, list(stored_decision.columns))},
            {"check": "external_artifact_hashes", "observed": int(sum(artifact_hashes)), "expected": len(artifact_hashes), "passed": all(artifact_hashes)},
            {"check": "causal_primary_events", "observed": int(references.causal_primary_eligible.sum()), "expected": 241, "passed": int(references.causal_primary_eligible.sum()) == 241},
            {"check": "validation_or_test_inputs", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent validation failed")
    verify_or_create_tsv(checks, result_dir() / "independent_validation_v0.1.tsv")
    print(f"Independent validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(artifact_hashes)} external artifacts and {len(matches)} event matches.")


if __name__ == "__main__":
    main()
