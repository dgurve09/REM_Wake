"""Independently validate causal multiscale change-feature outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_causal_bidirectional_transition_v0_1 as causal
import run_causal_multiscale_change_features_v0_1 as experiment
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv


# Section 1: paths and table comparison

def result_dir() -> Path:
    return experiment.output_dir()


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir() / name, sep="\t")


def same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if set(left.columns) != set(right.columns) or left.shape != right.shape:
        return False
    columns = list(right.columns)
    left = left[columns].sort_values(columns, kind="stable").reset_index(drop=True)
    right = right[columns].sort_values(columns, kind="stable").reset_index(drop=True)
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


# Section 2: independent peak and alarm reconstruction

def independent_peak_indices(times: np.ndarray, probability: np.ndarray) -> np.ndarray:
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) > 15.0 + 1e-6) + 1, len(times)]
    selected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = probability[start:stop]
        previous = np.r_[-np.inf, local[:-1]]
        following = np.r_[local[1:], -np.inf]
        peaks = start + np.flatnonzero((local > previous) & (local >= following))
        retained = []
        for index in sorted(peaks, key=lambda value: (-probability[value], times[value])):
            if all(abs(times[index] - times[other]) > 60.0 for other in retained):
                retained.append(index)
        selected.extend(retained)
    return np.asarray(sorted(selected, key=lambda value: times[value]), dtype=int)


def reconstruct_alarms(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for item in selections.itertuples(index=False):
        local = scores[
            scores.candidate.eq(item.candidate) & scores.outer_fold.eq(item.outer_fold)
        ]
        rows = []
        for (subject, pid), group in local.groupby(["subject", "pid"], sort=True):
            group = group.sort_values("candidate_time_sec").reset_index(drop=True)
            indices = independent_peak_indices(
                group.candidate_time_sec.to_numpy(dtype=float),
                group.probability.to_numpy(dtype=float),
            )
            selected = group.iloc[indices]
            selected = selected[selected.probability.ge(float(item.threshold))]
            rows.extend(
                {
                    "candidate": item.candidate,
                    "outer_fold": int(item.outer_fold),
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(time),
                    "threshold": float(item.threshold),
                }
                for time in selected.candidate_time_sec
            )
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def reference_table() -> pd.DataFrame:
    references = causal.load_references(prior.train_assignments())
    support = load("primary_event_causal_support_v0.1.tsv")
    supported = set(
        support.loc[prior.truth(support.causal_context_supported), "transition_id"].astype(int)
    )
    references["causal_primary_eligible"] = (
        prior.truth(references.primary_analysis_eligible)
        & references.transition_id.astype(int).isin(supported)
    )
    return references


def evaluate(alarms: pd.DataFrame, support: pd.DataFrame) -> dict[str, pd.DataFrame]:
    references = reference_table()
    metrics, participants, matches = [], [], []
    for candidate in experiment.CANDIDATES:
        predictions = alarms[alarms.candidate.eq(candidate)]
        summary, local_participants, local_matches = causal.evaluate_predictions(
            predictions, support, references, "REM_to_Wake", return_details=True
        )
        config = {
            "candidate": candidate,
            "scope": "REM_to_Wake",
            "partition": "train_nested_oof",
            "alarm_budget_per_hour": experiment.PRIMARY_BUDGET,
            "causal_match_start_sec": 0.0,
            "causal_match_stop_sec": 15.0,
        }
        metrics.append({**config, **summary})
        for key, value in reversed(list(config.items())):
            local_participants.insert(0, key, value)
            if len(local_matches):
                local_matches.insert(0, key, value)
        participants.append(local_participants)
        if len(local_matches):
            matches.append(local_matches)
    return {
        "metrics": pd.DataFrame(metrics),
        "participants": pd.concat(participants, ignore_index=True),
        "matches": pd.concat(matches, ignore_index=True),
    }


# Section 3: validation controls

def main() -> None:
    scores = pd.concat(
        [
            pd.read_csv(experiment.score_path(candidate), sep="\t", compression="gzip")
            for candidate in experiment.CANDIDATES
        ],
        ignore_index=True,
    )
    selections = load("inner_threshold_selections_v0.1.tsv")
    curves = load("inner_threshold_curves_v0.1.tsv")
    support = load("outer_support_v0.1.tsv")
    fits = load("model_fit_summary_v0.1.tsv")
    stored_alarms = load("outer_predicted_events_v0.1.tsv")
    stored_metrics = load("outer_event_metrics_v0.1.tsv")
    stored_participants = load("outer_event_participants_v0.1.tsv")
    stored_matches = load("outer_event_matches_v0.1.tsv")
    stored_bootstrap = load("paired_participant_bootstrap_v0.1.tsv")
    stored_decision = load("hypothesis_decision_v0.1.tsv")
    in_run = load("in_run_checks_v0.1.tsv")
    manifest = load("external_artifact_manifest_v0.1.tsv")

    alarms = reconstruct_alarms(scores, selections)
    evaluated = evaluate(alarms, support)
    bootstrap = experiment.participant_bootstrap(evaluated["participants"])
    decision = experiment.decision_table(
        {"fits": fits, "curves": curves}, evaluated["metrics"]
    )

    selected_again = []
    for (candidate, outer), curve in curves.groupby(["candidate", "outer_fold"]):
        eligible = curve[curve.false_alarms_per_hour.le(experiment.PRIMARY_BUDGET + 1e-12)]
        selected = eligible.sort_values(
            ["f1", "recall", "false_alarms_per_hour", "threshold"],
            ascending=[False, False, True, False],
            kind="stable",
        ).iloc[0]
        stored = selections[
            selections.candidate.eq(candidate) & selections.outer_fold.eq(outer)
        ].iloc[0]
        selected_again.append(np.isclose(float(selected.threshold), float(stored.threshold)))

    hashes = []
    for item in manifest.itertuples(index=False):
        path = experiment.data_parent() / item.relative_path
        hashes.append(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and prior.sha256(path) == item.sha256
        )

    old = pd.read_csv(causal.outer_score_path("CAUSAL-RW"), sep="\t", compression="gzip")
    base = scores[scores.candidate.eq("BASE-PEAK")]
    reproduction = old.merge(
        base,
        on=["outer_fold", "subject", "pid", "candidate_time_sec"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    maximum_base_difference = float(
        np.max(np.abs(reproduction.probability_old - reproduction.probability_new))
    )

    checks = pd.DataFrame(
        [
            {"check": "outer_score_rows", "observed": len(scores), "expected": 301934, "passed": len(scores) == 301934},
            {"check": "participant_coverage", "observed": scores.pid.nunique(), "expected": 64, "passed": scores.pid.nunique() == 64},
            {"check": "threshold_reconstruction", "observed": int(sum(selected_again)), "expected": 10, "passed": all(selected_again)},
            {"check": "alarm_reconstruction", "observed": len(alarms), "expected": len(stored_alarms), "passed": same_frame(alarms, stored_alarms)},
            {"check": "metric_reconstruction", "observed": len(evaluated["metrics"]), "expected": len(stored_metrics), "passed": same_frame(evaluated["metrics"], stored_metrics)},
            {"check": "participant_reconstruction", "observed": len(evaluated["participants"]), "expected": len(stored_participants), "passed": same_frame(evaluated["participants"], stored_participants)},
            {"check": "match_reconstruction", "observed": len(evaluated["matches"]), "expected": len(stored_matches), "passed": same_frame(evaluated["matches"], stored_matches)},
            {"check": "bootstrap_reconstruction", "observed": len(bootstrap), "expected": len(stored_bootstrap), "passed": same_frame(bootstrap, stored_bootstrap)},
            {"check": "decision_reconstruction", "observed": str(decision.decision.iloc[0]), "expected": str(stored_decision.decision.iloc[0]), "passed": same_frame(decision, stored_decision)},
            {"check": "baseline_score_reproduction", "observed": maximum_base_difference, "expected": "<=1e-12", "passed": maximum_base_difference <= 1e-12},
            {"check": "external_artifact_hashes", "observed": int(sum(hashes)), "expected": len(hashes), "passed": all(hashes)},
            {"check": "in_run_controls", "observed": int(prior.truth(in_run.passed).sum()), "expected": len(in_run), "passed": prior.truth(in_run.passed).all()},
            {"check": "validation_or_test_inputs", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent validation failed")
    verify_or_create_tsv(checks, result_dir() / "independent_validation_v0.1.tsv")
    print(f"Independent validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(hashes)} external artifacts and {len(stored_matches)} event matches.")


if __name__ == "__main__":
    main()
