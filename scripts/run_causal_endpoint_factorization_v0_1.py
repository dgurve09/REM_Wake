"""Run the frozen train-only causal endpoint factorization experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import run_causal_bidirectional_transition_v0_1 as causal
import run_full_night_risk_set_formulation_v0_1 as formulation
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import metric_values


# Section 1: frozen configuration

EXPERIMENT_DIR = "2026-10-15_causal_endpoint_factorization_v0.1"
DERIVED_DIR = "causal_endpoint_factorization_v0.1"
PROTOCOL_COMMIT = "a8544d8"
CANDIDATE = "CEF"
HEADS = ["rem_before", "wake_after"]
PRIMARY_BUDGET = 0.25
GRID_SEC = 15.0
C_VALUE = 0.1
MAX_ITER = 1000
BASE_SEED = 20261015
BOOTSTRAP_RESAMPLES = 2000


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return root().parent / "REM_W_data"


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root().as_posix()}", "rev-parse", "HEAD"],
        cwd=root(),
        text=True,
    ).strip()


def write_json(path: Path, value: dict) -> None:
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
        raise RuntimeError(f"Reviewed output changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def model_path(head: str, outer: int, phase: str) -> Path:
    return derived_dir() / "models" / head / f"outer_{outer}_{phase}.json.gz"


def outer_score_path() -> Path:
    return derived_dir() / "scores" / "cef_outer_oof.tsv.gz"


# Section 2: full-night hypnogram endpoint labels

def event_path(subject: str) -> Path:
    return (
        data_parent()
        / "boas_ds005555_v1.1.1"
        / subject
        / "eeg"
        / f"{subject}_task-Sleep_acq-psg_events.tsv"
    )


def endpoint_labels(subject: str, decision_times: np.ndarray) -> tuple[np.ndarray, ...]:
    events = pd.read_csv(
        event_path(subject), sep="\t", usecols=["onset", "duration", "stage_hum"]
    ).sort_values("onset")
    onsets = events.onset.to_numpy(dtype=float)
    durations = events.duration.to_numpy(dtype=float)
    stages = pd.to_numeric(events.stage_hum, errors="coerce").to_numpy(dtype=float)
    if not np.isclose(durations, 30.0).all() or np.any(np.diff(onsets) < 0):
        raise ValueError(f"Invalid hypnogram intervals: {subject}")

    boundaries = decision_times - 15.0
    before_index = np.searchsorted(onsets, boundaries, side="left") - 1
    after_index = np.searchsorted(onsets, boundaries, side="right") - 1
    before_safe = np.clip(before_index, 0, len(onsets) - 1)
    after_safe = np.clip(after_index, 0, len(onsets) - 1)

    before_valid = (
        (before_index >= 0)
        & (boundaries > onsets[before_safe] - 1e-9)
        & (boundaries <= onsets[before_safe] + durations[before_safe] + 1e-9)
        & np.isin(stages[before_safe], np.arange(5))
    )
    after_valid = (
        (after_index >= 0)
        & (boundaries >= onsets[after_safe] - 1e-9)
        & (boundaries < onsets[after_safe] + durations[after_safe] - 1e-9)
        & np.isin(stages[after_safe], np.arange(5))
    )
    supported = before_valid & after_valid
    before_stage = np.full(len(boundaries), -1, dtype=np.int8)
    after_stage = np.full(len(boundaries), -1, dtype=np.int8)
    before_stage[supported] = stages[before_safe[supported]].astype(np.int8)
    after_stage[supported] = stages[after_safe[supported]].astype(np.int8)
    rem_before = (before_stage == 4).astype(np.int8)
    wake_after = (after_stage == 0).astype(np.int8)
    return supported, before_stage, after_stage, rem_before, wake_after


def add_endpoint_labels(
    assignments: pd.DataFrame, recordings: dict
) -> tuple[pd.DataFrame, dict]:
    summary = []
    for item in assignments.itertuples(index=False):
        recording = recordings[item.subject]
        supported, before, after, rem, wake = endpoint_labels(
            item.subject, recording["times"]
        )
        recording.update(
            {
                "label_supported": supported,
                "before_stage": before,
                "after_stage": after,
                "rem_before_label": rem,
                "wake_after_label": wake,
            }
        )
        summary.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "grid_decisions": len(supported),
                "label_supported_decisions": int(supported.sum()),
                "rem_before_positive": int(rem[supported].sum()),
                "wake_after_positive": int(wake[supported].sum()),
                "rem_to_wake_pair_positive": int((rem[supported] & wake[supported]).sum()),
            }
        )
    return pd.DataFrame(summary), recordings


def full_training_arrays(recordings: dict) -> dict:
    values, pids, rem, wake = [], [], [], []
    for recording in recordings.values():
        supported = recording["label_supported"]
        values.append(recording["features"][supported])
        pids.append(np.full(int(supported.sum()), int(recording["pid"]), dtype=int))
        rem.append(recording["rem_before_label"][supported])
        wake.append(recording["wake_after_label"][supported])
    return {
        "values": np.concatenate(values).astype(np.float32, copy=False),
        "pids": np.concatenate(pids),
        "rem_before": np.concatenate(rem),
        "wake_after": np.concatenate(wake),
    }


# Section 3: event support and fitting

def prepare_references(assignments: pd.DataFrame, recordings: dict) -> pd.DataFrame:
    references = causal.load_references(assignments)
    supported_ids = []
    for event in references.itertuples(index=False):
        key = int(round(float(event.event_time_sec) * 2))
        available = {
            int(round(float(value) * 2)) for value in recordings[event.subject]["times"]
        }
        if key in available:
            supported_ids.append(int(event.transition_id))
    references["causal_primary_eligible"] = (
        prior.truth(references.primary_analysis_eligible)
        & references.transition_id.astype(int).isin(set(supported_ids))
    )
    return references


def fit_head(
    values: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=MAX_ITER,
        tol=1e-6,
        random_state=BASE_SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(values), labels)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    payload = {
        "configuration": {
            "C": C_VALUE,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "tol": 1e-6,
            "random_state": BASE_SEED,
        },
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
    }
    return {"scaler": scaler, "model": model}, payload


def fit_endpoint_models(
    outer: int,
    phase: str,
    fitting_pids: set[int],
    training: dict,
    feature_names: list[str],
) -> tuple[dict, list[dict]]:
    mask = np.isin(training["pids"], list(fitting_pids))
    values = training["values"][mask]
    fitted, rows = {}, []
    for head in HEADS:
        labels = training[head][mask]
        fitted[head], payload = fit_head(values, labels, feature_names)
        path = model_path(head, outer, phase)
        prior.verify_or_create_json_gzip(path, payload)
        rows.append(
            {
                "candidate": CANDIDATE,
                "head": head,
                "outer_fold": outer,
                "phase": phase,
                "training_rows": len(labels),
                "positive_rows": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "iterations": payload["iterations"],
                "converged": payload["converged"],
                "intercept": float(payload["intercept"][0]),
                "model_relative_path": path.relative_to(data_parent()).as_posix(),
                "model_bytes": path.stat().st_size,
                "model_sha256": prior.sha256(path),
            }
        )
    return fitted, rows


def score_recordings(
    fitted: dict,
    heldout: pd.DataFrame,
    recordings: dict,
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    frames, support, probability_rows = [], [], []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        head_probability = {}
        for head in HEADS:
            head_probability[head] = fitted[head]["model"].predict_proba(
                fitted[head]["scaler"].transform(recording["features"])
            )[:, 1]
        score = head_probability["rem_before"] * head_probability["wake_after"]
        frames.append(
            pd.DataFrame(
                {
                    "candidate": CANDIDATE,
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["times"],
                    "probability": score,
                    "rem_before_probability": head_probability["rem_before"],
                    "wake_after_probability": head_probability["wake_after"],
                    "label_supported": recording["label_supported"],
                    "rem_before_label": recording["rem_before_label"],
                    "wake_after_label": recording["wake_after_label"],
                }
            )
        )
        support.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(score),
                "supported_hours": len(score) * GRID_SEC / 3600.0,
            }
        )
        for head in HEADS:
            probability = head_probability[head]
            probability_rows.append(
                {
                    "candidate": CANDIDATE,
                    "head": head,
                    "outer_fold": outer,
                    "phase": phase,
                    "minimum_probability": float(probability.min()),
                    "maximum_probability": float(probability.max()),
                    "unique_probabilities": int(np.unique(probability).size),
                    "exact_endpoint_fraction": float(
                        np.mean((probability == 0.0) | (probability == 1.0))
                    ),
                    "all_probabilities_finite": bool(np.isfinite(probability).all()),
                }
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(support), probability_rows


# Section 4: participant-nested execution

def run_nested(
    assignments: pd.DataFrame,
    folds: pd.DataFrame,
    recordings: dict,
    training: dict,
    references: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    fold_by_pid = folds.set_index("pid").fold.astype(int).to_dict()
    fold_values = sorted(folds.fold.astype(int).unique())
    fit_rows, probability_rows, curve_frames, selections = [], [], [], []
    outer_scores, outer_support = [], []

    for outer in fold_values:
        inner_scores, inner_support = [], []
        for inner in [value for value in fold_values if value != outer]:
            fitting_pids = {
                int(pid) for pid, fold in fold_by_pid.items() if fold not in {outer, inner}
            }
            heldout = assignments[assignments.pid.map(fold_by_pid).eq(inner)]
            fitted, local_fits = fit_endpoint_models(
                outer, f"inner_{inner}", fitting_pids, training, feature_names
            )
            scores, support, local_probabilities = score_recordings(
                fitted, heldout, recordings, outer, f"inner_{inner}"
            )
            fit_rows.extend(local_fits)
            probability_rows.extend(local_probabilities)
            inner_scores.append(scores)
            inner_support.append(support)
            print(f"fit CEF outer={outer} phase=inner_{inner}", flush=True)

        curve = causal.threshold_curve(
            pd.concat(inner_scores, ignore_index=True),
            pd.concat(inner_support, ignore_index=True),
            references,
            "REM_to_Wake",
        )
        curve.insert(0, "candidate", CANDIDATE)
        curve.insert(1, "outer_fold", outer)
        curve_frames.append(curve)
        selections.append(
            {
                "candidate": CANDIDATE,
                "outer_fold": outer,
                "selection_scope": "REM_to_Wake",
                **formulation.select_threshold(curve, PRIMARY_BUDGET),
            }
        )

        fitting_pids = {int(pid) for pid, fold in fold_by_pid.items() if fold != outer}
        heldout = assignments[assignments.pid.map(fold_by_pid).eq(outer)]
        fitted, local_fits = fit_endpoint_models(
            outer, "outer_final", fitting_pids, training, feature_names
        )
        scores, support, local_probabilities = score_recordings(
            fitted, heldout, recordings, outer, "outer_final"
        )
        fit_rows.extend(local_fits)
        probability_rows.extend(local_probabilities)
        outer_scores.append(scores)
        outer_support.append(support)
        print(f"fit CEF outer={outer} phase=outer_final", flush=True)

    scores = pd.concat(outer_scores, ignore_index=True)
    support = pd.concat(outer_support, ignore_index=True)
    selection_table = pd.DataFrame(selections)
    alarms = []
    for item in selection_table.itertuples(index=False):
        local = scores[scores.outer_fold.eq(item.outer_fold)]
        predictions = causal.predictions_from_scores(local, float(item.threshold))
        predictions.insert(0, "candidate", CANDIDATE)
        predictions.insert(1, "outer_fold", int(item.outer_fold))
        predictions["threshold"] = float(item.threshold)
        alarms.append(predictions)
    return {
        "fits": pd.DataFrame(fit_rows),
        "probability_checks": pd.DataFrame(probability_rows),
        "curves": pd.concat(curve_frames, ignore_index=True),
        "selections": selection_table,
        "scores": scores,
        "support": support,
        "alarms": pd.concat(alarms, ignore_index=True),
    }


# Section 5: event and mechanism evaluation

def evaluate_outer(nested: dict, references: pd.DataFrame) -> dict[str, pd.DataFrame]:
    summary, participants, matches = causal.evaluate_predictions(
        nested["alarms"],
        nested["support"],
        references,
        "REM_to_Wake",
        return_details=True,
    )
    config = {
        "candidate": CANDIDATE,
        "scope": "REM_to_Wake",
        "partition": "train_nested_oof",
        "alarm_budget_per_hour": PRIMARY_BUDGET,
        "causal_match_start_sec": 0.0,
        "causal_match_stop_sec": 15.0,
    }
    metrics = pd.DataFrame([{**config, **summary}])
    for key, value in reversed(list(config.items())):
        participants.insert(0, key, value)
        if len(matches):
            matches.insert(0, key, value)
    return {"metrics": metrics, "participants": participants, "matches": matches}


def state_head_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    supported = scores[scores.label_supported.astype(bool)]
    rows = []
    for head in HEADS:
        labels = supported[f"{head}_label"].to_numpy(dtype=int)
        probability = supported[f"{head}_probability"].to_numpy(dtype=float)
        rows.append(
            {
                "candidate": CANDIDATE,
                "head": head,
                "partition": "train_outer_oof",
                "rows": len(labels),
                "positive_rows": int(labels.sum()),
                "positive_fraction": float(labels.mean()),
                "average_precision": float(average_precision_score(labels, probability)),
                "roc_auc": float(roc_auc_score(labels, probability)),
            }
        )
    return pd.DataFrame(rows)


def baseline_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    source = root() / "experiments/2026-10-11_causal_bidirectional_transition_v0.1"
    metrics = pd.read_csv(source / "outer_event_metrics_v0.1.tsv", sep="\t")
    participants = pd.read_csv(source / "outer_event_participants_v0.1.tsv", sep="\t")
    metrics = metrics[
        metrics.candidate.eq("CAUSAL-RW") & metrics.scope.eq("REM_to_Wake")
    ].copy()
    participants = participants[
        participants.candidate.eq("CAUSAL-RW") & participants.scope.eq("REM_to_Wake")
    ].copy()
    if len(metrics) != 1 or participants.pid.nunique() != 64:
        raise ValueError("Frozen CAUSAL-RW comparator is incomplete")
    return metrics, participants


def comparison_table(new_metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate", "reference_events", "predicted_events", "true_positive",
        "false_positive", "false_negative", "supported_hours", "precision", "recall",
        "f1", "false_alarms_per_hour",
    ]
    return pd.concat([baseline[columns], new_metrics[columns]], ignore_index=True)


def participant_bootstrap(new: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    paired = new[columns].merge(
        baseline[columns], on="pid", suffixes=("_cef", "_rw"), validate="one_to_one"
    ).sort_values("pid")
    if len(paired) != 64:
        raise ValueError("Incomplete paired participant comparison")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for indices in rng.integers(0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired))):
        sample = paired.iloc[indices]
        values = {}
        for side in ["cef", "rw"]:
            values[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()),
                int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()),
                float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "f1_difference": values["cef"]["f1"] - values["rw"]["f1"],
                "recall_difference": values["cef"]["recall"] - values["rw"]["recall"],
                "false_alarms_per_hour_difference": (
                    values["cef"]["false_alarms_per_hour"]
                    - values["rw"]["false_alarms_per_hour"]
                ),
            }
        )
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "comparison": "CEF_minus_CAUSAL-RW",
                "metric": metric,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BASE_SEED,
                "lower_95": float(frame[metric].quantile(0.025)),
                "median": float(frame[metric].quantile(0.5)),
                "upper_95": float(frame[metric].quantile(0.975)),
            }
            for metric in frame.columns
        ]
    )


def decision_table(nested: dict, comparison: pd.DataFrame) -> pd.DataFrame:
    by_name = comparison.set_index("candidate")
    f1_difference = float(by_name.loc[CANDIDATE, "f1"] - by_name.loc["CAUSAL-RW", "f1"])
    far_difference = float(
        by_name.loc[CANDIDATE, "false_alarms_per_hour"]
        - by_name.loc["CAUSAL-RW", "false_alarms_per_hour"]
    )
    fits = nested["fits"]
    checks = nested["probability_checks"]
    scores = nested["scores"].probability.to_numpy(dtype=float)
    numerical = bool(
        len(fits) == 50
        and fits.converged.all()
        and fits.iterations.lt(MAX_ITER).all()
        and checks.all_probabilities_finite.all()
        and checks.exact_endpoint_fraction.lt(0.01).all()
        and checks.unique_probabilities.gt(100).all()
        and np.isfinite(scores).all()
        and np.unique(scores).size > 100
        and np.mean((scores == 0.0) | (scores == 1.0)) < 0.01
    )
    point_gate = bool(f1_difference >= 0.03 and far_difference <= 0.05)
    decision = (
        "inconclusive_numerical_stop"
        if not numerical
        else "retain_for_new_confirmation" if point_gate else "stop_v0.1"
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-CEF",
                "comparison": "CEF_minus_CAUSAL-RW",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.03,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_false_alarm_difference": 0.05,
                "numerical_gate_pass": numerical,
                "point_gate_pass": numerical and point_gate,
                "decision": decision,
            }
        ]
    )


# Section 6: artifacts and execution

def manifest(fits: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = fits[
        ["candidate", "head", "outer_fold", "phase", "model_relative_path", "model_bytes", "model_sha256"]
    ].rename(
        columns={
            "model_relative_path": "relative_path",
            "model_bytes": "bytes",
            "model_sha256": "sha256",
        }
    )
    rows.insert(0, "artifact_type", "model")
    path = outer_score_path()
    prior.verify_or_create_gzip_tsv(path, scores)
    score_row = pd.DataFrame(
        [
            {
                "artifact_type": "outer_score",
                "candidate": CANDIDATE,
                "head": "combined",
                "outer_fold": "all",
                "phase": "outer_oof",
                "relative_path": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": prior.sha256(path),
            }
        ]
    )
    return pd.concat([rows, score_row], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output_dir().mkdir(parents=True, exist_ok=True)

    assignments = prior.train_assignments()
    folds = prior.frozen_fold_assignments(assignments)
    recordings, feature_summary, feature_names = causal.prepare_recordings(assignments)
    label_summary, recordings = add_endpoint_labels(assignments, recordings)
    references = prepare_references(assignments, recordings)
    supported_primary = int(references.causal_primary_eligible.sum())

    verify_or_create_tsv(feature_summary, output_dir() / "feature_grid_summary_v0.1.tsv")
    verify_or_create_tsv(label_summary, output_dir() / "endpoint_label_support_v0.1.tsv")
    audit = pd.DataFrame(
        [
            {"check": "train_recordings", "observed": len(assignments), "expected": 82, "passed": len(assignments) == 82},
            {"check": "train_pid", "observed": assignments.pid.nunique(), "expected": 64, "passed": assignments.pid.nunique() == 64},
            {"check": "causal_features", "observed": len(feature_names), "expected": 80, "passed": len(feature_names) == 80},
            {"check": "grid_decisions", "observed": int(label_summary.grid_decisions.sum()), "expected": 150967, "passed": int(label_summary.grid_decisions.sum()) == 150967},
            {"check": "future_signal_sec", "observed": float(feature_summary.future_signal_sec.max()), "expected": 0, "passed": feature_summary.future_signal_sec.eq(0).all()},
            {"check": "label_supported_positive", "observed": int(label_summary.label_supported_decisions.sum()), "expected": ">0", "passed": int(label_summary.label_supported_decisions.sum()) > 0},
            {"check": "rem_before_both_classes", "observed": int(label_summary.rem_before_positive.sum()), "expected": "between_0_and_supported", "passed": 0 < int(label_summary.rem_before_positive.sum()) < int(label_summary.label_supported_decisions.sum())},
            {"check": "wake_after_both_classes", "observed": int(label_summary.wake_after_positive.sum()), "expected": "between_0_and_supported", "passed": 0 < int(label_summary.wake_after_positive.sum()) < int(label_summary.label_supported_decisions.sum())},
            {"check": "supported_primary_events", "observed": supported_primary, "expected": 241, "passed": supported_primary == 241},
            {"check": "unauthorized_partitions", "observed": ";".join(sorted(assignments.partition.unique())), "expected": "train", "passed": set(assignments.partition) == {"train"}},
        ]
    )
    verify_or_create_tsv(audit, output_dir() / "pre_result_support_audit_v0.1.tsv")
    if not audit.passed.all():
        raise ValueError("Pre-result support audit failed")
    if args.audit_only:
        print("Pre-result support audit passed 10/10; no model was fitted.")
        print(label_summary.sum(numeric_only=True).to_string())
        return

    training = full_training_arrays(recordings)
    nested = run_nested(
        assignments, folds, recordings, training, references, feature_names
    )
    evaluated = evaluate_outer(nested, references)
    state_metrics = state_head_metrics(nested["scores"])
    baseline_metrics, baseline_participants = baseline_tables()
    comparison = comparison_table(evaluated["metrics"], baseline_metrics)
    bootstrap = participant_bootstrap(evaluated["participants"], baseline_participants)
    decision = decision_table(nested, comparison)
    artifacts = manifest(nested["fits"], nested["scores"])

    checks = pd.DataFrame(
        [
            {"check": "model_fit_rows", "observed": len(nested["fits"]), "expected": 50, "passed": len(nested["fits"]) == 50},
            {"check": "all_models_converged", "observed": bool(nested["fits"].converged.all()), "expected": True, "passed": bool(nested["fits"].converged.all())},
            {"check": "outer_score_rows", "observed": len(nested["scores"]), "expected": 150967, "passed": len(nested["scores"]) == 150967},
            {"check": "outer_pid", "observed": nested["scores"].pid.nunique(), "expected": 64, "passed": nested["scores"].pid.nunique() == 64},
            {"check": "outer_subjects", "observed": nested["scores"].subject.nunique(), "expected": 82, "passed": nested["scores"].subject.nunique() == 82},
            {"check": "outer_scores_finite", "observed": bool(np.isfinite(nested["scores"].probability).all()), "expected": True, "passed": bool(np.isfinite(nested["scores"].probability).all())},
            {"check": "baseline_pid", "observed": baseline_participants.pid.nunique(), "expected": 64, "passed": baseline_participants.pid.nunique() == 64},
            {"check": "bootstrap_rows", "observed": len(bootstrap), "expected": 3, "passed": len(bootstrap) == 3},
            {"check": "validation_or_test_opened", "observed": False, "expected": False, "passed": True},
        ]
    )
    if not checks.passed.all():
        raise ValueError("In-run controls failed")

    outputs = {
        "model_fit_summary_v0.1.tsv": nested["fits"],
        "heldout_probability_checks_v0.1.tsv": nested["probability_checks"],
        "inner_threshold_curves_v0.1.tsv": nested["curves"],
        "inner_threshold_selections_v0.1.tsv": nested["selections"],
        "outer_support_v0.1.tsv": nested["support"],
        "outer_predicted_events_v0.1.tsv": nested["alarms"],
        "outer_event_metrics_v0.1.tsv": evaluated["metrics"],
        "outer_event_participants_v0.1.tsv": evaluated["participants"],
        "outer_event_matches_v0.1.tsv": evaluated["matches"],
        "state_head_oof_metrics_v0.1.tsv": state_metrics,
        "causal_rw_comparison_v0.1.tsv": comparison,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decision_v0.1.tsv": decision,
        "external_artifact_manifest_v0.1.tsv": artifacts,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    write_json(
        output_dir() / "software_versions_v0.1.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "protocol_commit": PROTOCOL_COMMIT,
            "execution_commit": current_commit(),
        },
    )
    print(comparison.to_string(index=False))
    print(state_metrics.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
