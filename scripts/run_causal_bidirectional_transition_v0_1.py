"""Run the frozen train-only causal REM/Wake transition experiment."""

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
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

import run_full_night_risk_set_formulation_v0_1 as formulation
import run_interval_aware_localization_v0_1 as interval
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import metric_values


# Section 1: frozen configuration

EXPERIMENT_DIR = "2026-10-11_causal_bidirectional_transition_v0.1"
DERIVED_DIR = "causal_bidirectional_transition_v0.1"
PROTOCOL_COMMIT = "481e525"
CANDIDATES = ["CAUSAL-RW", "CAUSAL-BI"]
PRIMARY_BUDGET = 0.25
EXCLUSION_SEC = 135.0
GRID_SEC = 15.0
C_VALUE = 0.1
MAX_ITER = 1000
BASE_SEED = 20261011
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


def model_path(stage: str, name: str, outer: int, phase: str) -> Path:
    safe = name.lower().replace("-", "_")
    return derived_dir() / "models" / stage / safe / f"outer_{outer}_{phase}.json.gz"


def outer_score_path(candidate: str) -> Path:
    safe = candidate.lower().replace("-", "_")
    return derived_dir() / "scores" / f"{safe}_outer_oof.tsv.gz"


# Section 2: causal feature grid

def causal_context(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    starts = prior.context_start_indices(onsets)
    indices = starts[:, None] + np.arange(prior.CONTEXT_EPOCHS)[None, :]
    base = features[indices][:, :, prior.base_indices()]
    decisions = onsets[starts + prior.CONTEXT_EPOCHS - 1] + prior.EPOCH_SEC
    matrix = base.reshape(len(starts), -1).astype(np.float32, copy=False)
    return decisions, matrix


def prepare_recordings(assignments: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[str]]:
    scaler = prior.frozen_scalers()["H2"]
    recordings = {}
    summaries = []
    feature_names = None
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        onsets, features, names, base_summary = prior.load_or_create_features(
            item.subject, "H2", scaler
        )
        shifted_onsets, shifted_features, shifted_summary = interval.load_or_create_shifted_features(
            item.subject, onsets, scaler
        )
        base_times, base_context = causal_context(onsets, features)
        shifted_times, shifted_context = causal_context(shifted_onsets, shifted_features)
        times = np.concatenate([base_times, shifted_times])
        matrix = np.concatenate([base_context, shifted_context], axis=0)
        order = np.argsort(times)
        times, matrix = times[order], matrix[order]
        if len(times) != len(np.unique(np.rint(times * 2).astype(int))):
            raise ValueError(f"Duplicate causal decision: {item.subject}")
        if not np.isfinite(matrix).all() or matrix.shape[1] != 80:
            raise ValueError(f"Invalid causal feature matrix: {item.subject}")
        recordings[item.subject] = {
            "pid": int(item.pid),
            "times": times,
            "features": matrix,
        }
        summaries.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "base_decisions": len(base_times),
                "shifted_decisions": len(shifted_times),
                "grid_decisions": len(times),
                "supported_hours": len(times) * GRID_SEC / 3600.0,
                "earliest_decision_sec": float(times.min()),
                "latest_decision_sec": float(times.max()),
                "context_sec": 240.0,
                "future_signal_sec": 0.0,
                "base_cache_sha256": base_summary["cache_sha256"],
                "shifted_cache_sha256": shifted_summary["shifted_cache_sha256"],
            }
        )
        current_names = [
            f"t{lag:+d}_{names[index]}"
            for lag in range(-240, 0, 30)
            for index in prior.base_indices()
        ]
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError("Causal feature schema changed across recordings")
        print(f"features {number:02d}/{len(assignments)} {item.subject}", flush=True)
    return recordings, pd.DataFrame(summaries), feature_names


# Section 3: event membership and targets

def load_references(assignments: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(
        root() / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        root() / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    rows = membership[
        membership.subject.isin(set(assignments.subject))
        & membership.transition_type.isin(["REM_to_Wake", "Wake_to_REM"])
    ].merge(quality, on="transition_id", validate="one_to_one")
    if set(rows.partition) != {"train"}:
        raise ValueError("Unauthorized event partition")
    rows["event_time_sec"] = rows.nominal_boundary_sec.astype(float) + 15.0
    return rows.sort_values(["subject", "event_time_sec", "transition_id"]).reset_index(drop=True)


def build_targets(
    assignments: pd.DataFrame,
    recordings: dict,
    references: pd.DataFrame,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    negatives, negative_values = [], []
    positive_rows = {candidate: [] for candidate in CANDIDATES}
    positive_values = {candidate: [] for candidate in CANDIDATES}
    primary = references[prior.truth(references.primary_analysis_eligible)].copy()
    support_rows = []

    for item in assignments.itertuples(index=False):
        recording = recordings[item.subject]
        times = recording["times"]
        values = recording["features"]
        local_refs = references[references.subject.eq(item.subject)]
        nominal = local_refs.nominal_boundary_sec.to_numpy(dtype=float)
        eligible = (
            np.min(np.abs(times[:, None] - nominal[None, :]), axis=1) >= EXCLUSION_SEC
            if len(nominal)
            else np.ones(len(times), dtype=bool)
        )
        for index in np.flatnonzero(eligible):
            time = float(times[index])
            negatives.append(
                {
                    "sample_id": f"negative_{item.subject}_{int(round(time * 2))}",
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": time,
                    "label": 0,
                }
            )
            negative_values.append(values[index])

        lookup = {int(round(time * 2)): index for index, time in enumerate(times)}
        for event in primary[primary.subject.eq(item.subject)].itertuples(index=False):
            key = int(round(float(event.event_time_sec) * 2))
            index = lookup.get(key)
            support_rows.append(
                {
                    "transition_id": int(event.transition_id),
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "transition_type": event.transition_type,
                    "nominal_boundary_sec": float(event.nominal_boundary_sec),
                    "decision_time_sec": float(event.event_time_sec),
                    "causal_context_supported": index is not None,
                }
            )
            if index is None:
                continue
            targets = ["CAUSAL-BI"]
            if event.transition_type == "REM_to_Wake":
                targets.append("CAUSAL-RW")
            for candidate in targets:
                positive_rows[candidate].append(
                    {
                        "sample_id": f"transition_{int(event.transition_id)}",
                        "transition_id": int(event.transition_id),
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": float(event.event_time_sec),
                        "transition_type": event.transition_type,
                        "label": 1,
                    }
                )
                positive_values[candidate].append(values[index])

    targets = {
        "negative": {
            "rows": pd.DataFrame(negatives),
            "values": np.stack(negative_values).astype(np.float32),
        }
    }
    for candidate in CANDIDATES:
        targets[candidate] = {
            "rows": pd.DataFrame(positive_rows[candidate]),
            "values": np.stack(positive_values[candidate]).astype(np.float32),
        }
    support = pd.DataFrame(support_rows)
    summary = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "negative_rows": len(targets["negative"]["rows"]),
                "positive_rows": len(targets[candidate]["rows"]),
                "rem_to_wake_positive_rows": int(
                    targets[candidate]["rows"].transition_type.eq("REM_to_Wake").sum()
                ),
                "wake_to_rem_positive_rows": int(
                    targets[candidate]["rows"].transition_type.eq("Wake_to_REM").sum()
                ),
            }
            for candidate in CANDIDATES
        ]
    )
    if len(targets["CAUSAL-RW"]["rows"]) != 180:
        raise ValueError("Causal REM-to-Wake support changed")
    if len(targets["CAUSAL-BI"]["rows"]) != 242:
        raise ValueError("Causal bidirectional support changed")
    if not support.causal_context_supported.all():
        raise ValueError("A primary transition lacks causal context")
    return targets, summary, support


def training_arrays(targets: dict, candidate: str, fitting_pids: set[int]) -> tuple:
    negative = targets["negative"]
    positive = targets[candidate]
    negative_mask = negative["rows"].pid.isin(fitting_pids).to_numpy()
    positive_mask = positive["rows"].pid.isin(fitting_pids).to_numpy()
    rows = pd.concat(
        [negative["rows"].loc[negative_mask], positive["rows"].loc[positive_mask]],
        ignore_index=True,
    )
    values = np.concatenate(
        [negative["values"][negative_mask], positive["values"][positive_mask]], axis=0
    )
    return rows, values, rows.label.to_numpy(dtype=int)


# Section 4: fixed logistic fits

def fit_logistic(
    values: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    class_weight: str | None,
) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="lbfgs",
        class_weight=class_weight,
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
            "class_weight": class_weight,
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


def score_recordings(
    fitted: dict,
    heldout: pd.DataFrame,
    recordings: dict,
    candidate: str,
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, support = [], []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        probability = fitted["model"].predict_proba(
            fitted["scaler"].transform(recording["features"])
        )[:, 1]
        frames.append(
            pd.DataFrame(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["times"],
                    "probability": probability,
                }
            )
        )
        support.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(probability),
                "supported_hours": len(probability) * GRID_SEC / 3600.0,
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(support)


def run_stage_a_fit(
    candidate: str,
    outer: int,
    phase: str,
    fitting_pids: set[int],
    heldout: pd.DataFrame,
    targets: dict,
    recordings: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, values, labels = training_arrays(targets, candidate, fitting_pids)
    fitted, payload = fit_logistic(values, labels, feature_names, None)
    scores, support = score_recordings(
        fitted, heldout, recordings, candidate, outer, phase
    )
    probability = scores.probability.to_numpy(dtype=float)
    path = model_path("stage_a", candidate, outer, phase)
    prior.verify_or_create_json_gzip(path, payload)
    summary = {
        "stage": "A",
        "candidate": candidate,
        "outer_fold": outer,
        "phase": phase,
        "training_rows": len(rows),
        "positive_rows": int(labels.sum()),
        "iterations": payload["iterations"],
        "converged": payload["converged"],
        "intercept": float(payload["intercept"][0]),
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "unique_probabilities": int(np.unique(probability).size),
        "exact_endpoint_fraction": float(np.mean((probability == 0.0) | (probability == 1.0))),
        "all_values_finite": bool(
            np.isfinite(probability).all()
            and np.isfinite(payload["coefficient"]).all()
            and np.isfinite(payload["intercept"]).all()
        ),
        "model_relative_path": path.relative_to(data_parent()).as_posix(),
        "model_bytes": path.stat().st_size,
        "model_sha256": prior.sha256(path),
    }
    print(f"fit A {candidate} outer={outer} phase={phase}", flush=True)
    return scores, support, summary


# Section 5: causal alarm matching

def causal_matches(references: np.ndarray, predictions: np.ndarray) -> list[tuple[int, int]]:
    reference_order = np.argsort(references)
    prediction_order = np.argsort(predictions)
    matches = []
    left = right = 0
    while left < len(reference_order) and right < len(prediction_order):
        reference_index = int(reference_order[left])
        prediction_index = int(prediction_order[right])
        reference = float(references[reference_index])
        prediction = float(predictions[prediction_index])
        if prediction < reference - 1e-9:
            right += 1
        elif prediction > reference + 15.0 + 1e-9:
            left += 1
        else:
            matches.append((reference_index, prediction_index))
            left += 1
            right += 1
    return matches


def scope_mask(references: pd.DataFrame, scope: str) -> pd.Series:
    primary = prior.truth(references.primary_analysis_eligible)
    if scope == "REM_to_Wake":
        return primary & references.transition_type.eq("REM_to_Wake")
    if scope == "Wake_to_REM":
        return primary & references.transition_type.eq("Wake_to_REM")
    if scope == "bidirectional":
        return primary
    raise ValueError(scope)


def predictions_from_scores(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    for (subject, pid), group in scores.groupby(["subject", "pid"], sort=True):
        times = interval.collapsed_times(group.sort_values("candidate_time_sec"), threshold)
        rows.extend(
            {"subject": subject, "pid": int(pid), "event_time_sec": float(time)}
            for time in times
        )
    return pd.DataFrame(rows, columns=["subject", "pid", "event_time_sec"])


def evaluate_predictions(
    predictions: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    scope: str,
    return_details: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    eligible = references[scope_mask(references, scope)]
    ignored = references[~scope_mask(references, scope)]
    participant_rows, match_rows = [], []
    for item in support.itertuples(index=False):
        local_predictions = predictions[predictions.subject.eq(item.subject)].event_time_sec.to_numpy(
            dtype=float
        )
        local_eligible = eligible[eligible.subject.eq(item.subject)].event_time_sec.to_numpy(dtype=float)
        local_ignored = ignored[ignored.subject.eq(item.subject)].event_time_sec.to_numpy(dtype=float)
        matches = causal_matches(local_eligible, local_predictions)
        matched_prediction = {pair[1] for pair in matches}
        unmatched_indices = [
            index for index in range(len(local_predictions)) if index not in matched_prediction
        ]
        ignored_matches = causal_matches(local_ignored, local_predictions[unmatched_indices])
        true_positive = len(matches)
        false_negative = len(local_eligible) - true_positive
        false_positive = len(unmatched_indices) - len(ignored_matches)
        participant_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "ignored_predictions": len(ignored_matches),
                "predicted_events": len(local_predictions),
                "reference_events": len(local_eligible),
                "supported_hours": float(item.supported_hours),
            }
        )
        if return_details:
            for reference_index, prediction_index in matches:
                match_rows.append(
                    {
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "match_type": "eligible",
                        "reference_time_sec": float(local_eligible[reference_index]),
                        "prediction_time_sec": float(local_predictions[prediction_index]),
                    }
                )
            for reference_index, local_index in ignored_matches:
                prediction_index = unmatched_indices[local_index]
                match_rows.append(
                    {
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "match_type": "ignored",
                        "reference_time_sec": float(local_ignored[reference_index]),
                        "prediction_time_sec": float(local_predictions[prediction_index]),
                    }
                )
    participants = pd.DataFrame(participant_rows)
    by_pid = participants.groupby("pid", as_index=False).agg(
        true_positive=("true_positive", "sum"),
        false_positive=("false_positive", "sum"),
        false_negative=("false_negative", "sum"),
        ignored_predictions=("ignored_predictions", "sum"),
        predicted_events=("predicted_events", "sum"),
        reference_events=("reference_events", "sum"),
        supported_hours=("supported_hours", "sum"),
    )
    tp = int(by_pid.true_positive.sum())
    fp = int(by_pid.false_positive.sum())
    fn = int(by_pid.false_negative.sum())
    hours = float(by_pid.supported_hours.sum())
    summary = {
        "recordings": len(support),
        "pid": support.pid.nunique(),
        "reference_events": int(by_pid.reference_events.sum()),
        "predicted_events": int(by_pid.predicted_events.sum()),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "ignored_predictions": int(by_pid.ignored_predictions.sum()),
        "supported_hours": hours,
        **metric_values(tp, fp, fn, hours),
    }
    return summary, by_pid, pd.DataFrame(match_rows)


def threshold_curve(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    scope: str,
) -> pd.DataFrame:
    rows = []
    for threshold in formulation.threshold_grid():
        predictions = predictions_from_scores(scores, float(threshold))
        summary, _, _ = evaluate_predictions(predictions, support, references, scope)
        rows.append({"threshold": float(threshold), **summary})
    return pd.DataFrame(rows)


# Section 6: participant-grouped nested Stage A

def run_stage_a(
    assignments: pd.DataFrame,
    folds: pd.DataFrame,
    targets: dict,
    recordings: dict,
    references: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    fold_by_pid = folds.set_index("pid").fold.astype(int).to_dict()
    fold_values = sorted(folds.fold.astype(int).unique())
    fit_rows, curve_frames, selections = [], [], []
    outer_scores, outer_support = [], []
    for outer in fold_values:
        inner_scores = {candidate: [] for candidate in CANDIDATES}
        inner_support = {candidate: [] for candidate in CANDIDATES}
        for inner in [value for value in fold_values if value != outer]:
            fitting_pids = {
                int(pid) for pid, fold in fold_by_pid.items() if fold not in {outer, inner}
            }
            heldout = assignments[assignments.pid.map(fold_by_pid).eq(inner)]
            for candidate in CANDIDATES:
                scores, support, summary = run_stage_a_fit(
                    candidate,
                    outer,
                    f"inner_{inner}",
                    fitting_pids,
                    heldout,
                    targets,
                    recordings,
                    feature_names,
                )
                inner_scores[candidate].append(scores)
                inner_support[candidate].append(support)
                fit_rows.append(summary)

        for candidate in CANDIDATES:
            scope = "REM_to_Wake" if candidate == "CAUSAL-RW" else "bidirectional"
            curve = threshold_curve(
                pd.concat(inner_scores[candidate], ignore_index=True),
                pd.concat(inner_support[candidate], ignore_index=True),
                references,
                scope,
            )
            curve.insert(0, "candidate", candidate)
            curve.insert(1, "outer_fold", outer)
            curve.insert(2, "selection_scope", scope)
            curve_frames.append(curve)
            selected = formulation.select_threshold(curve, PRIMARY_BUDGET)
            selections.append(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "selection_scope": scope,
                    **selected,
                }
            )

        fitting_pids = {int(pid) for pid, fold in fold_by_pid.items() if fold != outer}
        heldout = assignments[assignments.pid.map(fold_by_pid).eq(outer)]
        for candidate in CANDIDATES:
            scores, support, summary = run_stage_a_fit(
                candidate,
                outer,
                "outer_final",
                fitting_pids,
                heldout,
                targets,
                recordings,
                feature_names,
            )
            outer_scores.append(scores)
            if candidate == CANDIDATES[0]:
                outer_support.append(support)
            fit_rows.append(summary)

    scores = pd.concat(outer_scores, ignore_index=True)
    support = pd.concat(outer_support, ignore_index=True)
    selection_table = pd.DataFrame(selections)
    alarm_frames = []
    for item in selection_table.itertuples(index=False):
        local = scores[
            scores.candidate.eq(item.candidate) & scores.outer_fold.eq(item.outer_fold)
        ]
        predictions = predictions_from_scores(local, float(item.threshold))
        predictions.insert(0, "candidate", item.candidate)
        predictions.insert(1, "outer_fold", int(item.outer_fold))
        predictions["threshold"] = float(item.threshold)
        alarm_frames.append(predictions)
    alarms = pd.concat(alarm_frames, ignore_index=True)
    return {
        "fits": pd.DataFrame(fit_rows),
        "curves": pd.concat(curve_frames, ignore_index=True),
        "selections": selection_table,
        "scores": scores,
        "support": support,
        "alarms": alarms,
    }


def evaluate_stage_a(
    stage_a: dict[str, pd.DataFrame], references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metric_rows, participant_frames, match_frames = [], [], []
    for candidate in CANDIDATES:
        predictions = stage_a["alarms"][stage_a["alarms"].candidate.eq(candidate)]
        for scope in ["bidirectional", "REM_to_Wake", "Wake_to_REM"]:
            summary, participants, matches = evaluate_predictions(
                predictions,
                stage_a["support"],
                references,
                scope,
                return_details=True,
            )
            config = {
                "candidate": candidate,
                "scope": scope,
                "partition": "train_nested_oof",
                "alarm_budget_per_hour": PRIMARY_BUDGET,
                "causal_match_start_sec": 0.0,
                "causal_match_stop_sec": 15.0,
            }
            metric_rows.append({**config, **summary})
            local = participants.copy()
            for key, value in reversed(list(config.items())):
                local.insert(0, key, value)
            participant_frames.append(local)
            if len(matches):
                local_matches = matches.copy()
                for key, value in reversed(list(config.items())):
                    local_matches.insert(0, key, value)
                match_frames.append(local_matches)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "participants": pd.concat(participant_frames, ignore_index=True),
        "matches": (
            pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
        ),
    }


# Section 7: oracle-event direction diagnostic

def run_stage_b(
    folds: pd.DataFrame,
    targets: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = targets["CAUSAL-BI"]["rows"].copy().reset_index(drop=True)
    values = targets["CAUSAL-BI"]["values"]
    labels = rows.transition_type.eq("REM_to_Wake").to_numpy(dtype=int)
    fold_by_pid = folds.set_index("pid").fold.astype(int).to_dict()
    score_frames, fit_rows = [], []
    for outer in sorted(folds.fold.astype(int).unique()):
        train = ~rows.pid.map(fold_by_pid).eq(outer).to_numpy()
        heldout = ~train
        fitted, payload = fit_logistic(
            values[train], labels[train], feature_names, "balanced"
        )
        probability = fitted["model"].predict_proba(
            fitted["scaler"].transform(values[heldout])
        )[:, 1]
        local = rows.loc[heldout, [
            "transition_id", "subject", "pid", "candidate_time_sec", "transition_type"
        ]].copy()
        local.insert(0, "outer_fold", outer)
        local["rem_to_wake_probability"] = probability
        local["predicted_transition_type"] = np.where(
            probability >= 0.5, "REM_to_Wake", "Wake_to_REM"
        )
        score_frames.append(local)
        path = model_path("stage_b", "direction", outer, "outer_final")
        prior.verify_or_create_json_gzip(path, payload)
        fit_rows.append(
            {
                "stage": "B",
                "candidate": "DIRECTION",
                "outer_fold": outer,
                "phase": "outer_final",
                "training_rows": int(train.sum()),
                "positive_rows": int(labels[train].sum()),
                "iterations": payload["iterations"],
                "converged": payload["converged"],
                "intercept": float(payload["intercept"][0]),
                "minimum_probability": float(probability.min()),
                "maximum_probability": float(probability.max()),
                "unique_probabilities": int(np.unique(probability).size),
                "exact_endpoint_fraction": float(
                    np.mean((probability == 0.0) | (probability == 1.0))
                ),
                "all_values_finite": bool(
                    np.isfinite(probability).all()
                    and np.isfinite(payload["coefficient"]).all()
                    and np.isfinite(payload["intercept"]).all()
                ),
                "model_relative_path": path.relative_to(data_parent()).as_posix(),
                "model_bytes": path.stat().st_size,
                "model_sha256": prior.sha256(path),
            }
        )
    scores = pd.concat(score_frames, ignore_index=True).sort_values("transition_id")
    actual = scores.transition_type.eq("REM_to_Wake").to_numpy(dtype=int)
    predicted = scores.predicted_transition_type.eq("REM_to_Wake").to_numpy(dtype=int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    recalls = recall_score(actual, predicted, labels=[0, 1], average=None, zero_division=0)
    metrics = pd.DataFrame(
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
    return scores, pd.DataFrame(fit_rows), metrics


# Section 8: uncertainty, decision, and outputs

def participant_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    local = participants[participants.scope.eq("REM_to_Wake")]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    left = local[local.candidate.eq("CAUSAL-BI")][columns].sort_values("pid")
    right = local[local.candidate.eq("CAUSAL-RW")][columns].sort_values("pid")
    paired = left.merge(right, on="pid", suffixes=("_bi", "_rw"), validate="one_to_one")
    if len(paired) != 64:
        raise ValueError("Incomplete paired participant table")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for indices in rng.integers(0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired))):
        sample = paired.iloc[indices]
        values = {}
        for side in ["bi", "rw"]:
            values[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()),
                int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()),
                float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "f1_difference": values["bi"]["f1"] - values["rw"]["f1"],
                "recall_difference": values["bi"]["recall"] - values["rw"]["recall"],
                "false_alarms_per_hour_difference": (
                    values["bi"]["false_alarms_per_hour"]
                    - values["rw"]["false_alarms_per_hour"]
                ),
            }
        )
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "comparison": "CAUSAL-BI_minus_CAUSAL-RW",
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


def decision_table(metrics: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[metrics.scope.eq("REM_to_Wake")].set_index("candidate")
    f1_difference = float(primary.loc["CAUSAL-BI", "f1"] - primary.loc["CAUSAL-RW", "f1"])
    far_difference = float(
        primary.loc["CAUSAL-BI", "false_alarms_per_hour"]
        - primary.loc["CAUSAL-RW", "false_alarms_per_hour"]
    )
    stage_a = fits[fits.stage.eq("A")]
    numerical = bool(
        len(stage_a) == 50
        and stage_a.converged.all()
        and stage_a.all_values_finite.all()
        and stage_a.iterations.lt(MAX_ITER).all()
        and stage_a.exact_endpoint_fraction.lt(0.01).all()
        and stage_a.unique_probabilities.gt(100).all()
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
                "hypothesis": "H-BI",
                "comparison": "CAUSAL-BI_minus_CAUSAL-RW",
                "scope": "REM_to_Wake",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.03,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_false_alarm_difference": 0.05,
                "numerical_gate_pass": numerical,
                "point_gate_pass": point_gate and numerical,
                "decision": decision,
            }
        ]
    )


def manifest(fits: pd.DataFrame, outer_scores: pd.DataFrame) -> pd.DataFrame:
    rows = fits[["stage", "candidate", "outer_fold", "phase", "model_relative_path", "model_bytes", "model_sha256"]].copy()
    rows = rows.rename(
        columns={
            "model_relative_path": "relative_path",
            "model_bytes": "bytes",
            "model_sha256": "sha256",
        }
    )
    rows.insert(0, "artifact_type", "model")
    score_rows = []
    for candidate in CANDIDATES:
        path = outer_score_path(candidate)
        local = outer_scores[outer_scores.candidate.eq(candidate)].copy()
        prior.verify_or_create_gzip_tsv(path, local)
        score_rows.append(
            {
                "artifact_type": "outer_score",
                "stage": "A",
                "candidate": candidate,
                "outer_fold": "all",
                "phase": "outer_oof",
                "relative_path": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": prior.sha256(path),
            }
        )
    return pd.concat([rows, pd.DataFrame(score_rows)], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output_dir().mkdir(parents=True, exist_ok=True)

    assignments = prior.train_assignments()
    folds = prior.frozen_fold_assignments(assignments)
    recordings, feature_summary, feature_names = prepare_recordings(assignments)
    references = load_references(assignments)
    targets, target_summary, event_support = build_targets(
        assignments, recordings, references
    )
    verify_or_create_tsv(output_dir() / "feature_grid_summary_v0.1.tsv", feature_summary)
    verify_or_create_tsv(output_dir() / "target_construction_summary_v0.1.tsv", target_summary)
    verify_or_create_tsv(output_dir() / "primary_event_causal_support_v0.1.tsv", event_support)
    audit = pd.DataFrame(
        [
            {"check": "train_recordings", "observed": len(assignments), "expected": 82, "passed": len(assignments) == 82},
            {"check": "train_pid", "observed": assignments.pid.nunique(), "expected": 64, "passed": assignments.pid.nunique() == 64},
            {"check": "causal_features", "observed": len(feature_names), "expected": 80, "passed": len(feature_names) == 80},
            {"check": "future_signal_sec", "observed": feature_summary.future_signal_sec.max(), "expected": 0, "passed": feature_summary.future_signal_sec.eq(0).all()},
            {"check": "rem_to_wake_events", "observed": target_summary.loc[target_summary.candidate.eq("CAUSAL-RW"), "positive_rows"].iloc[0], "expected": 180, "passed": len(targets["CAUSAL-RW"]["rows"]) == 180},
            {"check": "bidirectional_events", "observed": target_summary.loc[target_summary.candidate.eq("CAUSAL-BI"), "positive_rows"].iloc[0], "expected": 242, "passed": len(targets["CAUSAL-BI"]["rows"]) == 242},
            {"check": "complete_primary_event_support", "observed": int(event_support.causal_context_supported.sum()), "expected": 242, "passed": event_support.causal_context_supported.all()},
        ]
    )
    verify_or_create_tsv(output_dir() / "pre_result_support_audit_v0.1.tsv", audit)
    if not audit.passed.all():
        raise ValueError("Pre-result support audit failed")
    if args.audit_only:
        print("Pre-result support audit passed 7/7; no model was fitted.")
        return

    stage_a = run_stage_a(
        assignments, folds, targets, recordings, references, feature_names
    )
    evaluated = evaluate_stage_a(stage_a, references)
    stage_b_scores, stage_b_fits, stage_b_metrics = run_stage_b(
        folds, targets, feature_names
    )
    fits = pd.concat([stage_a["fits"], stage_b_fits], ignore_index=True)
    bootstrap = participant_bootstrap(evaluated["participants"])
    decision = decision_table(evaluated["metrics"], fits)
    artifacts = manifest(fits, stage_a["scores"])

    outputs = {
        "model_fit_summary_v0.1.tsv": fits,
        "inner_threshold_curves_v0.1.tsv": stage_a["curves"],
        "inner_threshold_selections_v0.1.tsv": stage_a["selections"],
        "outer_support_v0.1.tsv": stage_a["support"],
        "outer_predicted_events_v0.1.tsv": stage_a["alarms"],
        "outer_event_metrics_v0.1.tsv": evaluated["metrics"],
        "outer_event_participants_v0.1.tsv": evaluated["participants"],
        "outer_event_matches_v0.1.tsv": evaluated["matches"],
        "stage_b_direction_scores_v0.1.tsv": stage_b_scores,
        "stage_b_direction_metrics_v0.1.tsv": stage_b_metrics,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decision,
        "external_artifact_manifest_v0.1.tsv": artifacts,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(output_dir() / name, frame)
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
    print(evaluated["metrics"].to_string(index=False))
    print(stage_b_metrics.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
