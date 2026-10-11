"""Run the frozen Block 10 interval-aware localization experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import run_full_night_risk_set_formulation_v0_1 as formulation
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


EXPERIMENT_DIR = "2026-10-10_interval_aware_localization_v0.1"
DERIVED_DIR = "interval_aware_localization_v0.1"
PROTOCOL_COMMIT = "58ef4c3"
AMENDMENT_COMMIT = "ab4397d"
CANDIDATES = ["HARD15", "INT15"]
ALARM_BUDGETS = [0.10, 0.25, 0.50, 1.00]
PRIMARY_BUDGET = 0.25
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
GRID_SEC = 15.0
EXCLUSION_SEC = 135.0
C_VALUE = 0.1
MAX_ITER = 2000
BASE_SEED = 20261010
BOOTSTRAP_RESAMPLES = 2000
MAX_WORKERS = 2


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return root().parent / "REM_W_data"


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root().as_posix()}", "rev-parse", "HEAD"],
        cwd=root(), text=True,
    ).strip()


def write_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")


def shifted_feature_path(subject: str) -> Path:
    return derived_dir() / "features" / "h2_shift15" / f"{subject}.npz"


def model_path(candidate: str, outer: int, phase: str) -> Path:
    return derived_dir() / "models" / candidate.lower() / f"outer_{outer}_{phase}.json.gz"


def score_path(candidate: str, outer: int, phase: str) -> Path:
    return derived_dir() / "scores" / candidate.lower() / f"outer_{outer}_{phase}.tsv.gz"


# Section 1: construct a genuine 15-second feature grid

def expected_shifted_onsets(onsets: np.ndarray) -> np.ndarray:
    return onsets[:-1][np.isclose(np.diff(onsets), prior.EPOCH_SEC)] + GRID_SEC


def load_or_create_shifted_features(
    subject: str,
    base_onsets: np.ndarray,
    scaler: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    path = shifted_feature_path(subject)
    expected_onsets = expected_shifted_onsets(base_onsets)
    names = prior.rich_feature_names(["HB_1", "HB_2"])
    if path.exists():
        with np.load(path, allow_pickle=False) as values:
            onsets = values["onset"].astype(np.float64)
            features = values["features"].astype(np.float32)
            cached_names = values["feature_names"].astype(str).tolist()
        if not np.array_equal(onsets, expected_onsets) or cached_names != names:
            raise RuntimeError(f"Shifted feature cache changed: {subject}")
        status = "reused"
    else:
        raw = prior.read_uv(subject, "headband", ["HB_1", "HB_2"])
        signal = prior.normalize(
            prior.filter_resample(raw, prior.filter_sos()), ["HB_1", "HB_2"], scaler
        )
        blocks = prior.epoch_blocks(signal, expected_onsets)
        features = prior.calculate_rich_features(blocks, ["HB_1", "HB_2"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            onset=expected_onsets,
            features=features,
            feature_names=np.asarray(names),
        )
        onsets = expected_onsets
        status = "created"
    if not np.isfinite(features).all() or not prior.feature_bounds_pass(features):
        raise ValueError(f"Shifted feature validation failed: {subject}")
    summary = {
        "subject": subject,
        "shifted_epochs": len(onsets),
        "shifted_cache_status": status,
        "shifted_relative_path": path.relative_to(data_parent()).as_posix(),
        "shifted_cache_bytes": path.stat().st_size,
        "shifted_cache_sha256": prior.sha256(path),
    }
    return onsets, features, summary


def prepare_recordings(assignments: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[str]]:
    scaler = prior.frozen_scalers()["H2"]
    recordings, summaries = {}, []
    feature_names = None
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        onsets, features, names, base_summary = prior.load_or_create_features(
            item.subject, "H2", scaler
        )
        base_centers, base_context = prior.context_matrix(onsets, features)
        shifted_onsets, shifted_features, shifted_summary = load_or_create_shifted_features(
            item.subject, onsets, scaler
        )
        shifted_centers, shifted_context = prior.context_matrix(
            shifted_onsets, shifted_features
        )
        centers = np.concatenate([base_centers, shifted_centers])
        context = np.concatenate([base_context, shifted_context], axis=0)
        order = np.argsort(centers)
        centers, context = centers[order], context[order]
        if len(np.unique(np.rint(centers * 2).astype(int))) != len(centers):
            raise ValueError(f"Duplicate 15-second boundary: {item.subject}")
        steps = np.diff(centers) / GRID_SEC
        if not np.allclose(steps, np.round(steps)):
            raise ValueError(f"Off-grid boundary: {item.subject}")
        recordings[item.subject] = {
            "pid": int(item.pid), "centers": centers, "features": context,
        }
        current_names = [
            f"t{int((offset - 4) * prior.EPOCH_SEC):+d}_{name}"
            for offset in range(prior.CONTEXT_EPOCHS)
            for name in names
        ]
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError("Feature schema changed across recordings")
        summaries.append(
            {
                "pid": int(item.pid),
                "base_supported_boundaries": len(base_centers),
                "shifted_supported_boundaries": len(shifted_centers),
                "grid_supported_boundaries": len(centers),
                "supported_hours": len(centers) * GRID_SEC / 3600.0,
                "base_cache_sha256": base_summary["cache_sha256"],
                **shifted_summary,
            }
        )
        print(f"features {number:02d}/{len(assignments)} {item.subject}", flush=True)
    if feature_names is None or len(feature_names) != 360:
        raise ValueError("Unexpected context feature schema")
    return recordings, pd.DataFrame(summaries), feature_names


# Section 2: build shared negatives and the two fixed positive targets

def build_targets(assignments: pd.DataFrame, recordings: dict) -> tuple[dict, pd.DataFrame]:
    label_table = pd.read_csv(
        root() / "labels/transition_labels_v0.1/transition_labels_v0.1.tsv", sep="\t"
    )
    label_table = label_table[label_table.subject.isin(set(assignments.subject))]
    references = prior.reference_events(assignments)
    positives = references[prior.truth(references.primary_analysis_eligible)].copy()
    negative_rows, negative_values = [], []
    hard_rows, hard_values = [], []
    interval_rows, interval_values = [], []

    for item in assignments.itertuples(index=False):
        recording = recordings[item.subject]
        centers = recording["centers"]
        values = recording["features"]
        boundaries = label_table.loc[
            label_table.subject.eq(item.subject), "nominal_boundary_sec"
        ].to_numpy(dtype=float)
        if len(boundaries):
            eligible = np.min(np.abs(centers[:, None] - boundaries[None, :]), axis=1) >= EXCLUSION_SEC
        else:
            eligible = np.ones(len(centers), dtype=bool)
        for index in np.flatnonzero(eligible):
            time = float(centers[index])
            negative_rows.append(
                {
                    "sample_id": f"negative_{item.subject}_{int(round(time * 2))}",
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": time,
                    "label": 0,
                    "sample_weight": 1.0,
                    "label_offset_sec": np.nan,
                }
            )
            negative_values.append(values[index])

        lookup = {int(round(time * 2)): index for index, time in enumerate(centers)}
        local_positive = positives[positives.subject.eq(item.subject)]
        for event in local_positive.itertuples(index=False):
            nominal = float(event.nominal_boundary_sec)
            nominal_index = lookup.get(int(round(nominal * 2)))
            if nominal_index is None:
                raise ValueError(f"Missing hard target: {event.transition_id}")
            hard_rows.append(
                {
                    "sample_id": f"transition_{event.transition_id}_0",
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": nominal,
                    "label": 1,
                    "sample_weight": 1.0,
                    "label_offset_sec": 0.0,
                }
            )
            hard_values.append(values[nominal_index])
            for offset in [-15.0, 0.0, 15.0]:
                time = nominal + offset
                index = lookup.get(int(round(time * 2)))
                if index is None:
                    raise ValueError(f"Missing interval target: {event.transition_id}, {offset}")
                interval_rows.append(
                    {
                        "sample_id": f"transition_{event.transition_id}_{int(offset):+d}",
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": time,
                        "label": 1,
                        "sample_weight": 1.0 / 3.0,
                        "label_offset_sec": offset,
                    }
                )
                interval_values.append(values[index])

    targets = {
        "negative": {
            "rows": pd.DataFrame(negative_rows),
            "values": np.stack(negative_values).astype(np.float32),
        },
        "HARD15": {
            "rows": pd.DataFrame(hard_rows),
            "values": np.stack(hard_values).astype(np.float32),
        },
        "INT15": {
            "rows": pd.DataFrame(interval_rows),
            "values": np.stack(interval_values).astype(np.float32),
        },
    }
    if len(targets["negative"]["rows"]) != 147055:
        raise ValueError("15-second negative count changed")
    if len(targets["HARD15"]["rows"]) != 180 or len(targets["INT15"]["rows"]) != 540:
        raise ValueError("Positive target count changed")
    summary = pd.DataFrame(
        [
            {
                "candidate": candidate,
                "negative_rows": len(targets["negative"]["rows"]),
                "positive_rows": len(targets[candidate]["rows"]),
                "positive_weight": targets[candidate]["rows"].sample_weight.sum(),
                "total_weight": (
                    targets["negative"]["rows"].sample_weight.sum()
                    + targets[candidate]["rows"].sample_weight.sum()
                ),
            }
            for candidate in CANDIDATES
        ]
    )
    return targets, summary


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
    return (
        rows,
        values,
        rows.label.to_numpy(dtype=int),
        rows.sample_weight.to_numpy(dtype=float),
    )


# Section 3: shared preprocessing and stable stochastic-logistic fits

def score_recordings(
    fitted: dict,
    heldout: pd.DataFrame,
    recordings: dict,
    candidate: str,
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_frames, support_rows = [], []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        transformed = fitted["scaler"].transform(
            recording["features"][:, fitted["selected"]]
        )
        probability = fitted["model"].predict_proba(transformed)[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError(f"Nonfinite probability: {candidate}, {item.subject}")
        score_frames.append(
            pd.DataFrame(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["centers"],
                    "probability": probability,
                }
            )
        )
        support_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(probability),
                "supported_hours": len(probability) * GRID_SEC / 3600.0,
            }
        )
    return pd.concat(score_frames, ignore_index=True), pd.DataFrame(support_rows)


def cached_fit(candidate: str, outer: int, phase: str, heldout: pd.DataFrame) -> tuple | None:
    model_file = model_path(candidate, outer, phase)
    score_file = score_path(candidate, outer, phase)
    if model_file.exists() != score_file.exists():
        raise RuntimeError(f"Incomplete cached fit: {candidate}, {outer}, {phase}")
    if not model_file.exists():
        return None
    with gzip.open(model_file, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    scores = pd.read_csv(score_file, sep="\t", compression="gzip")
    if (
        payload["configuration"]["candidate"] != candidate
        or set(scores.subject) != set(heldout.subject)
        or not scores.outer_fold.eq(outer).all()
        or not scores.phase.eq(phase).all()
    ):
        raise RuntimeError(f"Cached fit identity changed: {candidate}, {outer}, {phase}")
    support = (
        scores.groupby(["subject", "pid", "outer_fold"], as_index=False)
        .size().rename(columns={"size": "supported_boundaries"})
    )
    support["supported_hours"] = support.supported_boundaries * GRID_SEC / 3600.0
    coefficient = np.asarray(payload["coefficient"], dtype=float)
    summary = {
        **payload["summary"],
        "model_relative_path": model_file.relative_to(data_parent()).as_posix(),
        "model_sha256": prior.sha256(model_file),
        "score_relative_path": score_file.relative_to(data_parent()).as_posix(),
        "score_sha256": prior.sha256(score_file),
        "nonzero_coefficients": int(np.sum(np.abs(coefficient) > 1e-12)),
    }
    return scores, support, summary


def run_fit_pair(
    outer: int,
    phase: str,
    fitting_pids: set[int],
    heldout: pd.DataFrame,
    targets: dict,
    recordings: dict,
    feature_names: list[str],
) -> dict:
    results = {}
    for candidate in CANDIDATES:
        cached = cached_fit(candidate, outer, phase, heldout)
        if cached is not None:
            results[candidate] = cached
    if len(results) == len(CANDIDATES):
        return results

    hard_rows, hard_values, _, _ = training_arrays(targets, "HARD15", fitting_pids)
    selected = prior.correlation_prune(hard_values)
    scaler = StandardScaler().fit(hard_values[:, selected])
    preprocessing_signature = hashlib.sha256(
        json.dumps(
            {
                "selected_indices": selected.tolist(),
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    for candidate in CANDIDATES:
        if candidate in results:
            continue
        rows, values, labels, weights = training_arrays(targets, candidate, fitting_pids)
        expected_weight = float(hard_rows.sample_weight.sum())
        if not np.isclose(weights.sum(), expected_weight):
            raise ValueError(f"Fitting weight mismatch: {candidate}, {outer}, {phase}")
        alpha = 1.0 / (C_VALUE * float(weights.sum()))
        model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=alpha,
            l1_ratio=0.5,
            learning_rate="optimal",
            average=True,
            shuffle=True,
            tol=1e-4,
            n_iter_no_change=10,
            max_iter=MAX_ITER,
            random_state=BASE_SEED,
        )
        model.fit(scaler.transform(values[:, selected]), labels, sample_weight=weights)
        coefficient = model.coef_[0]
        if not np.isfinite(coefficient).all() or not np.isfinite(model.intercept_).all():
            raise ValueError(f"Nonfinite fitted parameter: {candidate}, {outer}, {phase}")
        fitted = {"selected": selected, "scaler": scaler, "model": model}
        scores, support = score_recordings(
            fitted, heldout, recordings, candidate, outer, phase
        )
        converged = int(model.n_iter_) < MAX_ITER
        summary = {
            "candidate": candidate,
            "outer_fold": outer,
            "phase": phase,
            "training_rows": len(rows),
            "negative_rows": int((rows.label == 0).sum()),
            "positive_rows": int((rows.label == 1).sum()),
            "training_weight": float(weights.sum()),
            "positive_weight": float(weights[labels == 1].sum()),
            "input_features": values.shape[1],
            "retained_after_pruning": len(selected),
            "preprocessing_signature": preprocessing_signature,
            "nonzero_coefficients": int(np.sum(np.abs(coefficient) > 1e-12)),
            "alpha": alpha,
            "iterations": int(model.n_iter_),
            "converged": converged,
        }
        model_file = model_path(candidate, outer, phase)
        score_file = score_path(candidate, outer, phase)
        payload = {
            "configuration": {
                "candidate": candidate,
                "outer_fold": outer,
                "phase": phase,
                "loss": "log_loss",
                "penalty": "elasticnet",
                "l1_ratio": 0.5,
                "C_scale": C_VALUE,
                "alpha": alpha,
                "learning_rate": "optimal",
                "average": True,
                "tol": 1e-4,
                "n_iter_no_change": 10,
                "max_iter": MAX_ITER,
                "random_state": BASE_SEED,
            },
            "selected_indices": selected.tolist(),
            "selected_feature_names": [feature_names[index] for index in selected],
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "coefficient": coefficient.tolist(),
            "intercept": model.intercept_.tolist(),
            "iterations": int(model.n_iter_),
            "converged": converged,
            "summary": summary,
        }
        prior.verify_or_create_json_gzip(model_file, payload)
        prior.verify_or_create_gzip_tsv(score_file, scores)
        summary.update(
            {
                "model_relative_path": model_file.relative_to(data_parent()).as_posix(),
                "model_sha256": prior.sha256(model_file),
                "score_relative_path": score_file.relative_to(data_parent()).as_posix(),
                "score_sha256": prior.sha256(score_file),
            }
        )
        results[candidate] = (scores, support, summary)
    return results


# Section 4: 15-second alarm selection and event evaluation

def collapsed_times(group: pd.DataFrame, threshold: float) -> np.ndarray:
    times = group.candidate_time_sec.to_numpy(dtype=float)
    probabilities = group.probability.to_numpy(dtype=float)
    selected = np.flatnonzero(probabilities >= threshold)
    if len(selected) == 0:
        return np.asarray([], dtype=float)
    splits = np.flatnonzero(np.diff(times[selected]) > GRID_SEC + 1e-6) + 1
    result = []
    for run in np.split(selected, splits):
        local = probabilities[run]
        best = run[np.flatnonzero(np.isclose(local, local.max()))[0]]
        result.append(times[best])
    return np.asarray(result, dtype=float)


def fast_summary(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    threshold: float,
) -> dict:
    eligible, ignored = prior.local_event_inputs(references, "primary")
    score_groups = {
        subject: group.sort_values("candidate_time_sec")
        for subject, group in scores.groupby("subject")
    }
    eligible_groups = {
        subject: group.event_time_sec.to_numpy(dtype=float)
        for subject, group in eligible.groupby("subject")
    }
    ignored_groups = {
        subject: group.event_time_sec.to_numpy(dtype=float)
        for subject, group in ignored.groupby("subject")
    }
    true_positive = false_positive = false_negative = ignored_count = predicted = 0
    for item in support.itertuples(index=False):
        predictions = collapsed_times(score_groups[item.subject], threshold)
        refs = eligible_groups.get(item.subject, np.asarray([], dtype=float))
        ignored_refs = ignored_groups.get(item.subject, np.asarray([], dtype=float))
        matches = prior.optimal_matches(refs, predictions, 15.0)
        matched_predictions = {match[1] for match in matches}
        unmatched = [index for index in range(len(predictions)) if index not in matched_predictions]
        ignored_matches = prior.optimal_matches(ignored_refs, predictions[unmatched], 15.0)
        true_positive += len(matches)
        false_negative += len(refs) - len(matches)
        false_positive += len(unmatched) - len(ignored_matches)
        ignored_count += len(ignored_matches)
        predicted += len(predictions)
    hours = float(support.supported_hours.sum())
    return {
        "recordings": len(support),
        "pid": support.pid.nunique(),
        "reference_events": len(eligible),
        "predicted_events": predicted,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "ignored_predictions": ignored_count,
        "supported_hours": hours,
        **metric_values(true_positive, false_positive, false_negative, hours),
    }


def threshold_curve(
    scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"threshold": float(threshold), **fast_summary(scores, support, references, threshold)}
            for threshold in formulation.threshold_grid()
        ]
    )


def collapse_events(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in selections.itertuples(index=False):
        local = scores[
            scores.candidate.eq(item.candidate)
            & scores.outer_fold.eq(item.outer_fold)
        ]
        for (subject, pid), group in local.groupby(["subject", "pid"], sort=True):
            for time in collapsed_times(
                group.sort_values("candidate_time_sec"), float(item.threshold)
            ):
                rows.append(
                    {
                        "candidate": item.candidate,
                        "outer_fold": int(item.outer_fold),
                        "alarm_budget_per_hour": float(item.alarm_budget_per_hour),
                        "subject": subject,
                        "pid": int(pid),
                        "event_time_sec": float(time),
                        "threshold": float(item.threshold),
                    }
                )
    columns = [
        "candidate", "outer_fold", "alarm_budget_per_hour", "subject",
        "pid", "event_time_sec", "threshold",
    ]
    return pd.DataFrame(rows, columns=columns)


def evaluate_all(
    events: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    metric_rows, participant_frames, match_frames = [], [], []
    for budget in ALARM_BUDGETS:
        for candidate in CANDIDATES:
            predictions = events[
                events.candidate.eq(candidate)
                & np.isclose(events.alarm_budget_per_hour, budget)
            ][["subject", "pid", "event_time_sec"]]
            for membership in MEMBERSHIPS:
                eligible, ignored = prior.local_event_inputs(references, membership)
                for tolerance in TOLERANCES:
                    _, participants, matches, summary = evaluate_events(
                        eligible,
                        predictions,
                        ignored,
                        support[["subject", "pid", "supported_hours"]],
                        tolerance,
                    )
                    config = {
                        "candidate": candidate,
                        "partition": "train_nested_oof",
                        "membership": membership,
                        "tolerance_sec": tolerance,
                        "alarm_budget_per_hour": budget,
                    }
                    metric_rows.append({**config, **summary})
                    for frame, destination in [
                        (participants, participant_frames), (matches, match_frames)
                    ]:
                        if len(frame):
                            local = frame.copy()
                            for key, value in reversed(list(config.items())):
                                local.insert(0, key, value)
                            destination.append(local)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "participants": pd.concat(participant_frames, ignore_index=True),
        "matches": pd.concat(match_frames, ignore_index=True),
    }


# Section 5: paired uncertainty, timing mechanism, and frozen decision

def event_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        participants.membership.eq("primary")
        & participants.tolerance_sec.eq(15.0)
        & np.isclose(participants.alarm_budget_per_hour, PRIMARY_BUDGET)
    ]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    hard = primary[primary.candidate.eq("HARD15")][columns].sort_values("pid")
    interval = primary[primary.candidate.eq("INT15")][columns].sort_values("pid")
    paired = interval.merge(hard, on="pid", suffixes=("_interval", "_hard"), validate="one_to_one")
    if len(paired) != 64:
        raise ValueError("Incomplete participant bootstrap pairing")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for indices in rng.integers(0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired))):
        sample = paired.iloc[indices]
        values = {}
        for side in ["interval", "hard"]:
            values[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()),
                int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()),
                float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "recall_difference": values["interval"]["recall"] - values["hard"]["recall"],
                "f1_difference": values["interval"]["f1"] - values["hard"]["f1"],
                "false_alarms_per_hour_difference": (
                    values["interval"]["false_alarms_per_hour"]
                    - values["hard"]["false_alarms_per_hour"]
                ),
            }
        )
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "comparison": "INT15_minus_HARD15",
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


def localization_summary(matches: pd.DataFrame) -> pd.DataFrame:
    primary = matches[
        matches.membership.eq("primary")
        & matches.tolerance_sec.eq(15.0)
        & np.isclose(matches.alarm_budget_per_hour, PRIMARY_BUDGET)
        & matches.match_type.eq("eligible")
    ].copy()
    primary["signed_error_sec"] = primary.prediction_time_sec - primary.reference_time_sec
    rows = []
    for candidate in CANDIDATES:
        local = primary[primary.candidate.eq(candidate)]
        counts = local.signed_error_sec.value_counts().to_dict()
        rows.append(
            {
                "candidate": candidate,
                "matched_events": len(local),
                "offset_minus_15": int(counts.get(-15.0, 0)),
                "offset_zero": int(counts.get(0.0, 0)),
                "offset_plus_15": int(counts.get(15.0, 0)),
                "median_absolute_error_sec": (
                    float(local.absolute_error_sec.median()) if len(local) else np.nan
                ),
                "maximum_absolute_error_sec": (
                    float(local.absolute_error_sec.max()) if len(local) else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def decision_table(
    metrics: pd.DataFrame, bootstrap: pd.DataFrame, fit_summary: pd.DataFrame
) -> pd.DataFrame:
    primary = metrics[
        metrics.membership.eq("primary")
        & metrics.tolerance_sec.eq(15.0)
        & np.isclose(metrics.alarm_budget_per_hour, PRIMARY_BUDGET)
    ].set_index("candidate")
    f1_difference = float(primary.loc["INT15", "f1"] - primary.loc["HARD15", "f1"])
    far_difference = float(
        primary.loc["INT15", "false_alarms_per_hour"]
        - primary.loc["HARD15", "false_alarms_per_hour"]
    )
    converged = bool(fit_summary.converged.all() and len(fit_summary) == 50)
    point_pass = f1_difference >= 0.03 and far_difference <= 0.05
    intervals = bootstrap.set_index("metric")
    direction_supported = bool(
        intervals.loc["f1_difference", "lower_95"] > 0
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.05
    )
    material_supported = bool(
        intervals.loc["f1_difference", "lower_95"] >= 0.03
        and intervals.loc["false_alarms_per_hour_difference", "upper_95"] <= 0.05
    )
    decision = (
        "inconclusive_numerical_stop"
        if not converged
        else "retain_for_new_confirmation" if point_pass else "stop_v0.1"
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-INT",
                "comparison": "INT15_minus_HARD15",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.03,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_false_alarm_difference": 0.05,
                "convergence_control_pass": converged,
                "point_gate_pass": point_pass and converged,
                "direction_participant_supported": direction_supported,
                "material_participant_supported": material_supported,
                "decision": decision,
            }
        ]
    )


def artifact_manifest(feature_summary: pd.DataFrame, fit_summary: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "artifact_type": "shifted_feature",
            "relative_path": item.shifted_relative_path,
            "bytes": int(item.shifted_cache_bytes),
            "sha256": item.shifted_cache_sha256,
        }
        for item in feature_summary.itertuples(index=False)
    ]
    for item in fit_summary.itertuples(index=False):
        for kind, relative, digest in [
            ("model", item.model_relative_path, item.model_sha256),
            ("score", item.score_relative_path, item.score_sha256),
        ]:
            path = data_parent() / relative
            rows.append(
                {
                    "artifact_type": kind,
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            )
    return pd.DataFrame(rows).sort_values(["artifact_type", "relative_path"]).reset_index(drop=True)


# Section 6: complete nested execution and immutable reviewed outputs

def run(result_code_commit: str) -> None:
    git_commit = current_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    assignments = prior.train_assignments()
    folds = prior.frozen_fold_assignments(assignments)
    references = prior.reference_events(assignments)
    recordings, feature_summary, feature_names = prepare_recordings(assignments)
    targets, target_summary = build_targets(assignments, recordings)

    fit_rows = []
    curve_rows = []
    selection_rows = []
    outer_score_rows = []
    outer_support_rows = []
    all_pids = set(assignments.pid.astype(int))

    for outer in range(1, 6):
        outer_pids = set(folds.loc[folds.fold.eq(outer), "pid"].astype(int))
        task_specs = []
        for inner in sorted(set(range(1, 6)) - {outer}):
            inner_pids = set(folds.loc[folds.fold.eq(inner), "pid"].astype(int))
            task_specs.append(
                {
                    "phase": f"inner_{inner}",
                    "fitting_pids": all_pids - outer_pids - inner_pids,
                    "heldout": assignments[assignments.pid.isin(inner_pids)].copy(),
                }
            )
        task_specs.append(
            {
                "phase": "outer_final",
                "fitting_pids": all_pids - outer_pids,
                "heldout": assignments[assignments.pid.isin(outer_pids)].copy(),
            }
        )

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    run_fit_pair,
                    outer,
                    task["phase"],
                    task["fitting_pids"],
                    task["heldout"],
                    targets,
                    recordings,
                    feature_names,
                )
                for task in task_specs
            ]
            completed = [future.result() for future in futures]

        pooled = {
            candidate: {"scores": [], "support": []} for candidate in CANDIDATES
        }
        for task, result in zip(task_specs, completed):
            for candidate in CANDIDATES:
                scores, support, summary = result[candidate]
                fit_rows.append(summary)
                if task["phase"] == "outer_final":
                    outer_score_rows.append(scores)
                    if candidate == CANDIDATES[0]:
                        outer_support_rows.append(support)
                else:
                    pooled[candidate]["scores"].append(scores)
                    pooled[candidate]["support"].append(support)

        for candidate in CANDIDATES:
            scores = pd.concat(pooled[candidate]["scores"], ignore_index=True)
            support = pd.concat(pooled[candidate]["support"], ignore_index=True)
            curve = threshold_curve(scores, support, references)
            curve.insert(0, "outer_fold", outer)
            curve.insert(0, "candidate", candidate)
            curve_rows.append(curve)
            for budget in ALARM_BUDGETS:
                selected = formulation.select_threshold(curve, budget)
                selected["candidate"] = candidate
                selected["outer_fold"] = outer
                selection_rows.append(selected)
        print(f"completed outer fold {outer}", flush=True)

    fit_summary = pd.DataFrame(fit_rows).sort_values(
        ["outer_fold", "phase", "candidate"]
    ).reset_index(drop=True)
    curves = pd.concat(curve_rows, ignore_index=True)
    selections = pd.DataFrame(selection_rows).sort_values(
        ["outer_fold", "candidate", "alarm_budget_per_hour"]
    ).reset_index(drop=True)
    outer_scores = pd.concat(outer_score_rows, ignore_index=True)
    support = pd.concat(outer_support_rows, ignore_index=True).sort_values(
        ["outer_fold", "subject"]
    ).reset_index(drop=True)
    events = collapse_events(outer_scores, selections)
    evaluation = evaluate_all(events, support, references)
    bootstrap = event_bootstrap(evaluation["participants"])
    localization = localization_summary(evaluation["matches"])
    decisions = decision_table(evaluation["metrics"], bootstrap, fit_summary)
    manifest = artifact_manifest(feature_summary, fit_summary)

    shared_preprocessing = (
        fit_summary.groupby(["outer_fold", "phase"])["preprocessing_signature"]
        .nunique().eq(1).all()
    )
    convergence_failures = int((~fit_summary.converged.astype(bool)).sum())
    target_index = target_summary.set_index("candidate")
    checks = pd.DataFrame(
        [
            {"check": "train_membership", "passed": len(assignments) == 82 and assignments.pid.nunique() == 64, "detail": "82 recordings; 64 participant groups"},
            {"check": "partition_scope", "passed": set(assignments.partition) == {"train"}, "detail": "train partition only"},
            {"check": "fold_membership", "passed": len(folds) == 64 and folds.fold.nunique() == 5, "detail": "five frozen participant-grouped folds"},
            {
                "check": "feature_grid",
                "passed": int(feature_summary.base_supported_boundaries.sum()) == 75539 and int(feature_summary.shifted_supported_boundaries.sum()) == 75428 and int(feature_summary.grid_supported_boundaries.sum()) == 150967,
                "detail": "75,539 base + 75,428 shifted = 150,967 supported boundaries",
            },
            {
                "check": "target_accounting",
                "passed": int(target_summary.negative_rows.min()) == 147055 and target_index.loc["HARD15", "positive_rows"] == 180 and target_index.loc["INT15", "positive_rows"] == 540 and np.allclose(target_summary.positive_weight, 180.0),
                "detail": "147,055 negatives; both candidates carry positive weight 180",
            },
            {"check": "fit_accounting", "passed": len(fit_summary) == 50 and fit_summary.groupby("candidate").size().eq(25).all(), "detail": "25 nested fits per candidate"},
            {"check": "shared_preprocessing", "passed": shared_preprocessing, "detail": "candidate pairs share retained indices and scaler in every fit"},
            {
                "check": "outer_score_coverage",
                "passed": outer_scores.groupby("candidate").size().eq(150967).all() and outer_scores.groupby("candidate").pid.nunique().eq(64).all(),
                "detail": "150,967 boundaries and 64 participants per candidate",
            },
            {
                "check": "threshold_selection",
                "passed": len(selections) == 40 and (selections.false_alarms_per_hour <= selections.alarm_budget_per_hour + 1e-12).all(),
                "detail": "2 candidates x 5 outer folds x 4 alarm budgets",
            },
            {"check": "event_output_accounting", "passed": len(evaluation["metrics"]) == 32 and len(evaluation["participants"]) == 2048, "detail": "32 metric rows; 2,048 participant rows"},
            {
                "check": "external_hashes",
                "passed": all((data_parent() / row.relative_path).stat().st_size == int(row.bytes) and prior.sha256(data_parent() / row.relative_path) == row.sha256 for row in manifest.itertuples(index=False)),
                "detail": f"{len(manifest)} external artifacts verified",
            },
            {"check": "convergence", "passed": convergence_failures == 0, "detail": f"{convergence_failures} of 50 fits reached the iteration limit"},
        ]
    )
    required = checks.loc[~checks.check.eq("convergence"), "passed"].astype(bool)
    if not required.all():
        raise RuntimeError("A required in-run control failed:\n" + checks.to_string(index=False))

    output = output_dir()
    tables = {
        "feature_grid_summary_v0.1.tsv": feature_summary,
        "target_construction_summary_v0.1.tsv": target_summary,
        "model_fit_summary_v0.1.tsv": fit_summary,
        "inner_threshold_curves_v0.1.tsv": curves,
        "inner_threshold_selections_v0.1.tsv": selections,
        "outer_support_v0.1.tsv": support,
        "outer_predicted_events_v0.1.tsv": events,
        "outer_event_metrics_v0.1.tsv": evaluation["metrics"],
        "outer_event_participants_v0.1.tsv": evaluation["participants"],
        "outer_event_matches_v0.1.tsv": evaluation["matches"],
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "primary_localization_summary_v0.1.tsv": localization,
        "hypothesis_decisions_v0.1.tsv": decisions,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, table in tables.items():
        verify_or_create_tsv(table, output / name)

    versions = {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "git_commit": git_commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "protocol_amendment_commit": AMENDMENT_COMMIT,
        "random_seed": BASE_SEED,
    }
    write_text(output / "software_versions_v0.1.json", json.dumps(versions, indent=2, sort_keys=True) + "\n")

    primary = evaluation["metrics"][
        evaluation["metrics"].membership.eq("primary")
        & evaluation["metrics"].tolerance_sec.eq(15.0)
        & np.isclose(evaluation["metrics"].alarm_budget_per_hour, PRIMARY_BUDGET)
    ].set_index("candidate")
    lines = [
        "# Interval-Aware Localization Experiment v0.1", "",
        "Train-only participant-grouped nested evaluation of hard versus interval-distributed boundary labels.", "",
        "| Candidate | Precision | Recall | F1 | False alarms/hour |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        row = primary.loc[candidate]
        lines.append(f"| {candidate} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |")
    decision = decisions.iloc[0]
    lines += [
        "", f"INT15 minus HARD15 F1: {decision.f1_difference:+.4f}.",
        f"INT15 minus HARD15 false alarms/hour: {decision.false_alarms_per_hour_difference:+.4f}.",
        f"Frozen H-INT decision: `{decision.decision}`.",
        f"Convergence: {convergence_failures} of 50 fits reached the iteration limit.", "",
        "Validation and current-test data remained closed. Results are retrospective train-only nested estimates, not evidence of prospective or clinical performance.", "",
    ]
    write_text(output / "README.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
