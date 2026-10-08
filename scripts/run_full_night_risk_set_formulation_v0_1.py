"""Run the predeclared full-night risk-set formulation experiment."""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler

import build_background_windows_v0_1 as background
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-10-07_full_night_risk_set_formulation_v0.1"
DERIVED_DIR = "full_night_risk_set_formulation_v0.1"
PROTOCOL_COMMIT = "72a37eb"
INITIAL_SEQUENTIAL_COMMIT = "53b1b07"
BASE_SEED = 20261007
OUTER_FOLDS = 5
C_VALUE = 0.1
MAX_ITER = 3000
ALARM_BUDGETS = [0.10, 0.25, 0.50, 1.00]
PRIMARY_ALARM_BUDGET = 0.25
BOOTSTRAP_RESAMPLES = 2000
MAX_CONCURRENT_FITS = 4
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
CANDIDATES = ["SAMP-BAL", "RISK-BAL", "RISK-NAT"]
NEW_CANDIDATES = ["RISK-BAL", "RISK-NAT"]
EPSILON = np.finfo(float).eps


# Section 2: paths and immutable writers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    return data_parent() / "boas_ds005555_v1.1.1"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
    ).strip()


def model_path(candidate: str, outer: int, phase: str) -> Path:
    return derived_dir() / "models" / candidate.lower() / f"outer_{outer}_{phase}_v0.1.json.gz"


def score_path(candidate: str, outer: int, phase: str) -> Path:
    return derived_dir() / "scores" / candidate.lower() / f"outer_{outer}_{phase}_v0.1.tsv.gz"


def risk_table_path() -> Path:
    return derived_dir() / "full_risk_set_candidates_v0.1.tsv.gz"


def risk_score_path() -> Path:
    return derived_dir() / "outer_labeled_risk_set_scores_v0.1.tsv.gz"


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


# Section 3: wearable recordings and full risk set

def prepare_h2_recordings(assignments: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[str]]:
    scaler = prior.frozen_scalers()["H2"]
    recordings = {}
    summaries = []
    flattened_names = None
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        onsets, features, names, summary = prior.load_or_create_features(item.subject, "H2", scaler)
        centers, context = prior.context_matrix(onsets, features)
        recordings[item.subject] = {
            "pid": int(item.pid),
            "centers": centers,
            "F1": context,
        }
        current_names = [
            f"t{int((offset - 4) * prior.EPOCH_SEC):+d}_{name}"
            for offset in range(prior.CONTEXT_EPOCHS)
            for name in names
        ]
        if flattened_names is None:
            flattened_names = current_names
        elif flattened_names != current_names:
            raise ValueError(f"Feature schema changed at {item.subject}")
        summaries.append(
            {
                "pid": int(item.pid),
                **summary,
                "supported_boundaries": len(centers),
            }
        )
        print(f"features {number:02d}/{len(assignments)} {item.subject}", flush=True)
    if flattened_names is None or len(flattened_names) != 360:
        raise ValueError("Unexpected H2 feature schema")
    return recordings, pd.DataFrame(summaries), flattened_names


def regenerate_backgrounds(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = background.load_participants(dataset_root())
    labels = background.read_tsv(
        repo_root() / "labels/transition_labels_v0.1/transition_labels_v0.1.tsv"
    )
    eligible_rows = []
    subject_rows = []
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        subject_id = int(item.subject.replace("sub-", ""))
        eligible, summary, exclusions = background.inspect_subject(
            dataset_root(), participants, labels, subject_id
        )
        eligible_rows.extend(eligible)
        summary["excluded_centers"] = len(exclusions)
        subject_rows.append(summary)
        print(f"backgrounds {number:02d}/{len(assignments)} {item.subject}", flush=True)
    eligible = pd.DataFrame(eligible_rows)
    if len(eligible) != 73476:
        raise ValueError(f"Pre-audited eligible background count changed: {len(eligible)}")
    if set(eligible["subject"]) != set(assignments["subject"]):
        raise ValueError("Background regeneration left the authorized train membership")
    return eligible, pd.DataFrame(subject_rows)


def build_risk_set(
    assignments: pd.DataFrame,
    recordings: dict,
    eligible: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    pid_by_subject = assignments.set_index("subject")["pid"].astype(int).to_dict()
    rows = []
    feature_rows = []
    dropped = 0
    for item in eligible.itertuples(index=False):
        centers = recordings[item.subject]["centers"]
        index = np.searchsorted(centers, float(item.center_sec))
        retained = index < len(centers) and prior.time_key(centers[index]) == prior.time_key(item.center_sec)
        if not retained:
            dropped += 1
            continue
        rows.append(
            {
                "sample_id": f"risk_background_{item.subject}_{prior.time_key(item.center_sec)}",
                "subject": item.subject,
                "pid": int(pid_by_subject[item.subject]),
                "candidate_time_sec": float(item.center_sec),
                "label": 0,
                "source_tier": item.background_tier,
                "center_pair": item.center_pair,
            }
        )
        feature_rows.append(recordings[item.subject]["F1"][index])

    references = prior.reference_events(assignments)
    positives = references[prior.truth(references["primary_analysis_eligible"])].copy()
    for item in positives.itertuples(index=False):
        centers = recordings[item.subject]["centers"]
        index = np.searchsorted(centers, float(item.nominal_boundary_sec))
        retained = (
            index < len(centers)
            and prior.time_key(centers[index]) == prior.time_key(item.nominal_boundary_sec)
        )
        if not retained:
            raise ValueError(f"Primary positive lacks H2 context: {item.transition_id}")
        rows.append(
            {
                "sample_id": f"transition_{item.transition_id}",
                "subject": item.subject,
                "pid": int(item.pid),
                "candidate_time_sec": float(item.nominal_boundary_sec),
                "label": 1,
                "source_tier": "REM_to_Wake",
                "center_pair": "REM_to_Wake",
            }
        )
        feature_rows.append(recordings[item.subject]["F1"][index])

    risk = pd.DataFrame(rows)
    matrix = np.stack(feature_rows).astype(np.float32)
    order = risk.sort_values(["subject", "candidate_time_sec", "label"]).index.to_numpy()
    risk = risk.loc[order].reset_index(drop=True)
    matrix = matrix[order]
    if risk["sample_id"].duplicated().any():
        raise ValueError("Duplicate risk-set sample ID")
    if int(risk["label"].sum()) != 180:
        raise ValueError("Positive risk-set count changed")
    risk.attrs["dropped_backgrounds"] = dropped
    return risk, matrix


# Section 4: prior control and new models

def prior_score_file(outer: int, phase: str) -> Path:
    return (
        data_parent()
        / "derived/paired_enriched_feature_pruning_nested_v0.1/scores/h2/f1_en"
        / f"outer_{outer}_{phase}_c0p1_v0.1.tsv.gz"
    )


def prior_manifest() -> dict[str, str]:
    manifest = pd.read_csv(
        repo_root()
        / "experiments/2026-10-04_paired_enriched_feature_pruning_nested_v0.1"
        / "external_artifact_manifest_v0.1.tsv",
        sep="\t",
    )
    return dict(zip(manifest["relative_path"], manifest["sha256"]))


def load_prior_score(
    outer: int,
    phase: str,
    heldout: pd.DataFrame,
    fitting_pids: set[int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = prior_score_file(outer, phase)
    relative = path.relative_to(data_parent()).as_posix()
    expected = prior_manifest().get(relative)
    if expected is None or prior.sha256(path) != expected:
        raise RuntimeError(f"Prior control score hash mismatch: {relative}")
    scores = pd.read_csv(path, sep="\t", compression="gzip")
    if set(scores["subject"]) != set(heldout["subject"]):
        raise ValueError(f"Prior score membership mismatch: outer {outer}, {phase}")
    scores["candidate"] = "SAMP-BAL"
    support = (
        scores.groupby(["subject", "pid", "outer_fold"], as_index=False)
        .size()
        .rename(columns={"size": "supported_boundaries"})
    )
    support["supported_hours"] = support["supported_boundaries"] * prior.EPOCH_SEC / 3600.0
    sampled = prior.labeled_candidates(prior.train_assignments())
    fitting_rows = sampled[sampled["pid"].isin(fitting_pids)]
    summary = {
        "candidate": "SAMP-BAL",
        "outer_fold": outer,
        "phase": phase,
        "training_rows": len(fitting_rows),
        "positive_rows": int(fitting_rows["label"].sum()),
        "class_weight": "balanced",
        "input_features": 360,
        "retained_after_pruning": np.nan,
        "nonzero_coefficients": np.nan,
        "iterations": np.nan,
        "converged": True,
        "model_relative_path": "reused_prior_reviewed_model",
        "model_sha256": "recorded_in_prior_manifest",
        "score_relative_path": relative,
        "score_sha256": expected,
    }
    return scores, support, summary


def fit_new_model(
    values: np.ndarray,
    labels: np.ndarray,
    candidate: str,
    feature_names: list[str],
) -> tuple[dict, dict]:
    selected = prior.correlation_prune(values)
    scaler = StandardScaler().fit(values[:, selected])
    class_weight = "balanced" if candidate == "RISK-BAL" else None
    model = LogisticRegression(
        C=C_VALUE,
        class_weight=class_weight,
        solver="saga",
        penalty="elasticnet",
        l1_ratio=0.5,
        max_iter=MAX_ITER,
        tol=1e-4,
        random_state=BASE_SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(values[:, selected]), labels)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    nonzero = int(np.sum(np.abs(model.coef_[0]) > 1e-12))
    payload = {
        "configuration": {
            "candidate": candidate,
            "C": C_VALUE,
            "class_weight": class_weight,
            "solver": "saga",
            "penalty": "elasticnet",
            "l1_ratio": 0.5,
            "max_iter": MAX_ITER,
            "tol": 1e-4,
            "random_state": BASE_SEED,
            "variance_threshold": prior.VARIANCE_THRESHOLD,
            "correlation_threshold": prior.CORRELATION_THRESHOLD,
        },
        "selected_indices": selected.tolist(),
        "selected_feature_names": [feature_names[index] for index in selected],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
    }
    fitted = {"selected": selected, "scaler": scaler, "model": model}
    summary = {
        "input_features": values.shape[1],
        "retained_after_pruning": len(selected),
        "nonzero_coefficients": nonzero,
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
        "payload": payload,
    }
    return fitted, summary


def score_recordings(
    fitted: dict,
    heldout: pd.DataFrame,
    recordings: dict,
    candidate: str,
    outer: int,
    phase: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        selected = fitted["selected"]
        scaled = fitted["scaler"].transform(recording["F1"][:, selected])
        probability = fitted["model"].predict_proba(scaled)[:, 1]
        score_rows.append(
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
                "supported_hours": len(probability) * prior.EPOCH_SEC / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def run_new_fit(
    candidate: str,
    outer: int,
    phase: str,
    fit_mask: np.ndarray,
    heldout: pd.DataFrame,
    matrix: np.ndarray,
    labels: np.ndarray,
    recordings: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    model_file = model_path(candidate, outer, phase)
    score_file = score_path(candidate, outer, phase)
    if model_file.exists() != score_file.exists():
        raise RuntimeError(f"Incomplete cached fit: {model_file}")
    if model_file.exists():
        with gzip.open(model_file, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
        scores = pd.read_csv(score_file, sep="\t", compression="gzip")
        if (
            payload["configuration"]["candidate"] != candidate
            or set(scores["subject"]) != set(heldout["subject"])
            or not scores["outer_fold"].eq(outer).all()
            or not scores["phase"].eq(phase).all()
        ):
            raise RuntimeError(f"Cached fit identity mismatch: {model_file}")
        support = (
            scores.groupby(["subject", "pid", "outer_fold"], as_index=False)
            .size()
            .rename(columns={"size": "supported_boundaries"})
        )
        support["supported_hours"] = support["supported_boundaries"] * prior.EPOCH_SEC / 3600.0
        coefficients = np.asarray(payload["coefficient"], dtype=float)
        summary = {
            "input_features": matrix.shape[1],
            "retained_after_pruning": len(payload["selected_indices"]),
            "nonzero_coefficients": int(np.sum(np.abs(coefficients) > 1e-12)),
            "iterations": int(payload["iterations"]),
            "converged": bool(payload["converged"]),
        }
    else:
        fitted, summary = fit_new_model(
            matrix[fit_mask], labels[fit_mask], candidate, feature_names
        )
        scores, support = score_recordings(
            fitted, heldout, recordings, candidate, outer, phase
        )
        prior.verify_or_create_json_gzip(model_file, summary.pop("payload"))
        prior.verify_or_create_gzip_tsv(score_file, scores)

    summary.update(
        {
            "candidate": candidate,
            "outer_fold": outer,
            "phase": phase,
            "training_rows": int(fit_mask.sum()),
            "positive_rows": int(labels[fit_mask].sum()),
            "class_weight": "balanced" if candidate == "RISK-BAL" else "none",
            "model_relative_path": model_file.relative_to(data_parent()).as_posix(),
            "model_sha256": prior.sha256(model_file),
            "score_relative_path": score_file.relative_to(data_parent()).as_posix(),
            "score_sha256": prior.sha256(score_file),
        }
    )
    return scores, support, summary


# Section 5: alarm-budget selection and event evaluation

def threshold_grid() -> np.ndarray:
    logits = np.arange(-16.0, 16.001, 0.25)
    return np.unique(np.concatenate([1.0 / (1.0 + np.exp(-logits)), [1.0]]))


def threshold_curve(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for threshold in threshold_grid():
        rows.append(
            {
                "threshold": float(threshold),
                **prior.fast_primary_summary(scores, support, references, float(threshold)),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(curve: pd.DataFrame, budget: float) -> dict:
    eligible = curve[curve["false_alarms_per_hour"] <= budget + 1e-12]
    if eligible.empty:
        raise ValueError(f"No threshold satisfies FAR budget {budget}")
    selected = eligible.sort_values(
        ["recall", "precision", "f1", "threshold"],
        ascending=[False, False, False, False],
        kind="stable",
    ).iloc[0].to_dict()
    selected["alarm_budget_per_hour"] = budget
    selected["selection_rule"] = "max_recall_then_precision_then_f1_then_threshold_under_far_budget"
    return selected


def collapse_outer_events(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = selections[np.isclose(selections["alarm_budget_per_hour"], PRIMARY_ALARM_BUDGET)]
    for item in primary.itertuples(index=False):
        local = scores[
            scores["candidate"].eq(item.candidate)
            & scores["outer_fold"].eq(item.outer_fold)
        ]
        for (subject, pid), group in local.groupby(["subject", "pid"], sort=True):
            times = prior.collapsed_times(group.sort_values("candidate_time_sec"), item.threshold)
            for event_time in times:
                rows.append(
                    {
                        "candidate": item.candidate,
                        "outer_fold": int(item.outer_fold),
                        "subject": subject,
                        "pid": int(pid),
                        "event_time_sec": float(event_time),
                        "threshold": float(item.threshold),
                        "alarm_budget_per_hour": PRIMARY_ALARM_BUDGET,
                    }
                )
    columns = [
        "candidate", "outer_fold", "subject", "pid", "event_time_sec", "threshold",
        "alarm_budget_per_hour",
    ]
    return pd.DataFrame(rows, columns=columns)


def evaluate_all(
    events: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metric_rows = []
    participant_rows = []
    match_rows = []
    for candidate in CANDIDATES:
        predictions = events[events["candidate"].eq(candidate)]
        for membership in MEMBERSHIPS:
            eligible, ignored = prior.local_event_inputs(references, membership)
            for tolerance in TOLERANCES:
                _, participants, matches, summary = evaluate_events(
                    eligible,
                    predictions[["subject", "pid", "event_time_sec"]],
                    ignored,
                    support[["subject", "pid", "supported_hours"]],
                    tolerance,
                )
                config = {
                    "candidate": candidate,
                    "partition": "train_nested_oof",
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "alarm_budget_per_hour": PRIMARY_ALARM_BUDGET,
                }
                metric_rows.append({**config, **summary})
                for frame, destination in [(participants, participant_rows), (matches, match_rows)]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        destination.append(local)
    return {
        "metrics": pd.DataFrame(metric_rows),
        "participants": pd.concat(participant_rows, ignore_index=True),
        "matches": pd.concat(match_rows, ignore_index=True),
    }


# Section 6: labelled risk-set metrics and paired uncertainty

def labelled_risk_scores(outer_scores: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    frames = []
    keys = ["subject", "pid", "candidate_time_sec"]
    for candidate in CANDIDATES:
        local = outer_scores[outer_scores["candidate"].eq(candidate)][
            keys + ["outer_fold", "probability"]
        ]
        joined = risk.merge(local, on=keys, how="left", validate="one_to_one")
        if joined["probability"].isna().any():
            raise ValueError(f"Missing outer risk-set probabilities: {candidate}")
        joined.insert(0, "candidate", candidate)
        frames.append(joined)
    return pd.concat(frames, ignore_index=True)


def calibration_values(labels: np.ndarray, probability: np.ndarray) -> dict:
    clipped = np.clip(probability, EPSILON, 1.0 - EPSILON)
    return {
        "average_precision": float(average_precision_score(labels, probability)),
        "brier_score": float(brier_score_loss(labels, probability)),
        "log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
    }


def risk_metrics(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        local = scores[scores["candidate"].eq(candidate)]
        rows.append(
            {
                "candidate": candidate,
                "rows": len(local),
                "positive_rows": int(local["label"].sum()),
                "prevalence": float(local["label"].mean()),
                **calibration_values(
                    local["label"].to_numpy(dtype=int),
                    local["probability"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def event_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        participants["membership"].eq("primary")
        & participants["tolerance_sec"].eq(15.0)
    ]
    comparisons = [("RISK-BAL", "SAMP-BAL"), ("RISK-NAT", "RISK-BAL")]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    rows = []
    rng = np.random.default_rng(BASE_SEED)
    draws = rng.integers(0, 64, size=(BOOTSTRAP_RESAMPLES, 64))
    for left_name, right_name in comparisons:
        left = primary[primary["candidate"].eq(left_name)][columns].sort_values("pid")
        right = primary[primary["candidate"].eq(right_name)][columns].sort_values("pid")
        paired = left.merge(right, on="pid", suffixes=("_left", "_right"), validate="one_to_one")
        if len(paired) != 64:
            raise ValueError("Incomplete event bootstrap pairing")
        samples = []
        for indices in draws:
            sample = paired.iloc[indices]
            values = {}
            for side in ["left", "right"]:
                values[side] = metric_values(
                    int(sample[f"true_positive_{side}"].sum()),
                    int(sample[f"false_positive_{side}"].sum()),
                    int(sample[f"false_negative_{side}"].sum()),
                    float(sample[f"supported_hours_{side}"].sum()),
                )
            samples.append(
                {
                    "recall_difference": values["left"]["recall"] - values["right"]["recall"],
                    "f1_difference": values["left"]["f1"] - values["right"]["f1"],
                    "false_alarms_per_hour_difference": values["left"]["false_alarms_per_hour"]
                    - values["right"]["false_alarms_per_hour"],
                }
            )
        frame = pd.DataFrame(samples)
        for metric in frame.columns:
            rows.append(
                {
                    "analysis": "event",
                    "comparison": f"{left_name}_minus_{right_name}",
                    "metric": metric,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BASE_SEED,
                    "lower_95": float(frame[metric].quantile(0.025)),
                    "median": float(frame[metric].quantile(0.5)),
                    "upper_95": float(frame[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def risk_bootstrap(scores: pd.DataFrame) -> pd.DataFrame:
    wide = scores.pivot(
        index=["sample_id", "subject", "pid", "candidate_time_sec", "label"],
        columns="candidate",
        values="probability",
    ).reset_index()
    pid_values = np.asarray(sorted(wide["pid"].unique()), dtype=int)
    if len(pid_values) != 64:
        raise ValueError("Incomplete risk bootstrap membership")
    pid_to_index = {pid: index for index, pid in enumerate(pid_values)}
    row_pid = wide["pid"].map(pid_to_index).to_numpy(dtype=int)
    labels = wide["label"].to_numpy(dtype=int)
    probabilities = {candidate: wide[candidate].to_numpy(dtype=float) for candidate in CANDIDATES}
    rng = np.random.default_rng(BASE_SEED)
    samples = {"RISK-BAL_minus_SAMP-BAL": [], "RISK-NAT_minus_RISK-BAL": []}
    for _ in range(BOOTSTRAP_RESAMPLES):
        drawn = rng.integers(0, len(pid_values), size=len(pid_values))
        weights = np.bincount(drawn, minlength=len(pid_values))[row_pid]
        keep = weights > 0
        metrics = {}
        for candidate in CANDIDATES:
            probability = probabilities[candidate]
            metrics[candidate] = {
                "average_precision": float(
                    average_precision_score(
                        labels[keep], probability[keep], sample_weight=weights[keep]
                    )
                ),
                "brier_score": float(
                    np.average((probability[keep] - labels[keep]) ** 2, weights=weights[keep])
                ),
            }
        for left, right in [("RISK-BAL", "SAMP-BAL"), ("RISK-NAT", "RISK-BAL")]:
            samples[f"{left}_minus_{right}"].append(
                {
                    metric: metrics[left][metric] - metrics[right][metric]
                    for metric in ["average_precision", "brier_score"]
                }
            )
    rows = []
    for comparison, values in samples.items():
        frame = pd.DataFrame(values)
        for metric in frame.columns:
            rows.append(
                {
                    "analysis": "risk_set",
                    "comparison": comparison,
                    "metric": f"{metric}_difference",
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BASE_SEED,
                    "lower_95": float(frame[metric].quantile(0.025)),
                    "median": float(frame[metric].quantile(0.5)),
                    "upper_95": float(frame[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def decision_table(event_metrics: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    primary = event_metrics[
        event_metrics["membership"].eq("primary")
        & event_metrics["tolerance_sec"].eq(15.0)
    ].set_index("candidate")
    risk = calibration.set_index("candidate")
    risk_recall_gain = float(primary.loc["RISK-BAL", "recall"] - primary.loc["SAMP-BAL", "recall"])
    brier_reduction = float(
        (risk.loc["RISK-BAL", "brier_score"] - risk.loc["RISK-NAT", "brier_score"])
        / risk.loc["RISK-BAL", "brier_score"]
    )
    natural_recall_difference = float(
        primary.loc["RISK-NAT", "recall"] - primary.loc["RISK-BAL", "recall"]
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-RISK",
                "comparison": "RISK-BAL_minus_SAMP-BAL",
                "primary_value": risk_recall_gain,
                "required_primary_value": 0.05,
                "secondary_value": np.nan,
                "required_secondary_value": np.nan,
                "point_gate_pass": risk_recall_gain >= 0.05,
                "decision": "retain_for_new_confirmation" if risk_recall_gain >= 0.05 else "stop_v0.1",
            },
            {
                "hypothesis": "H-PRIOR",
                "comparison": "RISK-NAT_minus_RISK-BAL",
                "primary_value": brier_reduction,
                "required_primary_value": 0.10,
                "secondary_value": natural_recall_difference,
                "required_secondary_value": -0.05,
                "point_gate_pass": brier_reduction >= 0.10 and natural_recall_difference >= -0.05,
                "decision": (
                    "retain_for_new_confirmation"
                    if brier_reduction >= 0.10 and natural_recall_difference >= -0.05
                    else "stop_v0.1"
                ),
            },
        ]
    )


# Section 7: complete execution and reviewed outputs

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    assignments = prior.train_assignments()
    folds = prior.frozen_fold_assignments(assignments)
    references = prior.reference_events(assignments)
    recordings, feature_summary, feature_names = prepare_h2_recordings(assignments)
    eligible, background_summary = regenerate_backgrounds(assignments)
    risk, matrix = build_risk_set(assignments, recordings, eligible)
    labels = risk["label"].to_numpy(dtype=int)
    groups = risk["pid"].to_numpy(dtype=int)

    prior.verify_or_create_gzip_tsv(risk_table_path(), risk)
    fit_rows = []
    curve_rows = []
    selection_rows = []
    outer_score_rows = []
    outer_support_rows = []

    for outer in range(1, OUTER_FOLDS + 1):
        outer_pid = set(folds[folds["fold"].eq(outer)]["pid"].astype(int))
        pooled = {candidate: {"scores": [], "support": []} for candidate in CANDIDATES}
        fit_tasks = []
        for inner in sorted(set(range(1, OUTER_FOLDS + 1)) - {outer}):
            inner_pid = set(folds[folds["fold"].eq(inner)]["pid"].astype(int))
            heldout = assignments[assignments["pid"].isin(inner_pid)].copy()
            fitting_pids = set(assignments["pid"].astype(int)) - outer_pid - inner_pid
            scores, support, summary = load_prior_score(
                outer, f"inner_{inner}", heldout, fitting_pids
            )
            pooled["SAMP-BAL"]["scores"].append(scores)
            pooled["SAMP-BAL"]["support"].append(support)
            fit_rows.append(summary)
            fit_mask = ~np.isin(groups, list(outer_pid | inner_pid))
            for candidate in NEW_CANDIDATES:
                fit_tasks.append(
                    {
                        "candidate": candidate,
                        "outer": outer,
                        "phase": f"inner_{inner}",
                        "fit_mask": fit_mask,
                        "heldout": heldout,
                        "inner": inner,
                    }
                )

        outer_heldout = assignments[assignments["pid"].isin(outer_pid)].copy()
        outer_fit_mask = ~np.isin(groups, list(outer_pid))
        for candidate in NEW_CANDIDATES:
            fit_tasks.append(
                {
                    "candidate": candidate,
                    "outer": outer,
                    "phase": "outer_final",
                    "fit_mask": outer_fit_mask,
                    "heldout": outer_heldout,
                    "inner": None,
                }
            )

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FITS) as executor:
            futures = [
                executor.submit(
                    run_new_fit,
                    task["candidate"],
                    task["outer"],
                    task["phase"],
                    task["fit_mask"],
                    task["heldout"],
                    matrix,
                    labels,
                    recordings,
                    feature_names,
                )
                for task in fit_tasks
            ]
            fitted_results = [future.result() for future in futures]

        outer_new_results = []
        for task, result in zip(fit_tasks, fitted_results):
            scores, support, summary = result
            fit_rows.append(summary)
            if task["phase"] == "outer_final":
                outer_new_results.append(result)
            else:
                pooled[task["candidate"]]["scores"].append(scores)
                pooled[task["candidate"]]["support"].append(support)

        for candidate in CANDIDATES:
            scores = pd.concat(pooled[candidate]["scores"], ignore_index=True)
            support = pd.concat(pooled[candidate]["support"], ignore_index=True)
            curve = threshold_curve(scores, support, references)
            curve.insert(0, "outer_fold", outer)
            curve.insert(0, "candidate", candidate)
            curve_rows.append(curve)
            for budget in ALARM_BUDGETS:
                selected = select_threshold(curve, budget)
                selected["candidate"] = candidate
                selected["outer_fold"] = outer
                selection_rows.append(selected)

        fitting_pids = set(assignments["pid"].astype(int)) - outer_pid
        scores, support, summary = load_prior_score(
            outer, "outer_final", outer_heldout, fitting_pids
        )
        outer_score_rows.append(scores)
        outer_support_rows.append(support)
        fit_rows.append(summary)
        for scores, support, summary in outer_new_results:
            outer_score_rows.append(scores)
            outer_support_rows.append(support)
        print(f"completed outer fold {outer}", flush=True)

    fit_summary = pd.DataFrame(fit_rows)
    curves = pd.concat(curve_rows, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    outer_scores = pd.concat(outer_score_rows, ignore_index=True)
    support = (
        pd.concat(outer_support_rows, ignore_index=True)
        .drop_duplicates(["subject", "pid", "outer_fold"])
        .sort_values("subject")
    )
    events = collapse_outer_events(outer_scores, selections)
    evaluation = evaluate_all(events, support, references)
    labelled_scores = labelled_risk_scores(outer_scores, risk)
    prior.verify_or_create_gzip_tsv(risk_score_path(), labelled_scores)
    calibration = risk_metrics(labelled_scores)
    bootstrap = pd.concat(
        [event_bootstrap(evaluation["participants"]), risk_bootstrap(labelled_scores)],
        ignore_index=True,
    )
    decisions = decision_table(evaluation["metrics"], calibration)

    stage_pairs = (
        eligible.groupby(["background_tier", "center_pair"], as_index=False)
        .size()
        .rename(columns={"size": "eligible_rows"})
    )
    candidate_summary = pd.DataFrame(
        [
            {"metric": "eligible_backgrounds_before_context_intersection", "value": len(eligible)},
            {"metric": "backgrounds_dropped_at_context_intersection", "value": risk.attrs["dropped_backgrounds"]},
            {"metric": "retained_backgrounds", "value": int((risk["label"] == 0).sum())},
            {"metric": "retained_primary_events", "value": int(risk["label"].sum())},
            {"metric": "risk_set_rows", "value": len(risk)},
            {"metric": "risk_set_prevalence", "value": float(risk["label"].mean())},
            {"metric": "old_review_negative_rows", "value": 2563},
            {"metric": "old_review_raw_prevalence", "value": 180 / 2743},
        ]
    )
    artifact_rows = [
        {
            "relative_path": risk_table_path().relative_to(data_parent()).as_posix(),
            "bytes": risk_table_path().stat().st_size,
            "sha256": prior.sha256(risk_table_path()),
        },
        {
            "relative_path": risk_score_path().relative_to(data_parent()).as_posix(),
            "bytes": risk_score_path().stat().st_size,
            "sha256": prior.sha256(risk_score_path()),
        },
    ]
    for row in fit_summary.itertuples(index=False):
        if row.model_relative_path != "reused_prior_reviewed_model":
            model_file = data_parent() / row.model_relative_path
            artifact_rows.append(
                {
                    "relative_path": row.model_relative_path,
                    "bytes": model_file.stat().st_size,
                    "sha256": row.model_sha256,
                }
            )
        score_file = data_parent() / row.score_relative_path
        artifact_rows.append(
            {
                "relative_path": row.score_relative_path,
                "bytes": score_file.stat().st_size,
                "sha256": row.score_sha256,
            }
        )
    manifest = pd.DataFrame(artifact_rows).drop_duplicates("relative_path").sort_values("relative_path")

    convergence_failures = int((fit_summary["candidate"].isin(NEW_CANDIDATES) & ~fit_summary["converged"]).sum())
    checks = pd.DataFrame(
        [
            {"check": "train_membership", "passed": len(assignments) == 82 and assignments["pid"].nunique() == 64, "detail": "82 recordings; 64 pid groups"},
            {"check": "partition_scope", "passed": set(assignments["partition"]) == {"train"}, "detail": "train only"},
            {"check": "fold_membership", "passed": len(folds) == 64 and folds["fold"].nunique() == 5, "detail": "five frozen pid folds"},
            {"check": "positive_count", "passed": int(risk["label"].sum()) == 180, "detail": f"{int(risk['label'].sum())} retained positives"},
            {"check": "full_risk_set", "passed": int((risk["label"] == 0).sum()) == len(eligible) - risk.attrs["dropped_backgrounds"], "detail": f"{int((risk['label'] == 0).sum())} retained negatives"},
            {"check": "outer_candidate_coverage", "passed": outer_scores.groupby("candidate")["pid"].nunique().eq(64).all(), "detail": "64 pid groups per candidate"},
            {"check": "threshold_selection", "passed": len(selections) == 60, "detail": "3 candidates x 5 folds x 4 budgets"},
            {"check": "external_hashes", "passed": all(prior.sha256(data_parent() / row.relative_path) == row.sha256 for row in manifest.itertuples(index=False)), "detail": f"{len(manifest)} artifacts verified"},
            {"check": "convergence", "passed": convergence_failures == 0, "detail": f"{convergence_failures} new-fit convergence failures retained"},
        ]
    )
    if not checks.loc[~checks["check"].eq("convergence"), "passed"].all():
        raise RuntimeError("A required in-run control failed")

    output = output_dir()
    verify_or_create_tsv(feature_summary, output / "feature_cache_summary_v0.1.tsv")
    verify_or_create_tsv(background_summary, output / "background_subject_summary_v0.1.tsv")
    verify_or_create_tsv(stage_pairs, output / "background_stage_pair_summary_v0.1.tsv")
    verify_or_create_tsv(candidate_summary, output / "candidate_construction_summary_v0.1.tsv")
    verify_or_create_tsv(fit_summary, output / "model_fit_summary_v0.1.tsv")
    verify_or_create_tsv(curves, output / "inner_threshold_curves_v0.1.tsv")
    verify_or_create_tsv(selections, output / "inner_threshold_selections_v0.1.tsv")
    verify_or_create_tsv(support, output / "outer_support_v0.1.tsv")
    verify_or_create_tsv(events, output / "outer_predicted_events_v0.1.tsv")
    verify_or_create_tsv(evaluation["metrics"], output / "outer_event_metrics_v0.1.tsv")
    verify_or_create_tsv(evaluation["participants"], output / "outer_event_participants_v0.1.tsv")
    verify_or_create_tsv(evaluation["matches"], output / "outer_event_matches_v0.1.tsv")
    verify_or_create_tsv(calibration, output / "outer_risk_set_metrics_v0.1.tsv")
    verify_or_create_tsv(bootstrap, output / "paired_participant_bootstrap_v0.1.tsv")
    verify_or_create_tsv(decisions, output / "hypothesis_decisions_v0.1.tsv")
    verify_or_create_tsv(manifest, output / "external_artifact_manifest_v0.1.tsv")
    verify_or_create_tsv(checks, output / "in_run_checks_v0.1.tsv")

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "git_commit": git_commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "initial_sequential_code_commit": INITIAL_SEQUENTIAL_COMMIT,
        "execution_note": (
            "The first two completed fits used the identical frozen fit function at the initial "
            "sequential commit; remaining independent fits used concurrency-only orchestration."
        ),
    }
    verify_or_create_text(
        output / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )

    primary = evaluation["metrics"][
        evaluation["metrics"]["membership"].eq("primary")
        & evaluation["metrics"]["tolerance_sec"].eq(15.0)
    ].set_index("candidate")
    calibration_index = calibration.set_index("candidate")
    lines = [
        "# Full-Night Risk-Set Formulation Experiment v0.1",
        "",
        "Train-only participant-grouped nested results under the predeclared alarm-budget protocol.",
        "",
        "| Candidate | Precision | Recall | F1 | False alarms/hour | Average precision | Brier score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        event = primary.loc[candidate]
        risk_row = calibration_index.loc[candidate]
        lines.append(
            f"| {candidate} | {event.precision:.4f} | {event.recall:.4f} | {event.f1:.4f} | "
            f"{event.false_alarms_per_hour:.4f} | {risk_row.average_precision:.4f} | "
            f"{risk_row.brier_score:.6f} |"
        )
    lines += [
        "",
        f"Full risk set: {len(risk):,} rows, including {int(risk['label'].sum())} primary events.",
        f"Convergence: {convergence_failures} of {int(fit_summary['candidate'].isin(NEW_CANDIDATES).sum())} new fits reached the iteration limit; failures were retained.",
        "",
        "The frozen decisions are recorded in `hypothesis_decisions_v0.1.tsv`.",
        "Validation and current-test data remained closed. This retrospective formulation test does not establish real-time performance or resolve 30-second label uncertainty.",
        "",
    ]
    verify_or_create_text(output / "README.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
