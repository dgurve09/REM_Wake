"""Event matching and summary functions for stage-first REM-to-Wake baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def metric_values(tp: int, fp: int, fn: int, supported_hours: float) -> dict:
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else math.nan
    false_alarms_per_hour = fp / supported_hours if supported_hours > 0 else math.nan
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": false_alarms_per_hour,
    }


def optimal_matches(
    reference_times: list[float] | np.ndarray,
    prediction_times: list[float] | np.ndarray,
    tolerance_sec: float,
) -> list[tuple[int, int, float]]:
    """Maximize match count, then minimize total absolute timing error."""
    references = np.asarray(reference_times, dtype=float)
    predictions = np.asarray(prediction_times, dtype=float)
    reference_order = np.argsort(references, kind="stable")
    prediction_order = np.argsort(predictions, kind="stable")
    references = references[reference_order]
    predictions = predictions[prediction_order]
    n_reference = len(references)
    n_prediction = len(predictions)

    scores = [[(0, 0.0) for _ in range(n_prediction + 1)] for _ in range(n_reference + 1)]
    actions = [["stop" for _ in range(n_prediction + 1)] for _ in range(n_reference + 1)]

    for reference_index in range(n_reference - 1, -1, -1):
        for prediction_index in range(n_prediction - 1, -1, -1):
            candidates = [
                (scores[reference_index + 1][prediction_index], "skip_reference", 0),
                (scores[reference_index][prediction_index + 1], "skip_prediction", 1),
            ]
            error = abs(references[reference_index] - predictions[prediction_index])
            if error <= tolerance_sec:
                next_score = scores[reference_index + 1][prediction_index + 1]
                candidates.append(((next_score[0] + 1, next_score[1] + error), "match", 2))
            best_score, best_action, _ = max(
                candidates,
                key=lambda item: (item[0][0], -item[0][1], item[2]),
            )
            scores[reference_index][prediction_index] = best_score
            actions[reference_index][prediction_index] = best_action

    matches = []
    reference_index = 0
    prediction_index = 0
    while reference_index < n_reference and prediction_index < n_prediction:
        action = actions[reference_index][prediction_index]
        if action == "match":
            error = abs(references[reference_index] - predictions[prediction_index])
            matches.append(
                (
                    int(reference_order[reference_index]),
                    int(prediction_order[prediction_index]),
                    float(error),
                )
            )
            reference_index += 1
            prediction_index += 1
        elif action == "skip_reference":
            reference_index += 1
        elif action == "skip_prediction":
            prediction_index += 1
        else:
            break
    return matches


def evaluate_events(
    references: pd.DataFrame,
    predictions: pd.DataFrame,
    ignored_references: pd.DataFrame,
    support: pd.DataFrame,
    tolerance_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate eligible references first, then one-to-one ignored references."""
    required_event = {"subject", "pid", "event_time_sec"}
    required_support = {"subject", "pid", "supported_hours"}
    for name, frame in [
        ("references", references),
        ("predictions", predictions),
        ("ignored_references", ignored_references),
    ]:
        if not required_event.issubset(frame.columns):
            raise ValueError(f"{name} is missing required event columns")
    if not required_support.issubset(support.columns):
        raise ValueError("support is missing required columns")
    if support["subject"].duplicated().any():
        raise ValueError("support must contain one row per recording")

    recording_rows = []
    match_rows = []
    subjects = support.sort_values("subject")["subject"].tolist()
    for subject in subjects:
        support_row = support[support["subject"] == subject].iloc[0]
        pid = int(support_row["pid"])
        reference_rows = references[references["subject"] == subject].reset_index(drop=True)
        prediction_rows = predictions[predictions["subject"] == subject].reset_index(drop=True)
        ignored_rows = ignored_references[
            ignored_references["subject"] == subject
        ].reset_index(drop=True)

        reference_times = reference_rows["event_time_sec"].to_numpy(dtype=float)
        prediction_times = prediction_rows["event_time_sec"].to_numpy(dtype=float)
        eligible_matches = optimal_matches(reference_times, prediction_times, tolerance_sec)
        matched_reference = {match[0] for match in eligible_matches}
        matched_prediction = {match[1] for match in eligible_matches}
        for reference_index, prediction_index, error in eligible_matches:
            match_rows.append(
                {
                    "subject": subject,
                    "pid": pid,
                    "match_type": "eligible",
                    "reference_time_sec": reference_times[reference_index],
                    "prediction_time_sec": prediction_times[prediction_index],
                    "absolute_error_sec": error,
                }
            )

        unmatched_prediction_indices = [
            index for index in range(len(prediction_times)) if index not in matched_prediction
        ]
        unmatched_prediction_times = prediction_times[unmatched_prediction_indices]
        ignored_times = ignored_rows["event_time_sec"].to_numpy(dtype=float)
        ignored_matches = optimal_matches(
            ignored_times, unmatched_prediction_times, tolerance_sec
        )
        ignored_prediction_local = {match[1] for match in ignored_matches}
        for ignored_index, prediction_local_index, error in ignored_matches:
            original_prediction_index = unmatched_prediction_indices[prediction_local_index]
            match_rows.append(
                {
                    "subject": subject,
                    "pid": pid,
                    "match_type": "ignored_quality",
                    "reference_time_sec": ignored_times[ignored_index],
                    "prediction_time_sec": prediction_times[original_prediction_index],
                    "absolute_error_sec": error,
                }
            )

        true_positive = len(eligible_matches)
        false_negative = len(reference_times) - len(matched_reference)
        ignored_prediction_count = len(ignored_matches)
        false_positive = len(unmatched_prediction_indices) - len(ignored_prediction_local)
        supported_hours = float(support_row["supported_hours"])
        row = {
            "subject": subject,
            "pid": pid,
            "tolerance_sec": tolerance_sec,
            "reference_events": len(reference_times),
            "predicted_events": len(prediction_times),
            "ignored_reference_events": len(ignored_times),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "ignored_predictions": ignored_prediction_count,
            "supported_hours": supported_hours,
        }
        row.update(metric_values(true_positive, false_positive, false_negative, supported_hours))
        recording_rows.append(row)

    recordings = pd.DataFrame(recording_rows)
    matches = pd.DataFrame(match_rows)
    participant_rows = []
    for pid, group in recordings.groupby("pid"):
        true_positive = int(group["true_positive"].sum())
        false_positive = int(group["false_positive"].sum())
        false_negative = int(group["false_negative"].sum())
        supported_hours = float(group["supported_hours"].sum())
        row = {
            "pid": int(pid),
            "recordings": len(group),
            "reference_events": int(group["reference_events"].sum()),
            "predicted_events": int(group["predicted_events"].sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "ignored_predictions": int(group["ignored_predictions"].sum()),
            "supported_hours": supported_hours,
        }
        row.update(metric_values(true_positive, false_positive, false_negative, supported_hours))
        participant_rows.append(row)
    participants = pd.DataFrame(participant_rows)

    true_positive = int(recordings["true_positive"].sum())
    false_positive = int(recordings["false_positive"].sum())
    false_negative = int(recordings["false_negative"].sum())
    supported_hours = float(recordings["supported_hours"].sum())
    summary = {
        "recordings": len(recordings),
        "pid": recordings["pid"].nunique(),
        "reference_events": int(recordings["reference_events"].sum()),
        "predicted_events": int(recordings["predicted_events"].sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "ignored_predictions": int(recordings["ignored_predictions"].sum()),
        "supported_hours": supported_hours,
    }
    summary.update(metric_values(true_positive, false_positive, false_negative, supported_hours))
    eligible_errors = matches.loc[
        matches["match_type"] == "eligible", "absolute_error_sec"
    ] if len(matches) else pd.Series(dtype=float)
    summary["median_absolute_error_sec"] = (
        float(eligible_errors.median()) if len(eligible_errors) else math.nan
    )
    summary["maximum_absolute_error_sec"] = (
        float(eligible_errors.max()) if len(eligible_errors) else math.nan
    )
    return recordings, participants, matches, summary


def participant_bootstrap(
    participants: pd.DataFrame,
    resamples: int = 2000,
    seed: int = 20260815,
) -> pd.DataFrame:
    if len(participants) == 0:
        raise ValueError("Cannot bootstrap an empty participant table")
    rng = np.random.default_rng(seed)
    rows = []
    values = participants.reset_index(drop=True)
    for bootstrap_index in range(resamples):
        sampled = values.iloc[rng.integers(0, len(values), size=len(values))]
        metrics = metric_values(
            int(sampled["true_positive"].sum()),
            int(sampled["false_positive"].sum()),
            int(sampled["false_negative"].sum()),
            float(sampled["supported_hours"].sum()),
        )
        rows.append({"bootstrap_index": bootstrap_index, **metrics})
    samples = pd.DataFrame(rows)
    result = []
    for metric in ["precision", "recall", "f1", "false_alarms_per_hour"]:
        finite = samples[metric].replace([np.inf, -np.inf], np.nan).dropna()
        result.append(
            {
                "metric": metric,
                "resamples": resamples,
                "seed": seed,
                "lower_95": float(finite.quantile(0.025)),
                "median": float(finite.quantile(0.5)),
                "upper_95": float(finite.quantile(0.975)),
            }
        )
    return pd.DataFrame(result)
