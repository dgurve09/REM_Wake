"""Diagnose alarm-run morphology after the frozen CEF result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import run_causal_endpoint_factorization_v0_1 as experiment
from reviewed_output import verify_or_create_tsv


# Section 1: inputs

def root() -> Path:
    return Path(__file__).resolve().parents[1]


def result_dir() -> Path:
    return root() / "experiments" / experiment.EXPERIMENT_DIR


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir() / name, sep="\t")


# Section 2: above-threshold runs

def recording_runs(group: pd.DataFrame, threshold: float) -> list[dict]:
    group = group.sort_values("candidate_time_sec").reset_index(drop=True)
    times = group.candidate_time_sec.to_numpy(dtype=float)
    scores = group.probability.to_numpy(dtype=float)
    above = scores >= threshold
    rows = []
    start = None
    for index in range(len(group)):
        contiguous = index > 0 and np.isclose(times[index] - times[index - 1], 15.0)
        if above[index] and (start is None or not contiguous):
            if start is not None:
                stop = index - 1
                rows.append(
                    {
                        "run_start_sec": times[start],
                        "run_stop_sec": times[stop],
                        "run_candidates": stop - start + 1,
                        "run_duration_sec": times[stop] - times[start] + 15.0,
                        "run_maximum_score": float(scores[start : stop + 1].max()),
                    }
                )
            start = index
        elif not above[index] and start is not None:
            stop = index - 1
            rows.append(
                {
                    "run_start_sec": times[start],
                    "run_stop_sec": times[stop],
                    "run_candidates": stop - start + 1,
                    "run_duration_sec": times[stop] - times[start] + 15.0,
                    "run_maximum_score": float(scores[start : stop + 1].max()),
                }
            )
            start = None
    if start is not None:
        stop = len(group) - 1
        rows.append(
            {
                "run_start_sec": times[start],
                "run_stop_sec": times[stop],
                "run_candidates": stop - start + 1,
                "run_duration_sec": times[stop] - times[start] + 15.0,
                "run_maximum_score": float(scores[start : stop + 1].max()),
            }
        )
    return rows


def build_run_tables(scores: pd.DataFrame, selections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows, recording_rows = [], []
    threshold_by_fold = selections.set_index("outer_fold").threshold.astype(float).to_dict()
    for (subject, pid, outer), group in scores.groupby(
        ["subject", "pid", "outer_fold"], sort=True
    ):
        threshold = threshold_by_fold[int(outer)]
        above = group.probability.to_numpy(dtype=float) >= threshold
        runs = recording_runs(group, threshold)
        recording_rows.append(
            {
                "subject": subject,
                "pid": int(pid),
                "outer_fold": int(outer),
                "threshold": threshold,
                "candidate_rows": len(group),
                "above_threshold_rows": int(above.sum()),
                "above_threshold_fraction": float(above.mean()),
                "runs": len(runs),
            }
        )
        for run_index, row in enumerate(runs, start=1):
            run_rows.append(
                {
                    "subject": subject,
                    "pid": int(pid),
                    "outer_fold": int(outer),
                    "threshold": threshold,
                    "run_index": run_index,
                    **row,
                }
            )
    return pd.DataFrame(run_rows), pd.DataFrame(recording_rows)


# Section 3: reviewed diagnostic

def main() -> None:
    scores = pd.read_csv(experiment.outer_score_path(), sep="\t", compression="gzip")
    selections = load("inner_threshold_selections_v0.1.tsv")
    alarms = load("outer_predicted_events_v0.1.tsv")
    runs, recordings = build_run_tables(scores, selections)

    pair_label = (
        scores.rem_before_label.astype(bool) & scores.wake_after_label.astype(bool)
    ).astype(int)
    discrimination = pd.DataFrame(
        [
            {
                "target": "hypnogram_REM_to_Wake_pair",
                "rows": len(scores),
                "positive_rows": int(pair_label.sum()),
                "positive_fraction": float(pair_label.mean()),
                "average_precision": float(
                    average_precision_score(pair_label, scores.probability)
                ),
                "roc_auc": float(roc_auc_score(pair_label, scores.probability)),
            }
        ]
    )

    minimum_grid_threshold = float(experiment.formulation.threshold_grid().min())
    summary = pd.DataFrame(
        [
            {
                "recordings": recordings.subject.nunique(),
                "candidate_rows": int(recordings.candidate_rows.sum()),
                "selected_folds_at_grid_minimum": int(
                    np.isclose(selections.threshold, minimum_grid_threshold).sum()
                ),
                "folds": selections.outer_fold.nunique(),
                "above_threshold_rows": int(recordings.above_threshold_rows.sum()),
                "above_threshold_fraction": float(
                    recordings.above_threshold_rows.sum() / recordings.candidate_rows.sum()
                ),
                "alarm_runs": len(runs),
                "stored_alarms": len(alarms),
                "median_run_duration_sec": float(runs.run_duration_sec.median()),
                "p95_run_duration_sec": float(runs.run_duration_sec.quantile(0.95)),
                "maximum_run_duration_sec": float(runs.run_duration_sec.max()),
                "runs_at_least_30_min": int(runs.run_duration_sec.ge(1800.0).sum()),
                "runs_at_least_2_hours": int(runs.run_duration_sec.ge(7200.0).sum()),
            }
        ]
    )

    joined = alarms.merge(
        scores[
            [
                "outer_fold", "subject", "pid", "candidate_time_sec",
                "rem_before_label", "wake_after_label", "probability",
            ]
        ],
        left_on=["outer_fold", "subject", "pid", "event_time_sec"],
        right_on=["outer_fold", "subject", "pid", "candidate_time_sec"],
        validate="one_to_one",
    )
    joined["endpoint_pair"] = np.select(
        [
            joined.rem_before_label.eq(1) & joined.wake_after_label.eq(1),
            joined.rem_before_label.eq(1),
            joined.wake_after_label.eq(1),
        ],
        ["REM_to_Wake", "REM_to_other", "other_to_Wake"],
        default="neither_endpoint",
    )
    alarm_pairs = (
        joined.groupby("endpoint_pair", as_index=False)
        .agg(alarms=("event_time_sec", "size"), median_score=("probability", "median"))
        .sort_values("endpoint_pair")
    )

    verify_or_create_tsv(summary, result_dir() / "alarm_morphology_diagnostic_v0.1.tsv")
    verify_or_create_tsv(discrimination, result_dir() / "pair_score_diagnostic_v0.1.tsv")
    verify_or_create_tsv(alarm_pairs, result_dir() / "predicted_alarm_endpoint_pairs_v0.1.tsv")
    print(summary.to_string(index=False))
    print(discrimination.to_string(index=False))
    print(alarm_pairs.to_string(index=False))


if __name__ == "__main__":
    main()
