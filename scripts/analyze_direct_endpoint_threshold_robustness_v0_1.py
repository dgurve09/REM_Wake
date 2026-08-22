"""Assess DE-D threshold stability without model or test access."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_direct_endpoint_factorization_v0_1 import candidate_score_path
from run_direct_event_baseline_v0_1 import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    MEMBERSHIPS,
    THRESHOLDS,
    TOLERANCES,
    collapse_alarms,
    data_parent,
    evaluate_events,
    local_event_inputs,
    metric_values,
    participant_bootstrap,
    reference_events,
    repo_root,
)


# Section 1: paths and frozen comparison values

VERSION = "v0.1"
COMPARATOR = "DE-D-LOPO"
EXPERIMENT_DIR = "2026-08-22_direct_endpoint_threshold_robustness_v0.1"
SELECTED_THRESHOLD = 0.74
DE_B_F1 = 0.11267605633802817
DE_B_FALSE_ALARMS_PER_HOUR = 1.4495633530303824


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def factorization_dir() -> Path:
    return repo_root() / "experiments/2026-08-22_direct_endpoint_factorization_v0.1"


def baseline_dir() -> Path:
    return repo_root() / "experiments/2026-08-22_direct_event_baseline_v0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Section 2: threshold success intervals

def threshold_success_intervals() -> tuple[pd.DataFrame, pd.DataFrame]:
    curve = pd.read_csv(factorization_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t")
    curve["beats_de_b_f1"] = curve["f1"] > DE_B_F1
    curve["beats_de_b_false_alarm_rate"] = (
        curve["false_alarms_per_hour"] < DE_B_FALSE_ALARMS_PER_HOUR
    )
    curve["meets_two_part_rule"] = (
        curve["beats_de_b_f1"] & curve["beats_de_b_false_alarm_rate"]
    )
    successful = curve[curve["meets_two_part_rule"]].sort_values("threshold").copy()
    if len(successful) == 0:
        intervals = pd.DataFrame(
            columns=[
                "interval_id",
                "threshold_start",
                "threshold_end",
                "threshold_count",
                "threshold_span",
                "contains_selected_threshold",
            ]
        )
        return curve, intervals
    successful["interval_id"] = (
        successful["threshold"].diff().fillna(0.01) > 0.0100001
    ).cumsum()
    rows = []
    for interval_id, group in successful.groupby("interval_id"):
        start = float(group["threshold"].min())
        end = float(group["threshold"].max())
        rows.append(
            {
                "interval_id": int(interval_id),
                "threshold_start": start,
                "threshold_end": end,
                "threshold_count": len(group),
                "threshold_span": end - start,
                "contains_selected_threshold": bool(
                    start <= SELECTED_THRESHOLD <= end
                ),
            }
        )
    return curve, pd.DataFrame(rows)


# Section 3: participant metrics at every threshold

def load_scores_and_support() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(candidate_score_path(), sep="\t", compression="gzip")
    support = pd.read_csv(factorization_dir() / "validation_event_support_v0.1.tsv", sep="\t")
    if set(scores["partition"]) != {"validation"} or set(support["partition"]) != {
        "validation"
    }:
        raise ValueError("Threshold analysis requires validation-only inputs")
    return scores, support


def participant_threshold_metrics(
    scores: pd.DataFrame, support: pd.DataFrame
) -> pd.DataFrame:
    references = reference_events()
    eligible, ignored = local_event_inputs(references, "validation", "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    rows = []
    for threshold in THRESHOLDS:
        alarms = collapse_alarms(scores, threshold)
        _, participants, _, _ = evaluate_events(
            eligible,
            alarms[["subject", "pid", "event_time_sec"]],
            ignored,
            local_support,
            15.0,
        )
        participants.insert(0, "threshold", threshold)
        rows.append(participants)
    result = pd.concat(rows, ignore_index=True)
    expected = len(THRESHOLDS) * support["pid"].nunique()
    if len(result) != expected:
        raise ValueError(f"Expected {expected} participant-threshold rows, found {len(result)}")
    return result


# Section 4: leave-one-participant-out threshold selection

def select_lopo_thresholds(participant_curve: pd.DataFrame) -> pd.DataFrame:
    pids = sorted(participant_curve["pid"].unique())
    rows = []
    for held_out_pid in pids:
        calibration = participant_curve[participant_curve["pid"] != held_out_pid]
        threshold_rows = []
        for threshold, group in calibration.groupby("threshold", sort=True):
            metrics = metric_values(
                int(group["true_positive"].sum()),
                int(group["false_positive"].sum()),
                int(group["false_negative"].sum()),
                float(group["supported_hours"].sum()),
            )
            threshold_rows.append(
                {
                    "held_out_pid": int(held_out_pid),
                    "calibration_pid": group["pid"].nunique(),
                    "threshold": float(threshold),
                    "true_positive": int(group["true_positive"].sum()),
                    "false_positive": int(group["false_positive"].sum()),
                    "false_negative": int(group["false_negative"].sum()),
                    "supported_hours": float(group["supported_hours"].sum()),
                    **metrics,
                }
            )
        local = pd.DataFrame(threshold_rows)
        selected = local.sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        rows.append(selected.to_dict())
    return pd.DataFrame(rows).sort_values("held_out_pid")


def apply_lopo_thresholds(
    scores: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    threshold_map = selected.set_index("held_out_pid")["threshold"].to_dict()
    for pid, group in scores.groupby("pid", sort=True):
        threshold = float(threshold_map[int(pid)])
        alarms = collapse_alarms(group, threshold)
        alarms["calibration_threshold"] = threshold
        alarms["held_out_pid"] = int(pid)
        rows.append(alarms)
    return pd.concat(rows, ignore_index=True)


# Section 5: aggregate held-out evaluation

def evaluate_lopo(
    alarms: pd.DataFrame, support: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    references = reference_events()
    local_support = support[["subject", "pid", "supported_hours"]]
    predictions = alarms[["subject", "pid", "event_time_sec"]]
    summaries = []
    bootstraps = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for membership in MEMBERSHIPS:
        eligible, ignored = local_event_inputs(references, "validation", membership)
        for tolerance in TOLERANCES:
            recordings, participants, matches, summary = evaluate_events(
                eligible, predictions, ignored, local_support, tolerance
            )
            config = {
                "comparator": COMPARATOR,
                "model_version": VERSION,
                "partition": "validation",
                "membership": membership,
                "tolerance_sec": tolerance,
                "threshold_mode": "leave_one_pid_out_calibration",
            }
            summaries.append({**config, **summary})
            bootstrap = participant_bootstrap(
                participants,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            )
            for key, value in reversed(list(config.items())):
                bootstrap.insert(0, key, value)
            bootstraps.append(bootstrap)
            for frame, collection in [
                (recordings, recordings_all),
                (participants, participants_all),
                (matches, matches_all),
            ]:
                if len(frame):
                    frame = frame.copy()
                    for key, value in reversed(list(config.items())):
                        if key not in frame.columns:
                            frame.insert(0, key, value)
                    collection.append(frame)
    return {
        "event_metrics": pd.DataFrame(summaries),
        "event_bootstrap": pd.concat(bootstraps, ignore_index=True),
        "event_recordings": pd.concat(recordings_all, ignore_index=True),
        "event_participants": pd.concat(participants_all, ignore_index=True),
        "event_matches": pd.concat(matches_all, ignore_index=True),
    }


# Section 6: paired LOPO DE-D versus frozen DE-B

def paired_bootstrap_vs_de_b(participants: pd.DataFrame) -> pd.DataFrame:
    de_d = participants[
        (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ].copy()
    de_b = pd.read_csv(
        baseline_dir() / "train_validation_event_participants_v0.1.tsv", sep="\t"
    )
    de_b = de_b[
        (de_b["comparator"] == "DE-B")
        & (de_b["partition"] == "validation")
        & (de_b["membership"] == "primary")
        & (de_b["tolerance_sec"] == 15.0)
    ].copy()
    columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
    ]
    paired = de_d[columns].merge(
        de_b[columns],
        on="pid",
        suffixes=("_de_d", "_de_b"),
        validate="one_to_one",
    )
    if len(paired) != 16:
        raise ValueError(f"Expected 16 paired validation participants, found {len(paired)}")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample_rows = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        metrics = {}
        for suffix in ["de_d", "de_b"]:
            metrics[suffix] = metric_values(
                int(sampled[f"true_positive_{suffix}"].sum()),
                int(sampled[f"false_positive_{suffix}"].sum()),
                int(sampled[f"false_negative_{suffix}"].sum()),
                float(sampled[f"supported_hours_{suffix}"].sum()),
            )
        sample_rows.append(
            {
                "event_f1_difference": metrics["de_d"]["f1"]
                - metrics["de_b"]["f1"],
                "false_alarms_per_hour_difference": metrics["de_d"][
                    "false_alarms_per_hour"
                ]
                - metrics["de_b"]["false_alarms_per_hour"],
            }
        )
    samples = pd.DataFrame(sample_rows)
    points = {}
    aggregate = {}
    for suffix in ["de_d", "de_b"]:
        aggregate[suffix] = metric_values(
            int(paired[f"true_positive_{suffix}"].sum()),
            int(paired[f"false_positive_{suffix}"].sum()),
            int(paired[f"false_negative_{suffix}"].sum()),
            float(paired[f"supported_hours_{suffix}"].sum()),
        )
    points["event_f1_difference"] = aggregate["de_d"]["f1"] - aggregate["de_b"]["f1"]
    points["false_alarms_per_hour_difference"] = aggregate["de_d"][
        "false_alarms_per_hour"
    ] - aggregate["de_b"]["false_alarms_per_hour"]
    rows = []
    for metric, point in points.items():
        rows.append(
            {
                "comparison": "DE-D-LOPO_minus_DE-B_validation",
                "metric": metric,
                "point_difference": point,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "lower_95": float(samples[metric].quantile(0.025)),
                "median": float(samples[metric].quantile(0.5)),
                "upper_95": float(samples[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


# Section 7: decision and records

def make_decision(
    intervals: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    selected_interval = intervals[intervals["contains_selected_threshold"]]
    interval_stable = len(selected_interval) == 1 and int(
        selected_interval.iloc[0]["threshold_count"]
    ) >= 5
    primary = metrics[
        (metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)
    ].iloc[0]
    lopo_beats_de_b = bool(
        (primary.f1 > DE_B_F1)
        and (primary.false_alarms_per_hour < DE_B_FALSE_ALARMS_PER_HOUR)
    )
    return pd.DataFrame(
        [
            {
                "selected_interval_contains_at_least_five_thresholds": interval_stable,
                "lopo_f1": primary.f1,
                "lopo_false_alarms_per_hour": primary.false_alarms_per_hour,
                "lopo_beats_de_b_on_both_metrics": lopo_beats_de_b,
                "threshold_robustness_supported": bool(
                    interval_stable and lopo_beats_de_b
                ),
            }
        ]
    )


def input_manifest() -> pd.DataFrame:
    paths = [
        (
            "analysis_protocol",
            "repo",
            repo_root() / "docs/evaluation/direct_endpoint_threshold_robustness_plan_v0.1.md",
        ),
        ("DE-D_validation_candidate_scores", "data_parent", candidate_score_path()),
        (
            "DE-D_validation_support",
            "repo",
            factorization_dir() / "validation_event_support_v0.1.tsv",
        ),
        (
            "DE-D_validation_threshold_curve",
            "repo",
            factorization_dir() / "validation_threshold_curve_v0.1.tsv",
        ),
        (
            "transition_membership",
            "repo",
            repo_root()
            / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        ),
        (
            "transition_quality",
            "repo",
            repo_root()
            / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        ),
        (
            "DE-B_validation_metrics",
            "repo",
            baseline_dir() / "train_validation_event_metrics_v0.1.tsv",
        ),
        (
            "DE-B_validation_participants",
            "repo",
            baseline_dir() / "train_validation_event_participants_v0.1.tsv",
        ),
    ]
    rows = []
    for role, base, path in paths:
        root = repo_root() if base == "repo" else data_parent()
        rows.append(
            {
                "artifact_role": role,
                "path_base": base,
                "relative_path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    intervals: pd.DataFrame,
    selected: pd.DataFrame,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    decision: pd.DataFrame,
) -> None:
    interval = intervals[intervals["contains_selected_threshold"]].iloc[0]
    primary = metrics[
        (metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)
    ].iloc[0]
    paired_index = paired.set_index("metric")
    f1 = paired_index.loc["event_f1_difference"]
    far = paired_index.loc["false_alarms_per_hour_difference"]
    decision_row = decision.iloc[0]
    text = f"""# DE-D Threshold Robustness v0.1

**Created:** 2026-08-22
**Status:** Post-result exploratory validation analysis
**Plan:** `docs/evaluation/direct_endpoint_threshold_robustness_plan_v0.1.md`
**Test or model access:** None

## Threshold Perturbation

The two-part DE-B improvement rule held from threshold {interval.threshold_start:.2f} through {interval.threshold_end:.2f}, covering {int(interval.threshold_count)} adjacent thresholds. The originally selected threshold was 0.74.

## Leave-One-Participant-Out Calibration

Fold-specific thresholds ranged from {selected.threshold.min():.2f} to {selected.threshold.max():.2f}, with median {selected.threshold.median():.2f}.

Aggregated held-out performance was precision {primary.precision:.4f}, recall {primary.recall:.4f}, event F1 {primary.f1:.4f}, and {primary.false_alarms_per_hour:.4f} false alarms per hour.

Compared with frozen DE-B validation, the paired participant-bootstrap F1-difference interval was {f1.lower_95:+.4f} to {f1.upper_95:+.4f}. The false-alarm-rate-difference interval was {far.lower_95:+.4f} to {far.upper_95:+.4f} per hour.

## Decision

The prespecified threshold-robustness rule was supported: **{decision_row.threshold_robustness_supported}**. This requires both a success interval of at least five adjacent thresholds and LOPO performance better than DE-B in both F1 and false alarms per hour.

This analysis uses saved validation probabilities and cannot establish independent test performance. No raw signal, feature array, fitted model, train row, or current-test artifact was accessed.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


# Section 8: execute

def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    annotated_curve, intervals = threshold_success_intervals()
    scores, support = load_scores_and_support()
    participant_curve = participant_threshold_metrics(scores, support)
    selected = select_lopo_thresholds(participant_curve)
    alarms = apply_lopo_thresholds(scores, selected)
    outputs = evaluate_lopo(alarms, support)
    paired = paired_bootstrap_vs_de_b(outputs["event_participants"])
    decision = make_decision(intervals, outputs["event_metrics"])

    annotated_curve.to_csv(
        output_dir() / "annotated_validation_threshold_curve_v0.1.tsv",
        sep="\t",
        index=False,
    )
    intervals.to_csv(
        output_dir() / "threshold_success_intervals_v0.1.tsv", sep="\t", index=False
    )
    participant_curve.to_csv(
        output_dir() / "participant_threshold_metrics_v0.1.tsv", sep="\t", index=False
    )
    selected.to_csv(
        output_dir() / "lopo_selected_thresholds_v0.1.tsv", sep="\t", index=False
    )
    alarms.to_csv(output_dir() / "lopo_predicted_events_v0.1.tsv", sep="\t", index=False)
    for name, frame in outputs.items():
        frame.to_csv(output_dir() / f"lopo_{name}_v0.1.tsv", sep="\t", index=False)
    paired.to_csv(
        output_dir() / "lopo_vs_de_b_paired_bootstrap_v0.1.tsv", sep="\t", index=False
    )
    decision.to_csv(output_dir() / "threshold_robustness_decision_v0.1.tsv", sep="\t", index=False)
    input_manifest().to_csv(
        output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    write_readme(intervals, selected, outputs["event_metrics"], paired, decision)
    print(intervals.to_string(index=False))
    print(selected[["held_out_pid", "threshold", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(outputs["event_metrics"].to_string(index=False))
    print(paired.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
