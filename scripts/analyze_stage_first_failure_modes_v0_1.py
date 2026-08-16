"""Explain stage-first event failures using only frozen saved result tables."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd


VERSION = "v0.1"
EXPERIMENT_DIR = "2026-08-15_stage_first_failure_analysis_v0.1"
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


# Section 1: paths and frozen result loading

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def experiments_root() -> Path:
    return repo_root() / "experiments"


def output_dir() -> Path:
    return experiments_root() / EXPERIMENT_DIR


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_dir() -> Path:
    return experiments_root() / "2026-08-15_stage_first_fixed_comparator_v0.1"


def feature_dir() -> Path:
    return experiments_root() / "2026-08-15_stage_first_feature_baseline_v0.1"


def analysis_input_paths() -> list[Path]:
    return [
        fixed_dir() / "stage_metrics_v0.1.tsv",
        fixed_dir() / "stage_class_metrics_v0.1.tsv",
        fixed_dir() / "event_metrics_v0.1.tsv",
        fixed_dir() / "event_participant_metrics_v0.1.tsv",
        feature_dir() / "train_validation_stage_predictions_v0.1.tsv",
        feature_dir() / "test_stage_predictions_v0.1.tsv",
        feature_dir() / "train_validation_stage_metrics_v0.1.tsv",
        feature_dir() / "test_stage_metrics_v0.1.tsv",
        feature_dir() / "train_validation_stage_class_metrics_v0.1.tsv",
        feature_dir() / "test_stage_class_metrics_v0.1.tsv",
        feature_dir() / "train_validation_event_metrics_v0.1.tsv",
        feature_dir() / "test_event_metrics_v0.1.tsv",
        feature_dir() / "train_validation_event_participants_v0.1.tsv",
        feature_dir() / "test_event_participants_v0.1.tsv",
        feature_dir() / "train_validation_event_matches_v0.1.tsv",
        feature_dir() / "test_event_matches_v0.1.tsv",
        repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
    ]


def analysis_input_manifest() -> pd.DataFrame:
    rows = []
    for path in analysis_input_paths():
        rows.append(
            {
                "analysis_version": VERSION,
                "path_relative_to_repo": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def transparent_predictions() -> pd.DataFrame:
    return pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_stage_predictions_v0.1.tsv"),
            read_tsv(feature_dir() / "test_stage_predictions_v0.1.tsv"),
        ],
        ignore_index=True,
    )


def all_stage_metrics() -> pd.DataFrame:
    fixed = read_tsv(fixed_dir() / "stage_metrics_v0.1.tsv").rename(
        columns={"comparator_version": "model_version"}
    )
    transparent = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_stage_metrics_v0.1.tsv"),
            read_tsv(feature_dir() / "test_stage_metrics_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return pd.concat([fixed, transparent], ignore_index=True)


def all_stage_class_metrics() -> pd.DataFrame:
    fixed = read_tsv(fixed_dir() / "stage_class_metrics_v0.1.tsv").rename(
        columns={"comparator_version": "model_version"}
    )
    transparent = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_stage_class_metrics_v0.1.tsv"),
            read_tsv(feature_dir() / "test_stage_class_metrics_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return pd.concat([fixed, transparent], ignore_index=True).sort_values(
        ["partition", "comparator", "stage_code"]
    )


def all_event_metrics() -> pd.DataFrame:
    fixed = read_tsv(fixed_dir() / "event_metrics_v0.1.tsv").rename(
        columns={"comparator_version": "model_version"}
    )
    transparent = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_metrics_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_metrics_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return pd.concat([fixed, transparent], ignore_index=True)


def all_participant_metrics() -> pd.DataFrame:
    fixed = read_tsv(fixed_dir() / "event_participant_metrics_v0.1.tsv").rename(
        columns={"comparator_version": "model_version"}
    )
    transparent = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_participants_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_participants_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return pd.concat([fixed, transparent], ignore_index=True)


def transparent_matches() -> pd.DataFrame:
    return pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_matches_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_matches_v0.1.tsv"),
        ],
        ignore_index=True,
    )


def rem_to_wake_references() -> pd.DataFrame:
    membership = read_tsv(
        repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv"
    )
    quality = read_tsv(
        repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv"
    )[["transition_id", "nominal_boundary_sec"]]
    rows = membership.merge(quality, on="transition_id", validate="one_to_one")
    rows = rows[rows["is_primary_label"].astype(str).str.lower().eq("true")].copy()
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    rows["primary_analysis_eligible"] = (
        rows["primary_analysis_eligible"].astype(str).str.lower().eq("true")
    )
    rows["expanded_quality_analysis_eligible"] = (
        rows["expanded_quality_analysis_eligible"].astype(str).str.lower().eq("true")
    )
    return rows


# Section 2: reference-side boundary mechanisms

def boundary_mechanism(previous, current) -> str:
    if pd.isna(previous) or pd.isna(current):
        return "prediction_coverage_missing"
    previous = int(previous)
    current = int(current)
    if previous == 4 and current == 0:
        return "detected_rem_to_wake"
    if previous != 4 and current == 0:
        return "preceding_rem_not_predicted"
    if previous == 4 and current != 0:
        return "following_wake_not_predicted"
    return "both_target_endpoints_not_predicted"


def analyze_reference_boundaries(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = rem_to_wake_references()
    rows = []
    for comparator in ["SF-B", "SF-C"]:
        local_predictions = predictions[predictions["comparator"] == comparator]
        lookup = {
            (row.subject, float(row.onset)): int(row.stage_pred)
            for row in local_predictions.itertuples(index=False)
        }
        for reference in references.itertuples(index=False):
            previous = lookup.get((reference.subject, float(reference.event_time_sec - 30.0)))
            current = lookup.get((reference.subject, float(reference.event_time_sec)))
            rows.append(
                {
                    "analysis_version": VERSION,
                    "comparator": comparator,
                    "partition": reference.partition,
                    "transition_id": int(reference.transition_id),
                    "subject": reference.subject,
                    "pid": int(reference.pid),
                    "event_time_sec": float(reference.event_time_sec),
                    "primary_analysis_eligible": bool(reference.primary_analysis_eligible),
                    "expanded_quality_analysis_eligible": bool(
                        reference.expanded_quality_analysis_eligible
                    ),
                    "predicted_previous_stage": previous,
                    "predicted_current_stage": current,
                    "boundary_mechanism": boundary_mechanism(previous, current),
                }
            )
    detailed = pd.DataFrame(rows)
    summaries = []
    for membership, column in [
        ("primary", "primary_analysis_eligible"),
        ("expanded", "expanded_quality_analysis_eligible"),
    ]:
        eligible = detailed[detailed[column]]
        grouped = (
            eligible.groupby(["comparator", "partition", "boundary_mechanism"])
            .size()
            .rename("reference_events")
            .reset_index()
        )
        totals = grouped.groupby(["comparator", "partition"])["reference_events"].transform("sum")
        grouped.insert(2, "membership", membership)
        grouped["total_reference_events"] = totals
        grouped["fraction_of_references"] = grouped["reference_events"] / totals
        summaries.append(grouped)
    return detailed, pd.concat(summaries, ignore_index=True)


def summarize_reference_endpoints(detailed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for membership, column in [
        ("primary", "primary_analysis_eligible"),
        ("expanded", "expanded_quality_analysis_eligible"),
    ]:
        eligible = detailed[detailed[column]]
        for endpoint, prediction_column, expected_stage in [
            ("preceding_rem", "predicted_previous_stage", 4),
            ("following_wake", "predicted_current_stage", 0),
        ]:
            local = eligible[["comparator", "partition", prediction_column]].copy()
            local["predicted_stage"] = local[prediction_column].map(
                lambda value: "missing" if pd.isna(value) else str(int(value))
            )
            summary = (
                local.groupby(["comparator", "partition", "predicted_stage"])
                .size()
                .rename("reference_events")
                .reset_index()
            )
            totals = summary.groupby(["comparator", "partition"])["reference_events"].transform("sum")
            summary.insert(2, "membership", membership)
            summary.insert(3, "endpoint", endpoint)
            summary.insert(4, "expected_stage_code", expected_stage)
            summary.insert(5, "expected_stage", STAGE_NAMES[expected_stage])
            summary["predicted_stage_name"] = summary["predicted_stage"].map(
                lambda value: "Missing" if value == "missing" else STAGE_NAMES[int(value)]
            )
            summary["total_reference_events"] = totals
            summary["fraction_of_references"] = summary["reference_events"] / totals
            rows.append(summary)
    return pd.concat(rows, ignore_index=True)


# Section 3: prediction-side false-alarm mechanisms

def true_pair_category(previous: int, current: int) -> str:
    if previous == 4 and current == 0:
        return "true_rem_to_wake"
    if previous == 4:
        return "human_rem_to_other"
    if current == 0:
        return "human_other_to_wake"
    if previous == current:
        return "no_human_stage_change"
    return "other_human_stage_transition"


def derive_predicted_events(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparator, partition, subject, pid), group in predictions.groupby(
        ["comparator", "partition", "subject", "pid"]
    ):
        group = group.sort_values("onset")
        onset = group["onset"].to_numpy(dtype=float)
        truth = group["stage_hum"].to_numpy(dtype=int)
        predicted = group["stage_pred"].to_numpy(dtype=int)
        indices = np.flatnonzero(
            np.isclose(np.diff(onset), 30.0)
            & (predicted[:-1] == 4)
            & (predicted[1:] == 0)
        ) + 1
        for index in indices:
            rows.append(
                {
                    "analysis_version": VERSION,
                    "comparator": comparator,
                    "partition": partition,
                    "subject": subject,
                    "pid": int(pid),
                    "prediction_time_sec": float(onset[index]),
                    "human_previous_stage": int(truth[index - 1]),
                    "human_current_stage": int(truth[index]),
                    "true_stage_pair_category": true_pair_category(
                        int(truth[index - 1]), int(truth[index])
                    ),
                }
            )
    events = pd.DataFrame(rows)
    matches = transparent_matches()
    matches = matches[
        (matches["membership"] == "primary")
        & np.isclose(matches["tolerance_sec"].astype(float), 15.0)
    ][
        [
            "comparator",
            "partition",
            "subject",
            "prediction_time_sec",
            "match_type",
        ]
    ].drop_duplicates()
    events = events.merge(
        matches,
        on=["comparator", "partition", "subject", "prediction_time_sec"],
        how="left",
        validate="one_to_one",
    )
    events["event_outcome"] = events["match_type"].map(
        {"eligible": "eligible_match", "ignored_quality": "ignored_quality_match"}
    ).fillna("false_positive")
    events = events.drop(columns="match_type")
    return events


def summarize_predicted_events(events: pd.DataFrame) -> pd.DataFrame:
    summary = (
        events.groupby(
            ["comparator", "partition", "event_outcome", "true_stage_pair_category"]
        )
        .size()
        .rename("predicted_events")
        .reset_index()
    )
    totals = summary.groupby(["comparator", "partition", "event_outcome"])[
        "predicted_events"
    ].transform("sum")
    summary["total_events_for_outcome"] = totals
    summary["fraction_within_outcome"] = summary["predicted_events"] / totals
    return summary


# Section 4: participant dispersion and sensitivity

def participant_concentration(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["membership"] == "primary")
        & np.isclose(participants["tolerance_sec"].astype(float), 15.0)
    ]
    rows = []
    for (comparator, partition), group in primary.groupby(["comparator", "partition"]):
        positive = group[group["reference_events"] > 0]
        top_count = max(1, math.ceil(0.20 * len(group)))
        sorted_false_positive = group["false_positive"].sort_values(ascending=False)
        total_false_positive = int(group["false_positive"].sum())
        top_share = (
            float(sorted_false_positive.head(top_count).sum() / total_false_positive)
            if total_false_positive
            else np.nan
        )
        rows.append(
            {
                "analysis_version": VERSION,
                "comparator": comparator,
                "partition": partition,
                "participants": len(group),
                "participants_with_reference": len(positive),
                "participants_with_false_positive": int((group["false_positive"] > 0).sum()),
                "reference_participants_with_true_positive": int(
                    (positive["true_positive"] > 0).sum()
                ),
                "reference_participants_with_zero_true_positive": int(
                    (positive["true_positive"] == 0).sum()
                ),
                "total_false_positive": total_false_positive,
                "top_20_percent_pid_count": top_count,
                "top_20_percent_false_positive_share": top_share,
                "median_participant_false_alarms_per_hour": float(
                    group["false_alarms_per_hour"].median()
                ),
                "maximum_participant_false_alarms_per_hour": float(
                    group["false_alarms_per_hour"].max()
                ),
                "median_f1_among_reference_participants": float(positive["f1"].median()),
            }
        )
    return pd.DataFrame(rows)


def sensitivity_contrasts(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparator, partition), group in events.groupby(["comparator", "partition"]):
        indexed = group.set_index(["membership", "tolerance_sec"])
        primary_15 = indexed.loc[("primary", 15.0)]
        primary_45 = indexed.loc[("primary", 45.0)]
        expanded_15 = indexed.loc[("expanded", 15.0)]
        row = {
            "analysis_version": VERSION,
            "comparator": comparator,
            "partition": partition,
        }
        for metric in ["precision", "recall", "f1", "false_alarms_per_hour"]:
            base = float(primary_15[metric])
            row[f"primary_15s_{metric}"] = base
            row[f"primary_45s_{metric}"] = float(primary_45[metric])
            row[f"primary_45s_minus_15s_{metric}"] = float(primary_45[metric]) - base
            row[f"expanded_15s_{metric}"] = float(expanded_15[metric])
            row[f"expanded_minus_primary_15s_{metric}"] = float(expanded_15[metric]) - base
        rows.append(row)
    return pd.DataFrame(rows)


def stage_event_discordance(stages: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    primary = events[
        (events["membership"] == "primary")
        & np.isclose(events["tolerance_sec"].astype(float), 15.0)
    ][
        [
            "comparator",
            "partition",
            "reference_events",
            "predicted_events",
            "supported_hours",
            "precision",
            "recall",
            "f1",
            "false_alarms_per_hour",
        ]
    ].rename(columns={"f1": "event_f1"})
    result = stages.merge(primary, on=["comparator", "partition"], validate="one_to_one")
    result["stage_macro_f1_minus_event_f1"] = result["macro_f1"] - result["event_f1"]
    return result.sort_values(["partition", "comparator"])


def rem_run_lengths(onset: np.ndarray, stages: np.ndarray) -> list[int]:
    lengths = []
    current = 0
    for index, stage in enumerate(stages):
        continues = (
            index > 0
            and np.isclose(onset[index] - onset[index - 1], 30.0)
            and stages[index - 1] == 4
        )
        if stage == 4:
            current = current + 1 if continues else 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def sequence_fragmentation(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    recording_rows = []
    pooled_runs: dict[tuple[str, str, str], list[int]] = {}
    for (comparator, partition, subject, pid), group in predictions.groupby(
        ["comparator", "partition", "subject", "pid"]
    ):
        group = group.sort_values("onset")
        onset = group["onset"].to_numpy(dtype=float)
        human = group["stage_hum"].to_numpy(dtype=int)
        predicted = group["stage_pred"].to_numpy(dtype=int)
        contiguous = np.isclose(np.diff(onset), 30.0)
        human_runs = rem_run_lengths(onset, human)
        predicted_runs = rem_run_lengths(onset, predicted)
        pooled_runs.setdefault((comparator, partition, "human"), []).extend(human_runs)
        pooled_runs.setdefault((comparator, partition, "predicted"), []).extend(predicted_runs)
        recording_rows.append(
            {
                "analysis_version": VERSION,
                "comparator": comparator,
                "partition": partition,
                "subject": subject,
                "pid": int(pid),
                "supported_hours": float(contiguous.sum() * 30.0 / 3600.0),
                "human_all_stage_transitions": int((contiguous & (human[:-1] != human[1:])).sum()),
                "predicted_all_stage_transitions": int(
                    (contiguous & (predicted[:-1] != predicted[1:])).sum()
                ),
                "human_rem_to_wake_transitions": int(
                    (contiguous & (human[:-1] == 4) & (human[1:] == 0)).sum()
                ),
                "predicted_rem_to_wake_transitions": int(
                    (contiguous & (predicted[:-1] == 4) & (predicted[1:] == 0)).sum()
                ),
                "human_rem_epochs": int((human == 4).sum()),
                "predicted_rem_epochs": int((predicted == 4).sum()),
                "human_rem_bouts": len(human_runs),
                "predicted_rem_bouts": len(predicted_runs),
            }
        )
    recordings = pd.DataFrame(recording_rows)
    summary_rows = []
    for (comparator, partition), group in recordings.groupby(["comparator", "partition"]):
        hours = float(group["supported_hours"].sum())
        human_runs = np.asarray(pooled_runs[(comparator, partition, "human")], dtype=float)
        predicted_runs = np.asarray(pooled_runs[(comparator, partition, "predicted")], dtype=float)
        human_transitions = int(group["human_all_stage_transitions"].sum())
        predicted_transitions = int(group["predicted_all_stage_transitions"].sum())
        human_bouts = int(group["human_rem_bouts"].sum())
        predicted_bouts = int(group["predicted_rem_bouts"].sum())
        summary_rows.append(
            {
                "analysis_version": VERSION,
                "comparator": comparator,
                "partition": partition,
                "recordings": len(group),
                "supported_hours": hours,
                "human_all_stage_transitions": human_transitions,
                "predicted_all_stage_transitions": predicted_transitions,
                "human_all_stage_transitions_per_hour": human_transitions / hours,
                "predicted_all_stage_transitions_per_hour": predicted_transitions / hours,
                "predicted_to_human_all_transition_rate_ratio": (
                    predicted_transitions / human_transitions if human_transitions else np.nan
                ),
                "human_rem_to_wake_transitions": int(group["human_rem_to_wake_transitions"].sum()),
                "predicted_rem_to_wake_transitions": int(
                    group["predicted_rem_to_wake_transitions"].sum()
                ),
                "human_rem_to_wake_transitions_per_hour": float(
                    group["human_rem_to_wake_transitions"].sum() / hours
                ),
                "predicted_rem_to_wake_transitions_per_hour": float(
                    group["predicted_rem_to_wake_transitions"].sum() / hours
                ),
                "human_rem_epochs": int(group["human_rem_epochs"].sum()),
                "predicted_rem_epochs": int(group["predicted_rem_epochs"].sum()),
                "human_rem_bouts": human_bouts,
                "predicted_rem_bouts": predicted_bouts,
                "predicted_to_human_rem_bout_count_ratio": (
                    predicted_bouts / human_bouts if human_bouts else np.nan
                ),
                "human_median_rem_bout_duration_sec": float(np.median(human_runs) * 30.0),
                "predicted_median_rem_bout_duration_sec": float(
                    np.median(predicted_runs) * 30.0
                ),
                "human_mean_rem_bout_duration_sec": float(np.mean(human_runs) * 30.0),
                "predicted_mean_rem_bout_duration_sec": float(np.mean(predicted_runs) * 30.0),
            }
        )
    return recordings, pd.DataFrame(summary_rows)


# Section 5: integrity checks and report

def integrity_checks(
    boundary_summary: pd.DataFrame,
    predicted_events: pd.DataFrame,
    event_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    primary = event_metrics[
        (event_metrics["membership"] == "primary")
        & np.isclose(event_metrics["tolerance_sec"].astype(float), 15.0)
        & event_metrics["comparator"].isin(["SF-B", "SF-C"])
    ]
    for item in primary.itertuples(index=False):
        mechanism_count = int(
            boundary_summary[
                (boundary_summary["comparator"] == item.comparator)
                & (boundary_summary["partition"] == item.partition)
                & (boundary_summary["membership"] == "primary")
            ]["reference_events"].sum()
        )
        predicted_count = len(
            predicted_events[
                (predicted_events["comparator"] == item.comparator)
                & (predicted_events["partition"] == item.partition)
            ]
        )
        rows.extend(
            [
                {
                    "analysis_version": VERSION,
                    "check": f"{item.comparator}_{item.partition}_reference_mechanism_count",
                    "observed": mechanism_count,
                    "expected": int(item.reference_events),
                    "passed": mechanism_count == int(item.reference_events),
                },
                {
                    "analysis_version": VERSION,
                    "check": f"{item.comparator}_{item.partition}_predicted_event_count",
                    "observed": predicted_count,
                    "expected": int(item.predicted_events),
                    "passed": predicted_count == int(item.predicted_events),
                },
            ]
        )
    return pd.DataFrame(rows)


def write_readme(
    boundary_summary: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    event_failure_summary: pd.DataFrame,
    participant_summary: pd.DataFrame,
    stage_classes: pd.DataFrame,
    sensitivity: pd.DataFrame,
    fragmentation: pd.DataFrame,
) -> None:
    boundary = boundary_summary[
        (boundary_summary["comparator"] == "SF-C")
        & (boundary_summary["partition"] == "test")
        & (boundary_summary["membership"] == "primary")
    ].set_index("boundary_mechanism")["reference_events"]
    endpoints = endpoint_summary[
        (endpoint_summary["comparator"] == "SF-C")
        & (endpoint_summary["partition"] == "test")
        & (endpoint_summary["membership"] == "primary")
    ]
    preceding = endpoints[endpoints["endpoint"] == "preceding_rem"].set_index(
        "predicted_stage_name"
    )["reference_events"]
    following = endpoints[endpoints["endpoint"] == "following_wake"].set_index(
        "predicted_stage_name"
    )["reference_events"]
    false_positive = event_failure_summary[
        (event_failure_summary["comparator"] == "SF-C")
        & (event_failure_summary["partition"] == "test")
        & (event_failure_summary["event_outcome"] == "false_positive")
    ].set_index("true_stage_pair_category")["predicted_events"]
    participant = participant_summary[
        (participant_summary["comparator"] == "SF-C")
        & (participant_summary["partition"] == "test")
    ].iloc[0]
    stage = stage_classes[
        (stage_classes["comparator"] == "SF-C")
        & (stage_classes["partition"] == "test")
    ].sort_values("recall")
    sensitivity_row = sensitivity[
        (sensitivity["comparator"] == "SF-C")
        & (sensitivity["partition"] == "test")
    ].iloc[0]
    fragmentation_row = fragmentation[
        (fragmentation["comparator"] == "SF-C")
        & (fragmentation["partition"] == "test")
    ].iloc[0]

    boundary_lines = [
        f"| {name} | {int(count)} | {count / boundary.sum():.4f} |"
        for name, count in boundary.items()
    ]
    false_positive_lines = [
        f"| {name} | {int(count)} | {count / false_positive.sum():.4f} |"
        for name, count in false_positive.items()
    ]
    stage_lines = [
        f"| {row.stage} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {int(row.support):,} |"
        for row in stage.itertuples(index=False)
    ]
    text = f"""# Stage-First Failure Analysis v0.1

**Created:** 2026-08-15
**Plan:** `docs/evaluation/stage_first_failure_analysis_plan_v0.1.md`
**Status:** Exploratory diagnostic after the primary result
**Raw signals, external features, or models opened:** No
**Frozen input tables hashed:** 18

## SF-C Test Reference-Side Mechanisms

| Mechanism | Reference events | Fraction |
|---|---:|---:|
{chr(10).join(boundary_lines)}

Only {int(boundary.get('detected_rem_to_wake', 0))} of {int(boundary.sum())} primary reference boundaries were represented as an exact predicted 4-to-0 pair. No primary test reference lacked prediction coverage. Of the 41 missed references, {int(boundary.get('preceding_rem_not_predicted', 0))} missed only the preceding REM endpoint, {int(boundary.get('following_wake_not_predicted', 0))} missed only the following Wake endpoint, and {int(boundary.get('both_target_endpoints_not_predicted', 0))} missed both.

At the preceding true REM epoch, SF-C predicted REM for {int(preceding.get('REM', 0))} of {int(preceding.sum())} references and predicted Wake for {int(preceding.get('Wake', 0))}. At the following true Wake epoch, it predicted Wake for {int(following.get('Wake', 0))} of {int(following.sum())} references. Failure to retain REM immediately before the boundary is therefore the dominant reference-side mechanism.

## SF-C Test False-Positive Human Stage Pairs

| Human pair category | False positives | Fraction |
|---|---:|---:|
{chr(10).join(false_positive_lines)}

These categories show whether a predicted 4-to-0 boundary occurred across a true non-target transition or without any human stage change. They describe the source of false alarms; they do not redefine the primary event score.

Human REM-to-other and other-to-Wake pairs account for {int(false_positive.get('human_rem_to_other', 0) + false_positive.get('human_other_to_wake', 0))} of {int(false_positive.sum())} false positives ({(false_positive.get('human_rem_to_other', 0) + false_positive.get('human_other_to_wake', 0)) / false_positive.sum():.4f}). A further {int(false_positive.get('no_human_stage_change', 0))} predictions occurred without any human stage change.

## SF-C Test Stage Classes

| Stage | Precision | Recall | F1 | Epochs |
|---|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

The weakest class by recall was {stage.iloc[0]['stage']} ({stage.iloc[0]['recall']:.4f}). Boundary detection requires both REM and Wake endpoints to be correct in sequence, so class-wise errors compound at the event level.

## Participant Dispersion

SF-C detected at least one event in {int(participant.reference_participants_with_true_positive)} of {int(participant.participants_with_reference)} test participants with a primary reference. {int(participant.reference_participants_with_zero_true_positive)} reference-positive participants had no true positive. All {int(participant.participants_with_false_positive)} test participants produced at least one false positive. The highest-FP 20% of participants contributed {participant.top_20_percent_false_positive_share:.4f} of all false positives. Median participant false alarms/hour was {participant.median_participant_false_alarms_per_hour:.4f}, and the maximum was {participant.maximum_participant_false_alarms_per_hour:.4f}.

## Sensitivity Interpretation

Moving from +/-15 to +/-45 seconds changed SF-C test event F1 by {sensitivity_row.primary_45s_minus_15s_f1:+.4f} and recall by {sensitivity_row.primary_45s_minus_15s_recall:+.4f}. Expanding quality membership at +/-15 seconds changed event F1 by {sensitivity_row.expanded_minus_primary_15s_f1:+.4f}. The persistent false-alarm rate indicates that coarse timing and the conservative quality tier are not the sole causes of poor performance.

## Sequence Fragmentation

Across the same supported SF-C test epochs, the human hypnogram contained {int(fragmentation_row.human_all_stage_transitions)} all-stage transitions ({fragmentation_row.human_all_stage_transitions_per_hour:.4f}/hour), while SF-C produced {int(fragmentation_row.predicted_all_stage_transitions)} ({fragmentation_row.predicted_all_stage_transitions_per_hour:.4f}/hour), a {fragmentation_row.predicted_to_human_all_transition_rate_ratio:.4f}-fold rate. SF-C produced {int(fragmentation_row.predicted_rem_bouts)} REM bouts versus {int(fragmentation_row.human_rem_bouts)} human REM bouts. Median REM-bout duration was {fragmentation_row.predicted_median_rem_bout_duration_sec:.1f} seconds for SF-C and {fragmentation_row.human_median_rem_bout_duration_sec:.1f} seconds for the human sequence.

This directly supports a fragmentation mechanism: independently classified epochs create too many short stage runs and therefore too many opportunities for spurious 4-to-0 boundaries.

## Technical Interpretation

The failure is not simply low average stage accuracy. Stage-first event derivation requires a particular two-epoch sequence, so endpoint errors create false negatives while isolated REM-to-Wake prediction flips create false positives. Participant dispersion and true-pair categories indicate whether the problem is broad or concentrated, but no subgroup is used to revise the frozen primary result.

## Decision

Retain the Block 5 conclusion: temporal context improves the transparent stage model but does not produce an adequate event detector. These diagnostics may motivate the already-planned direct-event hypothesis, but no direct model, threshold, or later-phase experiment is included here.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


# Section 6: execute the read-only diagnostic

def main() -> None:
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    predictions = transparent_predictions()
    stages = all_stage_metrics()
    stage_classes = all_stage_class_metrics()
    events = all_event_metrics()
    participants = all_participant_metrics()

    boundary_detail, boundary_summary = analyze_reference_boundaries(predictions)
    endpoint_summary = summarize_reference_endpoints(boundary_detail)
    predicted_event_detail = derive_predicted_events(predictions)
    event_failure_summary = summarize_predicted_events(predicted_event_detail)
    participant_summary = participant_concentration(participants)
    sensitivity = sensitivity_contrasts(events)
    discordance = stage_event_discordance(stages, events)
    fragmentation_detail, fragmentation_summary = sequence_fragmentation(predictions)
    checks = integrity_checks(boundary_summary, predicted_event_detail, events)
    manifest = analysis_input_manifest()

    stage_classes.to_csv(destination / "stage_class_comparison_v0.1.tsv", sep="\t", index=False)
    boundary_detail.to_csv(destination / "reference_boundary_mechanisms_v0.1.tsv", sep="\t", index=False)
    boundary_summary.to_csv(destination / "reference_boundary_mechanism_summary_v0.1.tsv", sep="\t", index=False)
    endpoint_summary.to_csv(destination / "reference_endpoint_prediction_summary_v0.1.tsv", sep="\t", index=False)
    predicted_event_detail.to_csv(destination / "predicted_event_failure_modes_v0.1.tsv", sep="\t", index=False)
    event_failure_summary.to_csv(destination / "predicted_event_failure_mode_summary_v0.1.tsv", sep="\t", index=False)
    participant_summary.to_csv(destination / "participant_concentration_summary_v0.1.tsv", sep="\t", index=False)
    sensitivity.to_csv(destination / "quality_timing_sensitivity_contrasts_v0.1.tsv", sep="\t", index=False)
    discordance.to_csv(destination / "stage_event_discordance_v0.1.tsv", sep="\t", index=False)
    fragmentation_detail.to_csv(destination / "sequence_fragmentation_by_recording_v0.1.tsv", sep="\t", index=False)
    fragmentation_summary.to_csv(destination / "sequence_fragmentation_summary_v0.1.tsv", sep="\t", index=False)
    checks.to_csv(destination / "diagnostic_integrity_checks_v0.1.tsv", sep="\t", index=False)
    manifest.to_csv(destination / "analysis_input_manifest_v0.1.tsv", sep="\t", index=False)
    write_readme(
        boundary_summary,
        endpoint_summary,
        event_failure_summary,
        participant_summary,
        stage_classes,
        sensitivity,
        fragmentation_summary,
    )

    print(checks.to_string(index=False))
    if not checks["passed"].all():
        raise SystemExit("At least one diagnostic integrity check failed")
    print(f"Wrote failure analysis to {destination}")


if __name__ == "__main__":
    main()
