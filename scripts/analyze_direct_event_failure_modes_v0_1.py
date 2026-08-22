"""Analyze residual frozen-test failure modes for direct baseline DE-B."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_direct_event_baseline_v0_1 import data_parent, feature_path, repo_root, score_path


# Section 1: paths and inputs

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-08-22_direct_event_failure_analysis_v0.1"
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def baseline_dir() -> Path:
    return repo_root() / "experiments/2026-08-22_direct_event_baseline_v0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_category(previous: int, current: int) -> str:
    if previous == 4 and current == 0:
        return "human_REM_to_Wake"
    if previous == 4:
        return "human_REM_to_other"
    if current == 0:
        return "human_other_to_Wake"
    if previous == current:
        return "human_no_stage_change"
    return "human_other_transition"


def stage_lookup(subject: str) -> dict[int, int]:
    with np.load(feature_path(subject), allow_pickle=False) as values:
        onset = values["onset"].astype(float)
        stage = values["stage"].astype(int)
    return {int(round(time * 1000.0)): int(code) for time, code in zip(onset, stage)}


def add_human_stage_pair(rows: pd.DataFrame) -> pd.DataFrame:
    results = []
    for subject, group in rows.groupby("subject", sort=True):
        lookup = stage_lookup(subject)
        for item in group.itertuples(index=False):
            key = int(round(float(item.event_time_sec) * 1000.0))
            previous = lookup.get(key - 30000)
            current = lookup.get(key)
            if previous is None or current is None:
                raise ValueError(f"Missing human stage pair for {subject} at {item.event_time_sec}")
            results.append(
                {
                    **item._asdict(),
                    "human_stage_from": previous,
                    "human_stage_from_label": STAGE_NAMES[previous],
                    "human_stage_to": current,
                    "human_stage_to_label": STAGE_NAMES[current],
                    "human_stage_pair_category": stage_category(previous, current),
                }
            )
    return pd.DataFrame(results)


def remwake_boundaries() -> pd.DataFrame:
    membership = pd.read_csv(
        repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    return membership.merge(quality, on="transition_id", validate="one_to_one")


def add_boundary_distance(false_positives: pd.DataFrame) -> pd.DataFrame:
    boundaries = remwake_boundaries()
    rows = []
    for item in false_positives.itertuples(index=False):
        local = boundaries[boundaries["subject"] == item.subject].copy()
        if len(local) == 0:
            rows.append(
                {
                    **item._asdict(),
                    "nearest_remwake_transition_id": pd.NA,
                    "nearest_remwake_type": "none_in_recording",
                    "signed_seconds_to_nearest_remwake": np.nan,
                    "absolute_seconds_to_nearest_remwake": np.nan,
                    "remwake_distance_bin": "no_remwake_reference_in_recording",
                    "inside_background_exclusion_zone": False,
                }
            )
            continue
        distance = local["nominal_boundary_sec"].astype(float) - float(item.event_time_sec)
        order = np.lexsort((local["nominal_boundary_sec"].to_numpy(), np.abs(distance)))
        nearest = local.iloc[int(order[0])]
        signed = float(nearest["nominal_boundary_sec"] - item.event_time_sec)
        absolute = abs(signed)
        if np.isclose(absolute, 0.0):
            distance_bin = "exact_boundary"
        elif absolute <= 45.0:
            distance_bin = "30_to_45_sec"
        elif absolute <= 135.0:
            distance_bin = "60_to_135_sec"
        else:
            distance_bin = "beyond_135_sec"
        rows.append(
            {
                **item._asdict(),
                "nearest_remwake_transition_id": int(nearest["transition_id"]),
                "nearest_remwake_type": nearest["transition_type"],
                "signed_seconds_to_nearest_remwake": signed,
                "absolute_seconds_to_nearest_remwake": absolute,
                "remwake_distance_bin": distance_bin,
                "inside_background_exclusion_zone": absolute <= 135.0,
            }
        )
    return pd.DataFrame(rows)


# Section 2: identify primary false positives

def load_de_b_alarms() -> pd.DataFrame:
    alarms = pd.read_csv(baseline_dir() / "test_predicted_events_v0.1.tsv", sep="\t")
    return alarms[alarms["comparator"] == "DE-B"].copy()


def primary_false_positives(alarms: pd.DataFrame) -> pd.DataFrame:
    matches = pd.read_csv(baseline_dir() / "test_event_matches_v0.1.tsv", sep="\t")
    matches = matches[
        (matches["comparator"] == "DE-B")
        & (matches["membership"] == "primary")
        & (matches["tolerance_sec"] == 15.0)
    ].copy()
    matched_keys = {
        (row.subject, round(float(row.prediction_time_sec), 6))
        for row in matches.itertuples(index=False)
    }
    false_positive = alarms[
        [
            (row.subject, round(float(row.event_time_sec), 6)) not in matched_keys
            for row in alarms.itertuples(index=False)
        ]
    ].copy()
    if len(false_positive) != 250:
        raise ValueError(f"Expected 250 DE-B primary false positives, found {len(false_positive)}")
    return false_positive


# Section 3: stage-pair, distance, participant, and timing summaries

def stage_pair_summary(
    false_positive: pd.DataFrame, supported: pd.DataFrame
) -> pd.DataFrame:
    supported_counts = supported.groupby("human_stage_pair_category").size().rename(
        "supported_candidates"
    )
    false_counts = false_positive.groupby("human_stage_pair_category").size().rename(
        "false_positive_alarms"
    )
    result = pd.concat([supported_counts, false_counts], axis=1).fillna(0).reset_index()
    result["supported_share"] = result["supported_candidates"] / result[
        "supported_candidates"
    ].sum()
    result["false_positive_share"] = result["false_positive_alarms"] / result[
        "false_positive_alarms"
    ].sum()
    result["false_positive_share_enrichment"] = (
        result["false_positive_share"] / result["supported_share"]
    )
    return result.sort_values("false_positive_alarms", ascending=False)


def distance_summary(false_positive: pd.DataFrame) -> pd.DataFrame:
    order = [
        "exact_boundary",
        "30_to_45_sec",
        "60_to_135_sec",
        "beyond_135_sec",
        "no_remwake_reference_in_recording",
    ]
    counts = false_positive.groupby("remwake_distance_bin").size().reindex(order, fill_value=0)
    return pd.DataFrame(
        {
            "remwake_distance_bin": counts.index,
            "false_positive_alarms": counts.to_numpy(dtype=int),
            "false_positive_share": counts.to_numpy(dtype=float) / len(false_positive),
        }
    )


def participant_summary(false_positive: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_metrics = pd.read_csv(
        baseline_dir() / "test_event_participants_v0.1.tsv", sep="\t"
    )
    participant_metrics = participant_metrics[
        (participant_metrics["comparator"] == "DE-B")
        & (participant_metrics["membership"] == "primary")
        & (participant_metrics["tolerance_sec"] == 15.0)
    ].copy()
    fp_scores = (
        false_positive.groupby("pid")["probability"]
        .agg(false_positive_score_median="median", false_positive_score_max="max")
        .reset_index()
    )
    participant = participant_metrics.merge(fp_scores, on="pid", how="left")
    participant[["false_positive_score_median", "false_positive_score_max"]] = participant[
        ["false_positive_score_median", "false_positive_score_max"]
    ].fillna(np.nan)
    ranked = participant.sort_values(["false_positive", "pid"], ascending=[False, True])
    top_four = int(ranked.head(4)["false_positive"].sum())
    summary = pd.DataFrame(
        [
            {
                "test_participants": len(participant),
                "participants_with_true_positive": int((participant["true_positive"] > 0).sum()),
                "participants_with_false_positive": int((participant["false_positive"] > 0).sum()),
                "top_20_percent_participant_count": 4,
                "top_20_percent_false_positives": top_four,
                "top_20_percent_false_positive_share": top_four
                / participant["false_positive"].sum(),
            }
        ]
    )
    return participant, summary


def timing_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = pd.read_csv(baseline_dir() / "test_event_matches_v0.1.tsv", sep="\t")
    matches = matches[
        (matches["comparator"] == "DE-B")
        & (matches["membership"] == "primary")
        & (matches["tolerance_sec"] == 45.0)
        & (matches["match_type"] == "eligible")
    ].copy()
    matches["signed_error_sec"] = (
        matches["prediction_time_sec"] - matches["reference_time_sec"]
    )
    matches["timing_direction"] = np.select(
        [matches["signed_error_sec"] < 0, matches["signed_error_sec"] > 0],
        ["early", "late"],
        default="exact",
    )
    matches["outside_primary_tolerance"] = matches["absolute_error_sec"] > 15.0
    summary = (
        matches.groupby(["outside_primary_tolerance", "timing_direction"])
        .size()
        .rename("matches")
        .reset_index()
    )
    return matches, summary


def contrast_summary() -> pd.DataFrame:
    metrics = pd.read_csv(baseline_dir() / "test_event_metrics_v0.1.tsv", sep="\t")
    metrics = metrics[
        (metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    a = metrics.loc["DE-A"]
    b = metrics.loc["DE-B"]
    return pd.DataFrame(
        [
            {
                "comparison": "DE-B_minus_DE-A",
                "true_positive_difference": int(b.true_positive - a.true_positive),
                "false_positive_difference": int(b.false_positive - a.false_positive),
                "false_negative_difference": int(b.false_negative - a.false_negative),
                "event_f1_difference": float(b.f1 - a.f1),
                "false_alarms_per_hour_difference": float(
                    b.false_alarms_per_hour - a.false_alarms_per_hour
                ),
            }
        ]
    )


# Section 4: input manifest and result record

def input_manifest(subjects: list[str]) -> pd.DataFrame:
    paths = [
        ("saved_test_alarms", baseline_dir() / "test_predicted_events_v0.1.tsv", "repo"),
        ("saved_test_matches", baseline_dir() / "test_event_matches_v0.1.tsv", "repo"),
        ("saved_test_metrics", baseline_dir() / "test_event_metrics_v0.1.tsv", "repo"),
        (
            "saved_test_participant_metrics",
            baseline_dir() / "test_event_participants_v0.1.tsv",
            "repo",
        ),
        ("continuous_test_scores", score_path("test"), "data_parent"),
        (
            "transition_membership",
            repo_root()
            / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
            "repo",
        ),
        (
            "transition_quality",
            repo_root()
            / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
            "repo",
        ),
    ]
    paths.extend(("frozen_stage_lookup", feature_path(subject), "data_parent") for subject in subjects)
    rows = []
    for role, path, base in paths:
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
    stage_summary: pd.DataFrame,
    distance: pd.DataFrame,
    participant: pd.DataFrame,
    timing: pd.DataFrame,
    contrast: pd.DataFrame,
) -> None:
    top_stage = stage_summary.iloc[0]
    inside = float(
        distance[
            distance["remwake_distance_bin"].isin(
                ["exact_boundary", "30_to_45_sec", "60_to_135_sec"]
            )
        ]["false_positive_share"].sum()
    )
    p = participant.iloc[0]
    c = contrast.iloc[0]
    additional = timing[timing["outside_primary_tolerance"]]
    early = int(additional.loc[additional["timing_direction"] == "early", "matches"].sum())
    late = int(additional.loc[additional["timing_direction"] == "late", "matches"].sum())
    text = f"""# Direct Event Failure Analysis v0.1

**Created:** 2026-08-22
**Status:** Post-result exploratory analysis
**Plan:** `docs/evaluation/direct_event_failure_analysis_plan_v0.1.md`

## Retained Primary Result

DE-B test event F1 was 0.1497 with precision 0.0909, recall 0.4237, and 1.2571 false alarms per supported hour. This analysis does not change the frozen model, threshold, or result.

## Residual False Alarms

The largest false-positive stage-pair category was `{top_stage.human_stage_pair_category}`, contributing {int(top_stage.false_positive_alarms)} of 250 false alarms ({top_stage.false_positive_share:.2%}). Its enrichment relative to all supported candidate boundaries was {top_stage.false_positive_share_enrichment:.2f}-fold.

Only {inside:.2%} of false positives occurred within 135 seconds of a human-derived REM/Wake boundary, the region excluded from reviewed background construction. The excluded boundary zone is therefore not the dominant observed source of false alarms.

False positives occurred in {int(p.participants_with_false_positive)} of {int(p.test_participants)} test participants. The four highest-burden participants contributed {p.top_20_percent_false_positive_share:.2%}. The failure is broad across participants with moderate concentration in the highest-burden group, rather than attributable to a single outlier.

## Context Tradeoff

Relative to DE-A, DE-B added {int(c.true_positive_difference)} true positives and {int(c.false_positive_difference):+d} false positives. Event F1 changed by {c.event_f1_difference:+.4f}, while false alarms per hour changed by {c.false_alarms_per_hour_difference:+.4f}. Eight-epoch context improved test recall and F1 despite failing the prespecified validation F1 comparison, but it also increased the alarm burden.

## Timing Sensitivity

At +/-45 seconds, {early + late} matches were outside the primary +/-15-second tolerance: {early} were one epoch early and {late} were one epoch late. This quantifies how much one-epoch boundary displacement contributes to the remaining error.

## Decision Boundary

Direct modeling produced measurable value over transparent stage-first SF-C, but fixed log-bandpower context did not resolve event precision. The next method must address a stated source of residual error and must be validated without modifying this frozen test result. A CNN is not automatically justified by the remaining error; its proposed representation must be tied to a specific hypothesis in a new protocol.

No raw EDF or fitted model was opened. All diagnostic inputs and frozen feature-stage lookup arrays are listed with SHA-256 hashes in `input_artifact_manifest_v0.1.tsv`.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


# Section 5: execute

def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    alarms = load_de_b_alarms()
    false_positive = primary_false_positives(alarms)
    false_positive = add_human_stage_pair(false_positive)
    false_positive = add_boundary_distance(false_positive)

    scores = pd.read_csv(score_path("test"), sep="\t", compression="gzip")
    scores = scores[scores["comparator"] == "DE-B"].copy()
    supported = scores.rename(columns={"candidate_time_sec": "event_time_sec"})
    supported = add_human_stage_pair(supported)

    stage_summary = stage_pair_summary(false_positive, supported)
    distance = distance_summary(false_positive)
    participant_rows, participant = participant_summary(false_positive)
    timing_rows, timing = timing_summary()
    contrast = contrast_summary()

    false_positive.to_csv(
        output_dir() / "de_b_test_false_positive_context_v0.1.tsv", sep="\t", index=False
    )
    stage_summary.to_csv(
        output_dir() / "de_b_false_positive_stage_pair_summary_v0.1.tsv",
        sep="\t",
        index=False,
    )
    distance.to_csv(
        output_dir() / "de_b_false_positive_distance_summary_v0.1.tsv",
        sep="\t",
        index=False,
    )
    participant_rows.to_csv(
        output_dir() / "de_b_participant_failure_summary_v0.1.tsv", sep="\t", index=False
    )
    participant.to_csv(
        output_dir() / "de_b_participant_concentration_v0.1.tsv", sep="\t", index=False
    )
    timing_rows.to_csv(
        output_dir() / "de_b_timing_match_details_v0.1.tsv", sep="\t", index=False
    )
    timing.to_csv(
        output_dir() / "de_b_timing_direction_summary_v0.1.tsv", sep="\t", index=False
    )
    contrast.to_csv(
        output_dir() / "direct_context_contrast_v0.1.tsv", sep="\t", index=False
    )
    subjects = sorted(
        supported["subject"].unique(), key=lambda value: int(value.replace("sub-", ""))
    )
    input_manifest(subjects).to_csv(
        output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    write_readme(stage_summary, distance, participant, timing, contrast)
    print(stage_summary.to_string(index=False))
    print(distance.to_string(index=False))
    print(participant.to_string(index=False))
    print(timing.to_string(index=False))
    print(contrast.to_string(index=False))


if __name__ == "__main__":
    main()
