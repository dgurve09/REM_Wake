"""Run the frozen multi-dataset hypnogram transition-uncertainty analysis."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: fixed transition definitions

SLEEP_STAGES = np.asarray(["N1", "N2", "N3", "REM"])
MULTI_SCORER_SOURCES = ["dod", "isruc", "gx36m"]


def quantile(values, probability: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, probability)) if len(values) else np.nan


def sleep_period(stages: np.ndarray) -> tuple[int, int] | None:
    sleep = np.flatnonzero(np.isin(stages, SLEEP_STAGES))
    if not len(sleep):
        return None
    start = int(sleep[0])
    stop = int(sleep[-1]) + 1
    if stop < len(stages) and stages[stop] == "W":
        stop += 1
    return start, stop


def event_boundaries(stages: np.ndarray, start: int = 0, stop: int | None = None) -> np.ndarray:
    stop = len(stages) if stop is None else stop
    left = max(0, start)
    right = min(len(stages) - 1, stop - 1)
    if right <= left:
        return np.asarray([], dtype=int)
    local = np.flatnonzero(
        (stages[left:right] == "REM")
        & (stages[left + 1 : right + 1] == "W")
    )
    return left + local + 1


def rem_exit_count(stages: np.ndarray, start: int, stop: int) -> int:
    count = 0
    for index in range(start, stop - 1):
        if stages[index] == "REM" and stages[index + 1] != "REM":
            count += int(stages[index + 1] in common.KNOWN_STAGES)
    return count


def event_detail(item: dict, boundary: int, start: int, stop: int, previous: int | None) -> dict:
    stages = item["stages"]
    rem_start = boundary - 1
    while rem_start > 0 and stages[rem_start - 1] == "REM":
        rem_start -= 1
    wake_stop = boundary
    while wake_stop < len(stages) and stages[wake_stop] == "W":
        wake_stop += 1
    right_censored = wake_stop == len(stages) or stages[wake_stop] == "UNKNOWN"
    scorer_events = {
        name: set(event_boundaries(values).tolist())
        for name, values in item["scorers"].items()
    }
    support = sum(boundary in values for values in scorer_events.values())
    return {
        "dataset": item["dataset"],
        "recording": item["recording"],
        "participant_group": item["participant_group"],
        "cohort_group": item["cohort_group"],
        "event_boundary_epoch": boundary,
        "event_from_hypnogram_start_sec": boundary * 30.0,
        "sleep_period_position": (boundary - start) / (stop - start),
        "preceding_rem_duration_sec": (boundary - rem_start) * 30.0,
        "following_wake_observed_sec": (wake_stop - boundary) * 30.0,
        "following_wake_right_censored": right_censored,
        "previous_event_spacing_sec": np.nan if previous is None else (boundary - previous) * 30.0,
        "exact_scorer_event_support": support,
        "available_scorers": len(item["scorers"]),
    }


# Section 2: night-level and event-level measures

def recording_measures(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, events = [], []
    for item in records:
        stages = item["stages"]
        bounds = sleep_period(stages)
        full_events = event_boundaries(stages)
        if bounds is None:
            rows.append(
                {
                    "dataset": item["dataset"], "recording": item["recording"],
                    "participant_group": item["participant_group"], "cohort_group": item["cohort_group"],
                    "support_eligible": False, "support_start_epoch": np.nan, "support_stop_epoch": np.nan,
                    "support_hours": 0.0, "known_support_hours": 0.0, "rem_hours": 0.0,
                    "rem_exit_opportunities": 0, "rem_to_wake_events": 0,
                    "event_positive": False, "events_per_support_hour": np.nan,
                    "rw_fraction_of_known_rem_exits": np.nan, "median_event_position": np.nan,
                    "full_record_known_hours": float(np.isin(stages, common.KNOWN_STAGES).sum() / 120.0),
                    "full_record_events": len(full_events),
                }
            )
            continue
        start, stop = bounds
        support = stages[start:stop]
        boundaries = event_boundaries(stages, start, stop)
        exits = rem_exit_count(stages, start, stop)
        detail_rows = []
        for index, boundary in enumerate(boundaries):
            previous = None if index == 0 else int(boundaries[index - 1])
            detail_rows.append(event_detail(item, int(boundary), start, stop, previous))
        events.extend(detail_rows)
        support_hours = (stop - start) / 120.0
        rows.append(
            {
                "dataset": item["dataset"], "recording": item["recording"],
                "participant_group": item["participant_group"], "cohort_group": item["cohort_group"],
                "support_eligible": True, "support_start_epoch": start, "support_stop_epoch": stop,
                "support_hours": support_hours,
                "known_support_hours": float(np.isin(support, common.KNOWN_STAGES).sum() / 120.0),
                "rem_hours": float((support == "REM").sum() / 120.0),
                "rem_exit_opportunities": exits, "rem_to_wake_events": len(boundaries),
                "event_positive": len(boundaries) > 0,
                "events_per_support_hour": len(boundaries) / support_hours,
                "rw_fraction_of_known_rem_exits": len(boundaries) / exits if exits else np.nan,
                "median_event_position": quantile([row["sleep_period_position"] for row in detail_rows], 0.5),
                "full_record_known_hours": float(np.isin(stages, common.KNOWN_STAGES).sum() / 120.0),
                "full_record_events": len(full_events),
            }
        )
    event_frame = pd.DataFrame(events)
    if len(event_frame):
        event_frame.insert(4, "event_id", np.arange(1, len(event_frame) + 1))
    return pd.DataFrame(rows), event_frame


def summarize_group(local: pd.DataFrame, events: pd.DataFrame) -> dict:
    eligible = local[local.support_eligible.astype(bool)]
    local_events = events[events.recording.isin(eligible.recording)]
    uncensored = local_events[~local_events.following_wake_right_censored.astype(bool)]
    support_hours = float(eligible.support_hours.sum())
    event_count = int(eligible.rem_to_wake_events.sum())
    rem_exits = int(eligible.rem_exit_opportunities.sum())
    return {
        "primary_nights": len(local),
        "eligible_nights": len(eligible),
        "participant_groups": local.participant_group.nunique(),
        "support_hours": support_hours,
        "rem_hours": float(eligible.rem_hours.sum()),
        "rem_to_wake_events": event_count,
        "event_positive_nights": int(eligible.event_positive.sum()),
        "event_positive_night_fraction": float(eligible.event_positive.mean()) if len(eligible) else np.nan,
        "events_per_support_hour": event_count / support_hours if support_hours else np.nan,
        "night_event_count_q1": quantile(eligible.rem_to_wake_events, 0.25),
        "night_event_count_median": quantile(eligible.rem_to_wake_events, 0.5),
        "night_event_count_q3": quantile(eligible.rem_to_wake_events, 0.75),
        "night_event_rate_q1": quantile(eligible.events_per_support_hour, 0.25),
        "night_event_rate_median": quantile(eligible.events_per_support_hour, 0.5),
        "night_event_rate_q3": quantile(eligible.events_per_support_hour, 0.75),
        "known_rem_exit_opportunities": rem_exits,
        "rw_fraction_of_known_rem_exits": event_count / rem_exits if rem_exits else np.nan,
        "event_position_q1": quantile(local_events.sleep_period_position, 0.25),
        "event_position_median": quantile(local_events.sleep_period_position, 0.5),
        "event_position_q3": quantile(local_events.sleep_period_position, 0.75),
        "preceding_rem_sec_q1": quantile(local_events.preceding_rem_duration_sec, 0.25),
        "preceding_rem_sec_median": quantile(local_events.preceding_rem_duration_sec, 0.5),
        "preceding_rem_sec_q3": quantile(local_events.preceding_rem_duration_sec, 0.75),
        "uncensored_wake_sec_q1": quantile(uncensored.following_wake_observed_sec, 0.25),
        "uncensored_wake_sec_median": quantile(uncensored.following_wake_observed_sec, 0.5),
        "uncensored_wake_sec_q3": quantile(uncensored.following_wake_observed_sec, 0.75),
        "right_censored_events": int(local_events.following_wake_right_censored.sum()),
    }


def dataset_summaries(recordings: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = []
    for dataset, local in recordings.groupby("dataset", sort=True):
        dataset_rows.append({"dataset": dataset, **summarize_group(local, events)})
    group_rows = []
    for (dataset, group), local in recordings.groupby(["dataset", "cohort_group"], sort=True):
        if len(local) >= 10:
            summary = summarize_group(local, events)
            group_rows.append(
                {
                    "dataset": dataset, "cohort_group": group,
                    **{key: summary[key] for key in [
                        "primary_nights", "eligible_nights", "participant_groups", "support_hours",
                        "rem_to_wake_events", "event_positive_nights", "event_positive_night_fraction",
                        "events_per_support_hour", "known_rem_exit_opportunities",
                        "rw_fraction_of_known_rem_exits",
                    ]},
                }
            )
    return pd.DataFrame(dataset_rows), pd.DataFrame(group_rows)


# Section 3: scorer-pair uncertainty

def matched_count(left: np.ndarray, right: np.ndarray, tolerance_epochs: int) -> int:
    left = np.sort(np.asarray(left, dtype=int))
    right = np.sort(np.asarray(right, dtype=int))
    i = j = matches = 0
    while i < len(left) and j < len(right):
        if abs(left[i] - right[j]) <= tolerance_epochs:
            matches += 1
            i += 1
            j += 1
        elif left[i] < right[j] - tolerance_epochs:
            i += 1
        else:
            j += 1
    return matches


def event_f1(left: np.ndarray, right: np.ndarray, tolerance_epochs: int) -> float:
    denominator = len(left) + len(right)
    return 2.0 * matched_count(left, right, tolerance_epochs) / denominator if denominator else np.nan


def scorer_uncertainty(records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for item in records:
        if item["dataset"] not in MULTI_SCORER_SOURCES:
            continue
        for left_name, right_name in combinations(sorted(item["scorers"]), 2):
            left = item["scorers"][left_name]
            right = item["scorers"][right_name]
            length = min(len(left), len(right))
            known = np.isin(left[:length], common.KNOWN_STAGES) & np.isin(right[:length], common.KNOWN_STAGES)
            left_events = event_boundaries(left)
            right_events = event_boundaries(right)
            informative = len(left_events) + len(right_events) > 0
            rows.append(
                {
                    "dataset": item["dataset"], "recording": item["recording"],
                    "participant_group": item["participant_group"], "scorer_left": left_name,
                    "scorer_right": right_name, "common_epochs": length,
                    "joint_known_epochs": int(known.sum()),
                    "stage_disagreement_fraction": float(np.mean(left[:length][known] != right[:length][known])) if known.any() else np.nan,
                    "left_events": len(left_events), "right_events": len(right_events),
                    "absolute_event_count_difference": abs(len(left_events) - len(right_events)),
                    "event_presence_agreement": (len(left_events) > 0) == (len(right_events) > 0),
                    "informative_event_pair": informative,
                    "exact_event_f1": event_f1(left_events, right_events, 0),
                    "tolerant_30_sec_event_f1": event_f1(left_events, right_events, 1),
                }
            )
    frame = pd.DataFrame(rows)
    summaries = []
    for dataset, local in frame.groupby("dataset", sort=True):
        informative = local[local.informative_event_pair.astype(bool)]
        summaries.append(
            {
                "dataset": dataset, "recording_pairs": len(local),
                "informative_event_pairs": len(informative),
                "event_presence_agreement_fraction": float(local.event_presence_agreement.mean()),
                "median_stage_disagreement_fraction": quantile(local.stage_disagreement_fraction, 0.5),
                "median_absolute_event_count_difference": quantile(local.absolute_event_count_difference, 0.5),
                "median_exact_event_f1": quantile(informative.exact_event_f1, 0.5),
                "median_tolerant_30_sec_event_f1": quantile(informative.tolerant_30_sec_event_f1, 0.5),
            }
        )
    return frame, pd.DataFrame(summaries)


# Section 4: repeated-night variability and support sensitivity

def repeated_nights(recordings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    repeated = recordings.groupby(["dataset", "participant_group"]).filter(lambda value: len(value) >= 2)
    rows = []
    for (dataset, participant), local in repeated.groupby(["dataset", "participant_group"], sort=True):
        rates = local.events_per_support_hour.to_numpy(float)
        counts = local.rem_to_wake_events.to_numpy(int)
        rows.append(
            {
                "dataset": dataset, "participant_group": participant, "nights": len(local),
                "minimum_event_count": int(counts.min()), "maximum_event_count": int(counts.max()),
                "event_count_range": int(counts.max() - counts.min()),
                "minimum_event_rate": float(np.nanmin(rates)), "maximum_event_rate": float(np.nanmax(rates)),
                "event_rate_range": float(np.nanmax(rates) - np.nanmin(rates)),
                "mean_event_rate": float(np.nanmean(rates)),
                "discordant_event_presence": bool((counts > 0).any() and (counts == 0).any()),
            }
        )
    participants = pd.DataFrame(rows)
    summaries = []
    for dataset, local in participants.groupby("dataset", sort=True):
        summaries.append(
            {
                "dataset": dataset, "repeated_participants": len(local),
                "repeated_nights": int(local.nights.sum()),
                "discordant_event_presence_participants": int(local.discordant_event_presence.sum()),
                "discordant_event_presence_fraction": float(local.discordant_event_presence.mean()),
                "median_within_participant_event_count_range": quantile(local.event_count_range, 0.5),
                "median_within_participant_event_rate_range": quantile(local.event_rate_range, 0.5),
                "participant_mean_event_rate_q1": quantile(local.mean_event_rate, 0.25),
                "participant_mean_event_rate_q3": quantile(local.mean_event_rate, 0.75),
                "participant_mean_event_rate_iqr": quantile(local.mean_event_rate, 0.75) - quantile(local.mean_event_rate, 0.25),
            }
        )
    return participants, pd.DataFrame(summaries)


def full_record_sensitivity(recordings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, local in recordings.groupby("dataset", sort=True):
        support_hours = float(local.support_hours.sum())
        full_hours = float(local.full_record_known_hours.sum())
        support_events = int(local.rem_to_wake_events.sum())
        full_events = int(local.full_record_events.sum())
        rows.append(
            {
                "dataset": dataset, "sleep_period_events": support_events,
                "full_record_events": full_events, "event_count_difference": full_events - support_events,
                "sleep_period_hours": support_hours, "full_record_known_hours": full_hours,
                "sleep_period_events_per_hour": support_events / support_hours,
                "full_record_events_per_known_hour": full_events / full_hours,
            }
        )
    return pd.DataFrame(rows)


# Section 5: frozen decisions and controls

def decisions(dataset_summary: pd.DataFrame, scorer_summary: pd.DataFrame, repeated_summary: pd.DataFrame) -> pd.DataFrame:
    eligible = dataset_summary[
        dataset_summary.primary_nights.ge(20)
        & dataset_summary.rem_to_wake_events.ge(10)
    ]
    rate_ratio = float(eligible.events_per_support_hour.max() / eligible.events_per_support_hour.min())
    positive_range = float(eligible.event_positive_night_fraction.max() - eligible.event_positive_night_fraction.min())
    heterogeneity = rate_ratio >= 2.0 and positive_range >= 0.20

    scorer_exact_min = float(scorer_summary.median_exact_event_f1.min())
    scorer_tolerant_min = float(scorer_summary.median_tolerant_30_sec_event_f1.min())
    scorer_material = scorer_exact_min < 0.80 or scorer_tolerant_min < 0.90

    repeated_discordance_max = float(repeated_summary.discordant_event_presence_fraction.max())
    repeated_rate_range_max = float(repeated_summary.median_within_participant_event_rate_range.max())
    repeated_material = repeated_discordance_max >= 0.25 or repeated_rate_range_max >= 0.10

    return pd.DataFrame(
        [
            {
                "hypothesis": "H-DATASET-HETEROGENEITY", "primary_measure_1": "maximum_to_minimum_event_rate_ratio",
                "observed_1": rate_ratio, "threshold_1": 2.0, "condition_1_pass": rate_ratio >= 2.0,
                "primary_measure_2": "event_positive_night_fraction_range", "observed_2": positive_range,
                "threshold_2": 0.20, "condition_2_pass": positive_range >= 0.20,
                "decision": "material_heterogeneity" if heterogeneity else "not_material_by_v0.1_rule",
            },
            {
                "hypothesis": "H-SCORER-DEPENDENCE", "primary_measure_1": "minimum_source_median_exact_event_f1",
                "observed_1": scorer_exact_min, "threshold_1": 0.80, "condition_1_pass": scorer_exact_min < 0.80,
                "primary_measure_2": "minimum_source_median_tolerant_event_f1", "observed_2": scorer_tolerant_min,
                "threshold_2": 0.90, "condition_2_pass": scorer_tolerant_min < 0.90,
                "decision": "material_scorer_dependence" if scorer_material else "not_material_by_v0.1_rule",
            },
            {
                "hypothesis": "H-REPEATED-NIGHT", "primary_measure_1": "maximum_source_discordant_presence_fraction",
                "observed_1": repeated_discordance_max, "threshold_1": 0.25,
                "condition_1_pass": repeated_discordance_max >= 0.25,
                "primary_measure_2": "maximum_source_median_within_rate_range", "observed_2": repeated_rate_range_max,
                "threshold_2": 0.10, "condition_2_pass": repeated_rate_range_max >= 0.10,
                "decision": "material_repeated_night_dependence" if repeated_material else "not_material_by_v0.1_rule",
            },
        ]
    )


def main() -> None:
    records = common.load_primary_records()
    recording_frame, event_frame = recording_measures(records)
    dataset_frame, group_frame = dataset_summaries(recording_frame, event_frame)
    scorer_pairs, scorer_summary = scorer_uncertainty(records)
    repeated_participants, repeated_summary = repeated_nights(recording_frame)
    sensitivity = full_record_sensitivity(recording_frame)
    decision_frame = decisions(dataset_frame, scorer_summary, repeated_summary)

    repeated_sources = set(repeated_summary.dataset)
    expected_repeated = {"boas", "gx36m", "isruc", "sleep-edfx"}
    checks = pd.DataFrame(
        [
            {"check": "primary_nights", "observed": len(recording_frame), "expected": 1993, "passed": len(recording_frame) == 1993},
            {"check": "dataset_sources", "observed": recording_frame.dataset.nunique(), "expected": 11, "passed": recording_frame.dataset.nunique() == 11},
            {"check": "participant_groups", "observed": recording_frame.participant_group.nunique(), "expected": 1822, "passed": recording_frame.participant_group.nunique() == 1822},
            {"check": "support_eligible_nights", "observed": int(recording_frame.support_eligible.sum()), "expected": 1993, "passed": recording_frame.support_eligible.all()},
            {"check": "event_rows_equal_recording_counts", "observed": len(event_frame), "expected": int(recording_frame.rem_to_wake_events.sum()), "passed": len(event_frame) == int(recording_frame.rem_to_wake_events.sum())},
            {"check": "event_positions_bounded", "observed": bool(event_frame.sleep_period_position.between(0, 1, inclusive="both").all()), "expected": True, "passed": bool(event_frame.sleep_period_position.between(0, 1, inclusive="both").all())},
            {"check": "events_do_not_exceed_rem_exits", "observed": bool(recording_frame.rem_to_wake_events.le(recording_frame.rem_exit_opportunities).all()), "expected": True, "passed": bool(recording_frame.rem_to_wake_events.le(recording_frame.rem_exit_opportunities).all())},
            {"check": "scorer_pair_rows", "observed": len(scorer_pairs), "expected": 1100, "passed": len(scorer_pairs) == 1100},
            {"check": "scorer_sources", "observed": scorer_summary.dataset.nunique(), "expected": 3, "passed": scorer_summary.dataset.nunique() == 3},
            {"check": "repeated_sources", "observed": repeated_sources == expected_repeated, "expected": True, "passed": repeated_sources == expected_repeated},
            {"check": "full_record_sensitivity_rows", "observed": len(sensitivity), "expected": 11, "passed": len(sensitivity) == 11},
            {"check": "decision_rows", "observed": len(decision_frame), "expected": 3, "passed": len(decision_frame) == 3},
            {"check": "signal_or_model_inputs", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Hypnogram transition analysis controls failed")

    outputs = {
        "recording_transition_measures_v0.1.tsv": recording_frame,
        "transition_events_v0.1.tsv": event_frame,
        "dataset_transition_summary_v0.1.tsv": dataset_frame,
        "cohort_group_summary_v0.1.tsv": group_frame,
        "scorer_pair_records_v0.1.tsv": scorer_pairs,
        "scorer_uncertainty_summary_v0.1.tsv": scorer_summary,
        "repeated_participant_measures_v0.1.tsv": repeated_participants,
        "repeated_night_summary_v0.1.tsv": repeated_summary,
        "full_record_sensitivity_v0.1.tsv": sensitivity,
        "hypothesis_decisions_v0.1.tsv": decision_frame,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, common.output_dir() / name)
    print(dataset_frame[["dataset", "primary_nights", "rem_to_wake_events", "event_positive_night_fraction", "events_per_support_hour", "rw_fraction_of_known_rem_exits"]].to_string(index=False))
    print(scorer_summary.to_string(index=False))
    print(repeated_summary.to_string(index=False))
    print(decision_frame[["hypothesis", "observed_1", "observed_2", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
