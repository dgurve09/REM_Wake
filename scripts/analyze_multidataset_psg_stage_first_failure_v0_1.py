"""Describe why the frozen multi-dataset stage-first event rule failed."""

from __future__ import annotations

import gzip

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: retained inputs

def load_scores() -> pd.DataFrame:
    manifest = pd.read_csv(
        common.output_dir() / "external_artifact_manifest_v0.1.tsv",
        sep="\t",
    )
    frames = []
    for item in manifest.query("artifact_type == 'stage_scores'").itertuples(index=False):
        path = features.data_parent() / item.relative_path
        if features.sha256(path) != item.sha256:
            raise ValueError(f"Stage-score hash changed: {path}")
        with gzip.open(path, "rt", encoding="utf-8") as source:
            frames.append(pd.read_csv(source, sep="\t"))
    return pd.concat(frames, ignore_index=True)


def load_alarms() -> pd.DataFrame:
    return pd.read_csv(common.output_dir() / "derived_event_alarms_v0.1.tsv", sep="\t")


# Section 2: sequence fragmentation

def contiguous_bounds(times: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(np.diff(times) > 15.0 + 1e-6) + 1, len(times)]


def sequence_counts(times: np.ndarray, stages: np.ndarray) -> tuple[int, list[float]]:
    transitions = 0
    rem_durations = []
    bounds = contiguous_bounds(times)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        local = stages[start:stop]
        transitions += int(np.count_nonzero(local[1:] != local[:-1]))
        rem = local == "REM"
        edges = np.diff(np.r_[False, rem, False].astype(int))
        starts = np.flatnonzero(edges == 1)
        stops = np.flatnonzero(edges == -1)
        rem_durations.extend(((stops - starts) * 15.0).tolist())
    return transitions, rem_durations


def fragmentation_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, dataset_scores in scores.groupby("dataset", sort=True):
        truth_transitions = 0
        predicted_transitions = 0
        truth_rem_durations = []
        predicted_rem_durations = []
        for _, group in dataset_scores.groupby("recording", sort=True):
            group = group.sort_values("candidate_time_sec")
            times = group.candidate_time_sec.to_numpy(float)
            truth_count, truth_bouts = sequence_counts(times, group.truth_stage.to_numpy(str))
            predicted_count, predicted_bouts = sequence_counts(times, group.predicted_stage.to_numpy(str))
            truth_transitions += truth_count
            predicted_transitions += predicted_count
            truth_rem_durations.extend(truth_bouts)
            predicted_rem_durations.extend(predicted_bouts)
        hours = len(dataset_scores) * 15.0 / 3600.0
        rows.append(
            {
                "dataset": dataset,
                "supported_hours": hours,
                "human_all_stage_transitions": truth_transitions,
                "predicted_all_stage_transitions": predicted_transitions,
                "human_transitions_per_hour": truth_transitions / hours,
                "predicted_transitions_per_hour": predicted_transitions / hours,
                "predicted_to_human_transition_ratio": predicted_transitions / truth_transitions,
                "human_rem_bouts": len(truth_rem_durations),
                "predicted_rem_bouts": len(predicted_rem_durations),
                "human_rem_bouts_per_hour": len(truth_rem_durations) / hours,
                "predicted_rem_bouts_per_hour": len(predicted_rem_durations) / hours,
                "predicted_to_human_rem_bout_ratio": len(predicted_rem_durations) / len(truth_rem_durations),
                "human_median_rem_bout_sec": float(np.median(truth_rem_durations)),
                "predicted_median_rem_bout_sec": float(np.median(predicted_rem_durations)),
            }
        )
    return pd.DataFrame(rows)


# Section 3: predictions aligned to corrected references

def reference_patterns(
    scores: pd.DataFrame,
    recordings: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail = []
    offsets = np.asarray([-30.0, -15.0, 0.0, 15.0])
    score_groups = {
        recording: group.set_index("candidate_time_sec")
        for recording, group in scores.groupby("recording", sort=True)
    }
    for recording, item in recordings.items():
        group = score_groups[recording]
        for reference in item["primary"]:
            times = reference + offsets
            supported = all(time in group.index for time in times)
            if supported:
                predicted = group.loc[times, "predicted_stage"].astype(str).tolist()
                pattern = "|".join(predicted)
            else:
                pattern = "UNSUPPORTED"
            detail.append(
                {
                    "dataset": item["dataset"],
                    "recording": recording,
                    "participant_group": item["participant_group"],
                    "reference_time_sec": float(reference),
                    "four_decisions_supported": supported,
                    "predicted_pattern": pattern,
                    "exact_confirm2_pattern": pattern == "REM|REM|W|W",
                    "hard_change_at_reference": supported
                    and predicted[1] == "REM"
                    and predicted[2] == "W",
                }
            )
    detail_frame = pd.DataFrame(detail)
    summary = (
        detail_frame.groupby("dataset", as_index=False)
        .agg(
            reference_events=("reference_time_sec", "size"),
            supported_references=("four_decisions_supported", "sum"),
            exact_confirm2_references=("exact_confirm2_pattern", "sum"),
            hard_change_at_reference=("hard_change_at_reference", "sum"),
        )
    )
    summary["exact_confirm2_fraction"] = (
        summary.exact_confirm2_references / summary.reference_events
    )
    summary["hard_change_at_reference_fraction"] = (
        summary.hard_change_at_reference / summary.reference_events
    )
    pattern_counts = (
        detail_frame.groupby(["dataset", "predicted_pattern"], as_index=False)
        .size()
        .rename(columns={"size": "reference_events"})
        .sort_values(["dataset", "reference_events", "predicted_pattern"], ascending=[True, False, True])
        .reset_index(drop=True)
    )
    return summary, pattern_counts


# Section 4: descriptive alarm displacement

def alarm_offsets(
    alarms: pd.DataFrame,
    recordings: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for dataset in common.direct.DATASETS:
        references = []
        recording_names = []
        for recording, item in recordings.items():
            if item["dataset"] == dataset:
                references.extend(item["primary"].astype(float).tolist())
                recording_names.extend([recording] * len(item["primary"]))
        for candidate in common.CANDIDATES:
            offsets = []
            covered = 0
            local = alarms[
                alarms.heldout_dataset.eq(dataset) & alarms.candidate.eq(candidate)
            ]
            alarm_map = {
                recording: np.sort(group.event_time_sec.to_numpy(float))
                for recording, group in local.groupby("recording", sort=True)
            }
            for recording, reference in zip(recording_names, references):
                values = alarm_map.get(recording, np.asarray([], dtype=float)) - reference
                values = values[np.abs(values) <= 120.0 + 1e-9]
                if len(values):
                    covered += 1
                    offsets.append(float(values[np.argmin(np.abs(values))]))
            rows.append(
                {
                    "dataset": dataset,
                    "candidate": candidate,
                    "reference_events": len(references),
                    "references_with_alarm_within_120_sec": covered,
                    "coverage_fraction": covered / len(references),
                    "nearest_offset_median_sec": float(np.median(offsets)) if offsets else np.nan,
                    "nearest_offset_q25_sec": float(np.quantile(offsets, 0.25)) if offsets else np.nan,
                    "nearest_offset_q75_sec": float(np.quantile(offsets, 0.75)) if offsets else np.nan,
                }
            )
    return pd.DataFrame(rows)


# Section 5: retained outputs and controls

def main() -> None:
    scores = load_scores()
    alarms = load_alarms()
    recordings = common.load_recordings()
    fragmentation = fragmentation_table(scores)
    pattern_summary, pattern_counts = reference_patterns(scores, recordings)
    offsets = alarm_offsets(alarms, recordings)

    checks = pd.DataFrame(
        [
            {"check": "score_rows", "observed": len(scores), "expected": 372361, "passed": len(scores) == 372361},
            {"check": "fragmentation_datasets", "observed": len(fragmentation), "expected": 3, "passed": len(fragmentation) == 3},
            {"check": "corrected_references", "observed": int(pattern_summary.reference_events.sum()), "expected": 482, "passed": int(pattern_summary.reference_events.sum()) == 482},
            {"check": "all_reference_windows_supported", "observed": int(pattern_summary.supported_references.sum()), "expected": 482, "passed": int(pattern_summary.supported_references.sum()) == 482},
            {"check": "pattern_counts_complete", "observed": int(pattern_counts.reference_events.sum()), "expected": 482, "passed": int(pattern_counts.reference_events.sum()) == 482},
            {"check": "alarm_offset_rows", "observed": len(offsets), "expected": 6, "passed": len(offsets) == 6},
            {"check": "no_new_model_or_rule", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Stage-first failure diagnostic controls failed")

    outputs = {
        "failure_sequence_fragmentation_v0.1.tsv": fragmentation,
        "failure_reference_pattern_summary_v0.1.tsv": pattern_summary,
        "failure_reference_pattern_counts_v0.1.tsv": pattern_counts,
        "failure_alarm_offset_diagnostics_v0.1.tsv": offsets,
        "failure_diagnostic_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, common.output_dir() / name)

    print(fragmentation.to_string(index=False))
    print(pattern_summary.to_string(index=False))
    print(offsets.to_string(index=False))


if __name__ == "__main__":
    main()
