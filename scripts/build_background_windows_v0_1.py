"""Build deterministic background-window candidates v0.1.

This script defines non-transition windows for later preprocessing review. It
does not train or evaluate a model, and it does not write raw signal arrays.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT_COUNT = 128
LABEL_VERSION = "v0.1"
EVENT_DURATION_SEC = 30.0
WINDOW_SEC = 240.0
HALF_WINDOW_SEC = WINDOW_SEC / 2.0
UNCERTAINTY_HALF_SEC = 15.0
REM_WAKE_EXCLUSION_SEC = HALF_WINDOW_SEC + UNCERTAINTY_HALF_SEC
SAMPLE_RATE_HZ = 256.0
MAX_REVIEW_ROWS_PER_GROUP = 2

STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    8: "PSG disconnection",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "labels" / "background_windows_v0.1"


def subject_event_path(root: Path, subject_id: int) -> Path:
    subject = f"sub-{subject_id}"
    return root / subject / "eeg" / f"{subject}_task-Sleep_acq-psg_events.tsv"


def load_participants(root: Path) -> pd.DataFrame:
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants["subject_id"] = participants["participant_id"].str.replace(
        "sub-", "", regex=False
    ).astype(int)
    return participants.set_index("subject_id")


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def stage_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return STAGE_NAMES.get(int(value), f"unknown_{int(value)}")


def stage_family(value: int) -> str:
    if value == 0:
        return "Wake"
    if value == 4:
        return "REM"
    if value in {1, 2, 3}:
        return "NREM"
    if value == 8:
        return "PSG disconnection"
    return "unknown"


def transition_type(stage_from: int, stage_to: int) -> str:
    if stage_from == 4 and stage_to == 0:
        return "REM_to_Wake"
    if stage_from == 0 and stage_to == 4:
        return "Wake_to_REM"
    if stage_from == stage_to:
        return "same_stage"
    return "other_stage_transition"


def first_exclusion_reason(
    onsets: list[float],
    durations: list[float],
    stages: list[object],
    index: int,
    rem_wake_boundaries: list[float],
    event_coverage_end_sec: float,
) -> tuple[str, float | str, tuple[int, int] | None]:
    previous_stage = stages[index]
    following_stage = stages[index + 1]

    if pd.isna(previous_stage) or pd.isna(following_stage):
        return "missing_stage_at_center", "", None

    previous_duration = durations[index]
    following_duration = durations[index + 1]
    if previous_duration != EVENT_DURATION_SEC or following_duration != EVENT_DURATION_SEC:
        return "non_30_sec_epoch_at_center", "", None

    stage_from = int(previous_stage)
    stage_to = int(following_stage)
    if stage_from == 8 or stage_to == 8:
        return "psg_disconnection_at_center", "", None

    if transition_type(stage_from, stage_to) in {"REM_to_Wake", "Wake_to_REM"}:
        return "rem_wake_boundary_center", 0.0, None

    center_sec = onsets[index + 1]
    window_start_sec = center_sec - HALF_WINDOW_SEC
    window_end_sec = center_sec + HALF_WINDOW_SEC
    if window_start_sec < 0 or window_end_sec > event_coverage_end_sec:
        return "edge_window_incomplete", "", None

    expected_epochs = int(WINDOW_SEC / EVENT_DURATION_SEC)
    start_index = index - (expected_epochs // 2 - 1)
    stop_index = start_index + expected_epochs
    if start_index < 0 or stop_index > len(stages):
        return "edge_window_incomplete", "", None

    window_stages = stages[start_index:stop_index]
    window_durations = durations[start_index:stop_index]
    window_onsets = onsets[start_index:stop_index]
    if len(window_stages) != expected_epochs:
        return "window_epoch_count_mismatch", "", None
    if any(pd.isna(value) for value in window_stages):
        return "missing_stage_in_window", "", None
    if any(value != EVENT_DURATION_SEC for value in window_durations):
        return "non_30_sec_epoch_in_window", "", None
    onset_steps = [
        window_onsets[position] - window_onsets[position - 1]
        for position in range(1, len(window_onsets))
    ]
    if any(value != EVENT_DURATION_SEC for value in onset_steps):
        return "onset_gap_in_window", "", None
    actual_window_start = window_onsets[0]
    actual_window_end = window_onsets[-1] + window_durations[-1]
    if actual_window_start != window_start_sec or actual_window_end != window_end_sec:
        return "window_time_mismatch", "", None
    if any(int(value) == 8 for value in window_stages):
        return "psg_disconnection_in_window", "", None

    if rem_wake_boundaries:
        min_distance = min(abs(center_sec - boundary) for boundary in rem_wake_boundaries)
    else:
        min_distance = float("inf")
    if min_distance < REM_WAKE_EXCLUSION_SEC:
        return "too_close_to_rem_wake_uncertainty", min_distance, None

    return "eligible", min_distance, (start_index, stop_index)


def inspect_subject(
    root: Path,
    participants: pd.DataFrame,
    labels: pd.DataFrame,
    subject_id: int,
) -> tuple[list[dict], dict, list[dict]]:
    subject = f"sub-{subject_id}"
    participant = participants.loc[subject_id]
    events_path = subject_event_path(root, subject_id)
    events = read_tsv(events_path)
    onsets = [float(value) for value in events["onset"]]
    durations = [float(value) for value in events["duration"]]
    stages = list(events["stage_hum"])
    event_coverage_end_sec = float(max(onsets) + durations[-1])
    subject_labels = labels[labels["subject"] == subject]
    rem_wake_boundaries = sorted(float(value) for value in subject_labels["nominal_boundary_sec"])

    eligible_rows = []
    exclusion_rows = []
    potential_centers = max(0, len(events) - 1)

    for index in range(potential_centers):
        reason, min_distance, window_bounds = first_exclusion_reason(
            onsets, durations, stages, index, rem_wake_boundaries, event_coverage_end_sec
        )
        if reason != "eligible":
            exclusion_rows.append(
                {
                    "subject": subject,
                    "pid": int(participant["pid"]),
                    "exclusion_reason": reason,
                }
            )
            continue

        start_index, stop_index = window_bounds
        stage_from = int(stages[index])
        stage_to = int(stages[index + 1])
        center_sec = onsets[index + 1]
        window_start_sec = center_sec - HALF_WINDOW_SEC
        window_end_sec = center_sec + HALF_WINDOW_SEC
        window_stage_values = [int(value) for value in stages[start_index:stop_index]]
        unique_stage_values = sorted(set(window_stage_values))
        center_pair = f"{stage_label(stage_from)}_to_{stage_label(stage_to)}"

        if len(unique_stage_values) == 1:
            background_tier = "strict_same_stage_window"
        else:
            background_tier = "nontarget_window_no_remwake_nearby"

        eligible_rows.append(
            {
                "label_version": LABEL_VERSION,
                "subject": subject,
                "participant_id": participant["participant_id"],
                "pid": int(participant["pid"]),
                "background_tier": background_tier,
                "center_pair": center_pair,
                "center_transition_type": transition_type(stage_from, stage_to),
                "stage_from": stage_from,
                "stage_from_label": stage_label(stage_from),
                "stage_to": stage_to,
                "stage_to_label": stage_label(stage_to),
                "center_epoch_index_zero_based": index + 1,
                "center_sec": center_sec,
                "uncertainty_exclusion_sec": REM_WAKE_EXCLUSION_SEC,
                "min_distance_to_remwake_boundary_sec": min_distance,
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "headband_start_sample": int(round(window_start_sec * SAMPLE_RATE_HZ)),
                "headband_stop_sample": int(round(window_end_sec * SAMPLE_RATE_HZ)),
                "psg_start_sample": int(round(window_start_sec * SAMPLE_RATE_HZ)),
                "psg_stop_sample": int(round(window_end_sec * SAMPLE_RATE_HZ)),
                "window_stage_values": ";".join(str(value) for value in unique_stage_values),
                "window_stage_labels": ";".join(stage_label(value) for value in unique_stage_values),
                "window_stage_families": ";".join(
                    sorted({stage_family(value) for value in unique_stage_values})
                ),
                "window_epoch_count": int(len(window_stage_values)),
                "background_decision": "candidate",
                "relative_events_path": events_path.relative_to(root).as_posix(),
            }
        )

    summary = {
        "subject": subject,
        "participant_id": participant["participant_id"],
        "pid": int(participant["pid"]),
        "potential_boundary_centers": potential_centers,
        "rem_wake_transition_boundaries": len(rem_wake_boundaries),
        "eligible_background_centers": len(eligible_rows),
        "strict_same_stage_backgrounds": sum(
            row["background_tier"] == "strict_same_stage_window" for row in eligible_rows
        ),
        "nontarget_no_remwake_backgrounds": sum(
            row["background_tier"] == "nontarget_window_no_remwake_nearby"
            for row in eligible_rows
        ),
    }
    return eligible_rows, summary, exclusion_rows


def select_review_windows(eligible: pd.DataFrame) -> pd.DataFrame:
    if eligible.empty:
        return eligible
    sort_cols = ["subject", "background_tier", "center_pair", "center_sec"]
    reviewed = (
        eligible.sort_values(sort_cols)
        .groupby(["subject", "background_tier", "center_pair"], group_keys=False)
        .head(MAX_REVIEW_ROWS_PER_GROUP)
        .copy()
    )
    reviewed.insert(0, "background_review_id", range(1, len(reviewed) + 1))
    return reviewed


def metric_rows(
    eligible: pd.DataFrame,
    recording_summary: pd.DataFrame,
    exclusions: pd.DataFrame,
    review: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"metric": "recordings_checked", "value": len(recording_summary)},
        {
            "metric": "potential_boundary_centers",
            "value": int(recording_summary["potential_boundary_centers"].sum()),
        },
        {"metric": "eligible_background_centers", "value": len(eligible)},
        {
            "metric": "strict_same_stage_backgrounds",
            "value": int((eligible["background_tier"] == "strict_same_stage_window").sum()),
        },
        {
            "metric": "nontarget_no_remwake_backgrounds",
            "value": int(
                (eligible["background_tier"] == "nontarget_window_no_remwake_nearby").sum()
            ),
        },
        {"metric": "review_candidate_rows_written", "value": len(review)},
        {
            "metric": "rem_wake_exclusion_sec",
            "value": REM_WAKE_EXCLUSION_SEC,
        },
    ]
    if not exclusions.empty:
        for reason, group in exclusions.groupby("exclusion_reason", sort=True):
            rows.append({"metric": f"excluded_{reason}", "value": int(len(group))})
    return pd.DataFrame(rows)


def write_readme(
    destination: Path,
    pool_summary: pd.DataFrame,
    review: pd.DataFrame,
) -> None:
    raw_metric_value = dict(zip(pool_summary["metric"], pool_summary["value"]))

    def metric_value(name: str) -> str:
        value = float(raw_metric_value[name])
        if value.is_integer():
            return str(int(value))
        return str(value)

    text = f"""# Background Windows v0.1

**Created:** 2026-07-09
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Transition labels:** `labels/transition_labels_v0.1/`
**Model training performed:** No

## 1. Purpose

This artifact defines deterministic non-transition background-window rules for later wearable EEG preprocessing.

The uncertainty addressed here is whether negative/background windows can be selected without overlapping the 30-second uncertainty interval around REM/Wake transition labels.

## 2. Method

- Use the same 240-second extraction window as the transition-label artifact.
- Treat every adjacent PSG epoch boundary as a possible background center.
- Exclude direct REM-to-Wake and Wake-to-REM centers.
- Exclude any candidate whose 240-second window intersects a REM/Wake boundary uncertainty interval.
- The exclusion radius is 135 seconds: 120 seconds half-window plus 15 seconds label uncertainty.
- Exclude edge windows, missing labels, non-30-second epochs, and PSG disconnection windows.
- Keep two background tiers:
  - `strict_same_stage_window`: all epochs in the window have the same PSG stage.
  - `nontarget_window_no_remwake_nearby`: the window may contain other stage changes but no REM/Wake boundary within the exclusion radius.

## 3. Result

| Item | Value |
|---|---:|
| Recordings checked | {metric_value('recordings_checked')} |
| Potential boundary centers | {metric_value('potential_boundary_centers')} |
| Eligible background centers | {metric_value('eligible_background_centers')} |
| Strict same-stage backgrounds | {metric_value('strict_same_stage_backgrounds')} |
| Non-target backgrounds with no nearby REM/Wake boundary | {metric_value('nontarget_no_remwake_backgrounds')} |
| Review candidate rows written | {metric_value('review_candidate_rows_written')} |
| REM/Wake exclusion radius, seconds | {metric_value('rem_wake_exclusion_sec')} |

## 4. Outputs

| File | Purpose |
|---|---|
| `background_window_pool_summary_v0.1.tsv` | Overall pool and exclusion counts |
| `recording_background_summary_v0.1.tsv` | Per-recording eligible background counts |
| `background_stage_pair_summary_v0.1.tsv` | Counts by background tier and center-stage pair |
| `background_review_windows_v0.1.tsv` | Deterministic review-sized candidate table; not a final training set |

## 5. Limitations

- The full eligible pool is summarized but not written as a large table.
- The review table is for preprocessing inspection and split-policy design, not model training.
- Background windows are still derived from 30-second PSG epochs.
- Signal amplitude or artifact quality is handled separately by the quality-flag artifact.

## 6. Decision

Use these rules as background-window specification `v0.1` for the label/preprocessing gate. Do not create final train/validation/test splits until quality flags and background sampling policy are reviewed together.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = dataset_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    participants = load_participants(root)
    labels = read_tsv(repo_root() / "labels" / "transition_labels_v0.1" / "transition_labels_v0.1.tsv")

    eligible_rows = []
    recording_rows = []
    exclusion_rows = []
    for subject_id in range(1, SUBJECT_COUNT + 1):
        eligible, recording_summary, exclusions = inspect_subject(
            root, participants, labels, subject_id
        )
        eligible_rows.extend(eligible)
        recording_rows.append(recording_summary)
        exclusion_rows.extend(exclusions)

    eligible = pd.DataFrame(eligible_rows)
    recording_summary = pd.DataFrame(recording_rows)
    exclusions = pd.DataFrame(exclusion_rows)
    if not eligible.empty:
        eligible.insert(0, "background_pool_id", range(1, len(eligible) + 1))

    review = select_review_windows(eligible)
    pool_summary = metric_rows(eligible, recording_summary, exclusions, review)
    stage_pair_summary = (
        eligible.groupby(["background_tier", "center_pair"], sort=True)
        .size()
        .reset_index(name="eligible_rows")
        if not eligible.empty
        else pd.DataFrame(columns=["background_tier", "center_pair", "eligible_rows"])
    )

    pool_summary.to_csv(
        destination / "background_window_pool_summary_v0.1.tsv", sep="\t", index=False
    )
    recording_summary.to_csv(
        destination / "recording_background_summary_v0.1.tsv", sep="\t", index=False
    )
    stage_pair_summary.to_csv(
        destination / "background_stage_pair_summary_v0.1.tsv", sep="\t", index=False
    )
    review.to_csv(destination / "background_review_windows_v0.1.tsv", sep="\t", index=False)
    write_readme(destination, pool_summary, review)

    print("Background windows v0.1")
    print(f"Potential centers: {int(recording_summary['potential_boundary_centers'].sum())}")
    print(f"Eligible background centers: {len(eligible)}")
    print(f"Review rows written: {len(review)}")
    print(f"Outputs: {destination}")


if __name__ == "__main__":
    main()
