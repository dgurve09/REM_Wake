"""Run the BOAS E0 REM/Wake transition inventory.

This script uses PSG `stage_hum` event files only. It derives deterministic
REM-to-Wake and Wake-to-REM adjacent-epoch candidates and summarizes label
quality. It does not train or evaluate a model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT_COUNT = 128
EVENT_DURATION_SEC = 30.0
INSPECTION_WINDOW_SEC = 240.0
HALF_WINDOW_SEC = INSPECTION_WINDOW_SEC / 2

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
    return repo_root() / "experiments" / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def subject_paths(root: Path, subject_id: int) -> dict[str, Path]:
    subject = f"sub-{subject_id}"
    prefix = root / subject / "eeg" / f"{subject}_task-Sleep"
    return {
        "psg_events": Path(f"{prefix}_acq-psg_events.tsv"),
        "psg_sidecar": Path(f"{prefix}_acq-psg_eeg.json"),
        "headband_sidecar": Path(f"{prefix}_acq-headband_eeg.json"),
    }


def load_participants(root: Path) -> pd.DataFrame:
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    participants["subject_id"] = participants["participant_id"].str.replace("sub-", "", regex=False).astype(int)
    return participants.set_index("subject_id")


def stage_label(value: object) -> str:
    if pd.isna(value):
        return "missing"
    return STAGE_NAMES.get(int(value), f"unknown_{int(value)}")


def count_disconnection_episodes(stage_hum: pd.Series) -> int:
    disconnection = stage_hum == 8
    starts = disconnection & ~disconnection.shift(fill_value=False)
    return int(starts.sum())


def transition_type(stage_from: int, stage_to: int) -> str:
    if stage_from == 4 and stage_to == 0:
        return "REM_to_Wake"
    if stage_from == 0 and stage_to == 4:
        return "Wake_to_REM"
    return ""


def inspect_recording(
    root: Path,
    participants: pd.DataFrame,
    subject_id: int,
) -> tuple[dict, dict, dict, list[dict]]:
    subject = f"sub-{subject_id}"
    paths = subject_paths(root, subject_id)
    participant = participants.loc[subject_id]
    psg_events = pd.read_csv(paths["psg_events"], sep="\t")
    psg_sidecar = read_json(paths["psg_sidecar"])
    headband_sidecar = read_json(paths["headband_sidecar"])

    psg_duration_sec = float(psg_sidecar["RecordingDuration"])
    headband_duration_sec = float(headband_sidecar["RecordingDuration"])
    psg_sampling_hz = float(psg_sidecar["SamplingFrequency"])
    headband_sampling_hz = float(headband_sidecar["SamplingFrequency"])

    event_coverage_end_sec = float(psg_events["onset"].max() + psg_events["duration"].iloc[-1])
    unlabeled_tail_sec = psg_duration_sec - event_coverage_end_sec
    duration_values = sorted(float(value) for value in psg_events["duration"].dropna().unique())
    stage_values = sorted(int(value) for value in psg_events["stage_hum"].dropna().unique())
    onset_step = psg_events["onset"].diff()
    expected_next_onset = psg_events["onset"].shift(1) + psg_events["duration"].shift(1)
    onset_gap_issue_count = int((psg_events["onset"][1:] != expected_next_onset[1:]).sum())

    stage_hum = psg_events["stage_hum"]
    missing_stage_count = int(stage_hum.isna().sum())
    non_30_sec_epoch_count = int((psg_events["duration"] != EVENT_DURATION_SEC).sum())
    disconnection_epoch_count = int((stage_hum == 8).sum())
    disconnection_episode_count = count_disconnection_episodes(stage_hum)
    disconnection_duration_sec = disconnection_epoch_count * EVENT_DURATION_SEC
    disconnection_onsets = psg_events.loc[stage_hum == 8, "onset"]

    candidates = []
    for index in range(len(psg_events) - 1):
        previous = psg_events.iloc[index]
        following = psg_events.iloc[index + 1]

        if pd.isna(previous["stage_hum"]) or pd.isna(following["stage_hum"]):
            continue

        stage_from = int(previous["stage_hum"])
        stage_to = int(following["stage_hum"])
        kind = transition_type(stage_from, stage_to)
        if not kind:
            continue

        prev_duration = float(previous["duration"])
        next_duration = float(following["duration"])
        if prev_duration != EVENT_DURATION_SEC or next_duration != EVENT_DURATION_SEC:
            continue

        boundary_sec = float(following["onset"])
        window_start_sec = boundary_sec - HALF_WINDOW_SEC
        window_end_sec = boundary_sec + HALF_WINDOW_SEC
        window_events = psg_events[
            (psg_events["onset"] >= window_start_sec)
            & (psg_events["onset"] < window_end_sec)
        ]
        window_disconnection_epochs = int((window_events["stage_hum"] == 8).sum())

        candidates.append(
            {
                "subject": subject,
                "participant_id": participant["participant_id"],
                "pid": int(participant["pid"]),
                "transition_type": kind,
                "is_primary_rem_to_wake": kind == "REM_to_Wake",
                "previous_epoch_index_zero_based": index,
                "next_epoch_index_zero_based": index + 1,
                "previous_onset_sec": float(previous["onset"]),
                "boundary_onset_sec": boundary_sec,
                "uncertainty_start_sec": boundary_sec - 15.0,
                "uncertainty_end_sec": boundary_sec + 15.0,
                "stage_from": stage_from,
                "stage_from_label": stage_label(stage_from),
                "stage_to": stage_to,
                "stage_to_label": stage_label(stage_to),
                "previous_duration_sec": prev_duration,
                "next_duration_sec": next_duration,
                "previous_begsample": int(previous["begsample"]),
                "next_begsample": int(following["begsample"]),
                "inspection_window_sec": INSPECTION_WINDOW_SEC,
                "psg_disconnection_epochs_in_window": window_disconnection_epochs,
                "has_psg_disconnection_in_window": window_disconnection_epochs > 0,
                "unlabeled_tail_sec": unlabeled_tail_sec,
                "relative_events_path": paths["psg_events"].relative_to(root).as_posix(),
            }
        )

    rem_to_wake_count = sum(1 for row in candidates if row["transition_type"] == "REM_to_Wake")
    wake_to_rem_count = sum(1 for row in candidates if row["transition_type"] == "Wake_to_REM")
    candidate_window_disconnection_count = sum(
        1 for row in candidates if row["has_psg_disconnection_in_window"]
    )

    recording_row = {
        "subject": subject,
        "participant_id": participant["participant_id"],
        "pid": int(participant["pid"]),
        "psg_event_rows": int(len(psg_events)),
        "psg_recording_duration_sec": psg_duration_sec,
        "headband_recording_duration_sec": headband_duration_sec,
        "duration_mismatch_sec": psg_duration_sec - headband_duration_sec,
        "event_coverage_end_sec": event_coverage_end_sec,
        "unlabeled_tail_sec": unlabeled_tail_sec,
        "stage_hum_missing_epochs": missing_stage_count,
        "stage_hum_disconnection_epochs": disconnection_epoch_count,
        "stage_hum_disconnection_episodes": disconnection_episode_count,
        "non_30_sec_epoch_count": non_30_sec_epoch_count,
        "onset_gap_issue_count": onset_gap_issue_count,
        "rem_to_wake_count": rem_to_wake_count,
        "wake_to_rem_count": wake_to_rem_count,
        "total_rem_wake_transition_count": rem_to_wake_count + wake_to_rem_count,
        "has_rem_to_wake": rem_to_wake_count > 0,
        "has_wake_to_rem": wake_to_rem_count > 0,
        "candidate_windows_with_disconnection": candidate_window_disconnection_count,
    }

    label_quality_row = {
        "subject": subject,
        "participant_id": participant["participant_id"],
        "pid": int(participant["pid"]),
        "psg_event_rows": int(len(psg_events)),
        "stage_hum_values": ";".join(str(value) for value in stage_values),
        "duration_values_sec": ";".join(str(value) for value in duration_values),
        "stage_hum_missing_epochs": missing_stage_count,
        "stage_hum_disconnection_epochs": disconnection_epoch_count,
        "stage_hum_disconnection_episodes": disconnection_episode_count,
        "non_30_sec_epoch_count": non_30_sec_epoch_count,
        "onset_gap_issue_count": onset_gap_issue_count,
        "psg_sampling_hz": psg_sampling_hz,
        "headband_sampling_hz": headband_sampling_hz,
        "sampling_mismatch_hz": psg_sampling_hz - headband_sampling_hz,
        "psg_recording_duration_sec": psg_duration_sec,
        "headband_recording_duration_sec": headband_duration_sec,
        "duration_mismatch_sec": psg_duration_sec - headband_duration_sec,
        "event_coverage_end_sec": event_coverage_end_sec,
        "unlabeled_tail_sec": unlabeled_tail_sec,
    }

    disconnection_row = {
        "subject": subject,
        "participant_id": participant["participant_id"],
        "pid": int(participant["pid"]),
        "psg_disconnection_epochs": disconnection_epoch_count,
        "psg_disconnection_duration_sec": disconnection_duration_sec,
        "psg_disconnection_episodes": disconnection_episode_count,
        "first_disconnection_onset_sec": (
            float(disconnection_onsets.min()) if len(disconnection_onsets) else ""
        ),
        "last_disconnection_onset_sec": (
            float(disconnection_onsets.max()) if len(disconnection_onsets) else ""
        ),
        "candidate_windows_with_disconnection": candidate_window_disconnection_count,
    }

    return recording_row, label_quality_row, disconnection_row, candidates


def participant_inventory(recordings: pd.DataFrame) -> pd.DataFrame:
    grouped_rows = []
    for pid, group in recordings.groupby("pid", sort=True):
        grouped_rows.append(
            {
                "pid": int(pid),
                "recording_count": int(len(group)),
                "subjects": ";".join(group["subject"]),
                "participant_ids": ";".join(group["participant_id"]),
                "rem_to_wake_count": int(group["rem_to_wake_count"].sum()),
                "wake_to_rem_count": int(group["wake_to_rem_count"].sum()),
                "total_rem_wake_transition_count": int(group["total_rem_wake_transition_count"].sum()),
                "recordings_with_rem_to_wake": int(group["has_rem_to_wake"].sum()),
                "recordings_with_wake_to_rem": int(group["has_wake_to_rem"].sum()),
                "recordings_with_any_rem_wake_transition": int(
                    (group["total_rem_wake_transition_count"] > 0).sum()
                ),
                "stage_hum_disconnection_epochs": int(group["stage_hum_disconnection_epochs"].sum()),
                "non_30_sec_epoch_count": int(group["non_30_sec_epoch_count"].sum()),
                "unlabeled_tail_min_sec": float(group["unlabeled_tail_sec"].min()),
                "unlabeled_tail_max_sec": float(group["unlabeled_tail_sec"].max()),
            }
        )
    return pd.DataFrame(grouped_rows)


def unlabeled_tail_summary(label_quality: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subject",
        "participant_id",
        "pid",
        "psg_recording_duration_sec",
        "event_coverage_end_sec",
        "unlabeled_tail_sec",
    ]
    tail = label_quality[columns].copy()
    tail["unlabeled_tail_fraction"] = (
        tail["unlabeled_tail_sec"] / tail["psg_recording_duration_sec"]
    )
    tail["unlabeled_tail_at_least_one_epoch"] = tail["unlabeled_tail_sec"] >= EVENT_DURATION_SEC
    return tail


def write_readme(
    out_dir: Path,
    recordings: pd.DataFrame,
    participants: pd.DataFrame,
    candidates: pd.DataFrame,
    label_quality: pd.DataFrame,
) -> None:
    primary = candidates[candidates["transition_type"] == "REM_to_Wake"]
    secondary = candidates[candidates["transition_type"] == "Wake_to_REM"]
    pids_with_primary = primary["pid"].nunique()
    pids_with_secondary = secondary["pid"].nunique()
    recordings_with_primary = int((recordings["rem_to_wake_count"] > 0).sum())
    recordings_with_secondary = int((recordings["wake_to_rem_count"] > 0).sum())
    disconnection_recordings = int((label_quality["stage_hum_disconnection_epochs"] > 0).sum())
    non_30_recordings = int((label_quality["non_30_sec_epoch_count"] > 0).sum())
    missing_label_recordings = int((label_quality["stage_hum_missing_epochs"] > 0).sum())
    tail_min = float(label_quality["unlabeled_tail_sec"].min())
    tail_max = float(label_quality["unlabeled_tail_sec"].max())
    candidate_window_flags = int(candidates["has_psg_disconnection_in_window"].sum())

    text = f"""# BOAS E0 Transition Inventory

**Work period:** 2026-06-29 to 2026-07-05
**Run date:** 2026-07-01
**Project phase:** Block 3 E0 feasibility audit
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Label specification:** `docs/labels/transition_label_spec_v0.1.md`
**Model training performed:** No

## 1. Purpose

This inventory tests whether BOAS contains enough direct REM/Wake transition events and participant-level spread to justify later preprocessing and model work.

The uncertainty being tested is not model performance. The uncertainty is whether the available human PSG labels contain enough usable REM-to-Wake boundaries, with adequate participant grouping, to support the planned REM-to-Wake wearable EEG project.

## 2. Method

- Read all 128 PSG event tables from the local BOAS metadata/event snapshot.
- Use `stage_hum` as the only reference label source.
- Count direct adjacent REM-to-Wake events: `stage_hum[t] = 4` and `stage_hum[t + 1] = 0`.
- Count direct adjacent Wake-to-REM events separately as secondary information.
- Record the nominal boundary as the onset of the second epoch.
- Record a 30-second uncertainty interval as `boundary_onset_sec - 15` to `boundary_onset_sec + 15`.
- Exclude unlabeled EDF tails from transition generation and report the tail duration.
- Record missing labels, non-30-second epochs, PSG disconnection epochs, timing gaps, and PSG/headband sidecar mismatches.

## 3. Main Count Summary

| Item | Value |
|---|---:|
| PSG recordings checked | {len(recordings)} |
| Unique `pid` values checked | {participants['pid'].nunique()} |
| REM-to-Wake candidates | {len(primary)} |
| Wake-to-REM candidates | {len(secondary)} |
| Recordings with at least one REM-to-Wake candidate | {recordings_with_primary} |
| Recordings with at least one Wake-to-REM candidate | {recordings_with_secondary} |
| `pid` values with at least one REM-to-Wake candidate | {pids_with_primary} |
| `pid` values with at least one Wake-to-REM candidate | {pids_with_secondary} |

## 4. Label-Quality Summary

| Item | Value |
|---|---:|
| Recordings with missing `stage_hum` epochs | {missing_label_recordings} |
| Recordings with non-30-second epochs | {non_30_recordings} |
| Recordings with PSG disconnection epochs | {disconnection_recordings} |
| Candidate windows containing PSG disconnection epochs | {candidate_window_flags} |
| Unlabeled tail minimum, seconds | {tail_min:.1f} |
| Unlabeled tail maximum, seconds | {tail_max:.1f} |

## 5. Outputs

| File | Purpose |
|---|---|
| `recording_transition_inventory.tsv` | Transition counts and quality fields by recording |
| `participant_transition_inventory.tsv` | Transition counts grouped by BOAS `pid` |
| `candidate_transition_events.tsv` | Row-level REM/Wake transition candidates and uncertainty intervals |
| `label_quality_summary.tsv` | Label, duration, timing, and sidecar consistency checks |
| `unlabeled_tail_summary.tsv` | Unlabeled recording tail per PSG event table |
| `psg_disconnection_summary.tsv` | PSG disconnection counts and timing by recording |
| `e0_decision_report.md` | Manual review of the inventory against E0 feasibility questions |

## 6. Interpretation Boundary

This is an event inventory and label-quality audit. It is not a model result, not a classifier-performance estimate, and not evidence of clinical utility.

The companion decision report reviews these counts against the E0 feasibility criteria before any model training.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = dataset_root()
    out_dir = output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(root)
    recording_rows = []
    label_quality_rows = []
    disconnection_rows = []
    candidate_rows = []

    for subject_id in range(1, SUBJECT_COUNT + 1):
        recording, label_quality, disconnection, candidates = inspect_recording(
            root, participants, subject_id
        )
        recording_rows.append(recording)
        label_quality_rows.append(label_quality)
        disconnection_rows.append(disconnection)
        candidate_rows.extend(candidates)

    recordings = pd.DataFrame(recording_rows)
    label_quality = pd.DataFrame(label_quality_rows)
    disconnections = pd.DataFrame(disconnection_rows)
    candidates = pd.DataFrame(candidate_rows)

    if not candidates.empty:
        candidates.insert(0, "transition_id", range(1, len(candidates) + 1))

    participant_counts = participant_inventory(recordings)
    tails = unlabeled_tail_summary(label_quality)

    recordings.to_csv(out_dir / "recording_transition_inventory.tsv", sep="\t", index=False)
    participant_counts.to_csv(out_dir / "participant_transition_inventory.tsv", sep="\t", index=False)
    candidates.to_csv(out_dir / "candidate_transition_events.tsv", sep="\t", index=False)
    label_quality.to_csv(out_dir / "label_quality_summary.tsv", sep="\t", index=False)
    tails.to_csv(out_dir / "unlabeled_tail_summary.tsv", sep="\t", index=False)
    disconnections.to_csv(out_dir / "psg_disconnection_summary.tsv", sep="\t", index=False)
    write_readme(out_dir, recordings, participants, candidates, label_quality)

    print("BOAS E0 transition inventory")
    print(f"PSG recordings checked: {len(recordings)}")
    print(f"Unique pid values checked: {participants['pid'].nunique()}")
    print(f"REM-to-Wake candidates: {(candidates['transition_type'] == 'REM_to_Wake').sum()}")
    print(f"Wake-to-REM candidates: {(candidates['transition_type'] == 'Wake_to_REM').sum()}")
    print(f"Recordings with REM-to-Wake: {(recordings['rem_to_wake_count'] > 0).sum()}")
    print(f"Recordings with Wake-to-REM: {(recordings['wake_to_rem_count'] > 0).sum()}")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
