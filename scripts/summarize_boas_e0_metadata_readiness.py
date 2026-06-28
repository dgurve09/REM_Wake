"""Summarize BOAS metadata readiness for the scheduled E0 audit.

This script checks downloaded metadata/event files only. It does not count
REM-to-Wake events and does not produce the E0 feasibility decision.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT_COUNT = 128


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def subject_paths(root: Path, subject_id: int) -> dict[str, Path]:
    subject = f"sub-{subject_id}"
    prefix = root / subject / "eeg" / f"{subject}_task-Sleep"
    return {
        "subject": Path(subject),
        "scan_metadata": root / subject / f"{subject}_scans.tsv",
        "headband_channels": Path(f"{prefix}_acq-headband_channels.tsv"),
        "headband_sidecar": Path(f"{prefix}_acq-headband_eeg.json"),
        "headband_events_json": Path(f"{prefix}_acq-headband_events.json"),
        "headband_events_tsv": Path(f"{prefix}_acq-headband_events.tsv"),
        "psg_channels": Path(f"{prefix}_acq-psg_channels.tsv"),
        "psg_sidecar": Path(f"{prefix}_acq-psg_eeg.json"),
        "psg_events_json": Path(f"{prefix}_acq-psg_events.json"),
        "psg_events_tsv": Path(f"{prefix}_acq-psg_events.tsv"),
    }


def file_inventory(root: Path) -> pd.DataFrame:
    rows = []
    root_files = {
        "dataset_description": root / "dataset_description.json",
        "participants_tsv": root / "participants.tsv",
        "participants_json": root / "participants.json",
        "readme": root / "README",
        "changes": root / "CHANGES",
    }

    for role, path in root_files.items():
        rows.append(file_row("root", role, path))

    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        paths = subject_paths(root, subject_id)
        for role, path in paths.items():
            if role == "subject":
                continue
            rows.append(file_row(subject, role, path))

    return pd.DataFrame(rows)


def file_row(subject: str, role: str, path: Path) -> dict:
    exists = path.exists()
    return {
        "subject": subject,
        "role": role,
        "exists": exists,
        "bytes": int(path.stat().st_size) if exists else 0,
        "relative_path": path.relative_to(dataset_root()).as_posix() if exists else "",
    }


def participant_summary(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = pd.read_csv(root / "participants.tsv", sep="\t")
    pid_counts = participants["pid"].value_counts().sort_index().reset_index()
    pid_counts.columns = ["pid", "recording_count"]
    summary = (
        pid_counts["recording_count"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    summary.columns = ["recordings_per_pid", "pid_count"]
    return participants, summary


def event_schema_summary(root: Path) -> pd.DataFrame:
    rows = []

    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        paths = subject_paths(root, subject_id)

        for acquisition in ["headband", "psg"]:
            events_path = paths[f"{acquisition}_events_tsv"]
            sidecar_path = paths[f"{acquisition}_sidecar"]
            channels_path = paths[f"{acquisition}_channels"]

            events = pd.read_csv(events_path, sep="\t")
            sidecar = read_json(sidecar_path)
            channels = pd.read_csv(channels_path, sep="\t")
            recording_duration = float(sidecar["RecordingDuration"])
            coverage_end = float(events["onset"].max() + events["duration"].iloc[-1])
            columns = list(events.columns)

            row = {
                "subject": subject,
                "acquisition": acquisition,
                "rows": int(len(events)),
                "columns": ";".join(columns),
                "has_stage_hum": "stage_hum" in columns,
                "has_stage_ai": "stage_ai" in columns,
                "duration_values": ";".join(
                    str(value) for value in sorted(events["duration"].unique())
                ),
                "onset_min_sec": float(events["onset"].min()),
                "onset_max_sec": float(events["onset"].max()),
                "coverage_end_sec": coverage_end,
                "recording_duration_sec": recording_duration,
                "unlabeled_tail_sec": recording_duration - coverage_end,
                "sampling_frequency_hz": float(sidecar["SamplingFrequency"]),
                "channel_rows": int(len(channels)),
            }

            if "stage_hum" in columns:
                row["stage_hum_disconnection_epochs"] = int((events["stage_hum"] == 8).sum())
                row["stage_hum_missing"] = int(events["stage_hum"].isna().sum())
            else:
                row["stage_hum_disconnection_epochs"] = ""
                row["stage_hum_missing"] = ""

            if "stage_ai" in columns:
                row["stage_ai_negative_epochs"] = int((events["stage_ai"] < 0).sum())
            else:
                row["stage_ai_negative_epochs"] = ""

            rows.append(row)

    return pd.DataFrame(rows)


def write_readme(
    out_dir: Path,
    inventory: pd.DataFrame,
    participants: pd.DataFrame,
    pid_summary: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    missing_files = int((~inventory["exists"]).sum())
    total_metadata_bytes = int(inventory["bytes"].sum())
    psg_events = events[events["acquisition"] == "psg"]
    headband_events = events[events["acquisition"] == "headband"]
    psg_stage_hum_complete = bool(psg_events["has_stage_hum"].all())
    headband_stage_hum_count = int(headband_events["has_stage_hum"].sum())
    duration_values = sorted(set(events["duration_values"]))
    sampling_values = sorted(set(events["sampling_frequency_hz"]))
    tail_min = float(events["unlabeled_tail_sec"].min())
    tail_max = float(events["unlabeled_tail_sec"].max())

    text = f"""# BOAS E0 Metadata Readiness

**Work period:** 2026-06-25 to 2026-06-28
**Finalized:** 2026-06-28
**Project phase:** Block 2 closeout, E0 readiness
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Raw EDF files downloaded:** No
**Full E0 transition inventory performed:** No
**Model training performed:** No

## 1. Purpose

This readiness check prepares the metadata/event inputs required for the scheduled E0 feasibility audit beginning on 2026-06-29. It verifies file availability and schema consistency without counting REM-to-Wake events or making the feasibility decision.

## 2. File Readiness

- Metadata/event files checked: {len(inventory)}
- Missing files: {missing_files}
- Total downloaded metadata/event size: {total_metadata_bytes:,} bytes
- EDF signal files are intentionally excluded from this acquisition.

## 3. Participant Readiness

- Recording rows in `participants.tsv`: {len(participants)}
- Unique `pid` values: {participants['pid'].nunique()}
- Repeated-participant grouping is available through `pid`.

Participant repeat summary:

| Recordings per pid | Number of pid values |
|---:|---:|
"""

    for _, row in pid_summary.iterrows():
        text += f"| {int(row['recordings_per_pid'])} | {int(row['pid_count'])} |\n"

    text += f"""
## 4. Event Schema Readiness

- PSG event files checked: {len(psg_events)}
- Headband event files checked: {len(headband_events)}
- All PSG event files contain `stage_hum`: {psg_stage_hum_complete}
- Headband event files containing `stage_hum`: {headband_stage_hum_count}
- Event duration values observed: {', '.join(duration_values)}
- Sampling frequencies observed in sidecars: {', '.join(str(x) for x in sampling_values)}
- Unlabeled tail range across event files: {tail_min:.1f} to {tail_max:.1f} seconds

Interpretation: the metadata/event inputs are ready for the scheduled E0 event inventory. Human-derived labels should come from PSG `stage_hum`, and headband event files should not be treated as human ground truth.

## 5. Outputs

| File | Purpose |
|---|---|
| `metadata_file_inventory.tsv` | File presence and byte size for E0 metadata inputs |
| `participant_pid_summary.tsv` | Number of recordings per participant identifier |
| `event_schema_summary.tsv` | Per-recording event schema, coverage, and label-column checks |

## 6. Boundary

This check does not count REM-to-Wake events, does not estimate participant-level feasibility, and does not train or evaluate any model. Those decisions remain assigned to the E0 feasibility audit.
"""

    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = dataset_root()
    out_dir = (
        repo_root()
        / "experiments"
        / "2026-06-25_to_2026-06-28_boas_e0_metadata_readiness"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = file_inventory(root)
    participants, pid_summary = participant_summary(root)
    events = event_schema_summary(root)

    inventory.to_csv(out_dir / "metadata_file_inventory.tsv", sep="\t", index=False)
    pid_summary.to_csv(out_dir / "participant_pid_summary.tsv", sep="\t", index=False)
    events.to_csv(out_dir / "event_schema_summary.tsv", sep="\t", index=False)
    write_readme(out_dir, inventory, participants, pid_summary, events)

    print("BOAS E0 metadata readiness")
    print(f"Files checked: {len(inventory)}")
    print(f"Missing files: {(~inventory['exists']).sum()}")
    print(f"Metadata/event bytes: {inventory['bytes'].sum():,}")
    print(f"Participant rows: {len(participants)}")
    print(f"Unique pid values: {participants['pid'].nunique()}")
    print(f"Event rows checked: {len(events)}")
    print(f"All PSG files have stage_hum: {events[events['acquisition'] == 'psg']['has_stage_hum'].all()}")
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
