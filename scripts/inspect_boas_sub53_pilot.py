"""Inspect the BOAS sub-53 pilot files without training a model.

This script checks EDF readability, header-level timing, event tables, and
stage-derived transition candidates for the limited pilot record.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mne
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
STAGE_NAMES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
    8: "PSG disconnection",
}


def default_data_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sidecar_row(path: Path) -> dict:
    sidecar = read_json(path)
    fields = [
        "EEGChannelCount",
        "EEGReference",
        "Manufacturer",
        "ManufacturersModelName",
        "RecordingDuration",
        "SamplingFrequency",
        "TaskName",
    ]
    row = {"file": path.name}
    for field in fields:
        row[field] = sidecar.get(field, "")
    return row


def edf_summary(path: Path) -> dict:
    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "channels": len(raw.ch_names),
        "channel_names": ";".join(raw.ch_names),
        "sfreq_hz": raw.info["sfreq"],
        "samples": raw.n_times,
        "duration_sec": raw.n_times / raw.info["sfreq"],
        "meas_date": str(raw.info.get("meas_date")),
    }


def event_summary(path: Path) -> tuple[pd.DataFrame, dict]:
    events = pd.read_csv(path, sep="\t")
    if "stage_hum" in events.columns:
        label_column = "stage_hum"
    elif "stage_ai" in events.columns:
        label_column = "stage_ai"
    else:
        label_column = ""

    if label_column:
        stage_counts = events[label_column].value_counts(dropna=False).sort_index()
        stage_count_text = "; ".join(
            f"{int(stage)}={int(count)}" for stage, count in stage_counts.items()
        )
    else:
        stage_count_text = "no stage column"

    summary = {
        "file": path.name,
        "rows": len(events),
        "columns": ";".join(events.columns),
        "stage_column_used": label_column or "none",
        "onset_min": events["onset"].min(),
        "onset_max": events["onset"].max(),
        "coverage_end_sec": events["onset"].max() + events["duration"].iloc[-1],
        "duration_values": ";".join(str(x) for x in sorted(events["duration"].unique())),
        "stage_counts": stage_count_text,
    }
    return events, summary


def transition_candidates(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ordered = events.sort_values("onset").reset_index(drop=True)

    for idx in range(len(ordered) - 1):
        current = ordered.loc[idx]
        following = ordered.loc[idx + 1]
        from_stage = int(current["stage_hum"])
        to_stage = int(following["stage_hum"])
        if (from_stage, to_stage) not in [(4, 0), (0, 4)]:
            continue

        boundary = float(following["onset"])
        rows.append(
            {
                "recording": "sub-53",
                "source_label": "stage_hum",
                "transition": f"{STAGE_NAMES[from_stage]}_to_{STAGE_NAMES[to_stage]}",
                "from_stage_code": from_stage,
                "from_stage": STAGE_NAMES[from_stage],
                "to_stage_code": to_stage,
                "to_stage": STAGE_NAMES[to_stage],
                "from_epoch_onset_sec": float(current["onset"]),
                "boundary_onset_sec": boundary,
                "to_epoch_onset_sec": float(following["onset"]),
                "epoch_duration_sec": float(following["duration"]),
                "uncertainty_start_sec": boundary - 15.0,
                "uncertainty_end_sec": boundary + 15.0,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    root = dataset_root()
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "experiments" / "2026-06-24_boas_sub53_pilot"
    out_dir.mkdir(parents=True, exist_ok=True)

    headband_edf = root / "sub-53/eeg/sub-53_task-Sleep_acq-headband_eeg.edf"
    psg_edf = root / "sub-53/eeg/sub-53_task-Sleep_acq-psg_eeg.edf"
    headband_json = root / "sub-53/eeg/sub-53_task-Sleep_acq-headband_eeg.json"
    psg_json = root / "sub-53/eeg/sub-53_task-Sleep_acq-psg_eeg.json"
    psg_events = root / "sub-53/eeg/sub-53_task-Sleep_acq-psg_events.tsv"
    headband_events = root / "sub-53/eeg/sub-53_task-Sleep_acq-headband_events.tsv"

    edf_rows = [edf_summary(headband_edf), edf_summary(psg_edf)]
    edf_table = pd.DataFrame(edf_rows)
    edf_table.to_csv(out_dir / "edf_header_summary.tsv", sep="\t", index=False)

    psg_event_table, psg_event_info = event_summary(psg_events)
    _, headband_event_info = event_summary(headband_events)
    event_table = pd.DataFrame([headband_event_info, psg_event_info])
    event_table.to_csv(out_dir / "event_table_summary.tsv", sep="\t", index=False)

    candidates = transition_candidates(psg_event_table)
    candidates.to_csv(out_dir / "sub53_stage_hum_transition_candidates.tsv", sep="\t", index=False)

    sidecars = pd.DataFrame([sidecar_row(headband_json), sidecar_row(psg_json)])
    sidecars.to_csv(out_dir / "sidecar_summary.tsv", sep="\t", index=False)

    print("EDF header summary")
    print(edf_table[["file", "bytes", "channels", "sfreq_hz", "samples", "duration_sec", "meas_date"]])
    print()
    print("Event table summary")
    print(event_table)
    print()
    print("Transition candidates")
    print(candidates)
    print()
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
