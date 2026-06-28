"""Download BOAS metadata and event files needed for the E0 audit.

This script intentionally does not download EDF signal files. It prepares the
small BIDS metadata and event tables needed to run the feasibility inventory on
or after the scheduled E0 start date.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"
SUBJECT_COUNT = 128


@dataclass(frozen=True)
class MetadataFile:
    relative_path: str
    role: str

    @property
    def url(self) -> str:
        return f"{S3_BASE}/{self.relative_path}"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def required_files() -> list[MetadataFile]:
    files = [
        MetadataFile("dataset_description.json", "root_metadata"),
        MetadataFile("participants.tsv", "root_metadata"),
        MetadataFile("participants.json", "root_metadata"),
        MetadataFile("README", "root_metadata"),
        MetadataFile("CHANGES", "root_metadata"),
    ]

    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        prefix = f"{subject}/eeg/{subject}_task-Sleep"
        files.extend(
            [
                MetadataFile(f"{subject}/{subject}_scans.tsv", "scan_metadata"),
                MetadataFile(f"{prefix}_acq-headband_channels.tsv", "headband_channels"),
                MetadataFile(f"{prefix}_acq-headband_eeg.json", "headband_sidecar"),
                MetadataFile(f"{prefix}_acq-headband_events.json", "headband_events_json"),
                MetadataFile(f"{prefix}_acq-headband_events.tsv", "headband_events_tsv"),
                MetadataFile(f"{prefix}_acq-psg_channels.tsv", "psg_channels"),
                MetadataFile(f"{prefix}_acq-psg_eeg.json", "psg_sidecar"),
                MetadataFile(f"{prefix}_acq-psg_events.json", "psg_events_json"),
                MetadataFile(f"{prefix}_acq-psg_events.tsv", "psg_events_tsv"),
            ]
        )

    return files


def download(item: MetadataFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(item.url) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file)
    partial.replace(target)


def main() -> None:
    root = dataset_root()
    files = required_files()
    downloaded = 0
    verified_existing = 0

    print(f"Dataset root: {root}")
    print(f"Dataset: {DATASET}, snapshot: {SNAPSHOT}")
    print(f"Required metadata/event files: {len(files)}")
    print("EDF files are intentionally excluded.")
    print()

    for index, item in enumerate(files, start=1):
        target = root / item.relative_path
        if target.exists() and target.stat().st_size > 0:
            verified_existing += 1
        else:
            download(item, target)
            downloaded += 1

        if index == 1 or index % 100 == 0 or index == len(files):
            print(
                f"Checked {index}/{len(files)} files "
                f"(downloaded={downloaded}, existing={verified_existing})"
            )

    print()
    print("E0 metadata acquisition complete.")
    print(f"Downloaded files: {downloaded}")
    print(f"Verified existing files: {verified_existing}")
    print("Raw EDF signal files were not downloaded.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
