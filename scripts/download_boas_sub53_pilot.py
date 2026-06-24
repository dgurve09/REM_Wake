"""Download the limited BOAS sub-53 pilot files outside Git.

The default data root is a sibling directory of the repository:
    <repo-parent>\\REM_W_data

Override it with:
    $env:REM_W_DATA_ROOT = "E:\\some\\other\\path"

The script downloads only the files needed for the first readability and
alignment check. It verifies expected byte sizes and EDF SHA-256 hashes where
the official git-annex keys provide them.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"


@dataclass(frozen=True)
class PilotFile:
    relative_path: str
    expected_size: int | None = None
    expected_sha256: str | None = None

    @property
    def url(self) -> str:
        return f"{S3_BASE}/{self.relative_path}"


FILES = [
    PilotFile("dataset_description.json"),
    PilotFile("participants.tsv"),
    PilotFile("participants.json"),
    PilotFile("README"),
    PilotFile("CHANGES"),
    PilotFile("sub-53/sub-53_scans.tsv"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-headband_channels.tsv"),
    PilotFile(
        "sub-53/eeg/sub-53_task-Sleep_acq-headband_eeg.edf",
        expected_size=92_199_424,
        expected_sha256="f7c2756bb5d1563d38fd1859f8c057b3d657d5a4485135c360b4b9e88f7822f0",
    ),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-headband_eeg.json"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-headband_events.json"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-headband_events.tsv"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-psg_channels.tsv"),
    PilotFile(
        "sub-53/eeg/sub-53_task-Sleep_acq-psg_eeg.edf",
        expected_size=143_421_184,
        expected_sha256="f17ea0541b6f94f8decef502c2b660ae756a630553dc23d2016f4be840de9705",
    ),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-psg_eeg.json"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-psg_events.json"),
    PilotFile("sub-53/eeg/sub-53_task-Sleep_acq-psg_events.tsv"),
]


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


def verify(path: Path, item: PilotFile) -> None:
    size = path.stat().st_size
    if item.expected_size is not None and size != item.expected_size:
        raise RuntimeError(
            f"Size mismatch for {item.relative_path}: expected "
            f"{item.expected_size}, found {size}"
        )

    if item.expected_sha256 is not None:
        actual = sha256(path)
        if actual != item.expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {item.relative_path}: expected "
                f"{item.expected_sha256}, found {actual}"
            )


def download(item: PilotFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")

    print(f"Downloading {item.relative_path}")
    with urllib.request.urlopen(item.url) as response, partial.open("wb") as file:
        shutil.copyfileobj(response, file)

    partial.replace(target)
    verify(target, item)


def main() -> None:
    root = dataset_root()
    print(f"Dataset root: {root}")
    print(f"Dataset: {DATASET}, snapshot: {SNAPSHOT}")
    print()

    for item in FILES:
        target = root / item.relative_path
        if target.exists():
            verify(target, item)
            print(f"Verified existing {item.relative_path}")
            continue
        download(item, target)

    print()
    print("Pilot acquisition complete.")
    print("Raw data remain outside the Git repository.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
