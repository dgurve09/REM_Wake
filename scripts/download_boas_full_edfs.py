"""Download all BOAS EDF signal files outside Git.

The script is resumable and verifies each EDF against the remote byte size
reported by OpenNeuro's S3 object endpoint. It writes only a small acquisition
manifest into the repository; raw EDF files remain outside Git.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
S3_BASE = f"https://s3.amazonaws.com/openneuro.org/{DATASET}"
SUBJECT_COUNT = 128
MIN_FREE_GB_AFTER_DOWNLOAD = 20.0


@dataclass(frozen=True)
class EdfFile:
    subject_id: int
    acquisition: str
    relative_path: str

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


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-07-04_boas_full_edf_acquisition"


def edf_files() -> list[EdfFile]:
    files = []
    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        prefix = f"{subject}/eeg/{subject}_task-Sleep"
        files.append(
            EdfFile(
                subject_id=subject_id,
                acquisition="headband",
                relative_path=f"{prefix}_acq-headband_eeg.edf",
            )
        )
        files.append(
            EdfFile(
                subject_id=subject_id,
                acquisition="psg",
                relative_path=f"{prefix}_acq-psg_eeg.edf",
            )
        )
    return files


def selected_files(subjects: list[int] | None) -> list[EdfFile]:
    files = edf_files()
    if not subjects:
        return files
    subject_set = set(subjects)
    return [item for item in files if item.subject_id in subject_set]


def remote_size(item: EdfFile, timeout: int = 60) -> int:
    request = urllib.request.Request(item.url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def local_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def partial_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".part")


def download_file(item: EdfFile, target: Path, expected_size: int) -> tuple[str, int]:
    target.parent.mkdir(parents=True, exist_ok=True)
    part = partial_path(target)

    if target.exists() and target.stat().st_size == expected_size:
        return "verified_existing", 0

    if target.exists() and target.stat().st_size != expected_size:
        if not part.exists() or part.stat().st_size < target.stat().st_size:
            target.replace(part)
        else:
            target.unlink()

    existing_part_size = local_size(part)
    mode = "ab" if existing_part_size else "wb"
    request = urllib.request.Request(item.url)
    if existing_part_size:
        request.add_header("Range", f"bytes={existing_part_size}-")

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            if existing_part_size and response.status != 206:
                existing_part_size = 0
                mode = "wb"
            with part.open(mode) as file:
                shutil.copyfileobj(response, file, length=1024 * 1024)
    except Exception:
        raise

    final_size = part.stat().st_size
    if final_size != expected_size:
        return "partial", final_size - existing_part_size

    part.replace(target)
    return "downloaded", final_size - existing_part_size


def manifest_row(item: EdfFile, root: Path, expected_size: int, status: str, bytes_changed: int) -> dict:
    target = root / item.relative_path
    part = partial_path(target)
    return {
        "dataset": DATASET,
        "snapshot": SNAPSHOT,
        "subject": f"sub-{item.subject_id}",
        "acquisition": item.acquisition,
        "relative_path": item.relative_path,
        "expected_bytes": expected_size,
        "local_bytes": local_size(target),
        "partial_bytes": local_size(part),
        "bytes_changed_this_run": bytes_changed,
        "status": status,
    }


def write_manifest(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "edf_acquisition_manifest.tsv", sep="\t", index=False)


def write_readme(rows: list[dict], out_dir: Path, manifest_only: bool) -> None:
    df = pd.DataFrame(rows)
    total_expected = int(df["expected_bytes"].sum()) if len(df) else 0
    total_local = int(df["local_bytes"].sum()) if len(df) else 0
    complete_files = int((df["local_bytes"] == df["expected_bytes"]).sum()) if len(df) else 0
    partial_files = int((df["partial_bytes"] > 0).sum()) if len(df) else 0
    missing_files = int((df["local_bytes"] == 0).sum()) if len(df) else 0

    status = "Manifest only" if manifest_only else "Acquisition run"
    text = f"""# BOAS Full EDF Acquisition

**Work date:** 2026-07-04
**Project phase:** Block 3 / early Block 4 signal-alignment preparation
**Dataset:** BOAS, OpenNeuro `ds005555`, snapshot `1.1.1`
**Raw EDF storage:** outside Git
**Status:** {status}
**Model training performed:** No

## 1. Purpose

This acquisition prepares the full BOAS PSG/headband EDF set for representative and full-dataset PSG-to-headband signal-alignment validation.

The technical uncertainty being prepared for is whether PSG-derived `stage_hum` transition labels remain temporally valid when mapped to wearable headband EEG across BOAS recordings, beyond the already completed `sub-53` pilot.

## 2. Acquisition Summary

| Item | Value |
|---|---:|
| EDF files in scope | {len(df)} |
| Files complete locally | {complete_files} |
| Files with partial download data | {partial_files} |
| Files with no complete local EDF | {missing_files} |
| Expected total EDF bytes | {total_expected:,} |
| Complete local EDF bytes | {total_local:,} |

## 3. Verification Method

Each EDF is checked against the remote object byte size reported by the OpenNeuro S3 endpoint. The script is resumable: incomplete downloads are retained as `.part` files outside Git and continued on the next run.

## 4. Outputs

| File | Purpose |
|---|---|
| `edf_acquisition_manifest.tsv` | Per-EDF expected size, local size, partial size, and acquisition status |

## 5. Boundary

This step acquires raw signal files only. It does not train a model and does not by itself validate signal alignment. Alignment validation still requires running signal-level checks after the EDF files are available.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def parse_subjects(subject_text: str | None) -> list[int] | None:
    if not subject_text:
        return None
    subjects = []
    for part in subject_text.split(","):
        value = part.strip().replace("sub-", "")
        if value:
            subjects.append(int(value))
    return subjects


def check_free_space(root: Path, remaining_bytes: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    required_free = remaining_bytes + int(MIN_FREE_GB_AFTER_DOWNLOAD * 1024**3)
    if usage.free < required_free:
        raise RuntimeError(
            f"Insufficient free space. Need remaining download bytes plus "
            f"{MIN_FREE_GB_AFTER_DOWNLOAD:.1f} GB buffer; free={usage.free:,}, "
            f"required={required_free:,}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true", help="Check remote sizes without downloading")
    parser.add_argument("--subjects", help="Comma-separated subject IDs, for example: 1,2,53")
    args = parser.parse_args()

    root = dataset_root()
    out_dir = output_dir()
    subjects = parse_subjects(args.subjects)
    files = selected_files(subjects)

    print(f"Dataset root: {root}")
    print(f"Dataset: {DATASET}, snapshot: {SNAPSHOT}")
    print(f"EDF files in scope: {len(files)}")
    print(f"Manifest only: {args.manifest_only}")
    print()

    rows = []
    expected_sizes: dict[str, int] = {}
    for index, item in enumerate(files, start=1):
        expected_sizes[item.relative_path] = remote_size(item)
        if index == 1 or index % 25 == 0 or index == len(files):
            print(f"Remote sizes checked: {index}/{len(files)}")

    remaining_bytes = 0
    for item in files:
        target = root / item.relative_path
        expected = expected_sizes[item.relative_path]
        if not target.exists() or target.stat().st_size != expected:
            remaining_bytes += max(expected - local_size(partial_path(target)), 0)

    print(f"Expected total EDF bytes: {sum(expected_sizes.values()):,}")
    print(f"Remaining bytes to download: {remaining_bytes:,}")
    check_free_space(root, remaining_bytes)

    for index, item in enumerate(files, start=1):
        expected = expected_sizes[item.relative_path]
        target = root / item.relative_path
        if args.manifest_only:
            status = "complete" if target.exists() and target.stat().st_size == expected else "not_complete"
            bytes_changed = 0
        else:
            start = time.time()
            try:
                status, bytes_changed = download_file(item, target, expected)
            except urllib.error.HTTPError as exc:
                status = f"http_error_{exc.code}"
                bytes_changed = 0
            except Exception as exc:
                status = f"error_{type(exc).__name__}"
                bytes_changed = 0
            elapsed = max(time.time() - start, 1e-6)
            mb_per_sec = bytes_changed / 1024 / 1024 / elapsed
            print(
                f"{index}/{len(files)} {item.relative_path} "
                f"{status} changed={bytes_changed:,} bytes speed={mb_per_sec:.2f} MiB/s"
            )

        rows.append(manifest_row(item, root, expected, status, bytes_changed))
        write_manifest(rows, out_dir)
        write_readme(rows, out_dir, args.manifest_only)

    print()
    print("BOAS EDF acquisition step complete.")
    print(f"Manifest: {out_dir / 'edf_acquisition_manifest.tsv'}")
    print("Raw EDF files remain outside Git.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
