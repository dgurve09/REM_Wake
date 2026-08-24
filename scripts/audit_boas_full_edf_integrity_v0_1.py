"""Verify all local BOAS EDF files against official git-annex SHA-256 keys."""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from pathlib import Path

import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SUBJECT_COUNT = 128
ANNEX_PATTERN = re.compile(r"SHA256E-s(?P<bytes>\d+)--(?P<sha256>[0-9a-f]{64})\.edf")
RAW_MIRROR = (
    f"https://raw.githubusercontent.com/OpenNeuroDatasets/{DATASET}/{SNAPSHOT}"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    default = repo_root().parent / "REM_W_data"
    return Path(os.environ.get("REM_W_DATA_ROOT", default)) / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-08-23_boas_full_edf_integrity_audit_v0.1"


def expected_files() -> list[tuple[str, str, str]]:
    rows = []
    for subject_id in range(1, SUBJECT_COUNT + 1):
        subject = f"sub-{subject_id}"
        prefix = f"{subject}/eeg/{subject}_task-Sleep"
        rows.append(
            (subject, "headband", f"{prefix}_acq-headband_eeg.edf")
        )
        rows.append((subject, "psg", f"{prefix}_acq-psg_eeg.edf"))
    return rows


def official_annex_key(relative_path: str) -> tuple[int, str, str]:
    url = f"{RAW_MIRROR}/{relative_path}"
    with urllib.request.urlopen(url, timeout=60) as response:
        pointer = response.read().decode("utf-8").strip()
    match = ANNEX_PATTERN.search(pointer)
    if match is None:
        raise ValueError(f"No SHA256E annex key found for {relative_path}")
    return int(match.group("bytes")), match.group("sha256"), url


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite changed reviewed output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def result_readme(result: pd.DataFrame) -> str:
    passed = int((result["status"] == "pass").sum())
    total_bytes = int(result["local_bytes"].sum())
    return f"""# BOAS Full EDF Integrity Audit v0.1

**Work date:** 2026-08-23
**Dataset:** OpenNeuro `{DATASET}`, snapshot `{SNAPSHOT}`
**Protocol:** `docs/data/boas_full_edf_integrity_audit_plan_v0.1.md`
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Expected EDF files | {len(result)} |
| PSG files | {int((result['acquisition'] == 'psg').sum())} |
| Headband files | {int((result['acquisition'] == 'headband').sum())} |
| Files matching official size and SHA-256 | {passed} |
| Local EDF bytes verified | {total_bytes:,} |

Every row is compared with the `SHA256E` key in the official OpenNeuro dataset mirror at tag `{SNAPSHOT}`. The raw EDF files remain outside Git.

## Decision

{'The full local EDF acquisition matches the official annex identities.' if passed == len(result) else 'At least one EDF identity check failed. Downstream raw-signal work must stop until the mismatch is resolved.'}

This audit verifies file identity only. It does not replace signal alignment, signal-quality, label, split, or model-output validation.
"""


def main() -> None:
    root = dataset_root()
    rows = []
    files = expected_files()
    if len({item[2] for item in files}) != 256:
        raise RuntimeError("Expected file list is not unique and complete")

    for index, (subject, acquisition, relative_path) in enumerate(files, start=1):
        expected_bytes, expected_sha256, pointer_url = official_annex_key(relative_path)
        local_path = root / Path(relative_path)
        local_exists = local_path.is_file()
        local_bytes = local_path.stat().st_size if local_exists else 0
        local_sha256 = sha256(local_path) if local_exists else ""
        size_match = local_exists and local_bytes == expected_bytes
        hash_match = local_exists and local_sha256 == expected_sha256
        rows.append(
            {
                "dataset": DATASET,
                "snapshot": SNAPSHOT,
                "subject": subject,
                "acquisition": acquisition,
                "relative_path": relative_path,
                "official_bytes": expected_bytes,
                "local_bytes": local_bytes,
                "official_sha256": expected_sha256,
                "local_sha256": local_sha256,
                "size_match": size_match,
                "sha256_match": hash_match,
                "status": "pass" if size_match and hash_match else "fail",
                "official_pointer_url": pointer_url,
            }
        )
        print(f"{index:3d}/{len(files)} {relative_path}: {rows[-1]['status']}")

    result = pd.DataFrame(rows)
    out_dir = output_dir()
    manifest_text = result.to_csv(sep="\t", index=False, lineterminator="\n")
    write_once(out_dir / "edf_sha256_verification_v0.1.tsv", manifest_text)
    write_once(out_dir / "README.md", result_readme(result))

    failed = result[result["status"] != "pass"]
    if len(failed):
        raise SystemExit(f"EDF integrity failures: {failed['relative_path'].tolist()}")
    print(f"Passed {len(result)}/{len(result)} EDF identity checks")


if __name__ == "__main__":
    main()
