"""Validate the reviewed multi-dataset PSG pool audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


# Section 1: paths and hashes

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parents[1]


def result_dir() -> Path:
    return repo_root() / "experiments" / "2026-10-25_multidataset_psg_pool_audit_v0.1"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(result_dir() / name, sep="\t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Section 2: independent consistency checks

def main() -> None:
    inventory = load("local_dataset_inventory_v0.1.tsv")
    recordings = load("recording_eligibility_v0.1.tsv")
    summary = load("dataset_summary_v0.1.tsv")
    channels = load("common_channel_mapping_v0.1.tsv")
    manifest = load("source_container_manifest_v0.1.tsv")
    audit = load("audit_checks_v0.1.tsv")

    hashes = []
    for item in manifest.itertuples(index=False):
        path = project_root() / item.relative_path
        hashes.append(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and sha256(path) == item.sha256
        )

    grouped = recordings.groupby("dataset", as_index=False).agg(
        recordings=("recording", "size"),
        participant_groups=("participant_group", "nunique"),
        rem_to_wake_events=("rem_to_wake_events", "sum"),
    )
    stored = summary[summary.dataset.ne("TOTAL")][
        ["dataset", "recordings", "participant_groups", "rem_to_wake_events"]
    ]
    summary_pass = grouped.equals(stored.reset_index(drop=True))
    expected_channels = {"BOAS": 3, "gx36m": 3, "ISRUC-3": 3}
    channel_counts = channels.groupby("dataset").canonical_channel.nunique().to_dict()

    checks = pd.DataFrame(
        [
            {"check": "inventory_sources", "observed": len(inventory), "expected": 11, "passed": len(inventory) == 11},
            {"check": "recording_rows", "observed": len(recordings), "expected": 196, "passed": len(recordings) == 196},
            {"check": "participant_groups", "observed": recordings.participant_group.nunique(), "expected": 130, "passed": recordings.participant_group.nunique() == 130},
            {"check": "event_count", "observed": int(recordings.rem_to_wake_events.sum()), "expected": 521, "passed": int(recordings.rem_to_wake_events.sum()) == 521},
            {"check": "dataset_summary", "observed": len(grouped), "expected": 3, "passed": summary_pass},
            {"check": "common_channel_schema", "observed": channel_counts, "expected": expected_channels, "passed": channel_counts == expected_channels},
            {"check": "channel_and_sampling_controls", "observed": bool(recordings.common_channels_present.all() and recordings.sampling_frequency_pass.all()), "expected": True, "passed": bool(recordings.common_channels_present.all() and recordings.sampling_frequency_pass.all())},
            {"check": "label_support", "observed": bool(recordings.label_within_signal_support.all()), "expected": True, "passed": bool(recordings.label_within_signal_support.all())},
            {"check": "maximum_unscored_tail", "observed": float(recordings.trailing_unscored_sec.max()), "expected": 166.0, "passed": float(recordings.trailing_unscored_sec.max()) == 166.0},
            {"check": "source_hashes", "observed": int(sum(hashes)), "expected": len(hashes), "passed": all(hashes)},
            {"check": "primary_audit", "observed": int(audit.passed.astype(bool).sum()), "expected": len(audit), "passed": audit.passed.astype(bool).all()},
            {"check": "model_outputs_present", "observed": 0, "expected": 0, "passed": not any(result_dir().glob("*model*"))},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Multi-dataset audit validation failed")

    path = result_dir() / "independent_validation_v0.1.tsv"
    value = checks.to_csv(sep="\t", index=False, lineterminator="\n")
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != value:
        raise RuntimeError("Independent validation output changed")
    path.write_text(value, encoding="utf-8")
    print(f"Independent validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(hashes)} external source hashes.")


if __name__ == "__main__":
    main()
