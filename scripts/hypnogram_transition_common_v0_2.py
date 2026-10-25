"""Shared loaders with safe gx36m numeric-stage parsing."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

import hypnogram_transition_common_v0_1 as previous


# Section 1: unchanged shared configuration

KNOWN_STAGES = previous.KNOWN_STAGES
EXPECTED_PRIMARY_NIGHTS = previous.EXPECTED_PRIMARY_NIGHTS
EXPECTED_SCORER_ROWS = previous.EXPECTED_SCORER_ROWS
repo_root = previous.repo_root
project_root = previous.project_root
hypno_root = previous.hypno_root
relative_source = previous.relative_source
sha256 = previous.sha256
read_stage_csv = previous.read_stage_csv
vote_consensus = previous.vote_consensus
agreement_consensus = previous.agreement_consensus
single_scorer_records = previous.single_scorer_records
dod_records = previous.dod_records
isruc_records = previous.isruc_records


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-10-25_multidataset_hypnogram_transition_uncertainty_v0.2"


# Section 2: corrected gx36m parsing

def read_gx_stage_file(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=int, ndmin=1)
    stages = np.full(len(raw), "UNKNOWN", dtype="U7")
    known = (raw >= 0) & (raw < len(KNOWN_STAGES))
    stages[known] = KNOWN_STAGES[raw[known]]
    return stages


def gx_records() -> list[dict]:
    staging = hypno_root() / "data" / "gx36m" / "staging"
    groups: dict[str, list[Path]] = {}
    for path in sorted(staging.glob("SLP*_Day*_Expert*.csv")):
        key = re.sub(r"_Expert[123]$", "", path.stem)
        groups.setdefault(key, []).append(path)

    rows = []
    for key, paths in sorted(groups.items()):
        scorers = {
            f"expert_{number}": read_gx_stage_file(path)
            for number, path in enumerate(sorted(paths), start=1)
        }
        stages = vote_consensus(scorers, minimum_votes=2)
        participant = key.split("_Day")[0]
        rows.append(
            {
                "dataset": "gx36m",
                "recording": f"gx36m-{key}",
                "participant_group": f"gx36m-{participant}",
                "cohort_group": "healthy",
                "night": key.split("_Day")[-1],
                "label_rule": "three_scorer_two_known_vote_majority",
                "stages": stages.astype("U7"),
                "scorers": scorers,
                "source_paths": paths,
                "grid_pass": True,
                "source_length_pass": True,
                "inserted_gap_epochs": 0,
                "metadata_epochs": len(stages),
            }
        )
    return rows


def load_primary_records() -> list[dict]:
    metadata = previous.pd.read_csv(hypno_root() / "metadata_all.csv")
    records = (
        single_scorer_records(metadata)
        + dod_records(metadata)
        + isruc_records(metadata)
        + gx_records()
    )
    return sorted(records, key=lambda item: (item["dataset"], item["recording"]))
