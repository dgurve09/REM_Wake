"""Shared loaders for the frozen multi-dataset hypnogram analysis."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd


# Section 1: fixed paths and vocabulary

KNOWN_STAGES = np.asarray(["W", "N1", "N2", "N3", "REM"])
EXPECTED_PRIMARY_NIGHTS = {
    "boas": 128,
    "cap-sleep": 108,
    "challenge-2018": 994,
    "dod": 80,
    "dreams": 20,
    "gx36m": 58,
    "hmc-sleep-staging": 151,
    "isleeps": 100,
    "isruc": 126,
    "sleep-accel": 31,
    "sleep-edfx": 197,
}
EXPECTED_SCORER_ROWS = {
    "boas": 128,
    "cap-sleep": 108,
    "challenge-2018": 994,
    "dod": 400,
    "dreams": 20,
    "gx36m": 174,
    "hmc-sleep-staging": 151,
    "isleeps": 100,
    "isruc": 252,
    "sleep-accel": 31,
    "sleep-edfx": 197,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parents[1]


def hypno_root() -> Path:
    return project_root() / "Project_hypno"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-10-25_multidataset_hypnogram_transition_uncertainty_v0.1"


def relative_source(path: Path) -> str:
    return path.relative_to(project_root()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Section 2: normalized stage readers and consensus

def read_stage_csv(path: Path) -> tuple[np.ndarray, bool, int, int]:
    frame = pd.read_csv(path)
    epochs = frame.epoch.to_numpy(int)
    onset = frame.onset_sec.to_numpy(float)
    duration = frame.duration_sec.to_numpy(float)
    origin = float(onset[0]) if len(onset) else 0.0
    offsets = np.rint((onset - origin) / 30.0).astype(int)
    grid_pass = (
        np.array_equal(epochs, np.arange(len(frame)))
        and np.allclose(duration, 30.0, rtol=0, atol=1e-6)
        and np.allclose(onset, origin + offsets * 30.0, rtol=0, atol=1e-6)
        and (len(offsets) == 0 or (offsets[0] == 0 and np.all(np.diff(offsets) > 0)))
    )
    values = frame.stage.fillna("UNKNOWN").astype(str).to_numpy()
    values = np.where(np.isin(values, KNOWN_STAGES), values, "UNKNOWN")
    if len(offsets):
        regular = np.full(int(offsets[-1]) + 1, "UNKNOWN", dtype="U7")
        regular[offsets] = values
    else:
        regular = np.asarray([], dtype="U7")
    return regular, bool(grid_pass), len(frame), len(regular) - len(frame)


def vote_consensus(scorers: dict[str, np.ndarray], minimum_votes: int) -> np.ndarray:
    length = min(len(value) for value in scorers.values())
    values = np.vstack([value[:length] for value in scorers.values()])
    counts = np.stack([(values == stage).sum(axis=0) for stage in KNOWN_STAGES])
    winner = counts.argmax(axis=0)
    return np.where(counts.max(axis=0) >= minimum_votes, KNOWN_STAGES[winner], "UNKNOWN")


def agreement_consensus(scorers: dict[str, np.ndarray]) -> np.ndarray:
    length = min(len(value) for value in scorers.values())
    values = np.vstack([value[:length] for value in scorers.values()])
    return np.where(np.all(values == values[0], axis=0), values[0], "UNKNOWN")


# Section 3: participant and recording identifiers

def sleep_edf_participant(subject: str) -> str:
    return f"sleep-edfx-{subject[:-1]}"


def isruc_recording(subject: str) -> str:
    return re.sub(r"_[12]$", "", subject)


def isruc_participant(recording: str) -> str:
    parts = recording.split("__")
    subgroup = parts[0].replace("ISURC_", "")
    return f"isruc-{subgroup}-{parts[1]}"


def source_csv(dataset: str, subject: str) -> Path:
    return hypno_root() / "data" / dataset / "csv_hypno" / f"{subject}.csv"


# Section 4: one primary record per night

def single_scorer_records(metadata: pd.DataFrame) -> list[dict]:
    excluded = {"dod", "isruc"}
    participants = pd.read_csv(hypno_root() / "data" / "boas" / "participants.tsv", sep="\t")
    boas_pid = dict(zip(participants.participant_id, participants.pid))
    rows = []
    for item in metadata[~metadata.dataset.isin(excluded)].itertuples(index=False):
        path = source_csv(item.dataset, item.subject)
        stages, grid_pass, source_rows, inserted = read_stage_csv(path)
        if item.dataset == "boas":
            participant = f"boas-pid-{int(boas_pid[item.subject])}"
            rule = "publisher_human_consensus"
        elif item.dataset == "sleep-edfx":
            participant = sleep_edf_participant(item.subject)
            rule = "provided_human_hypnogram"
        else:
            participant = f"{item.dataset}-{item.subject}"
            rule = "provided_human_hypnogram"
        rows.append(
            {
                "dataset": item.dataset,
                "recording": f"{item.dataset}-{item.subject}",
                "participant_group": participant,
                "cohort_group": str(item.group),
                "night": "" if pd.isna(item.night) else str(item.night),
                "label_rule": rule,
                "stages": stages,
                "scorers": {"provided": stages.copy()},
                "source_paths": [path],
                "grid_pass": grid_pass,
                "source_length_pass": source_rows == int(item.n_epochs),
                "inserted_gap_epochs": inserted,
                "metadata_epochs": int(item.n_epochs),
            }
        )
    return rows


def dod_records(metadata: pd.DataFrame) -> list[dict]:
    rows = []
    local = metadata[metadata.dataset.eq("dod")]
    for record, group in local.groupby("record", sort=True):
        scorers, paths, grids, lengths, inserted_gaps = {}, [], [], [], []
        for item in group.sort_values("scorer").itertuples(index=False):
            path = source_csv("dod", item.subject)
            stages, grid_pass, source_rows, inserted = read_stage_csv(path)
            scorers[str(item.scorer)] = stages
            paths.append(path)
            grids.append(grid_pass)
            lengths.append(source_rows == int(item.n_epochs))
            inserted_gaps.append(inserted)
        stages = vote_consensus(scorers, minimum_votes=3)
        rows.append(
            {
                "dataset": "dod",
                "recording": f"dod-{record}",
                "participant_group": f"dod-{record}",
                "cohort_group": str(group.group.iloc[0]),
                "night": "",
                "label_rule": "five_scorer_three_vote_majority",
                "stages": stages,
                "scorers": scorers,
                "source_paths": paths,
                "grid_pass": all(grids),
                "source_length_pass": all(lengths),
                "inserted_gap_epochs": max(inserted_gaps),
                "metadata_epochs": len(stages),
            }
        )
    return rows


def isruc_records(metadata: pd.DataFrame) -> list[dict]:
    local = metadata[metadata.dataset.eq("isruc")].copy()
    local["recording_key"] = local.subject.map(isruc_recording)
    rows = []
    for key, group in local.groupby("recording_key", sort=True):
        scorers, paths, grids, lengths, inserted_gaps = {}, [], [], [], []
        for number, item in enumerate(group.sort_values("subject").itertuples(index=False), start=1):
            path = source_csv("isruc", item.subject)
            stages, grid_pass, source_rows, inserted = read_stage_csv(path)
            scorers[f"expert_{number}"] = stages
            paths.append(path)
            grids.append(grid_pass)
            lengths.append(source_rows == int(item.n_epochs))
            inserted_gaps.append(inserted)
        stages = agreement_consensus(scorers)
        rows.append(
            {
                "dataset": "isruc",
                "recording": f"isruc-{key}",
                "participant_group": isruc_participant(key),
                "cohort_group": str(group.group.iloc[0]),
                "night": key.split("__")[2] if key.startswith("ISURC_2") else "1",
                "label_rule": "two_scorer_unanimous_else_unknown",
                "stages": stages,
                "scorers": scorers,
                "source_paths": paths,
                "grid_pass": all(grids),
                "source_length_pass": all(lengths),
                "inserted_gap_epochs": max(inserted_gaps),
                "metadata_epochs": len(stages),
            }
        )
    return rows


def gx_records() -> list[dict]:
    staging = hypno_root() / "data" / "gx36m" / "staging"
    groups: dict[str, list[Path]] = {}
    for path in sorted(staging.glob("SLP*_Day*_Expert*.csv")):
        key = re.sub(r"_Expert[123]$", "", path.stem)
        groups.setdefault(key, []).append(path)
    rows = []
    names = np.asarray(["W", "N1", "N2", "N3", "REM"])
    for key, paths in sorted(groups.items()):
        scorers = {
            f"expert_{number}": names[np.loadtxt(path, dtype=int, ndmin=1)]
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
                "label_rule": "three_scorer_two_vote_majority",
                "stages": stages.astype("U7"),
                "scorers": {name: value.astype("U7") for name, value in scorers.items()},
                "source_paths": paths,
                "grid_pass": True,
                "source_length_pass": True,
                "inserted_gap_epochs": 0,
                "metadata_epochs": len(stages),
            }
        )
    return rows


def load_primary_records() -> list[dict]:
    metadata = pd.read_csv(hypno_root() / "metadata_all.csv")
    records = single_scorer_records(metadata) + dod_records(metadata) + isruc_records(metadata) + gx_records()
    return sorted(records, key=lambda item: (item["dataset"], item["recording"]))


# Section 5: compact reviewed inventories

def recording_inventory(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": item["dataset"],
                "recording": item["recording"],
                "participant_group": item["participant_group"],
                "cohort_group": item["cohort_group"],
                "night": item["night"],
                "label_rule": item["label_rule"],
                "scorers": len(item["scorers"]),
                "epochs": len(item["stages"]),
                "known_epochs": int(np.isin(item["stages"], KNOWN_STAGES).sum()),
                "unknown_epochs": int((~np.isin(item["stages"], KNOWN_STAGES)).sum()),
                "inserted_gap_epochs": item["inserted_gap_epochs"],
                "grid_pass": item["grid_pass"],
                "source_length_pass": item["source_length_pass"],
                "source_paths": ";".join(relative_source(path) for path in item["source_paths"]),
            }
            for item in records
        ]
    )


def source_manifest() -> pd.DataFrame:
    metadata_path = hypno_root() / "metadata_all.csv"
    paths = [metadata_path]
    for dataset, expected in EXPECTED_SCORER_ROWS.items():
        if dataset == "gx36m":
            paths.extend(sorted((hypno_root() / "data" / dataset / "staging").glob("SLP*_Day*_Expert*.csv")))
        else:
            paths.extend(sorted((hypno_root() / "data" / dataset / "csv_hypno").glob("*.csv")))
    return pd.DataFrame(
        [
            {
                "relative_path": relative_source(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        ]
    )
