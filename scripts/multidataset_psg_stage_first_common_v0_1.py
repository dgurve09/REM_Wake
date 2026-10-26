"""Shared configuration for the multi-dataset PSG stage-first experiment."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_1 as direct
import run_multidataset_psg_external_generalization_v0_2 as correction


# Section 1: fixed configuration

EXPERIMENT_DIR = "2026-10-25_multidataset_psg_stage_first_v0.1"
DERIVED_DIR = "multidataset_psg_stage_first_v0.1"
STAGES = np.asarray(["W", "N1", "N2", "N3", "REM"])
STAGE_TO_INDEX = {stage: index for index, stage in enumerate(STAGES)}
CANDIDATES = ["STAGE-HARD", "STAGE-CONFIRM2"]
PRIMARY_CANDIDATE = "STAGE-CONFIRM2"
PRIMARY_EXTERNAL = ["gx36m", "ISRUC-3"]
SEED = 20261025
BOOTSTRAP_RESAMPLES = 2000


def output_dir() -> Path:
    return features.repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return features.data_parent() / "derived" / DERIVED_DIR


def load_recordings() -> dict[str, dict]:
    correction.configure_original_module()
    return direct.load_recordings()


# Section 2: causal target alignment

def stage_target(item: dict) -> tuple[np.ndarray, np.ndarray]:
    times = item["times"].astype(float)
    epoch_index = np.ceil(times / 30.0).astype(int) - 1
    valid_index = (epoch_index >= 0) & (epoch_index < len(item["stages"]))
    target = np.full(len(times), "UNKNOWN", dtype="U7")
    target[valid_index] = item["stages"][epoch_index[valid_index]]
    known = np.isin(target, STAGES)
    return target, known


def feature_cache_manifest_path() -> Path:
    return (
        features.repo_root()
        / "experiments/2026-10-25_multidataset_psg_external_generalization_v0.1/feature_generation_summary_v0.1.tsv"
    )
