"""Independently validate stored Block 7 train-only feature-gate outputs."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: fixed validation configuration

EXPERIMENT_DIR = "2026-09-06_block7_feature_generation_validation_v0.1"
DERIVED_DIR = "block7_feature_generation_validation_v0.1"
EXPECTED_RECORDINGS = 82
EXPECTED_PIDS = 64
EXPECTED_ARTIFACTS = 246
EXPECTED_MODALITIES = {"PSG-6": 30, "PSG-2": 10, "HB-2": 10}
EXPECTED_CHANNELS = {
    "PSG-6": ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"],
    "PSG-2": ["PSG_F3", "PSG_F4"],
    "HB-2": ["HB_1", "HB_2"],
}
EXPECTED_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
PSG_PARITY_TOLERANCE = 1e-10
HB_REPRODUCTION_TOLERANCE = 1e-6
EPOCH_SEC = 30.0
CONTEXT_LENGTH = 8


# Section 2: paths and frozen train membership

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def reference_hb_path(subject: str) -> Path:
    return (
        data_parent()
        / "derived"
        / "stage_first_feature_baseline_v0.1"
        / "recording_features"
        / f"{subject}_features_v0.1.npz"
    )


def train_assignment() -> pd.DataFrame:
    source = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    rows = []
    for item in source[source["partition"].eq("train")].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": "train"})
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_RECORDINGS or result["pid"].nunique() != EXPECTED_PIDS:
        raise ValueError("Frozen train membership differs from the expected 82 recordings/64 groups")
    return result.sort_values("subject").reset_index(drop=True)


# Section 3: small independent helpers

def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_onsets(onsets: np.ndarray) -> np.ndarray:
    centers = []
    for start in range(0, len(onsets) - CONTEXT_LENGTH + 1):
        local = onsets[start : start + CONTEXT_LENGTH]
        if np.allclose(np.diff(local), EPOCH_SEC, atol=1e-9, rtol=0.0):
            centers.append(float(local[4]))
    return np.asarray(centers, dtype=np.float64)


def read_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        required = {"onset", "stage", "features", "feature_names"}
        if set(values.files) != required:
            raise ValueError(f"Unexpected arrays in {path}")
        return {name: values[name].copy() for name in required}


def record_check(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append(
        {
            "check": name,
            "status": "pass" if passed else "fail",
            "detail": detail,
        }
    )


def verify_or_create_text(path: Path, expected: str) -> None:
    normalized = expected.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != normalized:
            raise RuntimeError(f"Reviewed validation output changed: {path}")
        return
    path.write_text(normalized, encoding="utf-8")


# Section 4: table-level validation

def table_checks(rows: list[dict], train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = output_dir()
    signal = pd.read_csv(out / "train_recording_signal_checks_v0.1.tsv", sep="\t")
    scalers = pd.read_csv(out / "train_robust_scalers_v0.1.tsv", sep="\t")
    synthetic = pd.read_csv(out / "synthetic_spectral_checks_v0.1.tsv", sep="\t")
    recording = pd.read_csv(out / "recording_feature_validation_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(out / "external_feature_manifest_v0.1.tsv", sep="\t")
    gate = pd.read_csv(out / "feature_gate_checks_v0.1.tsv", sep="\t")

    expected_subjects = set(train["subject"])
    expected_pairs = set(zip(train["subject"], train["pid"]))
    recording_pairs = set(zip(recording["subject"], recording["pid"]))
    manifest_pairs = set(zip(manifest["subject"], manifest["pid"]))

    record_check(
        rows,
        "train_membership_exact",
        len(recording) == EXPECTED_RECORDINGS
        and recording_pairs == expected_pairs
        and manifest_pairs == expected_pairs
        and set(signal["subject"]) == expected_subjects,
        "82 recordings and 64 pid groups reconstructed from frozen split",
    )
    record_check(
        rows,
        "partitions_train_only",
        signal["partition"].eq("train").all()
        and recording["partition"].eq("train").all()
        and manifest["partition"].eq("train").all(),
        "no validation or test partition row in retained outputs",
    )
    record_check(
        rows,
        "signal_checks_complete",
        len(signal) == 164
        and signal.groupby("scaler_owner")["subject"].nunique().to_dict()
        == {"HB-2": 82, "PSG-6": 82}
        and signal["recording_check"].eq("pass").all(),
        "two train acquisition paths per recording",
    )
    record_check(
        rows,
        "scaler_checks_complete",
        len(scalers) == 8
        and scalers["fit_partition"].eq("train").all()
        and np.isfinite(scalers["robust_scale_uv"]).all()
        and scalers["robust_scale_uv"].gt(0).all()
        and truth(scalers["reference_check_pass"]).all(),
        "six PSG and two wearable train-only scalers",
    )
    record_check(
        rows,
        "synthetic_checks_complete",
        len(synthetic) == 5 and truth(synthetic["check_pass"]).all(),
        "five predeclared band checks",
    )
    record_check(
        rows,
        "recording_table_passed",
        len(recording) == EXPECTED_RECORDINGS
        and recording["recording_check"].eq("pass").all()
        and truth(recording["all_features_finite"]).all(),
        "all per-recording retained checks passed",
    )
    record_check(
        rows,
        "manifest_cardinality",
        len(manifest) == EXPECTED_ARTIFACTS
        and manifest.groupby("subject")["modality"].nunique().eq(3).all()
        and set(manifest["modality"]) == set(EXPECTED_MODALITIES),
        "three feature artifacts per train recording",
    )
    record_check(
        rows,
        "original_gate_table_passed",
        len(gate) == 13 and gate["status"].eq("pass").all(),
        "13/13 result-producing gate checks retained",
    )
    return recording, manifest


# Section 5: external-array validation

def external_array_checks(
    rows: list[dict],
    recording: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    root = data_parent().resolve()
    file_hash_pass = True
    schema_pass = True
    onset_stage_pass = True
    psg_parity_pass = True
    hb_reproduction_pass = True
    context_pass = True
    table_hash_pass = True
    psg_maximum = 0.0
    hb_maximum = 0.0

    recording_index = recording.set_index("subject")
    for subject, group in manifest.groupby("subject"):
        features = {}
        hashes = {}
        for item in group.itertuples(index=False):
            path = (root / item.relative_external_path).resolve()
            if not path.is_relative_to(root) or not path.exists():
                file_hash_pass = False
                continue
            digest = sha256(path)
            hashes[item.modality] = digest
            file_hash_pass &= path.stat().st_size == int(item.bytes) and digest == item.sha256
            values = read_feature(path)
            features[item.modality] = values
            expected_names = np.asarray(
                [
                    f"{channel}_{band}_log10_mean_psd"
                    for channel in EXPECTED_CHANNELS[item.modality]
                    for band in EXPECTED_BANDS
                ]
            )
            schema_pass &= (
                values["features"].ndim == 2
                and values["features"].shape[1] == EXPECTED_MODALITIES[item.modality]
                and len(values["onset"]) == len(values["stage"]) == len(values["features"])
                and np.isfinite(values["features"]).all()
                and np.array_equal(values["feature_names"], expected_names)
            )

        if set(features) != set(EXPECTED_MODALITIES):
            onset_stage_pass = False
            continue
        psg6 = features["PSG-6"]
        psg2 = features["PSG-2"]
        hb2 = features["HB-2"]
        onset_stage_pass &= (
            np.array_equal(psg6["onset"], psg2["onset"])
            and np.array_equal(psg6["onset"], hb2["onset"])
            and np.array_equal(psg6["stage"], psg2["stage"])
            and np.array_equal(psg6["stage"], hb2["stage"])
        )
        psg_difference = float(
            np.max(np.abs(psg6["features"][:, :10] - psg2["features"]))
        )
        psg_maximum = max(psg_maximum, psg_difference)
        psg_parity_pass &= psg_difference <= PSG_PARITY_TOLERANCE

        reference = read_feature(reference_hb_path(subject))
        hb_difference = float(np.max(np.abs(hb2["features"] - reference["features"])))
        hb_maximum = max(hb_maximum, hb_difference)
        hb_reproduction_pass &= (
            np.array_equal(hb2["onset"], reference["onset"])
            and np.array_equal(hb2["stage"], reference["stage"])
            and np.array_equal(hb2["feature_names"], reference["feature_names"])
            and hb_difference <= HB_REPRODUCTION_TOLERANCE
        )

        contexts = [context_onsets(features[name]["onset"]) for name in EXPECTED_MODALITIES]
        context_pass &= np.array_equal(contexts[0], contexts[1]) and np.array_equal(
            contexts[0], contexts[2]
        )
        expected_hashes = {
            "PSG-6": recording_index.loc[subject, "psg6_sha256"],
            "PSG-2": recording_index.loc[subject, "psg2_sha256"],
            "HB-2": recording_index.loc[subject, "hb2_sha256"],
        }
        table_hash_pass &= hashes == expected_hashes

    record_check(rows, "external_files_rehashed", file_hash_pass, f"{EXPECTED_ARTIFACTS} size and SHA-256 checks")
    record_check(rows, "external_array_schema", schema_pass, "expected arrays, ordered names, and 30/10/10 dimensions")
    record_check(rows, "external_onset_stage_parity", onset_stage_pass, "onsets and stages agree across three modalities")
    record_check(rows, "external_psg_overlap_parity", psg_parity_pass, f"maximum absolute difference {psg_maximum:.12g}")
    record_check(rows, "external_hb_reproduction", hb_reproduction_pass, f"maximum absolute difference {hb_maximum:.12g}")
    record_check(rows, "external_context_parity", context_pass, "eight-epoch context centers agree")
    record_check(rows, "recording_manifest_hash_linkage", table_hash_pass, "recording table and external manifest hashes agree")


# Section 6: execute independent validation

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator-code-commit", required=True)
    args = parser.parse_args()
    if len(args.validator_code_commit) < 7:
        raise ValueError("A valid committed validator code hash is required")

    rows = []
    train = train_assignment()
    recording, manifest = table_checks(rows, train)
    external_array_checks(rows, recording, manifest)
    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")

    passed = int(checks["status"].eq("pass").sum())
    note = f"""# Block 7 Feature Output Validation v0.1

**Work date:** 2026-09-06
**Validator code commit:** `{args.validator_code_commit}`
**Scope:** Stored train-only tables and 246 external feature artifacts
**Raw EDF access:** No
**Model training performed:** No
**Validation or test data accessed:** No

The independent validator passed **{passed}/{len(checks)}** checks. It reconstructed train membership from the frozen split, rejected any non-train output row, rehashed every external feature file, reopened every array, and recomputed PSG overlap, wearable reproduction, and context parity.
"""
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", note)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 7 feature-output integrity check failed")
    print(f"Passed {passed}/{len(checks)} independent output-integrity checks")


if __name__ == "__main__":
    main()
