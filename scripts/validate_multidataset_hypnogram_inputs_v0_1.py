"""Independently validate the reviewed hypnogram input checkpoint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import audit_multidataset_hypnogram_inputs_v0_1 as audit
import hypnogram_transition_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: helpers

def load(name: str) -> pd.DataFrame:
    return pd.read_csv(common.output_dir() / name, sep="\t")


def assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, sort_by: list[str]) -> None:
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    actual = actual[expected.columns]
    if "night" in expected.columns:
        actual["night"] = pd.to_numeric(actual["night"], errors="coerce")
        expected["night"] = pd.to_numeric(expected["night"], errors="coerce")
    actual = actual.where(pd.notna(actual), "")
    expected = expected.where(pd.notna(expected), "")
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


# Section 2: reconstruction and file verification

def main() -> None:
    manifest = load("hypnogram_source_manifest_v0.1.tsv")
    saved_inventory = load("primary_recording_inventory_v0.1.tsv")
    saved_summary = load("input_dataset_summary_v0.1.tsv")
    saved_audit = load("pre_analysis_input_audit_v0.1.tsv")

    file_pass = []
    for item in manifest.itertuples(index=False):
        path = common.project_root() / Path(item.relative_path)
        file_pass.append(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and common.sha256(path) == item.sha256
        )

    records = common.load_primary_records()
    inventory = common.recording_inventory(records)
    summary = audit.dataset_summary(records, inventory)
    assert_frame(inventory, saved_inventory, ["dataset", "recording"])
    assert_frame(summary, saved_summary, ["dataset"])

    source_paths = {
        path
        for value in saved_inventory.source_paths
        for path in str(value).split(";")
    }
    manifest_paths = set(manifest.relative_path)
    metadata_path = common.relative_source(common.hypno_root() / "metadata_all.csv")
    gap_rows = saved_inventory[saved_inventory.inserted_gap_epochs.gt(0)]

    checks = pd.DataFrame(
        [
            {"check": "source_hashes", "observed": int(sum(file_pass)), "expected": len(manifest), "passed": all(file_pass)},
            {"check": "manifest_rows", "observed": len(manifest), "expected": 2556, "passed": len(manifest) == 2556},
            {"check": "primary_inventory_rows", "observed": len(inventory), "expected": 1993, "passed": len(inventory) == 1993},
            {"check": "participant_groups", "observed": inventory.participant_group.nunique(), "expected": 1822, "passed": inventory.participant_group.nunique() == 1822},
            {"check": "dataset_summary_rows", "observed": len(summary), "expected": 11, "passed": len(summary) == 11},
            {"check": "every_inventory_source_manifested", "observed": source_paths.issubset(manifest_paths), "expected": True, "passed": source_paths.issubset(manifest_paths)},
            {"check": "metadata_manifested", "observed": metadata_path in manifest_paths, "expected": True, "passed": metadata_path in manifest_paths},
            {"check": "gap_regularized_records", "observed": len(gap_rows), "expected": 2, "passed": len(gap_rows) == 2},
            {"check": "gap_regularized_epochs", "observed": int(gap_rows.inserted_gap_epochs.sum()), "expected": 66, "passed": int(gap_rows.inserted_gap_epochs.sum()) == 66},
            {"check": "primary_audit_checks", "observed": int(saved_audit.passed.astype(bool).sum()), "expected": len(saved_audit), "passed": saved_audit.passed.astype(bool).all()},
            {"check": "analysis_results_present", "observed": audit.result_outputs_present(), "expected": False, "passed": not audit.result_outputs_present()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent hypnogram input validation failed")
    verify_or_create_tsv(
        checks,
        common.output_dir() / "independent_input_validation_v0.1.tsv",
    )
    print(f"Independent input validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(manifest)} source hashes.")


if __name__ == "__main__":
    main()
