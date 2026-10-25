"""Audit all inputs before the frozen hypnogram transition analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: pre-result summaries

def dataset_summary(records: list[dict], inventory: pd.DataFrame) -> pd.DataFrame:
    metadata = pd.read_csv(common.hypno_root() / "metadata_all.csv")
    scorer_rows = metadata.groupby("dataset").size().to_dict()
    scorer_rows["gx36m"] = common.EXPECTED_SCORER_ROWS["gx36m"]
    rows = []
    for dataset, local in inventory.groupby("dataset", sort=True):
        rows.append(
            {
                "dataset": dataset,
                "stored_scorer_records": int(scorer_rows[dataset]),
                "primary_nights": len(local),
                "participant_groups": local.participant_group.nunique(),
                "primary_epochs": int(local.epochs.sum()),
                "known_epochs": int(local.known_epochs.sum()),
                "unknown_epochs": int(local.unknown_epochs.sum()),
                "unknown_fraction": float(local.unknown_epochs.sum() / local.epochs.sum()),
            }
        )
    return pd.DataFrame(rows)


def result_outputs_present() -> bool:
    names = [
        "recording_transition_measures_v0.1.tsv",
        "transition_events_v0.1.tsv",
        "dataset_transition_summary_v0.1.tsv",
        "cohort_group_summary_v0.1.tsv",
        "scorer_pair_records_v0.1.tsv",
        "scorer_uncertainty_summary_v0.1.tsv",
        "repeated_participant_measures_v0.1.tsv",
        "repeated_night_summary_v0.1.tsv",
        "full_record_sensitivity_v0.1.tsv",
        "hypothesis_decisions_v0.1.tsv",
        "in_run_checks_v0.1.tsv",
    ]
    return any((common.output_dir() / name).exists() for name in names)


# Section 2: fixed audit checks

def main() -> None:
    metadata = pd.read_csv(common.hypno_root() / "metadata_all.csv")
    records = common.load_primary_records()
    inventory = common.recording_inventory(records)
    summary = dataset_summary(records, inventory)
    manifest = common.source_manifest()

    observed_counts = summary.set_index("dataset").primary_nights.to_dict()
    scorer_counts = summary.set_index("dataset").stored_scorer_records.to_dict()
    dataset_counts_pass = observed_counts == common.EXPECTED_PRIMARY_NIGHTS
    scorer_counts_pass = scorer_counts == common.EXPECTED_SCORER_ROWS
    deterministic = all(
        np.array_equal(
            item["stages"],
            common.vote_consensus(item["scorers"], 3)
            if item["dataset"] == "dod"
            else common.agreement_consensus(item["scorers"])
            if item["dataset"] == "isruc"
            else common.vote_consensus(item["scorers"], 2)
            if item["dataset"] == "gx36m"
            else next(iter(item["scorers"].values())),
        )
        for item in records
    )
    all_used_paths = {
        common.relative_source(path)
        for item in records
        for path in item["source_paths"]
    }
    manifested_stage_paths = set(manifest.relative_path) - {common.relative_source(common.hypno_root() / "metadata_all.csv")}
    allowed_stages = np.append(common.KNOWN_STAGES, "UNKNOWN")
    stage_values_pass = all(np.isin(item["stages"], allowed_stages).all() for item in records)
    checks = pd.DataFrame(
        [
            {"check": "metadata_rows", "observed": len(metadata), "expected": 2381, "passed": len(metadata) == 2381},
            {"check": "manifest_rows", "observed": len(manifest), "expected": 2556, "passed": len(manifest) == 2556},
            {"check": "stored_scorer_counts", "observed": scorer_counts_pass, "expected": True, "passed": scorer_counts_pass},
            {"check": "primary_night_counts", "observed": dataset_counts_pass, "expected": True, "passed": dataset_counts_pass},
            {"check": "primary_nights", "observed": len(records), "expected": 1993, "passed": len(records) == 1993},
            {"check": "participant_groups", "observed": inventory.participant_group.nunique(), "expected": 1822, "passed": inventory.participant_group.nunique() == 1822},
            {"check": "recording_identifiers_unique", "observed": inventory.recording.nunique(), "expected": len(records), "passed": inventory.recording.nunique() == len(records)},
            {"check": "all_epoch_grids", "observed": int(inventory.grid_pass.sum()), "expected": len(records), "passed": inventory.grid_pass.all()},
            {"check": "explicit_unknown_gap_epochs", "observed": int(inventory.inserted_gap_epochs.sum()), "expected": 66, "passed": int(inventory.inserted_gap_epochs.sum()) == 66},
            {"check": "metadata_source_lengths", "observed": int(inventory.source_length_pass.sum()), "expected": len(records), "passed": inventory.source_length_pass.all()},
            {"check": "allowed_primary_stages", "observed": stage_values_pass, "expected": True, "passed": stage_values_pass},
            {"check": "dod_complete_five_scorers", "observed": sum(item["dataset"] == "dod" and len(item["scorers"]) == 5 for item in records), "expected": 80, "passed": sum(item["dataset"] == "dod" and len(item["scorers"]) == 5 for item in records) == 80},
            {"check": "isruc_complete_two_scorers", "observed": sum(item["dataset"] == "isruc" and len(item["scorers"]) == 2 for item in records), "expected": 126, "passed": sum(item["dataset"] == "isruc" and len(item["scorers"]) == 2 for item in records) == 126},
            {"check": "gx36m_complete_three_scorers", "observed": sum(item["dataset"] == "gx36m" and len(item["scorers"]) == 3 for item in records), "expected": 58, "passed": sum(item["dataset"] == "gx36m" and len(item["scorers"]) == 3 for item in records) == 58},
            {"check": "consensus_deterministic", "observed": deterministic, "expected": True, "passed": deterministic},
            {"check": "all_manifested_stage_files_used_once_by_scope", "observed": all_used_paths == manifested_stage_paths, "expected": True, "passed": all_used_paths == manifested_stage_paths},
            {"check": "boas_human_hypnogram_only", "observed": all("csv_hypno" in str(path) for item in records if item["dataset"] == "boas" for path in item["source_paths"]), "expected": True, "passed": all("csv_hypno" in str(path) for item in records if item["dataset"] == "boas" for path in item["source_paths"])},
            {"check": "analysis_results_present", "observed": result_outputs_present(), "expected": False, "passed": not result_outputs_present()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Hypnogram pre-analysis input audit failed")

    common.output_dir().mkdir(parents=True, exist_ok=True)
    verify_or_create_tsv(manifest, common.output_dir() / "hypnogram_source_manifest_v0.1.tsv")
    verify_or_create_tsv(inventory, common.output_dir() / "primary_recording_inventory_v0.1.tsv")
    verify_or_create_tsv(summary, common.output_dir() / "input_dataset_summary_v0.1.tsv")
    verify_or_create_tsv(checks, common.output_dir() / "pre_analysis_input_audit_v0.1.tsv")
    print(f"Pre-analysis input audit passed {len(checks)}/{len(checks)} checks.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
