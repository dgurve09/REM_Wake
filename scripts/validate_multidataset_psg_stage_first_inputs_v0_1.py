"""Independently validate the stage-first pre-fit checkpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


def main() -> None:
    saved_counts = pd.read_csv(common.output_dir() / "stage_class_counts_v0.1.tsv", sep="\t")
    saved_summary = pd.read_csv(common.output_dir() / "recording_stage_input_summary_v0.1.tsv", sep="\t")
    saved_support = pd.read_csv(common.output_dir() / "confirm2_event_support_v0.1.tsv", sep="\t")
    saved_audit = pd.read_csv(common.output_dir() / "pre_fit_input_audit_v0.1.tsv", sep="\t")
    recordings = common.load_recordings()

    rows = []
    for dataset in common.direct.DATASETS:
        for stage in common.STAGES:
            total = 0
            for item in recordings.values():
                if item["dataset"] != dataset:
                    continue
                times = item["times"].astype(float)
                indices = np.ceil(times / 30.0).astype(int) - 1
                valid = (indices >= 0) & (indices < len(item["stages"]))
                target = np.full(len(times), "UNKNOWN", dtype="U7")
                target[valid] = item["stages"][indices[valid]]
                total += int((target == stage).sum())
            rows.append({"dataset": dataset, "stage": stage, "rows": total})
    reconstructed = pd.DataFrame(rows)
    merged = saved_counts.merge(reconstructed, on=["dataset", "stage"], suffixes=("_saved", "_reconstructed"))

    cache_manifest = pd.read_csv(common.feature_cache_manifest_path(), sep="\t")
    cache_hashes = all(
        (features.data_parent() / item.relative_path).exists()
        and (features.data_parent() / item.relative_path).stat().st_size == int(item.bytes)
        and features.sha256(features.data_parent() / item.relative_path) == item.sha256
        for item in cache_manifest.itertuples(index=False)
    )
    checks = pd.DataFrame(
        [
            {"check": "stage_count_rows", "observed": len(merged), "expected": 15, "passed": len(merged) == 15},
            {"check": "stage_counts_reconstructed", "observed": bool(merged.rows_saved.eq(merged.rows_reconstructed).all()), "expected": True, "passed": bool(merged.rows_saved.eq(merged.rows_reconstructed).all())},
            {"check": "recording_rows", "observed": len(saved_summary), "expected": 196, "passed": len(saved_summary) == 196},
            {"check": "participant_groups", "observed": saved_summary.participant_group.nunique(), "expected": 130, "passed": saved_summary.participant_group.nunique() == 130},
            {"check": "corrected_events", "observed": int(saved_support.primary_events.sum()), "expected": 482, "passed": int(saved_support.primary_events.sum()) == 482},
            {"check": "confirm2_support", "observed": int(saved_support.confirm2_supported_events.sum()), "expected": 482, "passed": int(saved_support.confirm2_supported_events.sum()) == 482},
            {"check": "feature_cache_hashes", "observed": cache_hashes, "expected": True, "passed": cache_hashes},
            {"check": "primary_audit", "observed": int(saved_audit.passed.sum()), "expected": len(saved_audit), "passed": bool(saved_audit.passed.all())},
            {"check": "model_directory_absent", "observed": (common.derived_dir() / "models").exists(), "expected": False, "passed": not (common.derived_dir() / "models").exists()},
            {"check": "score_directory_absent", "observed": (common.derived_dir() / "scores").exists(), "expected": False, "passed": not (common.derived_dir() / "scores").exists()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent stage-first input validation failed")
    verify_or_create_tsv(
        checks,
        common.output_dir() / "independent_input_validation_v0.1.tsv",
    )
    print(f"Independent stage-first input validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
