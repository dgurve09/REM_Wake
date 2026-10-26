"""Audit stage-first targets and cached features before fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: target and feature summaries

def stage_counts(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset in common.direct.DATASETS:
        local = [item for item in recordings.values() if item["dataset"] == dataset]
        counts = {stage: 0 for stage in common.STAGES}
        unknown = 0
        decisions = 0
        for item in local:
            target, known = common.stage_target(item)
            decisions += len(target)
            unknown += int((~known).sum())
            for stage in common.STAGES:
                counts[stage] += int((target == stage).sum())
        for stage in common.STAGES:
            rows.append(
                {
                    "dataset": dataset,
                    "stage": stage,
                    "rows": counts[stage],
                    "fraction_of_known": counts[stage] / sum(counts.values()),
                    "dataset_decisions": decisions,
                    "dataset_unknown_rows": unknown,
                }
            )
    return pd.DataFrame(rows)


def recording_summary(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for recording, item in recordings.items():
        target, known = common.stage_target(item)
        times = item["times"].astype(float)
        exact = np.isclose(np.mod(times, 30.0), 0.0)
        midpoint = np.isclose(np.mod(times, 30.0), 15.0)
        rows.append(
            {
                "dataset": item["dataset"],
                "recording": recording,
                "participant_group": item["participant_group"],
                "decisions": len(times),
                "known_stage_rows": int(known.sum()),
                "unknown_stage_rows": int((~known).sum()),
                "exact_boundary_rows": int(exact.sum()),
                "midpoint_rows": int(midpoint.sum()),
                "primary_events": len(item["primary"]),
                "feature_rows": item["ADV3-CHANGE"].shape[0],
                "feature_columns": item["ADV3-CHANGE"].shape[1],
                "finite_features": bool(np.isfinite(item["ADV3-CHANGE"]).all()),
            }
        )
    return pd.DataFrame(rows)


def confirmation_support(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for recording, item in recordings.items():
        time_keys = set(np.rint(item["times"] * 2).astype(int))
        supported = 0
        for reference in item["primary"]:
            required = [reference - 30.0, reference - 15.0, reference, reference + 15.0]
            supported += int(all(int(round(value * 2)) in time_keys for value in required))
        rows.append(
            {
                "dataset": item["dataset"],
                "recording": recording,
                "primary_events": len(item["primary"]),
                "confirm2_supported_events": supported,
            }
        )
    return pd.DataFrame(rows)


def verify_feature_hashes() -> tuple[int, int]:
    manifest = pd.read_csv(common.feature_cache_manifest_path(), sep="\t")
    passed = 0
    for item in manifest.itertuples(index=False):
        path = features.data_parent() / item.relative_path
        passed += int(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and features.sha256(path) == item.sha256
        )
    return passed, len(manifest)


def model_outputs_present() -> bool:
    reviewed = [
        "stage_fit_summary_v0.1.tsv",
        "heldout_stage_metrics_v0.1.tsv",
        "heldout_event_metrics_v0.1.tsv",
        "hypothesis_decision_v0.1.tsv",
    ]
    return any((common.output_dir() / name).exists() for name in reviewed) or any(
        (common.derived_dir() / folder).exists() for folder in ["models", "scores"]
    )


# Section 2: reviewed pre-fit checkpoint

def main() -> None:
    recordings = common.load_recordings()
    counts = stage_counts(recordings)
    summary = recording_summary(recordings)
    support = confirmation_support(recordings)
    hashes_passed, hash_count = verify_feature_hashes()
    every_dataset_all_stages = bool(
        counts.groupby("dataset").stage.nunique().eq(len(common.STAGES)).all()
        and counts.rows.gt(0).all()
    )
    all_training_pools_have_stages = True
    for heldout in common.direct.DATASETS:
        local = counts[~counts.dataset.eq(heldout)].groupby("stage").rows.sum()
        all_training_pools_have_stages &= set(local.index) == set(common.STAGES) and local.gt(0).all()
    target_events = int(summary.primary_events.sum())
    supported_events = int(support.confirm2_supported_events.sum())
    checks = pd.DataFrame(
        [
            {"check": "recordings", "observed": len(summary), "expected": 196, "passed": len(summary) == 196},
            {"check": "participant_groups", "observed": summary.participant_group.nunique(), "expected": 130, "passed": summary.participant_group.nunique() == 130},
            {"check": "feature_cache_hashes", "observed": hashes_passed, "expected": hash_count, "passed": hashes_passed == hash_count == 196},
            {"check": "feature_columns", "observed": int(summary.feature_columns.min()), "expected": 444, "passed": summary.feature_columns.eq(444).all()},
            {"check": "feature_rows_match_decisions", "observed": bool(summary.feature_rows.eq(summary.decisions).all()), "expected": True, "passed": bool(summary.feature_rows.eq(summary.decisions).all())},
            {"check": "finite_features", "observed": bool(summary.finite_features.all()), "expected": True, "passed": bool(summary.finite_features.all())},
            {"check": "stage_rows", "observed": int(counts.rows.sum()), "expected": int(summary.known_stage_rows.sum()), "passed": int(counts.rows.sum()) == int(summary.known_stage_rows.sum())},
            {"check": "every_dataset_has_all_stages", "observed": every_dataset_all_stages, "expected": True, "passed": every_dataset_all_stages},
            {"check": "every_training_pool_has_all_stages", "observed": bool(all_training_pools_have_stages), "expected": True, "passed": bool(all_training_pools_have_stages)},
            {"check": "corrected_primary_events", "observed": target_events, "expected": 482, "passed": target_events == 482},
            {"check": "confirm2_temporal_support", "observed": supported_events, "expected": target_events, "passed": supported_events == target_events},
            {"check": "only_15_second_grid", "observed": bool(summary.exact_boundary_rows.add(summary.midpoint_rows).eq(summary.decisions).all()), "expected": True, "passed": bool(summary.exact_boundary_rows.add(summary.midpoint_rows).eq(summary.decisions).all())},
            {"check": "model_or_result_outputs_present", "observed": model_outputs_present(), "expected": False, "passed": not model_outputs_present()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Stage-first pre-fit audit failed")
    outputs = {
        "stage_class_counts_v0.1.tsv": counts,
        "recording_stage_input_summary_v0.1.tsv": summary,
        "confirm2_event_support_v0.1.tsv": support,
        "pre_fit_input_audit_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, common.output_dir() / name)
    print(f"Stage-first pre-fit audit passed {len(checks)}/{len(checks)} checks.")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
