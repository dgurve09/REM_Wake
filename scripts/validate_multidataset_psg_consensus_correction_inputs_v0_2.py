"""Independently validate the saved v0.2 label-correction checkpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_1 as original
import run_multidataset_psg_external_generalization_v0_2 as correction
from reviewed_output import verify_or_create_tsv


def read_reviewed(name: str) -> pd.DataFrame:
    return pd.read_csv(correction.output_dir() / name, sep="\t")


def independently_reconstruct_changes(eligibility: pd.DataFrame) -> pd.DataFrame:
    stage_names = np.asarray(["W", "N1", "N2", "N3", "REM"])
    rows = []
    for item in eligibility[eligibility.dataset.eq("gx36m")].itertuples(index=False):
        paths = original.source_paths(item.label_source)
        scorings = [np.loadtxt(path, dtype=int, ndmin=1) for path in paths]
        length = min(map(len, scorings))
        values = np.vstack([value[:length] for value in scorings])
        counts = np.stack([(values == stage).sum(axis=0) for stage in range(5)])
        winner = counts.argmax(axis=0)
        legacy = stage_names[winner]
        strict = np.where(counts.max(axis=0) >= 2, stage_names[winner], "UNKNOWN")
        rows.append(
            {
                "recording": item.recording,
                "participant_group": item.participant_group,
                "epochs": length,
                "raw_minus_one_labels": int((values == -1).sum()),
                "changed_epochs": int((legacy != strict).sum()),
                "legacy_unknown_epochs": int((legacy == "UNKNOWN").sum()),
                "strict_unknown_epochs": int((strict == "UNKNOWN").sum()),
                "legacy_events": len(original.transition_tables(legacy)[0]),
                "strict_events": len(original.transition_tables(strict)[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    eligibility = pd.read_csv(features.eligibility_path(), sep="\t")
    saved_changes = read_reviewed("label_correction_by_recording_v0.2.tsv")
    saved_summary = read_reviewed("label_correction_dataset_summary_v0.2.tsv")
    membership = read_reviewed("frozen_inner_fold_membership_v0.2.tsv")
    folds = read_reviewed("corrected_inner_fold_audit_v0.2.tsv")
    audit = read_reviewed("pre_fit_correction_audit_v0.2.tsv")
    scorer_summary = read_reviewed("gx_scorer_variant_correction_v0.2.tsv")
    changes = independently_reconstruct_changes(eligibility)

    pd.testing.assert_frame_equal(
        changes.sort_values("recording").reset_index(drop=True),
        saved_changes.sort_values("recording").reset_index(drop=True),
        check_dtype=False,
    )
    membership_unique = not membership.duplicated(["heldout_dataset", "participant_group"]).any()
    expected_training_groups = 0
    for heldout in original.DATASETS:
        expected_training_groups += eligibility[~eligibility.dataset.eq(heldout)].participant_group.nunique()
    cache_summary = pd.read_csv(features.output_dir() / "feature_generation_summary_v0.1.tsv", sep="\t")
    cache_hashes = all(
        (features.data_parent() / item.relative_path).exists()
        and (features.data_parent() / item.relative_path).stat().st_size == int(item.bytes)
        and features.sha256(features.data_parent() / item.relative_path) == item.sha256
        for item in cache_summary.itertuples(index=False)
    )
    strict_total = int(
        saved_summary[saved_summary.label_version.eq("v0.2_strict_2_of_3")].positive_targets.sum()
    )
    checks = pd.DataFrame(
        [
            {"check": "recording_change_table_reconstructed", "observed": len(changes), "expected": 58, "passed": len(changes) == 58},
            {"check": "raw_minus_one_labels", "observed": int(saved_changes.raw_minus_one_labels.sum()), "expected": 818, "passed": int(saved_changes.raw_minus_one_labels.sum()) == 818},
            {"check": "changed_epochs_reconstructed", "observed": int(changes.changed_epochs.sum()), "expected": 1479, "passed": int(changes.changed_epochs.sum()) == 1479},
            {"check": "event_counts_reconstructed", "observed": f"{changes.legacy_events.sum()}->{changes.strict_events.sum()}", "expected": "124->85", "passed": changes.legacy_events.sum() == 124 and changes.strict_events.sum() == 85},
            {"check": "corrected_total_targets", "observed": strict_total, "expected": 482, "passed": strict_total == 482},
            {"check": "scorer_unknown_epochs", "observed": int(scorer_summary.corrected_unknown_epochs.sum()), "expected": 818, "passed": int(scorer_summary.corrected_unknown_epochs.sum()) == 818},
            {"check": "fold_membership_unique", "observed": membership_unique, "expected": True, "passed": membership_unique},
            {"check": "fold_membership_rows", "observed": len(membership), "expected": expected_training_groups, "passed": len(membership) == expected_training_groups == 260},
            {"check": "folds_cover_both_classes", "observed": bool(folds.positive_targets.gt(0).all() and folds.negative_targets.gt(0).all()), "expected": True, "passed": bool(folds.positive_targets.gt(0).all() and folds.negative_targets.gt(0).all())},
            {"check": "feature_cache_hashes", "observed": cache_hashes, "expected": True, "passed": cache_hashes},
            {"check": "primary_audit_passed", "observed": bool(audit.passed.all()), "expected": True, "passed": bool(audit.passed.all())},
            {"check": "v0.2_models_absent", "observed": (correction.derived_dir() / "models").exists(), "expected": False, "passed": not (correction.derived_dir() / "models").exists()},
            {"check": "v0.2_scores_absent", "observed": (correction.derived_dir() / "scores").exists(), "expected": False, "passed": not (correction.derived_dir() / "scores").exists()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent v0.2 input validation failed")
    verify_or_create_tsv(
        checks,
        correction.output_dir() / "independent_input_validation_v0.2.tsv",
    )
    print(f"Independent v0.2 input validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
