"""Audit pooled PSG model inputs and participant-grouped splits before fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_multidataset_psg_external_generalization_v0_1 as experiment
from reviewed_output import verify_or_create_tsv


# Section 1: compact input summaries

def dataset_summary(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for dataset in experiment.DATASETS:
        local = [item for item in recordings.values() if item["dataset"] == dataset]
        rows.append(
            {
                "dataset": dataset,
                "recordings": len(local),
                "participant_groups": len({item["participant_group"] for item in local}),
                "decision_rows": int(sum(len(item["times"]) for item in local)),
                "scorable_rows": int(sum(item["score_mask"].sum() for item in local)),
                "positive_targets": int(sum(item["positive"].sum() for item in local)),
                "negative_targets": int(sum(item["negative"].sum() for item in local)),
                "ignored_boundaries": int(sum(len(item["ignored"]) for item in local)),
            }
        )
    return pd.DataFrame(rows)


def fold_summary(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for heldout in experiment.DATASETS:
        mapping = experiment.inner_fold_map(recordings, heldout)
        for fold in range(experiment.INNER_FOLDS):
            for dataset in experiment.DATASETS:
                if dataset == heldout:
                    continue
                local = [
                    item for item in recordings.values()
                    if item["dataset"] == dataset
                    and mapping[item["participant_group"]] == fold
                ]
                rows.append(
                    {
                        "heldout_dataset": heldout,
                        "inner_fold": fold,
                        "validation_dataset": dataset,
                        "recordings": len(local),
                        "participant_groups": len({item["participant_group"] for item in local}),
                        "positive_targets": int(sum(item["positive"].sum() for item in local)),
                        "negative_targets": int(sum(item["negative"].sum() for item in local)),
                    }
                )
    return pd.DataFrame(rows)


# Section 2: independent assertions

def model_outputs_present(output: Path, derived: Path) -> bool:
    reviewed_names = [
        "model_fit_summary_v0.1.tsv",
        "inner_threshold_curves_v0.1.tsv",
        "inner_threshold_selections_v0.1.tsv",
        "outer_support_v0.1.tsv",
        "outer_predicted_events_v0.1.tsv",
        "heldout_event_metrics_v0.1.tsv",
        "heldout_event_participants_v0.1.tsv",
        "score_ranking_diagnostics_v0.1.tsv",
        "scorer_sensitivity_v0.1.tsv",
        "paired_external_bootstrap_v0.1.tsv",
        "hypothesis_decision_v0.1.tsv",
        "external_artifact_manifest_v0.1.tsv",
        "in_run_checks_v0.1.tsv",
    ]
    return any((output / name).exists() for name in reviewed_names) or any(
        (derived / folder).exists() for folder in ["models", "scores"]
    )


def build_checks(
    recordings: dict[str, dict],
    summary: pd.DataFrame,
    folds: pd.DataFrame,
) -> pd.DataFrame:
    expected = {
        "BOAS": {"recordings": 128, "participant_groups": 100, "positive_targets": 365},
        "gx36m": {"recordings": 58, "participant_groups": 20, "positive_targets": 124},
        "ISRUC-3": {"recordings": 10, "participant_groups": 10, "positive_targets": 32},
    }
    dataset_counts_pass = all(
        all(int(summary.set_index("dataset").loc[name, key]) == value for key, value in values.items())
        for name, values in expected.items()
    )
    arrays_pass = all(
        len(item["times"]) == item["PSD3-BASE"].shape[0] == item["ADV3-CHANGE"].shape[0]
        and item["PSD3-BASE"].shape[1] == 120
        and item["ADV3-CHANGE"].shape[1] == 444
        and np.isfinite(item["PSD3-BASE"]).all()
        and np.isfinite(item["ADV3-CHANGE"]).all()
        and np.all(np.diff(item["times"]) == 15.0)
        for item in recordings.values()
    )
    masks_pass = all(
        not np.any(item["positive"] & item["negative"])
        and np.all(item["score_mask"][item["positive"]])
        and np.all(item["score_mask"][item["negative"]])
        for item in recordings.values()
    )
    split_pass = True
    split_coverage_pass = True
    for heldout in experiment.DATASETS:
        mapping = experiment.inner_fold_map(recordings, heldout)
        train_groups = {
            item["participant_group"] for item in recordings.values()
            if item["dataset"] != heldout
        }
        heldout_groups = {
            item["participant_group"] for item in recordings.values()
            if item["dataset"] == heldout
        }
        split_pass &= set(mapping) == train_groups and not (set(mapping) & heldout_groups)
        split_coverage_pass &= set(mapping.values()) == set(range(experiment.INNER_FOLDS))
    fold_support_pass = bool(
        folds.recordings.gt(0).all()
        and folds.participant_groups.gt(0).all()
        and folds.positive_targets.gt(0).all()
        and folds.negative_targets.gt(0).all()
    )
    no_outputs = not model_outputs_present(experiment.output_dir(), experiment.derived_dir())
    rows = [
        {"check": "recording_count", "observed": len(recordings), "expected": 196, "passed": len(recordings) == 196},
        {"check": "participant_group_count", "observed": len({item["participant_group"] for item in recordings.values()}), "expected": 130, "passed": len({item["participant_group"] for item in recordings.values()}) == 130},
        {"check": "decision_row_count", "observed": int(summary.decision_rows.sum()), "expected": 378054, "passed": int(summary.decision_rows.sum()) == 378054},
        {"check": "transition_target_count", "observed": int(summary.positive_targets.sum()), "expected": 521, "passed": int(summary.positive_targets.sum()) == 521},
        {"check": "dataset_specific_counts", "observed": dataset_counts_pass, "expected": True, "passed": dataset_counts_pass},
        {"check": "feature_shapes_times_and_values", "observed": arrays_pass, "expected": True, "passed": arrays_pass},
        {"check": "target_masks_disjoint_and_scorable", "observed": masks_pass, "expected": True, "passed": masks_pass},
        {"check": "heldout_groups_excluded_from_inner_mapping", "observed": split_pass, "expected": True, "passed": split_pass},
        {"check": "all_inner_folds_assigned", "observed": split_coverage_pass, "expected": True, "passed": split_coverage_pass},
        {"check": "each_fold_has_both_training_datasets_and_classes", "observed": fold_support_pass, "expected": True, "passed": fold_support_pass},
        {"check": "model_or_result_outputs_present", "observed": not no_outputs, "expected": False, "passed": no_outputs},
    ]
    return pd.DataFrame(rows)


# Section 3: reviewed pre-fit checkpoint

def main() -> None:
    recordings = experiment.load_recordings()
    summary = dataset_summary(recordings)
    folds = fold_summary(recordings)
    checks = build_checks(recordings, summary, folds)
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Pre-fit multi-dataset model-input audit failed")
    verify_or_create_tsv(summary, experiment.output_dir() / "model_input_summary_v0.1.tsv")
    verify_or_create_tsv(folds, experiment.output_dir() / "inner_fold_audit_v0.1.tsv")
    verify_or_create_tsv(checks, experiment.output_dir() / "pre_fit_model_input_audit_v0.1.tsv")
    print(f"Pre-fit model-input audit passed {len(checks)}/{len(checks)} checks.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
