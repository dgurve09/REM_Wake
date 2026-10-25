"""Audit the gx36m consensus correction before fitting any v0.2 model."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_1 as original
import run_multidataset_psg_external_generalization_v0_2 as correction
from reviewed_output import verify_or_create_tsv


# Section 1: compact summaries

def dataset_summary(recordings: dict[str, dict], version: str) -> pd.DataFrame:
    rows = []
    for dataset in original.DATASETS:
        local = [item for item in recordings.values() if item["dataset"] == dataset]
        rows.append(
            {
                "label_version": version,
                "dataset": dataset,
                "recordings": len(local),
                "participant_groups": len({item["participant_group"] for item in local}),
                "decision_rows": int(sum(len(item["times"]) for item in local)),
                "scorable_rows": int(sum(item["score_mask"].sum() for item in local)),
                "positive_targets": int(sum(item["positive"].sum() for item in local)),
                "negative_targets": int(sum(item["negative"].sum() for item in local)),
                "ignored_boundaries": int(sum(len(item["ignored"]) for item in local)),
                "unknown_epochs": int(sum((item["stages"] == "UNKNOWN").sum() for item in local)),
            }
        )
    return pd.DataFrame(rows)


def gx_recording_changes(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in eligibility[eligibility.dataset.eq("gx36m")].itertuples(index=False):
        legacy = correction.legacy_primary_stages(item)
        strict = correction.primary_stages(item)
        raw_minus_one = sum(
            int((np.loadtxt(path, dtype=int, ndmin=1) == -1).sum())
            for path in original.source_paths(item.label_source)
        )
        rows.append(
            {
                "recording": item.recording,
                "participant_group": item.participant_group,
                "epochs": len(strict),
                "raw_minus_one_labels": raw_minus_one,
                "changed_epochs": int((legacy != strict).sum()),
                "legacy_unknown_epochs": int((legacy == "UNKNOWN").sum()),
                "strict_unknown_epochs": int((strict == "UNKNOWN").sum()),
                "legacy_events": len(original.transition_tables(legacy)[0]),
                "strict_events": len(original.transition_tables(strict)[0]),
            }
        )
    return pd.DataFrame(rows)


def gx_scorer_summary(eligibility: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ["expert_1", "expert_2", "expert_3"]:
        unsafe_events = 0
        corrected_events = 0
        corrected_unknown = 0
        for item in eligibility[eligibility.dataset.eq("gx36m")].itertuples(index=False):
            paths = original.source_paths(item.label_source)
            number = int(variant.split("_")[-1]) - 1
            raw = np.loadtxt(paths[number], dtype=int, ndmin=1)
            unsafe = correction.STAGE_NAMES[raw]
            safe = correction.scorer_variants(item)[variant]
            unsafe_events += len(original.transition_tables(unsafe)[0])
            corrected_events += len(original.transition_tables(safe)[0])
            corrected_unknown += int((safe == "UNKNOWN").sum())
        rows.append(
            {
                "label_variant": variant,
                "unsafe_events": unsafe_events,
                "corrected_events": corrected_events,
                "corrected_unknown_epochs": corrected_unknown,
            }
        )
    return pd.DataFrame(rows)


def fold_membership(recordings: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for heldout in original.DATASETS:
        mapping = original.inner_fold_map(recordings, heldout)
        for participant_group, fold in sorted(mapping.items()):
            dataset = next(
                item["dataset"]
                for item in recordings.values()
                if item["participant_group"] == participant_group
            )
            rows.append(
                {
                    "heldout_dataset": heldout,
                    "training_dataset": dataset,
                    "participant_group": participant_group,
                    "inner_fold": fold,
                }
            )
    return pd.DataFrame(rows)


def corrected_fold_summary(recordings: dict[str, dict], membership: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (heldout, fold, dataset), group in membership.groupby(
        ["heldout_dataset", "inner_fold", "training_dataset"], sort=True
    ):
        groups = set(group.participant_group)
        local = [
            item
            for item in recordings.values()
            if item["dataset"] == dataset and item["participant_group"] in groups
        ]
        rows.append(
            {
                "heldout_dataset": heldout,
                "inner_fold": int(fold),
                "validation_dataset": dataset,
                "recordings": len(local),
                "participant_groups": len(groups),
                "positive_targets": int(sum(item["positive"].sum() for item in local)),
                "negative_targets": int(sum(item["negative"].sum() for item in local)),
            }
        )
    return pd.DataFrame(rows)


# Section 2: immutable feature and output controls

def verify_feature_caches() -> tuple[int, int]:
    summary = pd.read_csv(
        features.output_dir() / "feature_generation_summary_v0.1.tsv", sep="\t"
    )
    passed = 0
    for item in summary.itertuples(index=False):
        path = features.data_parent() / item.relative_path
        passed += int(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and features.sha256(path) == item.sha256
        )
    return passed, len(summary)


def result_artifacts_present() -> bool:
    reviewed = list(correction.output_dir().glob("*result*.tsv"))
    reviewed += list(correction.output_dir().glob("*metric*.tsv"))
    reviewed += list(correction.output_dir().glob("*decision*.tsv"))
    model_or_scores = any(
        (correction.derived_dir() / folder).exists() for folder in ["models", "scores"]
    )
    return bool(reviewed) or model_or_scores


# Section 3: pre-fit checkpoint

def main() -> None:
    eligibility = pd.read_csv(features.eligibility_path(), sep="\t")
    changes = gx_recording_changes(eligibility)
    scorer_summary = gx_scorer_summary(eligibility)
    caches_passed, cache_count = verify_feature_caches()

    original.primary_stages = correction.legacy_primary_stages
    legacy_recordings = original.load_recordings()
    legacy_summary = dataset_summary(legacy_recordings, "v0.1_argmax")
    membership = fold_membership(legacy_recordings)
    del legacy_recordings
    gc.collect()

    original.primary_stages = correction.primary_stages
    corrected_recordings = original.load_recordings()
    strict_summary = dataset_summary(corrected_recordings, "v0.2_strict_2_of_3")
    folds = corrected_fold_summary(corrected_recordings, membership)
    summary = pd.concat([legacy_summary, strict_summary], ignore_index=True)

    indexed = summary.set_index(["label_version", "dataset"])
    no_outputs = not result_artifacts_present()
    only_gx_changed = (
        int(indexed.loc[("v0.1_argmax", "BOAS"), "positive_targets"])
        == int(indexed.loc[("v0.2_strict_2_of_3", "BOAS"), "positive_targets"])
        and int(indexed.loc[("v0.1_argmax", "ISRUC-3"), "positive_targets"])
        == int(indexed.loc[("v0.2_strict_2_of_3", "ISRUC-3"), "positive_targets"])
    )
    fold_support = bool(folds.positive_targets.gt(0).all() and folds.negative_targets.gt(0).all())
    checks = pd.DataFrame(
        [
            {"check": "recordings", "observed": len(eligibility), "expected": 196, "passed": len(eligibility) == 196},
            {"check": "participant_groups", "observed": eligibility.participant_group.nunique(), "expected": 130, "passed": eligibility.participant_group.nunique() == 130},
            {"check": "feature_cache_hashes", "observed": caches_passed, "expected": cache_count, "passed": caches_passed == cache_count == 196},
            {"check": "gx_raw_minus_one_labels", "observed": int(changes.raw_minus_one_labels.sum()), "expected": 818, "passed": int(changes.raw_minus_one_labels.sum()) == 818},
            {"check": "gx_changed_epochs", "observed": int(changes.changed_epochs.sum()), "expected": 1479, "passed": int(changes.changed_epochs.sum()) == 1479},
            {"check": "gx_affected_recordings", "observed": int(changes.changed_epochs.gt(0).sum()), "expected": int(changes.changed_epochs.gt(0).sum()), "passed": int(changes.changed_epochs.gt(0).sum()) > 0},
            {"check": "strict_unknown_epochs", "observed": int(changes.strict_unknown_epochs.sum()), "expected": 1479, "passed": int(changes.strict_unknown_epochs.sum()) == 1479},
            {"check": "legacy_gx_events", "observed": int(changes.legacy_events.sum()), "expected": 124, "passed": int(changes.legacy_events.sum()) == 124},
            {"check": "strict_gx_events", "observed": int(changes.strict_events.sum()), "expected": 85, "passed": int(changes.strict_events.sum()) == 85},
            {"check": "corrected_total_targets", "observed": int(strict_summary.positive_targets.sum()), "expected": 482, "passed": int(strict_summary.positive_targets.sum()) == 482},
            {"check": "scorer_unknown_epochs", "observed": int(scorer_summary.corrected_unknown_epochs.sum()), "expected": 818, "passed": int(scorer_summary.corrected_unknown_epochs.sum()) == 818},
            {"check": "boas_and_isruc_unchanged", "observed": only_gx_changed, "expected": True, "passed": only_gx_changed},
            {"check": "frozen_fold_rows", "observed": len(membership), "expected": 260, "passed": len(membership) == 260},
            {"check": "each_corrected_fold_has_both_classes", "observed": fold_support, "expected": True, "passed": fold_support},
            {"check": "v0.2_model_or_result_outputs_present", "observed": not no_outputs, "expected": False, "passed": no_outputs},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Consensus-correction pre-fit audit failed")

    outputs = {
        "label_correction_by_recording_v0.2.tsv": changes,
        "label_correction_dataset_summary_v0.2.tsv": summary,
        "gx_scorer_variant_correction_v0.2.tsv": scorer_summary,
        "frozen_inner_fold_membership_v0.2.tsv": membership,
        "corrected_inner_fold_audit_v0.2.tsv": folds,
        "pre_fit_correction_audit_v0.2.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, correction.output_dir() / name)

    print(f"Pre-fit consensus-correction audit passed {len(checks)}/{len(checks)} checks.")
    print(summary.to_string(index=False))
    print(changes.sum(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
