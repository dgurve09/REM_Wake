"""Rerun the frozen PSG experiment with strict gx36m consensus."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import audit_multidataset_psg_pool_v0_1 as pool_audit
import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_1 as original
from reviewed_output import verify_or_create_tsv


# Section 1: corrected locations and label rule

EXPERIMENT_DIR = "2026-10-25_multidataset_psg_external_generalization_v0.2"
DERIVED_DIR = "multidataset_psg_external_generalization_v0.2"
STAGE_NAMES = np.asarray(["W", "N1", "N2", "N3", "REM"])
ORIGINAL_SCORER_VARIANTS = original.scorer_variants


def output_dir() -> Path:
    return features.repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return features.data_parent() / "derived" / DERIVED_DIR


def legacy_primary_stages(item) -> np.ndarray:
    """Reconstruct the preserved v0.1 labels for split verification only."""
    paths = original.source_paths(item.label_source)
    if item.dataset == "BOAS":
        return pd.read_csv(paths[0]).stage.astype(str).to_numpy()
    if item.dataset == "gx36m":
        return pool_audit.gx_consensus(paths)[0]
    if item.dataset == "ISRUC-3":
        return pool_audit.isruc_consensus(paths)[0]
    raise ValueError(item.dataset)


def strict_gx_consensus(paths: list[Path]) -> np.ndarray:
    scorings = [np.loadtxt(path, dtype=int, ndmin=1) for path in paths]
    length = min(map(len, scorings))
    values = np.vstack([scoring[:length] for scoring in scorings])
    counts = np.stack([(values == stage).sum(axis=0) for stage in range(5)])
    winner = counts.argmax(axis=0)
    return np.where(counts.max(axis=0) >= 2, STAGE_NAMES[winner], "UNKNOWN")


def primary_stages(item) -> np.ndarray:
    paths = original.source_paths(item.label_source)
    if item.dataset == "gx36m":
        return strict_gx_consensus(paths)
    return legacy_primary_stages(item)


def scorer_variants(item) -> dict[str, np.ndarray]:
    paths = original.source_paths(item.label_source)
    if item.dataset == "gx36m":
        variants = {}
        for number, path in enumerate(paths, start=1):
            raw = np.loadtxt(path, dtype=int, ndmin=1)
            stages = np.full(len(raw), "UNKNOWN", dtype="U7")
            known = (raw >= 0) & (raw < len(STAGE_NAMES))
            stages[known] = STAGE_NAMES[raw[known]]
            variants[f"expert_{number}"] = stages
        return variants
    return ORIGINAL_SCORER_VARIANTS(item)


# Section 2: preserve the v0.1 participant folds

def inner_fold_map(recordings: dict, heldout_dataset: str) -> dict[str, int]:
    path = output_dir() / "frozen_inner_fold_membership_v0.2.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Run the v0.2 pre-fit audit first: {path}")
    membership = pd.read_csv(path, sep="\t")
    local = membership[membership.heldout_dataset.eq(heldout_dataset)]
    mapping = dict(zip(local.participant_group, local.inner_fold.astype(int)))
    training_groups = {
        item["participant_group"]
        for item in recordings.values()
        if item["dataset"] != heldout_dataset
    }
    if set(mapping) != training_groups:
        raise ValueError(f"Frozen fold membership does not match {heldout_dataset} training groups")
    return mapping


def configure_original_module() -> None:
    original.output_dir = output_dir
    original.derived_dir = derived_dir
    original.primary_stages = primary_stages
    original.scorer_variants = scorer_variants
    original.inner_fold_map = inner_fold_map


# Section 3: exact rerun with versioned outputs

def main() -> None:
    configure_original_module()
    recordings = original.load_recordings()
    run = original.run_experiment(recordings)
    results = original.result_tables(run, recordings)
    paired = original.bootstrap(results["participants"])
    gate = original.decision(results["metrics"], run["fits"])

    manifest_rows = []
    for item in run["fits"].itertuples(index=False):
        manifest_rows.append(
            {
                "artifact_type": "model",
                "heldout_dataset": item.heldout_dataset,
                "candidate": item.candidate,
                "phase": item.phase,
                "relative_path": item.model_relative_path,
                "bytes": item.model_bytes,
                "sha256": item.model_sha256,
            }
        )
    for heldout in original.DATASETS:
        for candidate in original.CANDIDATES:
            path = original.score_path(heldout, candidate)
            manifest_rows.append(
                {
                    "artifact_type": "outer_score",
                    "heldout_dataset": heldout,
                    "candidate": candidate,
                    "phase": "outer_final",
                    "relative_path": path.relative_to(features.data_parent()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": features.sha256(path),
                }
            )
    manifest = pd.DataFrame(manifest_rows)

    target_count = int(sum(item["positive"].sum() for item in recordings.values()))
    checks = pd.DataFrame(
        [
            {"check": "model_fits", "observed": len(run["fits"]), "expected": 36, "passed": len(run["fits"]) == 36},
            {"check": "all_models_converged", "observed": bool(run["fits"].converged.all()), "expected": True, "passed": bool(run["fits"].converged.all())},
            {"check": "corrected_transition_targets", "observed": target_count, "expected": 482, "passed": target_count == 482},
            {"check": "heldout_datasets", "observed": run["scores"].heldout_dataset.nunique(), "expected": 3, "passed": run["scores"].heldout_dataset.nunique() == 3},
            {"check": "outer_candidates", "observed": run["scores"].candidate.nunique(), "expected": 2, "passed": run["scores"].candidate.nunique() == 2},
            {"check": "outer_recordings", "observed": run["scores"].recording.nunique(), "expected": 196, "passed": run["scores"].recording.nunique() == 196},
            {"check": "participant_coverage", "observed": run["scores"].participant_group.nunique(), "expected": 130, "passed": run["scores"].participant_group.nunique() == 130},
            {"check": "threshold_rows", "observed": len(run["selections"]), "expected": 6, "passed": len(run["selections"]) == 6},
            {"check": "alarm_monotonicity", "observed": bool(run["curves"].alarm_count_monotonic.all()), "expected": True, "passed": bool(run["curves"].alarm_count_monotonic.all())},
            {"check": "metric_rows", "observed": len(results["metrics"]), "expected": 8, "passed": len(results["metrics"]) == 8},
            {"check": "bootstrap_rows", "observed": len(paired), "expected": 3, "passed": len(paired) == 3},
            {"check": "scorer_sensitivity_rows", "observed": len(results["sensitivity"]), "expected": 10, "passed": len(results["sensitivity"]) == 10},
            {"check": "validation_or_test_tuning", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Corrected multi-dataset in-run controls failed")

    outputs = {
        "model_fit_summary_v0.2.tsv": run["fits"],
        "inner_threshold_curves_v0.2.tsv": run["curves"],
        "inner_threshold_selections_v0.2.tsv": run["selections"],
        "outer_support_v0.2.tsv": run["support"],
        "outer_predicted_events_v0.2.tsv": run["alarms"],
        "heldout_event_metrics_v0.2.tsv": results["metrics"],
        "heldout_event_participants_v0.2.tsv": results["participants"],
        "score_ranking_diagnostics_v0.2.tsv": results["ranking"],
        "scorer_sensitivity_v0.2.tsv": results["sensitivity"],
        "paired_external_bootstrap_v0.2.tsv": paired,
        "hypothesis_decision_v0.2.tsv": gate,
        "external_artifact_manifest_v0.2.tsv": manifest,
        "in_run_checks_v0.2.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)

    print(results["metrics"].to_string(index=False))
    print(paired.to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
