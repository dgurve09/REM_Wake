"""Validate split, metric, event-count, and external-artifact integrity."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score


VERSION = "v0.1"
COMPARATORS = ["SF-B", "SF-C"]
STAGES = [0, 1, 2, 3, 4]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def experiment_dir() -> Path:
    return repo_root() / "experiments/2026-08-15_stage_first_feature_baseline_v0.1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add(rows: list[dict], check: str, observed, expected, passed: bool) -> None:
    rows.append(
        {
            "validation_version": VERSION,
            "check": check,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
        }
    )


def validate_predictions(rows: list[dict]) -> None:
    destination = experiment_dir()
    train_validation = pd.read_csv(
        destination / "train_validation_stage_predictions_v0.1.tsv", sep="\t"
    )
    test = pd.read_csv(destination / "test_stage_predictions_v0.1.tsv", sep="\t")
    add(
        rows,
        "train_validation_contains_no_test_rows",
        ";".join(sorted(train_validation["partition"].unique())),
        "train;validation",
        set(train_validation["partition"]) == {"train", "validation"},
    )
    add(
        rows,
        "test_contains_only_test_rows",
        ";".join(sorted(test["partition"].unique())),
        "test",
        set(test["partition"]) == {"test"},
    )
    participant_sets = {
        partition: set(
            pd.concat([train_validation, test])
            .loc[lambda frame: frame["partition"] == partition, "pid"]
            .astype(int)
        )
        for partition in ["train", "validation", "test"]
    }
    overlap = (
        participant_sets["train"] & participant_sets["validation"]
    ) | (participant_sets["train"] & participant_sets["test"]) | (
        participant_sets["validation"] & participant_sets["test"]
    )
    add(rows, "participant_partition_overlap", len(overlap), 0, len(overlap) == 0)


def validate_stage_metrics(rows: list[dict]) -> None:
    destination = experiment_dir()
    predictions = pd.concat(
        [
            pd.read_csv(destination / "train_validation_stage_predictions_v0.1.tsv", sep="\t"),
            pd.read_csv(destination / "test_stage_predictions_v0.1.tsv", sep="\t"),
        ],
        ignore_index=True,
    )
    reported = pd.concat(
        [
            pd.read_csv(destination / "train_validation_stage_metrics_v0.1.tsv", sep="\t"),
            pd.read_csv(destination / "test_stage_metrics_v0.1.tsv", sep="\t"),
        ],
        ignore_index=True,
    )
    for item in reported.itertuples(index=False):
        group = predictions[
            (predictions["comparator"] == item.comparator)
            & (predictions["partition"] == item.partition)
        ]
        truth = group["stage_hum"].to_numpy(dtype=int)
        predicted = group["stage_pred"].to_numpy(dtype=int)
        recomputed = {
            "macro_f1": f1_score(truth, predicted, labels=STAGES, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(truth, predicted),
            "cohen_kappa": cohen_kappa_score(truth, predicted, labels=STAGES),
        }
        for metric, value in recomputed.items():
            expected = float(getattr(item, metric))
            add(
                rows,
                f"{item.comparator}_{item.partition}_{metric}",
                f"{value:.12f}",
                f"{expected:.12f}",
                np.isclose(value, expected, atol=1e-12),
            )


def validate_event_accounting(rows: list[dict]) -> None:
    destination = experiment_dir()
    for prefix in ["train_validation", "test"]:
        metrics = pd.read_csv(destination / f"{prefix}_event_metrics_v0.1.tsv", sep="\t")
        for item in metrics.itertuples(index=False):
            reference_balance = item.true_positive + item.false_negative
            prediction_balance = item.true_positive + item.false_positive + item.ignored_predictions
            add(
                rows,
                f"{item.comparator}_{item.partition}_{item.membership}_{int(item.tolerance_sec)}s_reference_balance",
                reference_balance,
                int(item.reference_events),
                reference_balance == int(item.reference_events),
            )
            add(
                rows,
                f"{item.comparator}_{item.partition}_{item.membership}_{int(item.tolerance_sec)}s_prediction_balance",
                prediction_balance,
                int(item.predicted_events),
                prediction_balance == int(item.predicted_events),
            )


def validate_external_artifacts(rows: list[dict]) -> None:
    manifest = pd.read_csv(experiment_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    hash_pass = 0
    for item in manifest.itertuples(index=False):
        path = data_parent() / Path(item.path_relative_to_data_parent)
        if path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256:
            hash_pass += 1
    add(rows, "external_artifact_hashes", hash_pass, len(manifest), hash_pass == len(manifest))
    add(rows, "external_artifact_count", len(manifest), 130, len(manifest) == 130)


def main() -> None:
    rows: list[dict] = []
    validate_predictions(rows)
    validate_stage_metrics(rows)
    validate_event_accounting(rows)
    validate_external_artifacts(rows)
    result = pd.DataFrame(rows)
    result.to_csv(experiment_dir() / "output_integrity_checks_v0.1.tsv", sep="\t", index=False)
    print(result.to_string(index=False))
    if not result["passed"].all():
        raise SystemExit("At least one stage-first output-integrity check failed")


if __name__ == "__main__":
    main()
