"""Independently validate reviewed alarm-budget sensitivity outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENT_DIR = "2026-10-08_full_night_risk_set_budget_sensitivity_v0.1"
SOURCE_DIR = "2026-10-07_full_night_risk_set_formulation_v0.1"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root().as_posix()}", "rev-parse", "HEAD"],
        cwd=root(), text=True,
    ).strip()


def equal(left: float, right: float) -> bool:
    if np.isnan(left) and np.isnan(right):
        return True
    return bool(np.isclose(left, right, rtol=1e-12, atol=1e-12))


def expected_metrics(true_positive: int, false_positive: int,
                     false_negative: int, supported_hours: float) -> dict[str, float]:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else np.nan
    recall = true_positive / recall_denominator if recall_denominator else np.nan
    f1_denominator = 2 * true_positive + false_positive + false_negative
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * true_positive / f1_denominator if f1_denominator else np.nan,
        "false_alarms_per_hour": false_positive / supported_hours,
    }


# Section 1: validate the complete source chain

def validate_sources(manifest: pd.DataFrame) -> None:
    if len(manifest) != 17:
        raise AssertionError("Expected 15 external scores and two repository inputs")
    for row in manifest.itertuples(index=False):
        if row.location_scope == "repository":
            path = root() / row.relative_path
        elif row.location_scope == "external_data":
            path = root().parent / "REM_W_data" / row.relative_path
        else:
            raise AssertionError(f"Unknown source location: {row.location_scope}")
        if not path.is_file():
            raise AssertionError(f"Missing source artifact: {path}")
        if path.stat().st_size != int(row.bytes) or sha256(path) != row.sha256:
            raise AssertionError(f"Source artifact changed: {path}")


# Section 2: recompute every aggregate from participant-level results

def validate_metrics(metrics: pd.DataFrame, participants: pd.DataFrame,
                     events: pd.DataFrame) -> None:
    keys = [
        "candidate", "partition", "membership", "tolerance_sec",
        "alarm_budget_per_hour",
    ]
    if len(metrics) != 48 or len(participants) != 3072:
        raise AssertionError("Unexpected metric or participant grid size")
    if metrics.duplicated(keys).any():
        raise AssertionError("Duplicate metric configuration")

    for metric in metrics.itertuples(index=False):
        mask = np.ones(len(participants), dtype=bool)
        for key in keys:
            value = getattr(metric, key)
            column = participants[key]
            mask &= np.isclose(column, value) if np.issubdtype(column.dtype, np.number) \
                else column.eq(value)
        local = participants.loc[mask]
        if len(local) != 64 or local.pid.nunique() != 64:
            raise AssertionError("Incomplete participant configuration")

        sums = {
            column: local[column].sum()
            for column in [
                "recordings", "reference_events", "predicted_events", "true_positive",
                "false_positive", "false_negative", "ignored_predictions", "supported_hours",
            ]
        }
        for column, value in sums.items():
            if not equal(float(value), float(getattr(metric, column))):
                raise AssertionError(f"Aggregate mismatch for {column}")
        recalculated = expected_metrics(
            int(sums["true_positive"]), int(sums["false_positive"]),
            int(sums["false_negative"]), float(sums["supported_hours"]),
        )
        for column, value in recalculated.items():
            if not equal(value, float(getattr(metric, column))):
                raise AssertionError(f"Metric mismatch for {column}")

        event_count = len(events[
            events.candidate.eq(metric.candidate)
            & np.isclose(events.alarm_budget_per_hour, metric.alarm_budget_per_hour)
        ])
        if event_count != int(metric.predicted_events):
            raise AssertionError("Predicted-event count mismatch")


# Section 3: reproduce the validated primary-budget table

def validate_primary(metrics: pd.DataFrame) -> None:
    source = pd.read_csv(
        root() / "experiments" / SOURCE_DIR / "outer_event_metrics_v0.1.tsv",
        sep="\t",
    )
    actual = metrics[np.isclose(metrics.alarm_budget_per_hour, 0.25)]
    actual = actual[source.columns].sort_values(
        ["candidate", "membership", "tolerance_sec"]
    ).reset_index(drop=True)
    expected = source.sort_values(
        ["candidate", "membership", "tolerance_sec"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12
    )


def run(result_code_commit: str) -> None:
    if not current_commit().startswith(result_code_commit):
        raise AssertionError("Validator code commit does not match current checkout")
    output = output_dir()
    metrics = pd.read_csv(output / "alarm_budget_event_metrics_v0.1.tsv", sep="\t")
    participants = pd.read_csv(
        output / "alarm_budget_event_participants_v0.1.tsv", sep="\t"
    )
    events = pd.read_csv(output / "alarm_budget_predicted_events_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(output / "source_artifact_manifest_v0.1.tsv", sep="\t")
    checks = pd.read_csv(output / "in_run_checks_v0.1.tsv", sep="\t")
    versions = json.loads(
        (output / "software_versions_v0.1.json").read_text(encoding="utf-8")
    )

    if not checks.passed.astype(bool).all():
        raise AssertionError("Sensitivity report retained a failed in-run check")
    if not versions["git_commit"].startswith(result_code_commit):
        raise AssertionError("Recorded result code commit does not match")
    validate_sources(manifest)
    validate_metrics(metrics, participants, events)
    validate_primary(metrics)
    print(
        f"Validated {len(metrics)} metric rows, {len(participants):,} participant rows, "
        f"{len(events):,} predicted events, and {len(manifest)} source hashes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.result_code_commit)
