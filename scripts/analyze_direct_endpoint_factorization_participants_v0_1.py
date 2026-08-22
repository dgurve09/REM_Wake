"""Paired participant analysis for validation-only endpoint factorization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_direct_event_baseline_v0_1 import metric_values, repo_root


# Section 1: paths and frozen inputs

EXPERIMENT_DIR = "2026-08-22_direct_endpoint_factorization_participant_analysis_v0.1"
RESAMPLES = 2000
SEED = 20260822


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def de_b_path() -> Path:
    return (
        repo_root()
        / "experiments/2026-08-22_direct_event_baseline_v0.1/train_validation_event_participants_v0.1.tsv"
    )


def de_d_path() -> Path:
    return (
        repo_root()
        / "experiments/2026-08-22_direct_endpoint_factorization_v0.1/validation_event_participants_v0.1.tsv"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Section 2: paired participant table

def load_participants() -> pd.DataFrame:
    de_b = pd.read_csv(de_b_path(), sep="\t")
    de_b = de_b[
        (de_b["comparator"] == "DE-B")
        & (de_b["partition"] == "validation")
        & (de_b["membership"] == "primary")
        & (de_b["tolerance_sec"] == 15.0)
    ].copy()
    de_d = pd.read_csv(de_d_path(), sep="\t")
    de_d = de_d[
        (de_d["comparator"] == "DE-D")
        & (de_d["partition"] == "validation")
        & (de_d["membership"] == "primary")
        & (de_d["tolerance_sec"] == 15.0)
    ].copy()
    columns = [
        "pid",
        "recordings",
        "reference_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
    ]
    paired = de_b[columns].merge(
        de_d[columns],
        on="pid",
        suffixes=("_de_b", "_de_d"),
        validate="one_to_one",
    )
    if len(paired) != 16:
        raise ValueError(f"Expected 16 validation participants, found {len(paired)}")
    for metric in [
        "true_positive",
        "false_positive",
        "false_negative",
        "f1",
        "false_alarms_per_hour",
    ]:
        paired[f"{metric}_difference"] = (
            paired[f"{metric}_de_d"] - paired[f"{metric}_de_b"]
        )
    return paired.sort_values("pid")


# Section 3: paired bootstrap and direction counts

def aggregate_metrics(rows: pd.DataFrame, suffix: str) -> dict:
    return metric_values(
        int(rows[f"true_positive_{suffix}"].sum()),
        int(rows[f"false_positive_{suffix}"].sum()),
        int(rows[f"false_negative_{suffix}"].sum()),
        float(rows[f"supported_hours_{suffix}"].sum()),
    )


def paired_bootstrap(paired: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(RESAMPLES):
        sampled = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        de_b = aggregate_metrics(sampled, "de_b")
        de_d = aggregate_metrics(sampled, "de_d")
        rows.append(
            {
                "event_f1_difference": de_d["f1"] - de_b["f1"],
                "false_alarms_per_hour_difference": de_d["false_alarms_per_hour"]
                - de_b["false_alarms_per_hour"],
            }
        )
    samples = pd.DataFrame(rows)
    point_b = aggregate_metrics(paired, "de_b")
    point_d = aggregate_metrics(paired, "de_d")
    points = {
        "event_f1_difference": point_d["f1"] - point_b["f1"],
        "false_alarms_per_hour_difference": point_d["false_alarms_per_hour"]
        - point_b["false_alarms_per_hour"],
    }
    summary = []
    for metric, point in points.items():
        summary.append(
            {
                "comparison": "DE-D_minus_DE-B_validation",
                "metric": metric,
                "point_difference": point,
                "resamples": RESAMPLES,
                "seed": SEED,
                "lower_95": float(samples[metric].quantile(0.025)),
                "median": float(samples[metric].quantile(0.5)),
                "upper_95": float(samples[metric].quantile(0.975)),
            }
        )
    return pd.DataFrame(summary)


def direction_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["true_positive", "false_positive", "f1", "false_alarms_per_hour"]:
        values = paired[f"{metric}_difference"]
        rows.append(
            {
                "metric": metric,
                "participants_decreased": int((values < 0).sum()),
                "participants_unchanged": int((values == 0).sum()),
                "participants_increased": int((values > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


# Section 4: records

def input_manifest() -> pd.DataFrame:
    rows = []
    for role, path in [
        ("DE-B_validation_participant_metrics", de_b_path()),
        ("DE-D_validation_participant_metrics", de_d_path()),
    ]:
        rows.append(
            {
                "artifact_role": role,
                "relative_path": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    paired: pd.DataFrame, bootstrap: pd.DataFrame, directions: pd.DataFrame
) -> None:
    index = bootstrap.set_index("metric")
    f1 = index.loc["event_f1_difference"]
    far = index.loc["false_alarms_per_hour_difference"]
    true_positive = directions[directions["metric"] == "true_positive"].iloc[0]
    false_positive = directions[directions["metric"] == "false_positive"].iloc[0]
    text = f"""# DE-D Paired Participant Analysis v0.1

**Created:** 2026-08-22
**Status:** Post-result exploratory validation analysis
**Plan:** `docs/evaluation/direct_endpoint_factorization_participant_plan_v0.1.md`
**Test access:** None

## Paired Bootstrap

DE-D minus DE-B validation event F1 was {f1.point_difference:+.4f}, with a paired participant-bootstrap 95% interval from {f1.lower_95:+.4f} to {f1.upper_95:+.4f}.

The false-alarm-rate difference was {far.point_difference:+.4f} per hour, with a 95% interval from {far.lower_95:+.4f} to {far.upper_95:+.4f}.

## Participant Direction

DE-D increased true-positive count for {int(true_positive.participants_increased)} participants, left it unchanged for {int(true_positive.participants_unchanged)}, and decreased it for {int(true_positive.participants_decreased)}. It reduced false-positive count for {int(false_positive.participants_decreased)} participants, left it unchanged for {int(false_positive.participants_unchanged)}, and increased it for {int(false_positive.participants_increased)}.

## Interpretation Boundary

The paired analysis characterizes participant dispersion within the validation set. It does not create independent confirmation because DE-D was designed after prior test evidence and selected on this validation partition. No test, raw signal, feature array, candidate score, or fitted model was accessed.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


# Section 5: execute

def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    paired = load_participants()
    bootstrap = paired_bootstrap(paired)
    directions = direction_summary(paired)
    paired.to_csv(output_dir() / "paired_participant_metrics_v0.1.tsv", sep="\t", index=False)
    bootstrap.to_csv(output_dir() / "paired_bootstrap_summary_v0.1.tsv", sep="\t", index=False)
    directions.to_csv(output_dir() / "participant_direction_summary_v0.1.tsv", sep="\t", index=False)
    input_manifest().to_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t", index=False)
    write_readme(paired, bootstrap, directions)
    print(bootstrap.to_string(index=False))
    print(directions.to_string(index=False))


if __name__ == "__main__":
    main()
