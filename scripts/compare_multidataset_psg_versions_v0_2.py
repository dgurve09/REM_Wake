"""Compare corrected PSG results with the preserved v0.1 result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_2 as correction
from reviewed_output import verify_or_create_tsv


def old_dir() -> Path:
    return features.repo_root() / "experiments" / "2026-10-25_multidataset_psg_external_generalization_v0.1"


def read(path: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(path / name, sep="\t")


def add_event_counts(metrics: pd.DataFrame, participants: pd.DataFrame) -> pd.DataFrame:
    counts = (
        participants.groupby(["heldout_dataset", "candidate"], as_index=False)
        .agg(
            true_positive=("true_positive", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
        )
    )
    metrics = metrics.merge(counts, on=["heldout_dataset", "candidate"], how="left")
    pooled = []
    for candidate in metrics.candidate.unique():
        local = counts[
            counts.heldout_dataset.isin(["gx36m", "ISRUC-3"])
            & counts.candidate.eq(candidate)
        ]
        pooled.append(
            {
                "heldout_dataset": "POOLED_EXTERNAL",
                "candidate": candidate,
                "true_positive": int(local.true_positive.sum()),
                "false_positive": int(local.false_positive.sum()),
                "false_negative": int(local.false_negative.sum()),
            }
        )
    pooled = pd.DataFrame(pooled).set_index(["heldout_dataset", "candidate"])
    metrics = metrics.set_index(["heldout_dataset", "candidate"])
    for index, item in pooled.iterrows():
        for column in ["true_positive", "false_positive", "false_negative"]:
            metrics.loc[index, column] = item[column]
    return metrics.reset_index()


def wide_comparison(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], measures: list[str]) -> pd.DataFrame:
    old = old[keys + measures].rename(columns={name: f"{name}_v0.1" for name in measures})
    new = new[keys + measures].rename(columns={name: f"{name}_v0.2" for name in measures})
    result = old.merge(new, on=keys, validate="one_to_one")
    for measure in measures:
        left = pd.to_numeric(result[f"{measure}_v0.1"], errors="coerce")
        right = pd.to_numeric(result[f"{measure}_v0.2"], errors="coerce")
        if left.notna().all() and right.notna().all():
            result[f"{measure}_difference"] = right.astype(float) - left.astype(float)
    return result


def main() -> None:
    new_dir = correction.output_dir()
    old_metrics = add_event_counts(
        read(old_dir(), "heldout_event_metrics_v0.1.tsv"),
        read(old_dir(), "heldout_event_participants_v0.1.tsv"),
    )
    new_metrics = add_event_counts(
        read(new_dir, "heldout_event_metrics_v0.2.tsv"),
        read(new_dir, "heldout_event_participants_v0.2.tsv"),
    )
    metric_measures = [
        "true_positive", "false_positive", "false_negative", "reference_events",
        "predicted_events", "precision", "recall", "f1", "false_alarms_per_hour",
    ]
    metrics = wide_comparison(
        old_metrics,
        new_metrics,
        ["heldout_dataset", "candidate"],
        metric_measures,
    )

    thresholds = wide_comparison(
        read(old_dir(), "inner_threshold_selections_v0.1.tsv"),
        read(new_dir, "inner_threshold_selections_v0.2.tsv"),
        ["heldout_dataset", "candidate"],
        ["threshold", "precision", "recall", "f1", "false_alarms_per_hour", "reference_events", "predicted_events"],
    )
    bootstrap = wide_comparison(
        read(old_dir(), "paired_external_bootstrap_v0.1.tsv"),
        read(new_dir, "paired_external_bootstrap_v0.2.tsv"),
        ["comparison", "metric"],
        ["lower_95", "median", "upper_95"],
    )
    sensitivity = wide_comparison(
        read(old_dir(), "scorer_sensitivity_v0.1.tsv"),
        read(new_dir, "scorer_sensitivity_v0.2.tsv"),
        ["heldout_dataset", "candidate", "label_variant"],
        ["reference_events", "precision", "recall", "f1", "false_alarms_per_hour"],
    )
    decisions = wide_comparison(
        read(old_dir(), "hypothesis_decision_v0.1.tsv"),
        read(new_dir, "hypothesis_decision_v0.2.tsv"),
        ["hypothesis"],
        ["advanced_f1", "baseline_f1", "f1_difference", "advanced_recall", "advanced_false_alarms_per_hour", "gx36m_f1_difference", "isruc_f1_difference", "point_gate_pass", "decision"],
    )

    old_targets = int(read(old_dir(), "model_input_summary_v0.1.tsv").positive_targets.sum())
    new_targets = int(
        read(new_dir, "label_correction_dataset_summary_v0.2.tsv")
        .query("label_version == 'v0.2_strict_2_of_3'")
        .positive_targets.sum()
    )
    pooled = new_metrics.set_index(["heldout_dataset", "candidate"])
    advanced = pooled.loc[("POOLED_EXTERNAL", "ADV3-CHANGE")]
    baseline = pooled.loc[("POOLED_EXTERNAL", "PSD3-BASE")]
    new_bootstrap = read(new_dir, "paired_external_bootstrap_v0.2.tsv").set_index("metric")
    f1_interval = new_bootstrap.loc["f1_difference"]
    old_gate = read(old_dir(), "hypothesis_decision_v0.1.tsv").iloc[0]
    new_gate = read(new_dir, "hypothesis_decision_v0.2.tsv").iloc[0]
    checks = pd.DataFrame(
        [
            {"check": "metric_comparison_rows", "observed": len(metrics), "expected": 8, "passed": len(metrics) == 8},
            {"check": "threshold_comparison_rows", "observed": len(thresholds), "expected": 6, "passed": len(thresholds) == 6},
            {"check": "bootstrap_comparison_rows", "observed": len(bootstrap), "expected": 3, "passed": len(bootstrap) == 3},
            {"check": "scorer_comparison_rows", "observed": len(sensitivity), "expected": 10, "passed": len(sensitivity) == 10},
            {"check": "target_count_correction", "observed": f"{old_targets}->{new_targets}", "expected": "521->482", "passed": old_targets == 521 and new_targets == 482},
            {"check": "corrected_external_reference_events", "observed": int(advanced.reference_events), "expected": 117, "passed": int(advanced.reference_events) == 117},
            {"check": "corrected_advanced_f1", "observed": float(advanced.f1), "expected": 0.22900763358778625, "passed": np.isclose(advanced.f1, 0.22900763358778625, rtol=0, atol=1e-12)},
            {"check": "corrected_baseline_f1", "observed": float(baseline.f1), "expected": 0.15789473684210525, "passed": np.isclose(baseline.f1, 0.15789473684210525, rtol=0, atol=1e-12)},
            {"check": "corrected_f1_interval_crosses_zero", "observed": bool(f1_interval.lower_95 <= 0 <= f1_interval.upper_95), "expected": True, "passed": bool(f1_interval.lower_95 <= 0 <= f1_interval.upper_95)},
            {"check": "point_gate_stable", "observed": f"{bool(old_gate.point_gate_pass)}->{bool(new_gate.point_gate_pass)}", "expected": "True->True", "passed": bool(old_gate.point_gate_pass) and bool(new_gate.point_gate_pass)},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("PSG version comparison failed")

    outputs = {
        "v0.1_v0.2_heldout_metric_comparison_v0.2.tsv": metrics,
        "v0.1_v0.2_threshold_comparison_v0.2.tsv": thresholds,
        "v0.1_v0.2_bootstrap_comparison_v0.2.tsv": bootstrap,
        "v0.1_v0.2_scorer_comparison_v0.2.tsv": sensitivity,
        "v0.1_v0.2_decision_comparison_v0.2.tsv": decisions,
        "version_comparison_checks_v0.2.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, new_dir / name)
    print(f"PSG version comparison passed {len(checks)}/{len(checks)} checks.")
    print(metrics[metrics.heldout_dataset.eq("POOLED_EXTERNAL")].to_string(index=False))


if __name__ == "__main__":
    main()
