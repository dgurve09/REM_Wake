"""Complete the validation-only Block 7 zero-shot hypothesis contrasts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import metric_values


# Section 1: frozen analysis configuration

SOURCE_EXPERIMENT = "2026-09-06_block7_transfer_validation_v0.1"
OUTPUT_EXPERIMENT = "2026-09-06_block7_zero_shot_hypothesis_analysis_v0.1"
PLAN_COMMIT = "03be56b"
INPUT_RESULT_COMMIT = "15dc9b4"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906
COMPARISONS = [
    ("P2-D", "P2-H2-Z"),
    ("H2-D", "P2-H2-Z"),
    ("P2-D", "H2-D"),
]


# Section 2: paths and immutable output helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def source_dir() -> Path:
    return repo_root() / "experiments" / SOURCE_EXPERIMENT


def output_dir() -> Path:
    return repo_root() / "experiments" / OUTPUT_EXPERIMENT


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def input_manifest() -> pd.DataFrame:
    paths = [
        source_dir() / "validation_event_metrics_v0.1.tsv",
        source_dir() / "validation_event_participants_v0.1.tsv",
        source_dir() / "paired_participant_bootstrap_v0.1.tsv",
    ]
    return pd.DataFrame(
        [
            {
                "input_role": path.stem,
                "repository_path": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "source_result_commit": INPUT_RESULT_COMMIT,
            }
            for path in paths
        ]
    )


# Section 3: frozen primary validation participant counts

def primary_participants() -> pd.DataFrame:
    rows = pd.read_csv(
        source_dir() / "validation_event_participants_v0.1.tsv", sep="\t"
    )
    result = rows[
        (rows["membership"] == "primary")
        & (rows["tolerance_sec"] == 15.0)
        & rows["comparator"].isin({name for pair in COMPARISONS for name in pair})
    ].copy()
    if result.groupby("comparator")["pid"].nunique().ne(16).any():
        raise ValueError("Expected 16 validation participants for every comparator")
    if set(result["partition"]) != {"validation"}:
        raise ValueError("This analysis is validation-only")
    return result


def aggregate_metrics(rows: pd.DataFrame) -> dict:
    return metric_values(
        int(rows["true_positive"].sum()),
        int(rows["false_positive"].sum()),
        int(rows["false_negative"].sum()),
        float(rows["supported_hours"].sum()),
    )


# Section 4: paired point estimates and participant bootstrap

def comparison_results(participants: pd.DataFrame):
    summary_rows = []
    bootstrap_rows = []
    participant_rows = []
    count_columns = [
        "pid",
        "true_positive",
        "false_positive",
        "false_negative",
        "supported_hours",
        "f1",
        "false_alarms_per_hour",
    ]

    for left_name, right_name in COMPARISONS:
        left = participants[participants["comparator"] == left_name][count_columns]
        right = participants[participants["comparator"] == right_name][count_columns]
        paired = left.merge(
            right, on="pid", suffixes=("_left", "_right"), validate="one_to_one"
        ).sort_values("pid")
        if len(paired) != 16:
            raise ValueError(f"Incomplete participant pairing for {left_name}/{right_name}")

        left_point = aggregate_metrics(
            paired.rename(
                columns={
                    "true_positive_left": "true_positive",
                    "false_positive_left": "false_positive",
                    "false_negative_left": "false_negative",
                    "supported_hours_left": "supported_hours",
                }
            )
        )
        right_point = aggregate_metrics(
            paired.rename(
                columns={
                    "true_positive_right": "true_positive",
                    "false_positive_right": "false_positive",
                    "false_negative_right": "false_negative",
                    "supported_hours_right": "supported_hours",
                }
            )
        )
        comparison = f"{left_name}_minus_{right_name}"
        summary_rows.append(
            {
                "comparison": comparison,
                "left_comparator": left_name,
                "right_comparator": right_name,
                "paired_pid": len(paired),
                "left_f1": left_point["f1"],
                "right_f1": right_point["f1"],
                "event_f1_difference": left_point["f1"] - right_point["f1"],
                "left_false_alarms_per_hour": left_point["false_alarms_per_hour"],
                "right_false_alarms_per_hour": right_point["false_alarms_per_hour"],
                "false_alarms_per_hour_difference": left_point[
                    "false_alarms_per_hour"
                ]
                - right_point["false_alarms_per_hour"],
            }
        )

        participant = paired[
            [
                "pid",
                "f1_left",
                "f1_right",
                "false_alarms_per_hour_left",
                "false_alarms_per_hour_right",
            ]
        ].copy()
        participant.insert(0, "comparison", comparison)
        participant["f1_difference"] = participant["f1_left"] - participant["f1_right"]
        participant["false_alarms_per_hour_difference"] = (
            participant["false_alarms_per_hour_left"]
            - participant["false_alarms_per_hour_right"]
        )
        participant_rows.append(participant)

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        indices = rng.integers(
            0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired))
        )
        samples = {}
        for side in ["left", "right"]:
            tp = paired[f"true_positive_{side}"].to_numpy(dtype=int)[indices].sum(axis=1)
            fp = paired[f"false_positive_{side}"].to_numpy(dtype=int)[indices].sum(axis=1)
            fn = paired[f"false_negative_{side}"].to_numpy(dtype=int)[indices].sum(axis=1)
            hours = paired[f"supported_hours_{side}"].to_numpy(dtype=float)[indices].sum(axis=1)
            samples[side] = {
                "event_f1_difference": 2 * tp / (2 * tp + fp + fn),
                "false_alarms_per_hour_difference": fp / hours,
            }
        point_values = {
            "event_f1_difference": left_point["f1"] - right_point["f1"],
            "false_alarms_per_hour_difference": left_point["false_alarms_per_hour"]
            - right_point["false_alarms_per_hour"],
        }
        for metric, point in point_values.items():
            values = samples["left"][metric] - samples["right"][metric]
            bootstrap_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "point_difference": point,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "lower_95": float(np.quantile(values, 0.025)),
                    "median": float(np.quantile(values, 0.5)),
                    "upper_95": float(np.quantile(values, 0.975)),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(bootstrap_rows),
        pd.concat(participant_rows, ignore_index=True),
    )


# Section 5: reproduction checks and interpretation

def analysis_checks(
    manifest: pd.DataFrame,
    participants: pd.DataFrame,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:
    source_bootstrap = pd.read_csv(
        source_dir() / "paired_participant_bootstrap_v0.1.tsv", sep="\t"
    )
    source = source_bootstrap[
        source_bootstrap["comparison"] == "P2-H2-Z_minus_H2-D"
    ].set_index("metric")
    control = bootstrap[
        bootstrap["comparison"] == "H2-D_minus_P2-H2-Z"
    ].set_index("metric")
    reproduction = True
    for metric in ["event_f1_difference", "false_alarms_per_hour_difference"]:
        reproduction &= np.allclose(
            [
                control.loc[metric, "point_difference"],
                control.loc[metric, "lower_95"],
                control.loc[metric, "median"],
                control.loc[metric, "upper_95"],
            ],
            [
                -source.loc[metric, "point_difference"],
                -source.loc[metric, "upper_95"],
                -source.loc[metric, "median"],
                -source.loc[metric, "lower_95"],
            ],
            atol=1e-12,
            rtol=0.0,
        )

    metrics = pd.read_csv(
        source_dir() / "validation_event_metrics_v0.1.tsv", sep="\t"
    )
    metrics = metrics[
        (metrics["membership"] == "primary")
        & (metrics["tolerance_sec"] == 15.0)
        & metrics["comparator"].isin(["P2-D", "H2-D", "P2-H2-Z"])
    ]
    threshold_map = metrics.set_index("comparator")["threshold"].to_dict()
    rows = [
        ("source_inputs_hashed", len(manifest) == 3 and manifest["sha256"].str.len().eq(64).all(), "three committed source tables"),
        ("validation_scope_only", set(participants["partition"]) == {"validation"}, "no test row"),
        ("participant_pairing", participants.groupby("comparator")["pid"].nunique().eq(16).all(), "16 pid per comparator"),
        ("thresholds_unchanged", threshold_map == {"P2-D": 0.99, "H2-D": 0.96, "P2-H2-Z": 0.99}, str(threshold_map)),
        ("comparison_set_complete", len(summary) == 3 and len(bootstrap) == 6, "three contrasts, two metrics"),
        ("source_control_reproduced", reproduction, "H2-D minus zero-shot sign-reverses committed control"),
        ("bootstrap_complete", bootstrap["resamples"].eq(BOOTSTRAP_RESAMPLES).all(), "2,000 resamples per contrast"),
    ]
    return pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in rows
        ]
    )


def write_readme(
    code_commit: str,
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    interval = bootstrap.set_index(["comparison", "metric"])
    rows = []
    for item in summary.itertuples(index=False):
        for metric in ["event_f1_difference", "false_alarms_per_hour_difference"]:
            value = getattr(item, metric)
            ci = interval.loc[(item.comparison, metric)]
            rows.append(
                f"| {item.comparison} | {metric} | {value:+.4f} | "
                f"{ci.lower_95:+.4f} to {ci.upper_95:+.4f} |"
            )
    p2_zero = summary[summary["comparison"] == "P2-D_minus_P2-H2-Z"].iloc[0]
    h2_zero = summary[summary["comparison"] == "H2-D_minus_P2-H2-Z"].iloc[0]
    p2_f1_interval = interval.loc[
        ("P2-D_minus_P2-H2-Z", "event_f1_difference")
    ]
    p2_far_interval = interval.loc[
        ("P2-D_minus_P2-H2-Z", "false_alarms_per_hour_difference")
    ]
    source_loss_supported = (
        p2_f1_interval.lower_95 > 0 and p2_far_interval.upper_95 < 0
    )
    source_interpretation = (
        "The paired intervals support both lower zero-shot F1 and higher zero-shot "
        "alarm burden relative to P2-D."
        if source_loss_supported
        else "At least one paired source-versus-zero-shot interval crosses zero, so "
        "the complete two-metric source-transfer loss remains inconclusive."
    )
    text = "\n".join(
        [
            "# Block 7 Zero-Shot Hypothesis Analysis v0.1",
            "",
            "**Work date:** 2026-09-06",
            "**Analysis status:** Post-result completion analysis",
            f"**Plan commit:** `{PLAN_COMMIT}`",
            f"**Input result commit:** `{INPUT_RESULT_COMMIT}`",
            f"**Analysis code commit:** `{code_commit}`",
            "**Input partition:** Validation only",
            "**Raw signal, feature, model, or test access:** No",
            "",
            "## Paired Results",
            "",
            "| Comparison | Metric | Point difference | Paired-bootstrap 95% interval |",
            "|---|---|---:|---:|",
            *rows,
            "",
            "## Interpretation",
            "",
            f"Relative to direct P2-D, strict zero-shot transfer changed event F1 by {-p2_zero.event_f1_difference:+.4f} and false alarms per hour by {-p2_zero.false_alarms_per_hour_difference:+.4f}. This is the direct source-to-target transfer cost under one unchanged model and threshold.",
            "",
            source_interpretation,
            "",
            f"Relative to direct H2-D, zero-shot transfer changed event F1 by {-h2_zero.event_f1_difference:+.4f} and false alarms per hour by {-h2_zero.false_alarms_per_hour_difference:+.4f}. The lower F1 occurred with fewer, not more, false alarms, so this comparison is a recall/alarm tradeoff rather than uniform degradation.",
            "",
            "Relative to direct wearable fitting, the evidence is mixed: zero-shot F1 is lower but alarm burden is also lower. The hypothesis of uniformly worse wearable event performance is therefore not supported as stated on this validation partition.",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} analysis checks passed. These are post-result validation-only contrasts and do not provide independent confirmation.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)


# Section 6: execute the fixed completion analysis

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-code-commit", required=True)
    args = parser.parse_args()
    if len(args.analysis_code_commit) < 7:
        raise ValueError("A committed analysis code hash is required")

    manifest = input_manifest()
    participants = primary_participants()
    summary, bootstrap, participant = comparison_results(participants)
    checks = analysis_checks(manifest, participants, summary, bootstrap)
    outputs = {
        "input_manifest_v0.1.tsv": manifest,
        "zero_shot_comparison_summary_v0.1.tsv": summary,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "participant_differences_v0.1.tsv": participant,
        "analysis_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    write_readme(args.analysis_code_commit, summary, bootstrap, checks)
    print(summary.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one zero-shot hypothesis-analysis check failed")


if __name__ == "__main__":
    main()
