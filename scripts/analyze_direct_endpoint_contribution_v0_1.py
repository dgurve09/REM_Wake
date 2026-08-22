"""Compare DE-D product scoring with its two endpoint heads on validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_direct_event_failure_modes_v0_1 import add_human_stage_pair
from run_direct_event_baseline_v0_1 import (
    THRESHOLDS,
    collapse_alarms,
    data_parent,
    evaluate_events,
    feature_path,
    local_event_inputs,
    reference_events,
    repo_root,
)


# Section 1: fixed analysis configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-08-22_direct_endpoint_contribution_analysis_v0.1"
FACTORIZATION_DIR = "2026-08-22_direct_endpoint_factorization_v0.1"
DERIVED_DIR = "direct_endpoint_factorization_v0.1"
COMPARATORS = {
    "DE-D-rem-only": "probability_rem_before",
    "DE-D-wake-only": "probability_wake_after",
    "DE-D-product": "probability",
}
STAGE_PAIR_CATEGORIES = [
    "human_REM_to_Wake",
    "human_REM_to_other",
    "human_other_to_Wake",
    "human_no_stage_change",
    "human_other_transition",
]


# Section 2: paths and hashes

def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def factorization_dir() -> Path:
    return repo_root() / "experiments" / FACTORIZATION_DIR


def candidate_score_path() -> Path:
    return (
        data_parent()
        / "derived"
        / DERIVED_DIR
        / "candidate_scores"
        / "validation_candidate_scores_v0.1.tsv.gz"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Section 3: load the frozen validation scores

def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(candidate_score_path(), sep="\t", compression="gzip")
    required = {
        "comparator",
        "model_version",
        "partition",
        "subject",
        "pid",
        "candidate_time_sec",
        "probability_rem_before",
        "probability_wake_after",
        "probability",
    }
    if not required.issubset(scores.columns):
        raise ValueError("Candidate scores have an unexpected schema")
    if set(scores["partition"]) != {"validation"}:
        raise ValueError("Candidate scores must contain validation rows only")
    if set(scores["comparator"]) != {"DE-D"}:
        raise ValueError("Expected only the frozen DE-D candidate scores")
    if scores.duplicated(["subject", "candidate_time_sec"]).any():
        raise ValueError("Duplicate validation candidate boundary")

    probability_columns = list(COMPARATORS.values())
    values = scores[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Invalid endpoint probability")
    product = (
        scores["probability_rem_before"].to_numpy(dtype=float)
        * scores["probability_wake_after"].to_numpy(dtype=float)
    )
    if not np.allclose(product, scores["probability"], atol=1e-12, rtol=0):
        raise ValueError("Saved product score does not equal the two endpoint scores")

    support = pd.read_csv(
        factorization_dir() / "validation_event_support_v0.1.tsv", sep="\t"
    )
    if set(support["partition"]) != {"validation"}:
        raise ValueError("Support must contain validation rows only")
    if int(support["supported_boundaries"].sum()) != len(scores):
        raise ValueError("Candidate-score count does not match saved support")
    return scores, support


def comparator_scores(scores: pd.DataFrame) -> pd.DataFrame:
    columns = ["partition", "subject", "pid", "candidate_time_sec"]
    rows = []
    for comparator, score_column in COMPARATORS.items():
        local = scores[columns].copy()
        local.insert(0, "comparator", comparator)
        local.insert(1, "model_version", VERSION)
        local["probability"] = scores[score_column].to_numpy(dtype=float)
        rows.append(local)
    return pd.concat(rows, ignore_index=True)


# Section 4: fixed threshold search and event evaluation

def threshold_analysis(
    scores: pd.DataFrame, support: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    references = reference_events()
    eligible, ignored = local_event_inputs(references, "validation", "primary")
    curve_rows = []
    selected_rows = []

    for comparator in COMPARATORS:
        local_scores = scores[scores["comparator"] == comparator]
        local_support = support[["subject", "pid", "supported_hours"]]
        comparator_rows = []
        for threshold in THRESHOLDS:
            alarms = collapse_alarms(local_scores, float(threshold))
            _, _, _, summary = evaluate_events(
                eligible,
                alarms[["subject", "pid", "event_time_sec"]],
                ignored,
                local_support,
                15.0,
            )
            row = {
                "comparator": comparator,
                "analysis_version": VERSION,
                "partition": "validation",
                "membership": "primary",
                "tolerance_sec": 15.0,
                "threshold": float(threshold),
                **summary,
            }
            comparator_rows.append(row)
            curve_rows.append(row)

        selected = pd.DataFrame(comparator_rows).sort_values(
            ["f1", "false_alarms_per_hour", "recall", "threshold"],
            ascending=[False, True, False, False],
            kind="stable",
        ).iloc[0]
        selected_rows.append(selected.to_dict())

    return pd.DataFrame(curve_rows), pd.DataFrame(selected_rows)


def verify_product_control(selected: pd.DataFrame) -> None:
    observed = selected[selected["comparator"] == "DE-D-product"].iloc[0]
    saved = pd.read_csv(
        factorization_dir() / "selected_threshold_v0.1.tsv", sep="\t"
    ).iloc[0]
    exact_columns = [
        "reference_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "false_negative",
        "ignored_predictions",
    ]
    for column in exact_columns:
        if int(observed[column]) != int(saved[column]):
            raise ValueError(f"Product control failed for {column}")
    numeric_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
    ]
    for column in numeric_columns:
        if not np.isclose(float(observed[column]), float(saved[column]), atol=1e-12):
            raise ValueError(f"Product control failed for {column}")


# Section 5: selected false-positive mechanisms

def selected_false_positive_categories(
    scores: pd.DataFrame, support: pd.DataFrame, selected: pd.DataFrame
) -> pd.DataFrame:
    references = reference_events()
    eligible, ignored = local_event_inputs(references, "validation", "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    rows = []

    for item in selected.itertuples(index=False):
        local_scores = scores[scores["comparator"] == item.comparator]
        alarms = collapse_alarms(local_scores, float(item.threshold))
        _, _, matches, summary = evaluate_events(
            eligible,
            alarms[["subject", "pid", "event_time_sec"]],
            ignored,
            local_support,
            15.0,
        )
        matched = {
            (match.subject, round(float(match.prediction_time_sec), 6))
            for match in matches.itertuples(index=False)
        }
        false_positives = alarms[
            [
                (alarm.subject, round(float(alarm.event_time_sec), 6))
                not in matched
                for alarm in alarms.itertuples(index=False)
            ]
        ].copy()
        if len(false_positives) != int(summary["false_positive"]):
            raise ValueError(f"False-positive accounting failed for {item.comparator}")
        context = add_human_stage_pair(false_positives)
        counts = context.groupby("human_stage_pair_category").size()
        for category in STAGE_PAIR_CATEGORIES:
            count = int(counts.get(category, 0))
            rows.append(
                {
                    "comparator": item.comparator,
                    "analysis_version": VERSION,
                    "threshold": float(item.threshold),
                    "human_stage_pair_category": category,
                    "false_positive_alarms": count,
                    "false_positive_share": (
                        count / len(false_positives) if len(false_positives) else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


# Section 6: decision and reproducibility records

def mechanism_decision(
    selected: pd.DataFrame, categories: pd.DataFrame
) -> pd.DataFrame:
    metrics = selected.set_index("comparator")
    product = metrics.loc["DE-D-product"]
    single = metrics.loc[["DE-D-rem-only", "DE-D-wake-only"]]
    higher_f1 = bool((float(product.f1) > single["f1"].astype(float)).all())
    lower_far = bool(
        (
            float(product.false_alarms_per_hour)
            < single["false_alarms_per_hour"].astype(float)
        ).all()
    )
    dominates_product = single[
        (single["f1"].astype(float) >= float(product.f1))
        & (
            single["false_alarms_per_hour"].astype(float)
            <= float(product.false_alarms_per_hour)
        )
    ].index.tolist()

    category_counts = categories.pivot(
        index="comparator",
        columns="human_stage_pair_category",
        values="false_positive_alarms",
    )
    rem_mechanism = bool(
        category_counts.loc["DE-D-rem-only", "human_REM_to_other"]
        > category_counts.loc["DE-D-product", "human_REM_to_other"]
    )
    wake_mechanism = bool(
        category_counts.loc["DE-D-wake-only", "human_other_to_Wake"]
        > category_counts.loc["DE-D-product", "human_other_to_Wake"]
    )

    if higher_f1 and lower_far:
        outcome = "both_endpoint_contribution_supported"
    elif dominates_product:
        outcome = "explicit_conjunction_not_supported"
    else:
        outcome = "endpoint_contribution_inconclusive"
    return pd.DataFrame(
        [
            {
                "analysis_version": VERSION,
                "product_higher_f1_than_both_heads": higher_f1,
                "product_lower_far_than_both_heads": lower_far,
                "single_head_dominating_product": ";".join(dominates_product),
                "rem_only_has_more_rem_to_other_fp": rem_mechanism,
                "wake_only_has_more_other_to_wake_fp": wake_mechanism,
                "decision": outcome,
            }
        ]
    )


def input_manifest(scores: pd.DataFrame) -> pd.DataFrame:
    repository_files = [
        ("preanalysis_plan", repo_root() / "docs/evaluation/direct_endpoint_contribution_plan_v0.1.md"),
        ("analysis_script", Path(__file__).resolve()),
        ("validation_support", factorization_dir() / "validation_event_support_v0.1.tsv"),
        (
            "transition_membership",
            repo_root()
            / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        ),
        (
            "transition_boundary_times",
            repo_root()
            / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        ),
    ]
    data_files = [("validation_candidate_scores", candidate_score_path())]
    data_files.extend(
        ("validation_onset_stage_source", feature_path(subject))
        for subject in sorted(scores["subject"].unique())
    )

    rows = []
    for role, path in repository_files:
        rows.append(
            {
                "scope": "repository",
                "artifact_role": role,
                "relative_path": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    for role, path in data_files:
        rows.append(
            {
                "scope": "data_parent",
                "artifact_role": role,
                "relative_path": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["scope", "artifact_role", "relative_path"])


def integrity_checks(
    scores: pd.DataFrame,
    curve: pd.DataFrame,
    selected: pd.DataFrame,
    categories: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    checks = [
        ("validation_scores_only", set(scores["partition"]) == {"validation"}, f"rows={len(scores)}"),
        ("three_fixed_comparators", set(selected["comparator"]) == set(COMPARATORS), f"comparators={len(selected)}"),
        ("complete_threshold_grid", len(curve) == len(COMPARATORS) * len(THRESHOLDS), f"rows={len(curve)}"),
        ("product_control_reproduced", True, "saved DE-D threshold and metrics reproduced"),
        (
            "event_accounting",
            bool((selected["reference_events"] == selected["true_positive"] + selected["false_negative"]).all()),
            "reference_events=TP+FN",
        ),
        (
            "false_positive_category_accounting",
            bool(
                categories.groupby("comparator")["false_positive_alarms"].sum().astype(int).to_dict()
                == selected.set_index("comparator")["false_positive"].astype(int).to_dict()
            ),
            "category totals equal selected false positives",
        ),
        (
            "no_test_model_train_or_raw_input",
            not manifest["artifact_role"].str.contains("test|model|train|raw", case=False, regex=True).any(),
            f"input_rows={len(manifest)}",
        ),
    ]
    result = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    if not result["passed"].all():
        failed = result.loc[~result["passed"], "check"].tolist()
        raise ValueError(f"Integrity checks failed: {failed}")
    return result


def write_readme(selected: pd.DataFrame, decision: pd.DataFrame) -> None:
    metrics = selected.set_index("comparator")
    result = decision.iloc[0]
    lines = [
        "# Direct Endpoint Contribution Analysis v0.1",
        "",
        "**Created:** 2026-08-22  ",
        "**Status:** Completed post-result validation diagnostic  ",
        "**Protocol:** `docs/evaluation/direct_endpoint_contribution_plan_v0.1.md`  ",
        "**Repository base:** `ab76720`  ",
        "**Test access:** None",
        "",
        "## Selected Validation Results",
        "",
        "| Comparator | Threshold | Precision | Recall | Event F1 | False alarms/hour |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for comparator in COMPARATORS:
        row = metrics.loc[comparator]
        lines.append(
            f"| {comparator} | {row.threshold:.2f} | {row.precision:.4f} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Decision: **{result.decision}**.",
            "",
            f"The product had higher F1 than both heads: **{bool(result.product_higher_f1_than_both_heads)}**. "
            f"It had a lower false-alarm rate than both heads: **{bool(result.product_lower_far_than_both_heads)}**.",
            "",
            "The product control reproduced the previously saved DE-D threshold and event metrics exactly. "
            "This analysis used saved validation probabilities only and did not load fitted models, raw EEG, train rows, or test artifacts.",
            "",
            "## Interpretation Boundary",
            "",
            "The result describes mechanism on the already used validation partition. It does not update the frozen test result or establish new-cohort performance. DE-D still requires a new locked or external evaluation.",
        ]
    )
    output_dir().joinpath("README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# Section 7: run the analysis

def main() -> None:
    frozen_scores, support = load_inputs()
    scores = comparator_scores(frozen_scores)
    curve, selected = threshold_analysis(scores, support)
    verify_product_control(selected)
    categories = selected_false_positive_categories(scores, support, selected)
    decision = mechanism_decision(selected, categories)
    manifest = input_manifest(frozen_scores)
    checks = integrity_checks(frozen_scores, curve, selected, categories, manifest)

    output_dir().mkdir(parents=True, exist_ok=False)
    curve.to_csv(output_dir() / "validation_threshold_curve_v0.1.tsv", sep="\t", index=False)
    selected.to_csv(output_dir() / "selected_event_metrics_v0.1.tsv", sep="\t", index=False)
    categories.to_csv(output_dir() / "false_positive_category_summary_v0.1.tsv", sep="\t", index=False)
    decision.to_csv(output_dir() / "mechanism_decision_v0.1.tsv", sep="\t", index=False)
    manifest.to_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t", index=False)
    checks.to_csv(output_dir() / "output_integrity_checks_v0.1.tsv", sep="\t", index=False)
    write_readme(selected, decision)

    print(selected[["comparator", "threshold", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(decision.to_string(index=False))
    print(f"Passed {len(checks)}/{len(checks)} in-run integrity checks")


if __name__ == "__main__":
    main()
