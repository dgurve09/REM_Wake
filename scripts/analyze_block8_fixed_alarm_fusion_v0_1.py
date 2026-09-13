"""Evaluate fixed fusion rules on frozen Block 7 validation alarms."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-13_block8_fixed_alarm_fusion_v0.1"
PROTOCOL_COMMIT = "36b002c"
DIRECT = ["P6-D", "P2-D", "H2-D"]
FUSIONS = {
    "P6-H2-OR": ["P6-D", "H2-D"],
    "P6-P2-H2-OR": DIRECT,
    "P6-P2-H2-2OF3": DIRECT,
}
MIN_CONTRIBUTORS = {
    "P6-H2-OR": 1,
    "P6-P2-H2-OR": 1,
    "P6-P2-H2-2OF3": 2,
}
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
MAX_CLUSTER_SPAN_SEC = 30.0
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260913


# Section 2: paths and immutable source records

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def input_paths() -> dict[str, Path]:
    transfer = repo_root() / "experiments/2026-09-06_block7_transfer_validation_v0.1"
    return {
        "reviewed_predicted_events": transfer / "validation_predicted_events_v0.1.tsv",
        "reviewed_event_metrics": transfer / "validation_event_metrics_v0.1.tsv",
        "reviewed_support": transfer / "validation_support_v0.1.tsv",
        "transition_membership": repo_root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        "transition_quality": repo_root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for role, path in paths.items():
        rows.append(
            {
                "artifact_role": role,
                "path_relative_to_repository": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows).sort_values("artifact_role").reset_index(drop=True)


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


# Section 3: validation-only sources

def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, ...]:
    predictions = pd.read_csv(paths["reviewed_predicted_events"], sep="\t")
    metrics = pd.read_csv(paths["reviewed_event_metrics"], sep="\t")
    support_all = pd.read_csv(paths["reviewed_support"], sep="\t")
    membership = pd.read_csv(paths["transition_membership"], sep="\t")
    quality = pd.read_csv(
        paths["transition_quality"],
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )

    if set(predictions["partition"]) != {"validation"}:
        raise ValueError("Predicted-event input is not validation-only")
    if set(metrics["partition"]) != {"validation"}:
        raise ValueError("Metric input is not validation-only")
    if set(support_all["partition"]) != {"validation"}:
        raise ValueError("Support input is not validation-only")

    predictions = predictions[predictions["comparator"].isin(DIRECT)].copy()
    predictions = predictions.sort_values(
        ["comparator", "subject", "event_time_sec"], kind="stable"
    ).reset_index(drop=True)
    predictions["source_alarm_id"] = [
        f"{row.comparator}:{row.subject}:{float(row.event_time_sec):.1f}"
        for row in predictions.itertuples(index=False)
    ]
    if predictions["source_alarm_id"].duplicated().any():
        raise ValueError("Source alarm identity is not unique")

    direct_support = support_all[support_all["comparator"].isin(DIRECT)].copy()
    parity = direct_support.pivot(
        index=["subject", "pid"], columns="comparator", values="supported_hours"
    )
    if list(parity.columns.sort_values()) != sorted(DIRECT):
        raise ValueError("Direct support is incomplete")
    if not np.allclose(parity.max(axis=1), parity.min(axis=1), atol=1e-12):
        raise ValueError("Direct comparators do not share temporal support")
    support = (
        direct_support[direct_support["comparator"].eq("P6-D")]
        [["subject", "pid", "supported_hours"]]
        .sort_values("subject")
        .reset_index(drop=True)
    )

    references = membership[
        membership["partition"].eq("validation")
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    references["event_time_sec"] = references["nominal_boundary_sec"].astype(float)
    return predictions, metrics, direct_support, support, references


def reference_sets(
    references: pd.DataFrame, membership_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_column = (
        "primary_analysis_eligible"
        if membership_name == "primary"
        else "expanded_quality_analysis_eligible"
    )
    eligible = truth(references[eligible_column])
    columns = ["subject", "pid", "event_time_sec"]
    return references.loc[eligible, columns], references.loc[~eligible, columns]


# Section 4: fixed alarm clustering and fusion

def cluster_subject_alarms(subject_rows: pd.DataFrame) -> list[pd.DataFrame]:
    ordered = subject_rows.sort_values(
        ["event_time_sec", "comparator", "source_alarm_id"], kind="stable"
    ).reset_index(drop=True)
    clusters = []
    start = 0
    while start < len(ordered):
        first_time = float(ordered.iloc[start]["event_time_sec"])
        stop = start + 1
        while stop < len(ordered):
            next_time = float(ordered.iloc[stop]["event_time_sec"])
            if next_time - first_time > MAX_CLUSTER_SPAN_SEC:
                break
            stop += 1
        clusters.append(ordered.iloc[start:stop].copy())
        start = stop
    return clusters


def representative_time(times: np.ndarray) -> float:
    median = float(np.median(times))
    return float(min(times, key=lambda value: (abs(float(value) - median), float(value))))


def build_fusion_clusters(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    comparator_order = {name: index for index, name in enumerate(DIRECT)}
    for fusion, sources in FUSIONS.items():
        local = predictions[predictions["comparator"].isin(sources)]
        for subject, subject_rows in local.groupby("subject", sort=True):
            pid_values = subject_rows["pid"].unique()
            if len(pid_values) != 1:
                raise ValueError(f"Multiple pid values for {subject}")
            for cluster_index, cluster in enumerate(cluster_subject_alarms(subject_rows)):
                times = cluster["event_time_sec"].to_numpy(dtype=float)
                contributors = sorted(
                    cluster["comparator"].unique(), key=comparator_order.get
                )
                retained = len(contributors) >= MIN_CONTRIBUTORS[fusion]
                rows.append(
                    {
                        "fusion": fusion,
                        "partition": "validation",
                        "subject": subject,
                        "pid": int(pid_values[0]),
                        "cluster_index": cluster_index,
                        "event_time_sec": representative_time(times),
                        "cluster_start_sec": float(times.min()),
                        "cluster_stop_sec": float(times.max()),
                        "cluster_span_sec": float(times.max() - times.min()),
                        "contributor_count": len(contributors),
                        "contributors": ";".join(contributors),
                        "source_alarm_count": len(cluster),
                        "source_alarm_ids": ";".join(cluster["source_alarm_id"]),
                        "retained": retained,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["fusion", "subject", "cluster_index"], kind="stable"
    )


def retained_fused_events(clusters: pd.DataFrame) -> pd.DataFrame:
    return clusters[truth(clusters["retained"])][
        [
            "fusion",
            "partition",
            "subject",
            "pid",
            "event_time_sec",
            "contributors",
            "contributor_count",
            "source_alarm_count",
            "cluster_start_sec",
            "cluster_stop_sec",
        ]
    ].rename(columns={"fusion": "method"})


# Section 5: event evaluation

def evaluate_methods(
    predictions: pd.DataFrame,
    fused: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    methods = DIRECT + list(FUSIONS)
    metric_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for method in methods:
        if method in DIRECT:
            method_events = predictions[predictions["comparator"].eq(method)][
                ["subject", "pid", "event_time_sec"]
            ]
            role = "frozen_direct_control"
        else:
            method_events = fused[fused["method"].eq(method)][
                ["subject", "pid", "event_time_sec"]
            ]
            role = "fixed_alarm_fusion"

        for membership_name in MEMBERSHIPS:
            eligible, ignored = reference_sets(references, membership_name)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible,
                    method_events,
                    ignored,
                    support,
                    tolerance,
                )
                prefix = {
                    "method": method,
                    "method_role": role,
                    "partition": "validation",
                    "membership": membership_name,
                    "tolerance_sec": tolerance,
                }
                metric_rows.append({**prefix, **summary})
                for frame, target in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    for row in frame.to_dict("records"):
                        target.append({**prefix, **row})

    return {
        "metrics": pd.DataFrame(metric_rows),
        "recordings": pd.DataFrame(recording_rows),
        "participants": pd.DataFrame(participant_rows),
        "matches": pd.DataFrame(match_rows),
    }


def comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    controls = metrics[metrics["method"].eq("P6-D")].set_index(
        ["membership", "tolerance_sec"]
    )
    rows = []
    for row in metrics[metrics["method"].isin(FUSIONS)].itertuples(index=False):
        control = controls.loc[(row.membership, row.tolerance_sec)]
        rows.append(
            {
                "fusion": row.method,
                "membership": row.membership,
                "tolerance_sec": row.tolerance_sec,
                "f1": row.f1,
                "false_alarms_per_hour": row.false_alarms_per_hour,
                "precision": row.precision,
                "recall": row.recall,
                "p6_d_f1": control.f1,
                "p6_d_false_alarms_per_hour": control.false_alarms_per_hour,
                "f1_difference": row.f1 - control.f1,
                "false_alarms_per_hour_difference": row.false_alarms_per_hour
                - control.false_alarms_per_hour,
            }
        )
    return pd.DataFrame(rows)


# Section 6: participant-grouped paired contrasts

def aggregate_participants(frame: pd.DataFrame, weights: pd.Series) -> dict:
    local = frame.set_index("pid").loc[weights.index]
    return metric_values(
        int(np.dot(local["true_positive"], weights)),
        int(np.dot(local["false_positive"], weights)),
        int(np.dot(local["false_negative"], weights)),
        float(np.dot(local["supported_hours"], weights)),
    )


def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pids = np.sort(participants["pid"].unique())

    def run_contrast(
        comparison: str,
        first: pd.DataFrame,
        second: pd.DataFrame,
        metric: str,
    ) -> None:
        first = first.sort_values("pid")
        second = second.sort_values("pid")
        if not np.array_equal(first["pid"], second["pid"]):
            raise ValueError(f"Participant mismatch for {comparison}")
        unit_weights = pd.Series(1, index=pids)
        point = aggregate_participants(first, unit_weights)[metric] - aggregate_participants(
            second, unit_weights
        )[metric]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        values = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            selected = rng.choice(pids, size=len(pids), replace=True)
            weights = pd.Series(selected).value_counts().reindex(pids, fill_value=0)
            value = aggregate_participants(first, weights)[metric] - aggregate_participants(
                second, weights
            )[metric]
            values.append(value)
        values = np.asarray(values, dtype=float)
        rows.append(
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

    for tolerance in TOLERANCES:
        p6 = participants[
            participants["method"].eq("P6-D")
            & participants["membership"].eq("primary")
            & participants["tolerance_sec"].eq(tolerance)
        ]
        for fusion in FUSIONS:
            local = participants[
                participants["method"].eq(fusion)
                & participants["membership"].eq("primary")
                & participants["tolerance_sec"].eq(tolerance)
            ]
            for metric in ["f1", "false_alarms_per_hour"]:
                run_contrast(
                    f"{fusion}_minus_P6-D_at_{int(tolerance)}sec",
                    local,
                    p6,
                    metric,
                )

    consensus15 = participants[
        participants["method"].eq("P6-P2-H2-2OF3")
        & participants["membership"].eq("primary")
        & participants["tolerance_sec"].eq(15.0)
    ]
    consensus45 = participants[
        participants["method"].eq("P6-P2-H2-2OF3")
        & participants["membership"].eq("primary")
        & participants["tolerance_sec"].eq(45.0)
    ]
    run_contrast(
        "P6-P2-H2-2OF3_45sec_minus_15sec",
        consensus45,
        consensus15,
        "f1",
    )
    return pd.DataFrame(rows)


# Section 7: fixed decisions and integrity checks

def primary_row(metrics: pd.DataFrame, method: str, tolerance: float = 15.0) -> pd.Series:
    return metrics[
        metrics["method"].eq(method)
        & metrics["membership"].eq("primary")
        & metrics["tolerance_sec"].eq(tolerance)
    ].iloc[0]


def hypothesis_decisions(metrics: pd.DataFrame) -> pd.DataFrame:
    p6 = primary_row(metrics, "P6-D")
    pair_or = primary_row(metrics, "P6-H2-OR")
    all_or = primary_row(metrics, "P6-P2-H2-OR")
    consensus15 = primary_row(metrics, "P6-P2-H2-2OF3")
    consensus45 = primary_row(metrics, "P6-P2-H2-2OF3", 45.0)
    tests = [
        {
            "hypothesis": "H8.14_two_modality_or_advancement",
            "value_1": pair_or.f1 - p6.f1,
            "criterion_1": ">=0.05_f1",
            "value_2": pair_or.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_2": "<=0.25_far_per_hour",
            "supported": (pair_or.f1 - p6.f1 >= 0.05)
            and (
                pair_or.false_alarms_per_hour - p6.false_alarms_per_hour <= 0.25
            ),
        },
        {
            "hypothesis": "H8.15_consensus_advancement",
            "value_1": consensus15.f1 - p6.f1,
            "criterion_1": ">=0_f1",
            "value_2": consensus15.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_2": "<=0_far_per_hour",
            "supported": (consensus15.f1 - p6.f1 >= 0)
            and (
                consensus15.false_alarms_per_hour - p6.false_alarms_per_hour <= 0
            ),
        },
        {
            "hypothesis": "H8.16_consensus_boundary_sensitivity",
            "value_1": consensus45.f1 - consensus15.f1,
            "criterion_1": ">=0.05_f1",
            "value_2": np.nan,
            "criterion_2": "not_applicable",
            "supported": consensus45.f1 - consensus15.f1 >= 0.05,
        },
        {
            "hypothesis": "H8.17_all_direct_or_alarm_penalty",
            "value_1": all_or.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_1": ">=0.50_far_per_hour",
            "value_2": np.nan,
            "criterion_2": "not_applicable",
            "supported": all_or.false_alarms_per_hour
            - p6.false_alarms_per_hour
            >= 0.50,
        },
    ]
    result = pd.DataFrame(tests)
    result["decision"] = result["supported"].map({True: "pass", False: "fail"})
    fusion_advance = bool(
        result.loc[
            result["hypothesis"].isin(
                [
                    "H8.14_two_modality_or_advancement",
                    "H8.15_consensus_advancement",
                ]
            ),
            "supported",
        ].any()
    )
    result = pd.concat(
        [
            result,
            pd.DataFrame(
                [
                    {
                        "hypothesis": "overall_fixed_fusion_advance",
                        "value_1": float(fusion_advance),
                        "criterion_1": "H8.14_or_H8.15",
                        "value_2": np.nan,
                        "criterion_2": "not_applicable",
                        "supported": fusion_advance,
                        "decision": "advance" if fusion_advance else "stop",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return result


def integrity_checks(
    predictions: pd.DataFrame,
    frozen_metrics: pd.DataFrame,
    direct_support: pd.DataFrame,
    support: pd.DataFrame,
    clusters: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    bootstrap: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    metrics = outputs["metrics"]
    direct_rows = metrics[metrics["method"].isin(DIRECT)].copy()
    source_rows = frozen_metrics[
        frozen_metrics["comparator"].isin(DIRECT)
    ].copy()
    merge = direct_rows.merge(
        source_rows,
        left_on=["method", "membership", "tolerance_sec"],
        right_on=["comparator", "membership", "tolerance_sec"],
        suffixes=("_new", "_source"),
        validate="one_to_one",
    )
    metric_columns = [
        "reference_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "false_negative",
        "ignored_predictions",
        "supported_hours",
        "precision",
        "recall",
        "f1",
        "false_alarms_per_hour",
        "median_absolute_error_sec",
        "maximum_absolute_error_sec",
    ]
    reproduced = all(
        np.allclose(
            merge[f"{name}_new"],
            merge[f"{name}_source"],
            atol=1e-12,
            rtol=1e-12,
            equal_nan=True,
        )
        for name in metric_columns
    )

    source_accounting = True
    for fusion, sources in FUSIONS.items():
        expected = len(predictions[predictions["comparator"].isin(sources)])
        observed = int(clusters.loc[clusters["fusion"].eq(fusion), "source_alarm_count"].sum())
        source_accounting &= expected == observed

    consensus = clusters[clusters["fusion"].eq("P6-P2-H2-2OF3")]
    consensus_rule = (
        truth(consensus["retained"]) == (consensus["contributor_count"] >= 2)
    ).all()
    recordings = outputs["recordings"]
    event_accounting = (
        recordings["true_positive"] + recordings["false_negative"]
        == recordings["reference_events"]
    ).all() and (
        recordings["true_positive"]
        + recordings["false_positive"]
        + recordings["ignored_predictions"]
        == recordings["predicted_events"]
    ).all()
    support_parity = direct_support.pivot(
        index=["subject", "pid"], columns="comparator", values="supported_hours"
    )
    no_test_path = ~manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    rows = [
        ("validation_membership", len(support) == 20 and support["pid"].nunique() == 16, "20 recordings; 16 pid groups"),
        ("identical_direct_support", np.allclose(support_parity.max(axis=1), support_parity.min(axis=1), atol=1e-12), "same supported hours for all direct comparators"),
        ("frozen_direct_metrics_reproduced", reproduced and len(merge) == 12, "all 12 direct-model metric rows"),
        ("source_alarm_accounting", source_accounting, "all source alarms enter each applicable clustering once"),
        ("cluster_span", bool((clusters["cluster_span_sec"] <= MAX_CLUSTER_SPAN_SEC).all()), "all cluster spans at most 30 seconds"),
        ("consensus_rule", bool(consensus_rule), "retained exactly when at least two models contribute"),
        ("event_accounting", bool(event_accounting), "TP/FN and TP/FP/ignored identities hold"),
        ("participant_bootstrap", len(bootstrap) == 13 and (bootstrap["resamples"] == BOOTSTRAP_RESAMPLES).all(), "13 paired contrasts; 5000 resamples"),
        ("source_paths_exclude_test", bool(no_test_path), "no test path in source manifest"),
        ("validation_rows_only", set(clusters["partition"]) == {"validation"}, "all fusion clusters restricted to validation"),
    ]
    result = pd.DataFrame(rows, columns=["check", "status", "detail"])
    result["status"] = result["status"].map({True: "pass", False: "fail"})
    return result


# Section 8: reviewed output summary

def readme(metrics: pd.DataFrame, decisions: pd.DataFrame, checks: pd.DataFrame) -> str:
    rows = {
        method: primary_row(metrics, method)
        for method in ["P6-D", "H2-D", *FUSIONS]
    }
    decision_lines = "\n".join(
        f"- `{row.hypothesis}`: {row.decision}"
        for row in decisions.itertuples(index=False)
    )
    metric_lines = "\n".join(
        f"| `{name}` | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        for name, row in rows.items()
    )
    return f"""# Block 8 Fixed Alarm Fusion v0.1

**Work date:** 2026-09-13
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Partition:** Reused development validation only
**Model fitting or threshold selection:** None
**Test data accessed:** No

## Primary +/-15-Second Results

| Method | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|
{metric_lines}

## Frozen Decisions

{decision_lines}

## Boundary

The fusion rules operate on already frozen alarms. They are diagnostic combinations, not independently validated detectors. A result can explain whether complementary reference recovery survives false-alarm accounting, but it cannot authorize deployment or test access.

All {(checks['status'] == 'pass').sum()}/{len(checks)} in-run checks passed.
"""


def main() -> None:
    paths = input_paths()
    manifest = source_manifest(paths)
    predictions, frozen_metrics, direct_support, support, references = load_inputs(paths)
    clusters = build_fusion_clusters(predictions)
    fused = retained_fused_events(clusters)
    outputs = evaluate_methods(predictions, fused, support, references)
    comparison = comparisons(outputs["metrics"])
    bootstrap = paired_bootstrap(outputs["participants"])
    decisions = hypothesis_decisions(outputs["metrics"])
    checks = integrity_checks(
        predictions,
        frozen_metrics,
        direct_support,
        support,
        clusters,
        outputs,
        bootstrap,
        manifest,
    )
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"In-run checks failed: {failed}")

    reviewed = {
        "input_artifact_manifest_v0.1.tsv": manifest,
        "fusion_clusters_v0.1.tsv": clusters,
        "fused_predicted_events_v0.1.tsv": fused,
        "event_metrics_v0.1.tsv": outputs["metrics"],
        "event_recordings_v0.1.tsv": outputs["recordings"],
        "event_participants_v0.1.tsv": outputs["participants"],
        "event_matches_v0.1.tsv": outputs["matches"],
        "fusion_comparisons_v0.1.tsv": comparison,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decisions,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    verify_or_create_text(
        output_dir() / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )
    verify_or_create_text(
        output_dir() / "README.md",
        readme(outputs["metrics"], decisions, checks),
    )

    for method in FUSIONS:
        row = primary_row(outputs["metrics"], method)
        print(
            f"{method}: F1={row.f1:.6f}; recall={row.recall:.6f}; "
            f"FAR/h={row.false_alarms_per_hour:.6f}"
        )
    print(f"All {len(checks)}/{len(checks)} in-run checks passed")


if __name__ == "__main__":
    main()
