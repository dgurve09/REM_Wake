"""Independently reconstruct Block 8 fixed alarm-fusion outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: independent fixed configuration

EXPERIMENT_DIR = "2026-09-13_block8_fixed_alarm_fusion_v0.1"
DIRECT = ["P6-D", "P2-D", "H2-D"]
FUSION_SOURCES = {
    "P6-H2-OR": ["P6-D", "H2-D"],
    "P6-P2-H2-OR": DIRECT,
    "P6-P2-H2-2OF3": DIRECT,
}
MINIMUM_MODELS = {
    "P6-H2-OR": 1,
    "P6-P2-H2-OR": 1,
    "P6-P2-H2-2OF3": 2,
}
RESAMPLES = 5000
SEED = 20260913


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def source_paths() -> dict[str, Path]:
    transfer = root() / "experiments/2026-09-06_block7_transfer_validation_v0.1"
    return {
        "reviewed_predicted_events": transfer / "validation_predicted_events_v0.1.tsv",
        "reviewed_event_metrics": transfer / "validation_event_metrics_v0.1.tsv",
        "reviewed_support": transfer / "validation_support_v0.1.tsv",
        "transition_membership": root()
        / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        "transition_quality": root()
        / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


# Section 2: independent source and fusion reconstruction

def load_sources() -> tuple[pd.DataFrame, ...]:
    paths = source_paths()
    events = pd.read_csv(paths["reviewed_predicted_events"], sep="\t")
    frozen_metrics = pd.read_csv(paths["reviewed_event_metrics"], sep="\t")
    all_support = pd.read_csv(paths["reviewed_support"], sep="\t")
    membership = pd.read_csv(paths["transition_membership"], sep="\t")
    quality = pd.read_csv(
        paths["transition_quality"],
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    events = events[events["comparator"].isin(DIRECT)].sort_values(
        ["comparator", "subject", "event_time_sec"], kind="stable"
    ).reset_index(drop=True)
    events["source_alarm_id"] = [
        f"{row.comparator}:{row.subject}:{float(row.event_time_sec):.1f}"
        for row in events.itertuples(index=False)
    ]
    direct_support = all_support[all_support["comparator"].isin(DIRECT)].copy()
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
    return events, frozen_metrics, direct_support, support, references


def choose_cluster_time(values: np.ndarray) -> float:
    middle = float(np.median(values))
    return float(min(values, key=lambda value: (abs(float(value) - middle), float(value))))


def reconstruct_clusters(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = {name: index for index, name in enumerate(DIRECT)}
    for fusion, sources in FUSION_SOURCES.items():
        local = events[events["comparator"].isin(sources)]
        for subject, group in local.groupby("subject", sort=True):
            group = group.sort_values(
                ["event_time_sec", "comparator", "source_alarm_id"], kind="stable"
            ).reset_index(drop=True)
            cluster_index = 0
            start = 0
            while start < len(group):
                first = float(group.iloc[start]["event_time_sec"])
                stop = start + 1
                while stop < len(group):
                    if float(group.iloc[stop]["event_time_sec"]) - first > 30.0:
                        break
                    stop += 1
                cluster = group.iloc[start:stop]
                times = cluster["event_time_sec"].to_numpy(dtype=float)
                contributors = sorted(
                    cluster["comparator"].unique(), key=order.get
                )
                rows.append(
                    {
                        "fusion": fusion,
                        "partition": "validation",
                        "subject": subject,
                        "pid": int(cluster["pid"].iloc[0]),
                        "cluster_index": cluster_index,
                        "event_time_sec": choose_cluster_time(times),
                        "cluster_start_sec": float(times.min()),
                        "cluster_stop_sec": float(times.max()),
                        "cluster_span_sec": float(times.max() - times.min()),
                        "contributor_count": len(contributors),
                        "contributors": ";".join(contributors),
                        "source_alarm_count": len(cluster),
                        "source_alarm_ids": ";".join(cluster["source_alarm_id"]),
                        "retained": len(contributors) >= MINIMUM_MODELS[fusion],
                    }
                )
                cluster_index += 1
                start = stop
    return pd.DataFrame(rows).sort_values(
        ["fusion", "subject", "cluster_index"], kind="stable"
    )


def reconstruct_fused(clusters: pd.DataFrame) -> pd.DataFrame:
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


# Section 3: independent event reconstruction

def split_references(
    references: pd.DataFrame, membership_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = (
        "primary_analysis_eligible"
        if membership_name == "primary"
        else "expanded_quality_analysis_eligible"
    )
    eligible = truth(references[column])
    keep = ["subject", "pid", "event_time_sec"]
    return references.loc[eligible, keep], references.loc[~eligible, keep]


def reconstruct_events(
    source_events: pd.DataFrame,
    fused: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    metrics_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for method in DIRECT + list(FUSION_SOURCES):
        if method in DIRECT:
            predictions = source_events[source_events["comparator"].eq(method)][
                ["subject", "pid", "event_time_sec"]
            ]
            role = "frozen_direct_control"
        else:
            predictions = fused[fused["method"].eq(method)][
                ["subject", "pid", "event_time_sec"]
            ]
            role = "fixed_alarm_fusion"
        for membership_name in ["primary", "expanded"]:
            eligible, ignored = split_references(references, membership_name)
            for tolerance in [15.0, 45.0]:
                recordings, participants, matches, summary = evaluate_events(
                    eligible, predictions, ignored, support, tolerance
                )
                prefix = {
                    "method": method,
                    "method_role": role,
                    "partition": "validation",
                    "membership": membership_name,
                    "tolerance_sec": tolerance,
                }
                metrics_rows.append({**prefix, **summary})
                for frame, destination in [
                    (recordings, recording_rows),
                    (participants, participant_rows),
                    (matches, match_rows),
                ]:
                    destination.extend(
                        [{**prefix, **row} for row in frame.to_dict("records")]
                    )
    return {
        "event_metrics_v0.1.tsv": pd.DataFrame(metrics_rows),
        "event_recordings_v0.1.tsv": pd.DataFrame(recording_rows),
        "event_participants_v0.1.tsv": pd.DataFrame(participant_rows),
        "event_matches_v0.1.tsv": pd.DataFrame(match_rows),
    }


def reconstruct_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    controls = metrics[metrics["method"].eq("P6-D")].set_index(
        ["membership", "tolerance_sec"]
    )
    rows = []
    for row in metrics[metrics["method"].isin(FUSION_SOURCES)].itertuples(index=False):
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


# Section 4: independent paired bootstrap

def aggregate(frame: pd.DataFrame, weights: pd.Series) -> dict:
    local = frame.set_index("pid").loc[weights.index]
    return metric_values(
        int(np.dot(local["true_positive"], weights)),
        int(np.dot(local["false_positive"], weights)),
        int(np.dot(local["false_negative"], weights)),
        float(np.dot(local["supported_hours"], weights)),
    )


def reconstruct_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pids = np.sort(participants["pid"].unique())

    def contrast(label: str, first: pd.DataFrame, second: pd.DataFrame, metric: str) -> None:
        first = first.sort_values("pid")
        second = second.sort_values("pid")
        ones = pd.Series(1, index=pids)
        point = aggregate(first, ones)[metric] - aggregate(second, ones)[metric]
        generator = np.random.default_rng(SEED)
        values = []
        for _ in range(RESAMPLES):
            sample = generator.choice(pids, size=len(pids), replace=True)
            weights = pd.Series(sample).value_counts().reindex(pids, fill_value=0)
            values.append(
                aggregate(first, weights)[metric] - aggregate(second, weights)[metric]
            )
        values = np.asarray(values)
        rows.append(
            {
                "comparison": label,
                "metric": metric,
                "point_difference": point,
                "resamples": RESAMPLES,
                "seed": SEED,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
        )

    for tolerance in [15.0, 45.0]:
        p6 = participants[
            participants["method"].eq("P6-D")
            & participants["membership"].eq("primary")
            & participants["tolerance_sec"].eq(tolerance)
        ]
        for fusion in FUSION_SOURCES:
            local = participants[
                participants["method"].eq(fusion)
                & participants["membership"].eq("primary")
                & participants["tolerance_sec"].eq(tolerance)
            ]
            for metric in ["f1", "false_alarms_per_hour"]:
                contrast(
                    f"{fusion}_minus_P6-D_at_{int(tolerance)}sec",
                    local,
                    p6,
                    metric,
                )
    c15 = participants[
        participants["method"].eq("P6-P2-H2-2OF3")
        & participants["membership"].eq("primary")
        & participants["tolerance_sec"].eq(15.0)
    ]
    c45 = participants[
        participants["method"].eq("P6-P2-H2-2OF3")
        & participants["membership"].eq("primary")
        & participants["tolerance_sec"].eq(45.0)
    ]
    contrast("P6-P2-H2-2OF3_45sec_minus_15sec", c45, c15, "f1")
    return pd.DataFrame(rows)


# Section 5: independent fixed decisions

def metric_row(metrics: pd.DataFrame, method: str, tolerance: float = 15.0) -> pd.Series:
    return metrics[
        metrics["method"].eq(method)
        & metrics["membership"].eq("primary")
        & metrics["tolerance_sec"].eq(tolerance)
    ].iloc[0]


def reconstruct_decisions(metrics: pd.DataFrame) -> pd.DataFrame:
    p6 = metric_row(metrics, "P6-D")
    pair_or = metric_row(metrics, "P6-H2-OR")
    all_or = metric_row(metrics, "P6-P2-H2-OR")
    c15 = metric_row(metrics, "P6-P2-H2-2OF3")
    c45 = metric_row(metrics, "P6-P2-H2-2OF3", 45.0)
    rows = [
        {
            "hypothesis": "H8.14_two_modality_or_advancement",
            "value_1": pair_or.f1 - p6.f1,
            "criterion_1": ">=0.05_f1",
            "value_2": pair_or.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_2": "<=0.25_far_per_hour",
            "supported": (pair_or.f1 - p6.f1 >= 0.05)
            and (pair_or.false_alarms_per_hour - p6.false_alarms_per_hour <= 0.25),
        },
        {
            "hypothesis": "H8.15_consensus_advancement",
            "value_1": c15.f1 - p6.f1,
            "criterion_1": ">=0_f1",
            "value_2": c15.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_2": "<=0_far_per_hour",
            "supported": (c15.f1 - p6.f1 >= 0)
            and (c15.false_alarms_per_hour - p6.false_alarms_per_hour <= 0),
        },
        {
            "hypothesis": "H8.16_consensus_boundary_sensitivity",
            "value_1": c45.f1 - c15.f1,
            "criterion_1": ">=0.05_f1",
            "value_2": np.nan,
            "criterion_2": "not_applicable",
            "supported": c45.f1 - c15.f1 >= 0.05,
        },
        {
            "hypothesis": "H8.17_all_direct_or_alarm_penalty",
            "value_1": all_or.false_alarms_per_hour - p6.false_alarms_per_hour,
            "criterion_1": ">=0.50_far_per_hour",
            "value_2": np.nan,
            "criterion_2": "not_applicable",
            "supported": all_or.false_alarms_per_hour - p6.false_alarms_per_hour >= 0.50,
        },
    ]
    decisions = pd.DataFrame(rows)
    decisions["decision"] = decisions["supported"].map({True: "pass", False: "fail"})
    advance = bool(decisions.iloc[:2]["supported"].any())
    return pd.concat(
        [
            decisions,
            pd.DataFrame(
                [
                    {
                        "hypothesis": "overall_fixed_fusion_advance",
                        "value_1": float(advance),
                        "criterion_1": "H8.14_or_H8.15",
                        "value_2": np.nan,
                        "criterion_2": "not_applicable",
                        "supported": advance,
                        "decision": "advance" if advance else "stop",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )


# Section 6: output comparison and report

def frames_match(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    for name in expected.columns:
        left = expected[name].reset_index(drop=True)
        right = actual[name].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                left.to_numpy(dtype=float),
                pd.to_numeric(right).to_numpy(dtype=float),
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            ):
                return False
        elif not left.astype(str).equals(right.astype(str)):
            return False
    return True


def main() -> None:
    output = output_dir()
    paths = source_paths()
    expected_manifest = pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_repository": path.relative_to(root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in paths.items()
        ]
    ).sort_values("artifact_role").reset_index(drop=True)
    recorded_manifest = pd.read_csv(output / "input_artifact_manifest_v0.1.tsv", sep="\t")

    events, frozen_metrics, direct_support, support, references = load_sources()
    clusters = reconstruct_clusters(events)
    fused = reconstruct_fused(clusters)
    event_outputs = reconstruct_events(events, fused, support, references)
    comparisons = reconstruct_comparisons(event_outputs["event_metrics_v0.1.tsv"])
    bootstrap = reconstruct_bootstrap(event_outputs["event_participants_v0.1.tsv"])
    decisions = reconstruct_decisions(event_outputs["event_metrics_v0.1.tsv"])

    expected = {
        "fusion_clusters_v0.1.tsv": clusters,
        "fused_predicted_events_v0.1.tsv": fused,
        **event_outputs,
        "fusion_comparisons_v0.1.tsv": comparisons,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decisions,
    }
    comparisons_ok = {
        name: frames_match(frame, pd.read_csv(output / name, sep="\t"))
        for name, frame in expected.items()
    }

    frozen = frozen_metrics[frozen_metrics["comparator"].isin(DIRECT)]
    rebuilt = event_outputs["event_metrics_v0.1.tsv"]
    rebuilt = rebuilt[rebuilt["method"].isin(DIRECT)]
    frozen_keys = frozen.set_index(["comparator", "membership", "tolerance_sec"])
    direct_control_ok = True
    for row in rebuilt.itertuples(index=False):
        source = frozen_keys.loc[(row.method, row.membership, row.tolerance_sec)]
        for name in [
            "true_positive",
            "false_positive",
            "false_negative",
            "ignored_predictions",
            "f1",
            "false_alarms_per_hour",
        ]:
            direct_control_ok &= bool(
                np.isclose(getattr(row, name), source[name], atol=1e-12, equal_nan=True)
            )

    no_test_path = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    checks = pd.DataFrame(
        [
            ("input_manifest_rehashed", frames_match(expected_manifest, recorded_manifest), "five frozen source artifacts"),
            ("source_validation_only", set(events["partition"]) == {"validation"}, "source alarms restricted to validation"),
            ("fusion_clusters_reconstructed", comparisons_ok["fusion_clusters_v0.1.tsv"], f"rows={len(clusters)}"),
            ("fused_events_reconstructed", comparisons_ok["fused_predicted_events_v0.1.tsv"], f"rows={len(fused)}"),
            ("direct_controls_reproduced", direct_control_ok, "all direct-model controls"),
            ("event_outputs_reconstructed", all(comparisons_ok[name] for name in ["event_metrics_v0.1.tsv", "event_recordings_v0.1.tsv", "event_participants_v0.1.tsv", "event_matches_v0.1.tsv"]), "metrics, recording, participant, and match rows"),
            ("fusion_comparisons_reconstructed", comparisons_ok["fusion_comparisons_v0.1.tsv"], "12 fixed comparisons"),
            ("paired_bootstrap_reconstructed", comparisons_ok["paired_participant_bootstrap_v0.1.tsv"], "13 contrasts; 5000 resamples"),
            ("hypothesis_decisions_reconstructed", comparisons_ok["hypothesis_decisions_v0.1.tsv"], "four hypotheses and overall decision"),
            ("source_paths_exclude_test", bool(no_test_path), "no test path in source manifest"),
        ],
        columns=["check", "status", "detail"],
    )
    checks["status"] = checks["status"].map({True: "pass", False: "fail"})
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Independent validation failed: {failed}")
    verify_or_create_tsv(checks, output / "output_integrity_checks_v0.1.tsv")
    report = f"""# Block 8 Fixed Alarm-Fusion Output Validation

**Validation date:** 2026-09-13
**Method:** Independent source rehashing, alarm clustering, event evaluation, participant bootstrap, and decision reconstruction

All {len(checks)}/{len(checks)} independent checks passed.

The validator did not open raw signals, feature arrays, model objects, full probability tables, or current-test artifacts.
"""
    report_path = output / "OUTPUT_VALIDATION.md"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8").replace("\r\n", "\n") != report:
            raise RuntimeError("Reviewed validation report changed")
    else:
        report_path.write_text(report, encoding="utf-8")
    print(f"All {len(checks)}/{len(checks)} independent checks passed")


if __name__ == "__main__":
    main()
