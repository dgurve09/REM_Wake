"""Independently reconstruct Block 8 cross-modality overlap outputs."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: fixed validator configuration

EXPERIMENT_DIR = "2026-09-13_block8_cross_modality_event_overlap_v0.1"
COMPARATORS = ["P6-D", "P2-D", "H2-D", "P2-H2-Z"]
DIRECT = ["P6-D", "P2-D", "H2-D"]
RESAMPLES = 5000
SEED = 20260913


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def sources() -> dict[str, Path]:
    transfer = root() / "experiments/2026-09-06_block7_transfer_validation_v0.1"
    return {
        "reviewed_matches": transfer / "validation_event_matches_v0.1.tsv",
        "reviewed_metrics": transfer / "validation_event_metrics_v0.1.tsv",
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


def column(comparator: str) -> str:
    return comparator.lower().replace("-", "_") + "_detected"


# Section 2: independent reference reconstruction

def rebuild_matrix() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = sources()
    matches = pd.read_csv(paths["reviewed_matches"], sep="\t")
    metrics = pd.read_csv(paths["reviewed_metrics"], sep="\t")
    support = pd.read_csv(paths["reviewed_support"], sep="\t")
    membership = pd.read_csv(paths["transition_membership"], sep="\t")
    quality = pd.read_csv(
        paths["transition_quality"],
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    refs = membership[
        membership["partition"].eq("validation")
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    refs["event_time_sec"] = refs["nominal_boundary_sec"].astype(float)

    rows = []
    for membership_name, eligible_name in [
        ("primary", "primary_analysis_eligible"),
        ("expanded", "expanded_quality_analysis_eligible"),
    ]:
        eligible = refs[truth(refs[eligible_name])]
        for tolerance in [15.0, 45.0]:
            frame = eligible[
                ["transition_id", "subject", "pid", "event_time_sec"]
            ].copy()
            frame.insert(0, "tolerance_sec", tolerance)
            frame.insert(0, "membership", membership_name)
            for comparator in COMPARATORS:
                local = matches[
                    matches["comparator"].eq(comparator)
                    & matches["membership"].eq(membership_name)
                    & matches["tolerance_sec"].eq(tolerance)
                    & matches["match_type"].eq("eligible")
                ]
                detected_keys = set(
                    zip(
                        local["subject"].astype(str),
                        local["reference_time_sec"].astype(float),
                    )
                )
                frame[column(comparator)] = [
                    int((subject, float(event_time)) in detected_keys)
                    for subject, event_time in zip(
                        frame["subject"], frame["event_time_sec"]
                    )
                ]
            psg = frame["p6_d_detected"].astype(bool) | frame[
                "p2_d_detected"
            ].astype(bool)
            wearable = frame["h2_d_detected"].astype(bool)
            frame["direct_psg_detected"] = psg.astype(int)
            frame["direct_union_detected"] = (psg | wearable).astype(int)
            frame["direct_modality_category"] = np.select(
                [psg & wearable, psg & ~wearable, ~psg & wearable],
                ["both_psg_and_wearable", "psg_only", "wearable_only"],
                default="shared_miss",
            )
            rows.append(frame)
    matrix = pd.concat(rows, ignore_index=True).sort_values(
        ["membership", "tolerance_sec", "subject", "event_time_sec"],
        kind="stable",
    )
    return matrix, metrics, support


# Section 3: independent summary reconstruction

def rebuild_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"], sort=True
    ):
        recall = {name: float(frame[column(name)].mean()) for name in COMPARATORS}
        counts = frame["direct_modality_category"].value_counts()
        best = max(recall[name] for name in DIRECT)
        union = float(frame["direct_union_detected"].mean())
        rows.append(
            {
                "membership": membership_name,
                "tolerance_sec": tolerance,
                "reference_events": len(frame),
                "p6_d_recall": recall["P6-D"],
                "p2_d_recall": recall["P2-D"],
                "h2_d_recall": recall["H2-D"],
                "p2_h2_z_recall": recall["P2-H2-Z"],
                "direct_psg_union_recall": float(frame["direct_psg_detected"].mean()),
                "direct_union_recall": union,
                "best_individual_direct_recall": best,
                "direct_union_gain_over_best": union - best,
                "both_psg_and_wearable": int(counts.get("both_psg_and_wearable", 0)),
                "psg_only": int(counts.get("psg_only", 0)),
                "wearable_only": int(counts.get("wearable_only", 0)),
                "shared_miss": int(counts.get("shared_miss", 0)),
                "psg_only_fraction": float(frame["direct_modality_category"].eq("psg_only").mean()),
                "wearable_only_fraction": float(frame["direct_modality_category"].eq("wearable_only").mean()),
                "shared_miss_fraction": float(frame["direct_modality_category"].eq("shared_miss").mean()),
            }
        )
    return pd.DataFrame(rows)


def rebuild_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"], sort=True
    ):
        for first, second in itertools.combinations(COMPARATORS, 2):
            a = frame[column(first)].astype(bool)
            b = frame[column(second)].astype(bool)
            union = int((a | b).sum())
            rows.append(
                {
                    "membership": membership_name,
                    "tolerance_sec": tolerance,
                    "first": first,
                    "second": second,
                    "both_detected": int((a & b).sum()),
                    "first_only": int((a & ~b).sum()),
                    "second_only": int((~a & b).sum()),
                    "both_missed": int((~a & ~b).sum()),
                    "union_detected": union,
                    "jaccard_detected": float((a & b).sum() / union)
                    if union
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


# Section 4: independent participant bootstrap

def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["shared_miss_indicator"] = result["direct_modality_category"].eq(
        "shared_miss"
    ).astype(int)
    result["psg_only_indicator"] = result["direct_modality_category"].eq(
        "psg_only"
    ).astype(int)
    result["wearable_only_indicator"] = result["direct_modality_category"].eq(
        "wearable_only"
    ).astype(int)
    return result


def weighted_statistics(
    frame15: pd.DataFrame, frame45: pd.DataFrame, weights: np.ndarray
) -> dict[str, float]:
    denominator = float(weights.sum())

    def mean(name: str, frame: pd.DataFrame = frame15) -> float:
        return float(np.dot(frame[name].to_numpy(dtype=float), weights) / denominator)

    direct_recalls = [mean(column(name)) for name in DIRECT]
    union15 = mean("direct_union_detected")
    union45 = mean("direct_union_detected", frame45)
    return {
        "shared_miss_fraction": mean("shared_miss_indicator"),
        "psg_only_fraction": mean("psg_only_indicator"),
        "wearable_only_fraction": mean("wearable_only_indicator"),
        "direct_union_recall": union15,
        "direct_union_gain_over_best": union15 - max(direct_recalls),
        "boundary_union_recall_gain": union45 - union15,
        "psg_union_minus_h2_recall": mean("direct_psg_detected")
        - mean("h2_d_detected"),
    }


def rebuild_bootstrap(matrix: pd.DataFrame) -> pd.DataFrame:
    frame15 = add_indicators(
        matrix[
            matrix["membership"].eq("primary")
            & matrix["tolerance_sec"].eq(15.0)
        ].sort_values("transition_id")
    ).reset_index(drop=True)
    frame45 = add_indicators(
        matrix[
            matrix["membership"].eq("primary")
            & matrix["tolerance_sec"].eq(45.0)
        ].sort_values("transition_id")
    ).reset_index(drop=True)
    point = weighted_statistics(frame15, frame45, np.ones(len(frame15)))
    pids = np.sort(frame15["pid"].unique())
    rng = np.random.default_rng(SEED)
    samples = {name: [] for name in point}
    for _ in range(RESAMPLES):
        selected = rng.choice(pids, size=len(pids), replace=True)
        counts = pd.Series(selected).value_counts()
        weights = frame15["pid"].map(counts).fillna(0).to_numpy(dtype=float)
        values = weighted_statistics(frame15, frame45, weights)
        for name, value in values.items():
            samples[name].append(value)
    rows = []
    for name, point_value in point.items():
        values = np.asarray(samples[name])
        rows.append(
            {
                "metric": name,
                "point": point_value,
                "resamples": RESAMPLES,
                "seed": SEED,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def rebuild_decisions(summary: pd.DataFrame) -> pd.DataFrame:
    p15 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(15.0)
    ].iloc[0]
    p45 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(45.0)
    ].iloc[0]
    tests = [
        ("H8.10_shared_direct_failure", float(p15.shared_miss_fraction), 0.50),
        ("H8.11_wearable_specific_recovery_gap", float(p15.psg_only_fraction), 0.15),
        ("H8.12_direct_model_complementarity", float(p15.direct_union_gain_over_best), 0.10),
        (
            "H8.13_boundary_tolerance_sensitivity",
            float(p45.direct_union_recall - p15.direct_union_recall),
            0.10,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "hypothesis": name,
                "observed": observed,
                "pass_threshold": threshold,
                "supported": observed >= threshold,
                "decision": "pass" if observed >= threshold else "fail",
            }
            for name, observed, threshold in tests
        ]
    )


# Section 5: exact output comparison and validation report

def frames_match(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    for name in expected.columns:
        left = expected[name]
        right = actual[name]
        if pd.api.types.is_numeric_dtype(left):
            if not np.allclose(
                left.to_numpy(dtype=float),
                pd.to_numeric(right).to_numpy(dtype=float),
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            ):
                return False
        elif not left.astype(str).reset_index(drop=True).equals(
            right.astype(str).reset_index(drop=True)
        ):
            return False
    return True


def main() -> None:
    output = output_dir()
    paths = sources()
    recorded_manifest = pd.read_csv(
        output / "input_artifact_manifest_v0.1.tsv", sep="\t"
    )
    rebuilt_manifest = pd.DataFrame(
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

    matrix, metrics, support = rebuild_matrix()
    summary = rebuild_summary(matrix)
    pairs = rebuild_pairs(matrix)
    bootstrap = rebuild_bootstrap(matrix)
    decisions = rebuild_decisions(summary)

    recorded_matrix = pd.read_csv(output / "reference_detection_matrix_v0.1.tsv", sep="\t")
    recorded_summary = pd.read_csv(output / "overlap_summary_v0.1.tsv", sep="\t")
    recorded_pairs = pd.read_csv(output / "pairwise_detection_overlap_v0.1.tsv", sep="\t")
    recorded_bootstrap = pd.read_csv(output / "participant_bootstrap_v0.1.tsv", sep="\t")
    recorded_decisions = pd.read_csv(output / "hypothesis_decisions_v0.1.tsv", sep="\t")

    frozen_counts = metrics.set_index(
        ["comparator", "membership", "tolerance_sec"]
    )["true_positive"]
    tp_ok = True
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"]
    ):
        for comparator in COMPARATORS:
            tp_ok &= int(frame[column(comparator)].sum()) == int(
                frozen_counts.loc[(comparator, membership_name, tolerance)]
            )

    source_partitions_ok = (
        set(pd.read_csv(paths["reviewed_matches"], sep="\t")["partition"])
        == {"validation"}
        and set(metrics["partition"]) == {"validation"}
        and set(support["partition"]) == {"validation"}
    )
    no_test_paths = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    compact_outputs = all(
        path.stat().st_size < 1_000_000 for path in output.glob("*") if path.is_file()
    )

    checks = pd.DataFrame(
        [
            ("input_manifest_rehashed", frames_match(rebuilt_manifest, recorded_manifest), "five source artifacts"),
            ("source_validation_only", source_partitions_ok, "reviewed matches, metrics, and support"),
            ("reference_matrix_reconstructed", frames_match(matrix, recorded_matrix), f"rows={len(matrix)}"),
            ("frozen_true_positives_reproduced", tp_ok, "all 16 metric cells"),
            ("overlap_summary_reconstructed", frames_match(summary, recorded_summary), "four analysis scopes"),
            ("pairwise_overlap_reconstructed", frames_match(pairs, recorded_pairs), f"rows={len(pairs)}"),
            ("participant_bootstrap_reconstructed", frames_match(bootstrap, recorded_bootstrap), "seven metrics; 5000 resamples"),
            ("hypothesis_decisions_reconstructed", frames_match(decisions, recorded_decisions), "four fixed decisions"),
            ("source_paths_exclude_test", bool(no_test_paths), "no test path in source manifest"),
            ("compact_reviewed_outputs", compact_outputs, "no output file reaches 1 MB"),
        ],
        columns=["check", "status", "detail"],
    )
    checks["status"] = checks["status"].map({True: "pass", False: "fail"})
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Independent validation failed: {failed}")

    verify_or_create_tsv(checks, output / "output_integrity_checks_v0.1.tsv")
    report = f"""# Block 8 Cross-Modality Event-Overlap Output Validation

**Validation date:** 2026-09-13
**Method:** Independent source rehashing, reference reconstruction, overlap analysis, grouped bootstrap, and decision reconstruction

All {len(checks)}/{len(checks)} independent checks passed.

The validator did not open raw signals, feature arrays, model objects, full probability tables, or test artifacts.
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
