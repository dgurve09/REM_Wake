"""Analyze which frozen validation reference events each modality detects."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-13_block8_cross_modality_event_overlap_v0.1"
PROTOCOL_COMMIT = "a7ac70f"
COMPARATORS = ["P6-D", "P2-D", "H2-D", "P2-H2-Z"]
DIRECT_COMPARATORS = ["P6-D", "P2-D", "H2-D"]
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260913


# Section 2: paths and immutable source hashing

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def input_paths() -> dict[str, Path]:
    transfer = repo_root() / "experiments/2026-09-06_block7_transfer_validation_v0.1"
    return {
        "reviewed_matches": transfer / "validation_event_matches_v0.1.tsv",
        "reviewed_metrics": transfer / "validation_event_metrics_v0.1.tsv",
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


# Section 3: frozen validation references and match indicators

def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def load_inputs(paths: dict[str, Path]) -> tuple[pd.DataFrame, ...]:
    matches = pd.read_csv(paths["reviewed_matches"], sep="\t")
    metrics = pd.read_csv(paths["reviewed_metrics"], sep="\t")
    support = pd.read_csv(paths["reviewed_support"], sep="\t")
    membership = pd.read_csv(paths["transition_membership"], sep="\t")
    quality = pd.read_csv(
        paths["transition_quality"],
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )

    if set(matches["partition"]) != {"validation"}:
        raise ValueError("Reviewed matches contain a non-validation partition")
    if set(metrics["partition"]) != {"validation"}:
        raise ValueError("Reviewed metrics contain a non-validation partition")
    if set(support["partition"]) != {"validation"}:
        raise ValueError("Reviewed support contains a non-validation partition")

    references = membership[
        membership["partition"].eq("validation")
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
    ].merge(quality, on="transition_id", validate="one_to_one")
    references["event_time_sec"] = references["nominal_boundary_sec"].astype(float)
    return matches, metrics, support, references


def detection_matrix(
    matches: pd.DataFrame, references: pd.DataFrame
) -> pd.DataFrame:
    all_rows = []
    for membership_name in MEMBERSHIPS:
        eligible_column = (
            "primary_analysis_eligible"
            if membership_name == "primary"
            else "expanded_quality_analysis_eligible"
        )
        eligible = references[truth(references[eligible_column])].copy()
        for tolerance in TOLERANCES:
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
                keys = set(
                    zip(
                        local["subject"].astype(str),
                        local["reference_time_sec"].astype(float),
                    )
                )
                column = comparator.lower().replace("-", "_") + "_detected"
                frame[column] = [
                    int((subject, float(event_time)) in keys)
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
            all_rows.append(frame)

    return pd.concat(all_rows, ignore_index=True).sort_values(
        ["membership", "tolerance_sec", "subject", "event_time_sec"],
        kind="stable",
    )


# Section 4: overlap summaries

def scope_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"], sort=True
    ):
        recalls = {
            comparator: float(
                frame[comparator.lower().replace("-", "_") + "_detected"].mean()
            )
            for comparator in COMPARATORS
        }
        category_counts = frame["direct_modality_category"].value_counts()
        best_direct = max(recalls[name] for name in DIRECT_COMPARATORS)
        direct_union = float(frame["direct_union_detected"].mean())
        rows.append(
            {
                "membership": membership_name,
                "tolerance_sec": tolerance,
                "reference_events": len(frame),
                "p6_d_recall": recalls["P6-D"],
                "p2_d_recall": recalls["P2-D"],
                "h2_d_recall": recalls["H2-D"],
                "p2_h2_z_recall": recalls["P2-H2-Z"],
                "direct_psg_union_recall": float(frame["direct_psg_detected"].mean()),
                "direct_union_recall": direct_union,
                "best_individual_direct_recall": best_direct,
                "direct_union_gain_over_best": direct_union - best_direct,
                "both_psg_and_wearable": int(
                    category_counts.get("both_psg_and_wearable", 0)
                ),
                "psg_only": int(category_counts.get("psg_only", 0)),
                "wearable_only": int(category_counts.get("wearable_only", 0)),
                "shared_miss": int(category_counts.get("shared_miss", 0)),
                "psg_only_fraction": float(
                    frame["direct_modality_category"].eq("psg_only").mean()
                ),
                "wearable_only_fraction": float(
                    frame["direct_modality_category"].eq("wearable_only").mean()
                ),
                "shared_miss_fraction": float(
                    frame["direct_modality_category"].eq("shared_miss").mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def pairwise_overlap(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"], sort=True
    ):
        for first, second in itertools.combinations(COMPARATORS, 2):
            a = frame[first.lower().replace("-", "_") + "_detected"].astype(bool)
            b = frame[second.lower().replace("-", "_") + "_detected"].astype(bool)
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


# Section 5: participant-grouped uncertainty

def bootstrap_values(frame15: pd.DataFrame, frame45: pd.DataFrame) -> dict[str, float]:
    weights = frame15["bootstrap_weight"].to_numpy(dtype=float)
    denominator = float(weights.sum())

    def average(column: str, frame: pd.DataFrame = frame15) -> float:
        return float(np.dot(frame[column].to_numpy(dtype=float), weights) / denominator)

    direct_recalls = [average(name.lower().replace("-", "_") + "_detected") for name in DIRECT_COMPARATORS]
    union15 = average("direct_union_detected")
    union45 = average("direct_union_detected", frame45)
    return {
        "shared_miss_fraction": average("shared_miss_indicator"),
        "psg_only_fraction": average("psg_only_indicator"),
        "wearable_only_fraction": average("wearable_only_indicator"),
        "direct_union_recall": union15,
        "direct_union_gain_over_best": union15 - max(direct_recalls),
        "boundary_union_recall_gain": union45 - union15,
        "psg_union_minus_h2_recall": average("direct_psg_detected")
        - average("h2_d_detected"),
    }


def participant_bootstrap(matrix: pd.DataFrame) -> pd.DataFrame:
    primary15 = matrix[
        matrix["membership"].eq("primary") & matrix["tolerance_sec"].eq(15.0)
    ].copy()
    primary45 = matrix[
        matrix["membership"].eq("primary") & matrix["tolerance_sec"].eq(45.0)
    ].copy()
    primary15 = primary15.sort_values("transition_id").reset_index(drop=True)
    primary45 = primary45.sort_values("transition_id").reset_index(drop=True)
    if not primary15["transition_id"].equals(primary45["transition_id"]):
        raise ValueError("Primary reference identity differs across tolerances")

    for frame in [primary15, primary45]:
        frame["shared_miss_indicator"] = frame["direct_modality_category"].eq(
            "shared_miss"
        ).astype(int)
        frame["psg_only_indicator"] = frame["direct_modality_category"].eq(
            "psg_only"
        ).astype(int)
        frame["wearable_only_indicator"] = frame["direct_modality_category"].eq(
            "wearable_only"
        ).astype(int)

    primary15["bootstrap_weight"] = 1.0
    primary45["bootstrap_weight"] = 1.0
    point = bootstrap_values(primary15, primary45)

    pids = np.sort(primary15["pid"].unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = {name: [] for name in point}
    for _ in range(BOOTSTRAP_RESAMPLES):
        selected = rng.choice(pids, size=len(pids), replace=True)
        counts = pd.Series(selected).value_counts()
        weights = primary15["pid"].map(counts).fillna(0).astype(float)
        primary15["bootstrap_weight"] = weights
        primary45["bootstrap_weight"] = weights
        values = bootstrap_values(primary15, primary45)
        for name, value in values.items():
            samples[name].append(value)

    rows = []
    for name, point_value in point.items():
        values = np.asarray(samples[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "point": point_value,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "lower_95": float(np.quantile(values, 0.025)),
                "median": float(np.quantile(values, 0.5)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


# Section 6: fixed decisions and checks

def decisions(summary: pd.DataFrame) -> pd.DataFrame:
    primary15 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(15.0)
    ].iloc[0]
    primary45 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(45.0)
    ].iloc[0]
    values = [
        (
            "H8.10_shared_direct_failure",
            float(primary15.shared_miss_fraction),
            0.50,
            float(primary15.shared_miss_fraction) >= 0.50,
        ),
        (
            "H8.11_wearable_specific_recovery_gap",
            float(primary15.psg_only_fraction),
            0.15,
            float(primary15.psg_only_fraction) >= 0.15,
        ),
        (
            "H8.12_direct_model_complementarity",
            float(primary15.direct_union_gain_over_best),
            0.10,
            float(primary15.direct_union_gain_over_best) >= 0.10,
        ),
        (
            "H8.13_boundary_tolerance_sensitivity",
            float(primary45.direct_union_recall - primary15.direct_union_recall),
            0.10,
            float(primary45.direct_union_recall - primary15.direct_union_recall)
            >= 0.10,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "hypothesis": name,
                "observed": observed,
                "pass_threshold": threshold,
                "supported": supported,
                "decision": "pass" if supported else "fail",
            }
            for name, observed, threshold, supported in values
        ]
    )


def checks(
    matrix: pd.DataFrame,
    summary: pd.DataFrame,
    metrics: pd.DataFrame,
    support: pd.DataFrame,
    bootstrap: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    primary15 = matrix[
        matrix["membership"].eq("primary") & matrix["tolerance_sec"].eq(15.0)
    ]
    frozen_counts = metrics.set_index(
        ["comparator", "membership", "tolerance_sec"]
    )["true_positive"]
    reproduced = True
    for (membership_name, tolerance), frame in matrix.groupby(
        ["membership", "tolerance_sec"]
    ):
        for comparator in COMPARATORS:
            column = comparator.lower().replace("-", "_") + "_detected"
            reproduced &= int(frame[column].sum()) == int(
                frozen_counts.loc[(comparator, membership_name, tolerance)]
            )

    category_columns = [
        "both_psg_and_wearable",
        "psg_only",
        "wearable_only",
        "shared_miss",
    ]
    category_sum = summary[category_columns].sum(axis=1)
    union_valid = (
        summary["direct_union_recall"]
        >= summary[["p6_d_recall", "p2_d_recall", "h2_d_recall"]].max(axis=1)
    ).all()
    no_test_path = ~manifest["path_relative_to_repository"].str.contains(
        r"(^|/)test($|/)", case=False, regex=True
    ).any()
    rows = [
        ("validation_support", len(support) == 80 and support["subject"].nunique() == 20 and support["pid"].nunique() == 16, "four comparators; 20 recordings; 16 pid groups"),
        ("primary_reference_count", len(primary15) == 37, f"references={len(primary15)}"),
        ("unique_reference_identity", not matrix.duplicated(["membership", "tolerance_sec", "transition_id"]).any(), "unique within every analysis scope"),
        ("frozen_true_positives_reproduced", reproduced, "all comparator, membership, and tolerance cells"),
        ("category_accounting", bool((category_sum == summary["reference_events"]).all()), "four categories sum to every scope denominator"),
        ("union_monotonic", bool(union_valid), "direct union recall is not below any component"),
        ("participant_bootstrap", len(bootstrap) == 7 and (bootstrap["resamples"] == BOOTSTRAP_RESAMPLES).all() and np.isfinite(bootstrap[["point", "lower_95", "median", "upper_95"]]).all().all(), "seven metrics; 5000 participant-grouped resamples"),
        ("source_paths_exclude_test", bool(no_test_path), "no test path in source manifest"),
        ("validation_rows_only", set(support["partition"]) == {"validation"}, "reviewed support restricted to validation"),
    ]
    result = pd.DataFrame(rows, columns=["check", "status", "detail"])
    result["status"] = result["status"].map({True: "pass", False: "fail"})
    return result


# Section 7: reviewed outputs

def readme(summary: pd.DataFrame, decision: pd.DataFrame, check: pd.DataFrame) -> str:
    p15 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(15.0)
    ].iloc[0]
    p45 = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(45.0)
    ].iloc[0]
    categories = (
        f"{int(p15.both_psg_and_wearable)} both, {int(p15.psg_only)} PSG-only, "
        f"{int(p15.wearable_only)} wearable-only, and {int(p15.shared_miss)} shared misses"
    )
    decision_lines = "\n".join(
        f"- `{row.hypothesis}`: {row.decision} (observed {row.observed:.4f}; gate {row.pass_threshold:.2f})"
        for row in decision.itertuples(index=False)
    )
    return f"""# Block 8 Cross-Modality Event Overlap v0.1

**Work date:** 2026-09-13
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Partition:** Reused development validation only
**Model fitting or threshold selection:** None
**Test data accessed:** No

## Primary Result

Across {int(p15.reference_events)} primary references at +/-15 seconds, the direct models produced {categories}. Six-channel PSG recall was {p15.p6_d_recall:.4f}, two-channel PSG recall was {p15.p2_d_recall:.4f}, wearable recall was {p15.h2_d_recall:.4f}, and their non-deployable union recall was {p15.direct_union_recall:.4f}.

The union improved over the best individual direct-model recall by {p15.direct_union_gain_over_best:.4f}. Expanding tolerance to +/-45 seconds changed union recall by {p45.direct_union_recall - p15.direct_union_recall:+.4f}.

## Frozen Decisions

{decision_lines}

## Boundary

The union is a recoverability diagnostic, not an ensemble result, because its false-alarm burden was not evaluated. This analysis cannot improve or validate the detector. It identifies whether the next uncertainty concerns shared representation, boundary timing, or wearable-specific information loss.

All {(check['status'] == 'pass').sum()}/{len(check)} in-run checks passed. The current test partition remained closed.
"""


def main() -> None:
    paths = input_paths()
    manifest = source_manifest(paths)
    matches, metrics, support, references = load_inputs(paths)
    matrix = detection_matrix(matches, references)
    summary = scope_summary(matrix)
    pairs = pairwise_overlap(matrix)
    bootstrap = participant_bootstrap(matrix)
    decision = decisions(summary)
    check = checks(matrix, summary, metrics, support, bootstrap, manifest)
    if not check["status"].eq("pass").all():
        raise RuntimeError("One or more in-run integrity checks failed")

    output = output_dir()
    outputs = {
        "input_artifact_manifest_v0.1.tsv": manifest,
        "reference_detection_matrix_v0.1.tsv": matrix,
        "overlap_summary_v0.1.tsv": summary,
        "pairwise_detection_overlap_v0.1.tsv": pairs,
        "participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": decision,
        "in_run_checks_v0.1.tsv": check,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output / name)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    verify_or_create_text(
        output / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )
    verify_or_create_text(output / "README.md", readme(summary, decision, check))

    primary = summary[
        summary["membership"].eq("primary") & summary["tolerance_sec"].eq(15.0)
    ].iloc[0]
    print(f"Primary references: {int(primary.reference_events)}")
    print(f"Direct union recall: {primary.direct_union_recall:.6f}")
    print(f"Shared-miss fraction: {primary.shared_miss_fraction:.6f}")
    print(f"PSG-only fraction: {primary.psg_only_fraction:.6f}")
    print(f"All {len(check)}/{len(check)} in-run checks passed")


if __name__ == "__main__":
    main()
