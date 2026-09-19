"""Independently validate the Block 8 participant evidence-strength outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: independent source configuration

EXPERIMENT_DIR = "2026-09-18_block8_evidence_strength_v0.1"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def paths() -> dict[str, Path]:
    experiments = root() / "experiments"
    folders = {
        "single": experiments / "2026-09-11_block8_single_channel_robustness_v0.1",
        "raw": experiments / "2026-09-11_block8_raw_signal_noise_v0.1",
        "augmentation": experiments / "2026-09-12_block8_noise_augmented_training_v0.1",
        "overlap": experiments / "2026-09-13_block8_cross_modality_event_overlap_v0.1",
        "fusion": experiments / "2026-09-13_block8_fixed_alarm_fusion_v0.1",
    }
    return {
        "single_point": folders["single"] / "primary_ablation_comparisons_v0.1.tsv",
        "single_loo": folders["single"] / "leave_one_pid_out_summary_v0.1.tsv",
        "raw_decisions": folders["raw"] / "hypothesis_decisions_v0.1.tsv",
        "raw_bootstrap": folders["raw"] / "paired_participant_bootstrap_v0.1.tsv",
        "augmentation_decisions": folders["augmentation"] / "hypothesis_decisions_v0.1.tsv",
        "augmentation_bootstrap": folders["augmentation"] / "paired_participant_bootstrap_v0.1.tsv",
        "overlap_decisions": folders["overlap"] / "hypothesis_decisions_v0.1.tsv",
        "overlap_bootstrap": folders["overlap"] / "participant_bootstrap_v0.1.tsv",
        "fusion_decisions": folders["fusion"] / "hypothesis_decisions_v0.1.tsv",
        "fusion_bootstrap": folders["fusion"] / "paired_participant_bootstrap_v0.1.tsv",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(frame: pd.DataFrame, **criteria: object) -> pd.Series:
    selected = frame.copy()
    for column, value in criteria.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {criteria}, found {len(selected)}")
    return selected.iloc[0]


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"Expected Boolean value, found {value!r}")
    return text == "true"


# Section 2: independent source reconstruction

def rebuild_manifest(source_paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_role": role,
                "path_relative_to_repository": path.relative_to(root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in source_paths.items()
        ]
    ).sort_values("source_role").reset_index(drop=True)


def grade(point: float, lower: float, upper: float, operator: str, gate: float) -> tuple[bool, bool, bool, str, str]:
    if operator == ">=":
        point_pass = point >= gate
        range_pass = lower >= gate
        range_fail = upper < gate
    else:
        point_pass = point <= gate
        range_pass = upper <= gate
        range_fail = lower > gate

    if point_pass and range_pass:
        label = "gate_supported_by_uncertainty"
    elif point_pass:
        label = "point_gate_only"
    elif range_fail:
        label = "gate_failure_supported_by_uncertainty"
    else:
        label = "point_failure_with_gate_overlap"

    if lower == 0 and upper == 0:
        relation = "exact_zero"
    elif lower > 0:
        relation = "positive"
    elif upper < 0:
        relation = "negative"
    else:
        relation = "spans_zero"
    return point_pass, range_pass, range_fail, label, relation


def expected_claim_values(source_paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    single_point = pd.read_csv(source_paths["single_point"], sep="\t")
    single_loo = pd.read_csv(source_paths["single_loo"], sep="\t")
    raw = pd.read_csv(source_paths["raw_bootstrap"], sep="\t")
    aug_decisions = pd.read_csv(source_paths["augmentation_decisions"], sep="\t")
    aug = pd.read_csv(source_paths["augmentation_bootstrap"], sep="\t")
    overlap_decisions = pd.read_csv(source_paths["overlap_decisions"], sep="\t")
    overlap = pd.read_csv(source_paths["overlap_bootstrap"], sep="\t")
    fusion = pd.read_csv(source_paths["fusion_bootstrap"], sep="\t")

    expected: dict[str, dict[str, object]] = {}

    def add(
        claim_id: str,
        point: float,
        lower: float,
        upper: float,
        operator: str,
        gate_value: float,
    ) -> None:
        point_pass, range_pass, range_fail, label, relation = grade(
            float(point), float(lower), float(upper), operator, float(gate_value)
        )
        expected[claim_id] = {
            "point_estimate": float(point),
            "uncertainty_lower": float(lower),
            "uncertainty_upper": float(upper),
            "gate_operator": operator,
            "gate_value": float(gate_value),
            "point_passes_gate": point_pass,
            "uncertainty_supports_gate": range_pass,
            "uncertainty_supports_failure": range_fail,
            "evidence_grade": label,
            "zero_relation": relation,
        }

    for claim_id, channel in [("ES01", "HB_1"), ("ES02", "HB_2")]:
        point = one(single_point, ablated_channel=channel)
        loo = one(single_loo, ablated_channel=channel)
        add(claim_id, point.f1_difference, loo.f1_difference_min, loo.f1_difference_max, "<=", -0.03)

    raw_specs = [
        ("ES03", "H2-BOTH-20DB_minus_H2-CLEAN", "event_f1_difference", ">=", -0.03),
        ("ES04", "H2-BOTH-20DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", "<=", 0.50),
        ("ES05", "H2-BOTH-0DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03),
        ("ES06", "H2-HB1-10DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03),
        ("ES07", "H2-HB1-10DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", ">=", 0.50),
        ("ES08", "H2-HB2-10DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03),
        ("ES09", "H2-HB2-10DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", ">=", 0.50),
    ]
    for claim_id, comparison, metric, operator, gate_value in raw_specs:
        row = one(raw, comparison=comparison, metric=metric)
        add(claim_id, row.point_difference, row.lower_95, row.upper_95, operator, gate_value)

    aug_specs = [
        ("ES10", "H2-CLEAN", "event_f1_difference", ">=", -0.03),
        ("ES11", "H2-CLEAN", "false_alarms_per_hour_difference", "<=", 0.50),
        ("ES12", "H2-HB1-10DB", "event_f1_difference", ">=", 0.03),
        ("ES13", "H2-HB1-10DB", "false_alarms_per_hour_difference", "<=", -0.50),
        ("ES14", "H2-CLEAN", "event_f1_difference", ">=", 0.05),
    ]
    for claim_id, condition, metric, operator, gate_value in aug_specs:
        row = one(
            aug,
            comparison_kind="between_model",
            comparison=f"H2-NA_minus_H2-D__{condition}",
            metric=metric,
        )
        add(claim_id, row.point_difference, row.lower_95, row.upper_95, operator, gate_value)

    gap = one(aug_decisions, hypothesis="H8.8_degradation_gap_reduction")
    expected["ES15"] = {
        "point_estimate": float(gap.value_1),
        "secondary_point_estimate": float(gap.value_2),
        "gate_operator": "joint_>=",
        "gate_value": 0.03,
        "secondary_gate_value": 0.50,
        "point_passes_gate": as_bool(gap.supported),
        "evidence_grade": "no_direct_uncertainty_interval",
        "zero_relation": "not_available",
    }

    overlap_specs = [
        ("ES16", "H8.10_shared_direct_failure", "shared_miss_fraction"),
        ("ES17", "H8.11_wearable_specific_recovery_gap", "psg_only_fraction"),
        ("ES18", "H8.12_direct_model_complementarity", "direct_union_gain_over_best"),
        ("ES19", "H8.13_boundary_tolerance_sensitivity", "boundary_union_recall_gain"),
    ]
    for claim_id, hypothesis, metric in overlap_specs:
        decision = one(overlap_decisions, hypothesis=hypothesis)
        row = one(overlap, metric=metric)
        add(claim_id, row.point, row.lower_95, row.upper_95, ">=", decision.pass_threshold)

    fusion_specs = [
        ("ES20", "P6-H2-OR_minus_P6-D_at_15sec", "f1", ">=", 0.05),
        ("ES21", "P6-H2-OR_minus_P6-D_at_15sec", "false_alarms_per_hour", "<=", 0.25),
        ("ES22", "P6-P2-H2-2OF3_minus_P6-D_at_15sec", "f1", ">=", 0.0),
        ("ES23", "P6-P2-H2-2OF3_minus_P6-D_at_15sec", "false_alarms_per_hour", "<=", 0.0),
        ("ES24", "P6-P2-H2-2OF3_45sec_minus_15sec", "f1", ">=", 0.05),
        ("ES25", "P6-P2-H2-OR_minus_P6-D_at_15sec", "false_alarms_per_hour", ">=", 0.50),
    ]
    for claim_id, comparison, metric, operator, gate_value in fusion_specs:
        row = one(fusion, comparison=comparison, metric=metric)
        add(claim_id, row.point_difference, row.lower_95, row.upper_95, operator, gate_value)

    return expected


# Section 3: output validation

def numeric_equal(left: object, right: object) -> bool:
    return bool(np.isclose(float(left), float(right), atol=1e-12, rtol=1e-12))


def manifest_matches(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    return (
        expected["source_role"].equals(actual["source_role"])
        and expected["path_relative_to_repository"].equals(actual["path_relative_to_repository"])
        and np.array_equal(expected["bytes"].to_numpy(), actual["bytes"].to_numpy())
        and expected["sha256"].equals(actual["sha256"])
    )


def main() -> None:
    source_paths = paths()
    expected_manifest = rebuild_manifest(source_paths)
    expected = expected_claim_values(source_paths)
    output = output_dir()

    recorded_manifest = pd.read_csv(output / "input_artifact_manifest_v0.1.tsv", sep="\t")
    evidence = pd.read_csv(output / "evidence_strength_v0.1.tsv", sep="\t")
    summary = pd.read_csv(output / "evidence_strength_summary_v0.1.tsv", sep="\t")
    flags = pd.read_csv(output / "interpretation_flags_v0.1.tsv", sep="\t")
    decisions = pd.read_csv(output / "block8_decisions_v0.1.tsv", sep="\t")
    analysis_checks = pd.read_csv(output / "analysis_checks_v0.1.tsv", sep="\t")

    evidence_index = evidence.set_index("claim_id")
    ids_match = evidence["claim_id"].tolist() == [f"ES{i:02d}" for i in range(1, 26)]
    values_match = True
    classifications_match = True
    for claim_id, values in expected.items():
        actual = evidence_index.loc[claim_id]
        for column in ["point_estimate", "gate_value"]:
            values_match &= numeric_equal(actual[column], values[column])
        if claim_id == "ES15":
            values_match &= numeric_equal(
                actual["secondary_point_estimate"], values["secondary_point_estimate"]
            )
            values_match &= numeric_equal(
                actual["secondary_gate_value"], values["secondary_gate_value"]
            )
        else:
            for column in ["uncertainty_lower", "uncertainty_upper"]:
                values_match &= numeric_equal(actual[column], values[column])

        for column in ["gate_operator", "evidence_grade", "zero_relation"]:
            classifications_match &= str(actual[column]) == str(values[column])
        classifications_match &= as_bool(actual["point_passes_gate"]) == bool(
            values["point_passes_gate"]
        )
        if claim_id != "ES15":
            classifications_match &= as_bool(
                actual["uncertainty_supports_gate"]
            ) == bool(values["uncertainty_supports_gate"])
            classifications_match &= as_bool(
                actual["uncertainty_supports_failure"]
            ) == bool(values["uncertainty_supports_failure"])

    expected_grade_counts = pd.Series(
        [values["evidence_grade"] for values in expected.values()]
    ).value_counts()
    recorded_grade_counts = summary[summary["dimension"].eq("evidence_grade")].set_index(
        "category"
    )["count"]
    summary_matches = all(
        int(recorded_grade_counts.get(grade_name, 0)) == int(count)
        for grade_name, count in expected_grade_counts.items()
    ) and int(recorded_grade_counts.sum()) == 25

    expected_flag_ids = [
        claim_id
        for claim_id, values in expected.items()
        if values["evidence_grade"] != "gate_supported_by_uncertainty"
    ]
    flags_match = flags["claim_id"].tolist() == expected_flag_ids

    decision_index = decisions.set_index("decision")
    decisions_match = (
        decision_index.loc["deployable_wearable_method_advances", "outcome"]
        == "do_not_advance"
        and decision_index.loc["boundary_uncertainty_priority", "outcome"]
        == "retain_priority"
        and decision_index.loc[
            "boundary_materiality_participant_supported", "outcome"
        ]
        == "not_supported"
        and decision_index.loc["block8_closeout", "outcome"] == "remain_closed"
    )
    recorded_checks_valid = len(analysis_checks) == 11 and analysis_checks[
        "status"
    ].eq("pass").all()
    no_test = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    no_binary = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"\.(?:edf|npz|npy|joblib|pth|pt|pkl|gz)$", case=False, regex=True
    ).any()

    rows = [
        ("source_manifest_rehashed", manifest_matches(expected_manifest, recorded_manifest), "ten frozen source tables"),
        ("claim_ids_reconstructed", ids_match, "ES01 through ES25"),
        ("source_values_reconstructed", values_match, "points, ranges, and gates match sources"),
        ("classifications_reconstructed", classifications_match, "gate grades and zero relations match"),
        ("summary_reconstructed", summary_matches, "grade counts sum to 25"),
        ("interpretation_flags_reconstructed", flags_match, "all non-fully-supported claims flagged"),
        ("block_decisions_reconstructed", decisions_match, "nonadvance, boundary nuance, and closeout retained"),
        ("analysis_checks_valid", recorded_checks_valid, "11/11 primary checks pass"),
        ("source_paths_exclude_test", bool(no_test), "no current-test path"),
        ("compact_sources_only", bool(no_binary), "no raw, model, feature-array, or compressed-score source"),
    ]
    checks = pd.DataFrame(rows, columns=["check", "status", "detail"])
    checks["status"] = checks["status"].map({True: "pass", False: "fail"})
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Independent evidence-strength validation failed: {failed}")

    verify_or_create_tsv(checks, output / "output_integrity_checks_v0.1.tsv")
    report = f"""# Block 8 Evidence-Strength Output Validation

**Validation date:** 2026-09-18
**Method:** Independent source rehashing, claim-value reconstruction, gate classification, summary reconstruction, and decision verification

All {len(checks)}/{len(checks)} independent checks passed.

The validator opened only the ten compact frozen source tables. It did not access raw signals, feature arrays, models, full scores, new resamples, or current-test artifacts.
"""
    report_path = output / "OUTPUT_VALIDATION.md"
    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if existing != report:
            raise RuntimeError("Reviewed validation report changed")
    else:
        report_path.write_text(report, encoding="utf-8")
    print(f"All {len(checks)}/{len(checks)} independent checks passed")


if __name__ == "__main__":
    main()
