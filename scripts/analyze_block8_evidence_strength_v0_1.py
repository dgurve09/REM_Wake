"""Grade frozen Block 8 conclusions against stored participant uncertainty."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: frozen paths

VERSION = "v0.1"
PROTOCOL_COMMIT = "777e641"
EXPERIMENT_DIR = "2026-09-18_block8_evidence_strength_v0.1"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def source_paths() -> dict[str, Path]:
    experiments = root() / "experiments"
    single = experiments / "2026-09-11_block8_single_channel_robustness_v0.1"
    raw = experiments / "2026-09-11_block8_raw_signal_noise_v0.1"
    augmentation = experiments / "2026-09-12_block8_noise_augmented_training_v0.1"
    overlap = experiments / "2026-09-13_block8_cross_modality_event_overlap_v0.1"
    fusion = experiments / "2026-09-13_block8_fixed_alarm_fusion_v0.1"
    return {
        "single_point": single / "primary_ablation_comparisons_v0.1.tsv",
        "single_loo": single / "leave_one_pid_out_summary_v0.1.tsv",
        "raw_decisions": raw / "hypothesis_decisions_v0.1.tsv",
        "raw_bootstrap": raw / "paired_participant_bootstrap_v0.1.tsv",
        "augmentation_decisions": augmentation / "hypothesis_decisions_v0.1.tsv",
        "augmentation_bootstrap": augmentation / "paired_participant_bootstrap_v0.1.tsv",
        "overlap_decisions": overlap / "hypothesis_decisions_v0.1.tsv",
        "overlap_bootstrap": overlap / "participant_bootstrap_v0.1.tsv",
        "fusion_decisions": fusion / "hypothesis_decisions_v0.1.tsv",
        "fusion_bootstrap": fusion / "paired_participant_bootstrap_v0.1.tsv",
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "source_role": role,
                "path_relative_to_repository": path.relative_to(root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in paths.items()
        ]
    ).sort_values("source_role").reset_index(drop=True)


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def one(frame: pd.DataFrame, **criteria: object) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in criteria.items():
        mask &= frame[column].eq(value)
    selected = frame[mask]
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


# Section 2: fixed gate classification

GRADES = [
    "gate_supported_by_uncertainty",
    "point_gate_only",
    "gate_failure_supported_by_uncertainty",
    "point_failure_with_gate_overlap",
    "no_direct_uncertainty_interval",
]


def classify_gate(
    point: float, lower: float, upper: float, operator: str, gate: float
) -> tuple[bool, bool, bool, str]:
    if operator == ">=":
        point_passes = point >= gate
        range_supports = lower >= gate
        range_supports_failure = upper < gate
    elif operator == "<=":
        point_passes = point <= gate
        range_supports = upper <= gate
        range_supports_failure = lower > gate
    else:
        raise ValueError(f"Unsupported gate operator: {operator}")

    if point_passes and range_supports:
        grade = "gate_supported_by_uncertainty"
    elif point_passes:
        grade = "point_gate_only"
    elif range_supports_failure:
        grade = "gate_failure_supported_by_uncertainty"
    else:
        grade = "point_failure_with_gate_overlap"
    return point_passes, range_supports, range_supports_failure, grade


def zero_relation(lower: float, upper: float) -> str:
    if lower == 0 and upper == 0:
        return "exact_zero"
    if lower > 0:
        return "positive"
    if upper < 0:
        return "negative"
    return "spans_zero"


def interpretation(grade: str, relation: str) -> str:
    messages = {
        "gate_supported_by_uncertainty": "full stored range meets the original gate",
        "point_gate_only": "aggregate point passes but stored range overlaps the gate",
        "gate_failure_supported_by_uncertainty": "full stored range remains on the failing side of the gate",
        "point_failure_with_gate_overlap": "aggregate point fails but stored range overlaps the gate",
        "no_direct_uncertainty_interval": "exact derived contrast has no stored participant interval",
    }
    if relation == "not_available":
        return messages[grade]
    return f"{messages[grade]}; range is {relation} relative to zero"


def ranged_claim(
    claim_id: str,
    experiment: str,
    hypothesis: str,
    claim: str,
    metric: str,
    point: float,
    lower: float,
    upper: float,
    uncertainty_method: str,
    operator: str,
    gate: float,
    source_role: str,
) -> dict[str, object]:
    point_passes, range_supports, range_failure, grade = classify_gate(
        float(point), float(lower), float(upper), operator, float(gate)
    )
    relation = zero_relation(float(lower), float(upper))
    return {
        "claim_id": claim_id,
        "experiment": experiment,
        "original_hypothesis": hypothesis,
        "claim": claim,
        "metric": metric,
        "point_estimate": float(point),
        "secondary_point_estimate": np.nan,
        "uncertainty_method": uncertainty_method,
        "uncertainty_lower": float(lower),
        "uncertainty_upper": float(upper),
        "gate_operator": operator,
        "gate_value": float(gate),
        "secondary_gate_value": np.nan,
        "point_passes_gate": point_passes,
        "uncertainty_supports_gate": range_supports,
        "uncertainty_supports_failure": range_failure,
        "evidence_grade": grade,
        "zero_relation": relation,
        "source_role": source_role,
        "interpretation": interpretation(grade, relation),
    }


# Section 3: reconstruct the 25 predeclared claims

def evidence_strength(paths: dict[str, Path]) -> pd.DataFrame:
    single_point = pd.read_csv(paths["single_point"], sep="\t")
    single_loo = pd.read_csv(paths["single_loo"], sep="\t")
    raw_decisions = pd.read_csv(paths["raw_decisions"], sep="\t")
    raw_boot = pd.read_csv(paths["raw_bootstrap"], sep="\t")
    augmentation_decisions = pd.read_csv(paths["augmentation_decisions"], sep="\t")
    augmentation_boot = pd.read_csv(paths["augmentation_bootstrap"], sep="\t")
    overlap_decisions = pd.read_csv(paths["overlap_decisions"], sep="\t")
    overlap_boot = pd.read_csv(paths["overlap_bootstrap"], sep="\t")
    fusion_decisions = pd.read_csv(paths["fusion_decisions"], sep="\t")
    fusion_boot = pd.read_csv(paths["fusion_bootstrap"], sep="\t")

    for hypothesis in [
        "H8.3_mild_noise_tolerance",
        "H8.5_channel_noise_asymmetry",
    ]:
        one(raw_decisions, hypothesis=hypothesis)
    for hypothesis in [
        "H8.14_two_modality_or_advancement",
        "H8.15_consensus_advancement",
        "H8.16_consensus_boundary_sensitivity",
        "H8.17_all_direct_or_alarm_penalty",
    ]:
        one(fusion_decisions, hypothesis=hypothesis)

    rows: list[dict[str, object]] = []

    def add_single(claim_id: str, channel: str) -> None:
        point = one(single_point, ablated_channel=channel)
        loo = one(single_loo, ablated_channel=channel)
        rows.append(
            ranged_claim(
                claim_id,
                "single_channel",
                f"single_channel_{channel.lower()}_material_loss",
                f"{channel} neutralization has a material F1 loss",
                "event_f1_difference",
                point.f1_difference,
                loo.f1_difference_min,
                loo.f1_difference_max,
                "leave_one_pid_out_range",
                "<=",
                -0.03,
                "single_loo",
            )
        )

    add_single("ES01", "HB_1")
    add_single("ES02", "HB_2")

    def raw_claim(
        claim_id: str,
        hypothesis: str,
        claim: str,
        comparison: str,
        metric: str,
        operator: str,
        gate: float,
    ) -> None:
        row = one(raw_boot, comparison=comparison, metric=metric)
        rows.append(
            ranged_claim(
                claim_id,
                "raw_noise",
                hypothesis,
                claim,
                metric,
                row.point_difference,
                row.lower_95,
                row.upper_95,
                "participant_bootstrap_95",
                operator,
                gate,
                "raw_bootstrap",
            )
        )

    raw_claim("ES03", "H8.3", "Mild both-channel noise preserves F1", "H2-BOTH-20DB_minus_H2-CLEAN", "event_f1_difference", ">=", -0.03)
    raw_claim("ES04", "H8.3", "Mild both-channel noise preserves FAR", "H2-BOTH-20DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", "<=", 0.50)
    raw_claim("ES05", "severe_noise_stress", "Severe both-channel noise causes material F1 loss", "H2-BOTH-0DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03)
    raw_claim("ES06", "H8.5", "Isolated HB_1 noise causes material F1 loss", "H2-HB1-10DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03)
    raw_claim("ES07", "H8.5", "Isolated HB_1 noise causes material FAR increase", "H2-HB1-10DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", ">=", 0.50)
    raw_claim("ES08", "H8.5", "Isolated HB_2 noise causes material F1 loss", "H2-HB2-10DB_minus_H2-CLEAN", "event_f1_difference", "<=", -0.03)
    raw_claim("ES09", "H8.5", "Isolated HB_2 noise causes material FAR increase", "H2-HB2-10DB_minus_H2-CLEAN", "false_alarms_per_hour_difference", ">=", 0.50)

    def augmentation_claim(
        claim_id: str,
        hypothesis: str,
        claim: str,
        condition: str,
        metric: str,
        operator: str,
        gate: float,
    ) -> None:
        comparison = f"H2-NA_minus_H2-D__{condition}"
        row = one(
            augmentation_boot,
            comparison_kind="between_model",
            comparison=comparison,
            metric=metric,
        )
        rows.append(
            ranged_claim(
                claim_id,
                "noise_augmentation",
                hypothesis,
                claim,
                metric,
                row.point_difference,
                row.lower_95,
                row.upper_95,
                "participant_bootstrap_95",
                operator,
                gate,
                "augmentation_bootstrap",
            )
        )

    augmentation_claim("ES10", "H8.6", "Augmentation preserves clean F1", "H2-CLEAN", "event_f1_difference", ">=", -0.03)
    augmentation_claim("ES11", "H8.6", "Augmentation preserves clean FAR", "H2-CLEAN", "false_alarms_per_hour_difference", "<=", 0.50)
    augmentation_claim("ES12", "H8.7", "Augmentation materially improves HB_1-noise F1", "H2-HB1-10DB", "event_f1_difference", ">=", 0.03)
    augmentation_claim("ES13", "H8.7", "Augmentation materially reduces HB_1-noise FAR", "H2-HB1-10DB", "false_alarms_per_hour_difference", "<=", -0.50)
    augmentation_claim("ES14", "H8.9", "Augmentation meaningfully advances clean F1", "H2-CLEAN", "event_f1_difference", ">=", 0.05)

    gap = one(augmentation_decisions, hypothesis="H8.8_degradation_gap_reduction")
    rows.append(
        {
            "claim_id": "ES15",
            "experiment": "noise_augmentation",
            "original_hypothesis": "H8.8",
            "claim": "Augmentation reduces the joint HB_1 degradation gap",
            "metric": "f1_gap_reduction",
            "point_estimate": float(gap.value_1),
            "secondary_point_estimate": float(gap.value_2),
            "uncertainty_method": "no_direct_difference_in_differences_interval",
            "uncertainty_lower": np.nan,
            "uncertainty_upper": np.nan,
            "gate_operator": "joint_>=",
            "gate_value": 0.03,
            "secondary_gate_value": 0.50,
            "point_passes_gate": as_bool(gap.supported),
            "uncertainty_supports_gate": np.nan,
            "uncertainty_supports_failure": np.nan,
            "evidence_grade": "no_direct_uncertainty_interval",
            "zero_relation": "not_available",
            "source_role": "augmentation_decisions",
            "interpretation": interpretation(
                "no_direct_uncertainty_interval", "not_available"
            ),
        }
    )

    def overlap_claim(
        claim_id: str, hypothesis: str, claim: str, metric: str
    ) -> None:
        decision = one(overlap_decisions, hypothesis=hypothesis)
        interval = one(overlap_boot, metric=metric)
        rows.append(
            ranged_claim(
                claim_id,
                "event_overlap",
                hypothesis.split("_")[0],
                claim,
                metric,
                interval.point,
                interval.lower_95,
                interval.upper_95,
                "participant_bootstrap_95",
                ">=",
                decision.pass_threshold,
                "overlap_bootstrap",
            )
        )

    overlap_claim("ES16", "H8.10_shared_direct_failure", "Shared direct-model misses are material", "shared_miss_fraction")
    overlap_claim("ES17", "H8.11_wearable_specific_recovery_gap", "PSG-only recovery is material", "psg_only_fraction")
    overlap_claim("ES18", "H8.12_direct_model_complementarity", "Direct-model union gain is material", "direct_union_gain_over_best")
    overlap_claim("ES19", "H8.13_boundary_tolerance_sensitivity", "Boundary-tolerance gain is material", "boundary_union_recall_gain")

    def fusion_claim(
        claim_id: str,
        hypothesis: str,
        claim: str,
        comparison: str,
        metric: str,
        operator: str,
        gate: float,
    ) -> None:
        interval = one(fusion_boot, comparison=comparison, metric=metric)
        rows.append(
            ranged_claim(
                claim_id,
                "alarm_fusion",
                hypothesis,
                claim,
                metric,
                interval.point_difference,
                interval.lower_95,
                interval.upper_95,
                "participant_bootstrap_95",
                operator,
                gate,
                "fusion_bootstrap",
            )
        )

    fusion_claim("ES20", "H8.14", "Two-modality OR meaningfully improves F1", "P6-H2-OR_minus_P6-D_at_15sec", "f1", ">=", 0.05)
    fusion_claim("ES21", "H8.14", "Two-modality OR keeps FAR increase within its bound", "P6-H2-OR_minus_P6-D_at_15sec", "false_alarms_per_hour", "<=", 0.25)
    fusion_claim("ES22", "H8.15", "Laboratory consensus preserves or improves F1", "P6-P2-H2-2OF3_minus_P6-D_at_15sec", "f1", ">=", 0.0)
    fusion_claim("ES23", "H8.15", "Laboratory consensus does not increase FAR", "P6-P2-H2-2OF3_minus_P6-D_at_15sec", "false_alarms_per_hour", "<=", 0.0)
    fusion_claim("ES24", "H8.16", "Wider tolerance materially improves consensus F1", "P6-P2-H2-2OF3_45sec_minus_15sec", "f1", ">=", 0.05)
    fusion_claim("ES25", "H8.17", "All-direct OR causes a material alarm penalty", "P6-P2-H2-OR_minus_P6-D_at_15sec", "false_alarms_per_hour", ">=", 0.50)

    return pd.DataFrame(rows).sort_values("claim_id").reset_index(drop=True)


# Section 4: fixed summaries and decisions

def strength_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for grade in GRADES:
        rows.append(
            {
                "dimension": "evidence_grade",
                "category": grade,
                "count": int(evidence["evidence_grade"].eq(grade).sum()),
            }
        )
    for relation in ["positive", "negative", "spans_zero", "exact_zero", "not_available"]:
        rows.append(
            {
                "dimension": "zero_relation",
                "category": relation,
                "count": int(evidence["zero_relation"].eq(relation).sum()),
            }
        )
    for experiment, count in evidence["experiment"].value_counts().sort_index().items():
        rows.append(
            {
                "dimension": "experiment",
                "category": experiment,
                "count": int(count),
            }
        )
    return pd.DataFrame(rows)


def interpretation_flags(evidence: pd.DataFrame) -> pd.DataFrame:
    grades = [
        "point_gate_only",
        "gate_failure_supported_by_uncertainty",
        "point_failure_with_gate_overlap",
        "no_direct_uncertainty_interval",
    ]
    columns = [
        "claim_id",
        "experiment",
        "claim",
        "evidence_grade",
        "zero_relation",
        "interpretation",
    ]
    return evidence[evidence["evidence_grade"].isin(grades)][columns].reset_index(
        drop=True
    )


def block_decisions(evidence: pd.DataFrame) -> pd.DataFrame:
    indexed = evidence.set_index("claim_id")
    clean_advance_ruled_out = (
        indexed.loc["ES14", "evidence_grade"]
        == "gate_failure_supported_by_uncertainty"
    )
    boundary_direction_supported = indexed.loc["ES19", "zero_relation"] == "positive"
    boundary_materiality_supported = (
        indexed.loc["ES19", "evidence_grade"]
        == "gate_supported_by_uncertainty"
    )
    return pd.DataFrame(
        [
            {
                "decision": "deployable_wearable_method_advances",
                "outcome": "do_not_advance",
                "supported": clean_advance_ruled_out,
                "basis": "clean F1 advancement gate is ruled out by the stored participant interval; PSG fusion is nondeployable",
            },
            {
                "decision": "boundary_uncertainty_priority",
                "outcome": "retain_priority",
                "supported": boundary_direction_supported,
                "basis": "boundary effect is directionally positive, although the interval does not clear the 0.10 material gate",
            },
            {
                "decision": "boundary_materiality_participant_supported",
                "outcome": "not_supported",
                "supported": boundary_materiality_supported,
                "basis": "participant interval lower bound is above zero but below the predeclared 0.10 material gate",
            },
            {
                "decision": "block8_closeout",
                "outcome": "remain_closed",
                "supported": clean_advance_ruled_out and boundary_direction_supported,
                "basis": "evidence-strength analysis changes interpretation strength but does not create a wearable advance",
            },
        ]
    )


# Section 5: integrity checks and report

def analysis_checks(
    source_manifest: pd.DataFrame,
    evidence: pd.DataFrame,
    summary: pd.DataFrame,
    flags: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    ranged = evidence[~evidence["uncertainty_lower"].isna()]
    inside = (
        ranged["uncertainty_lower"].le(ranged["point_estimate"])
        & ranged["point_estimate"].le(ranged["uncertainty_upper"])
    )
    no_test = ~source_manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    expected_ids = [f"ES{i:02d}" for i in range(1, 26)]
    indexed = evidence.set_index("claim_id")
    decision_index = decisions.set_index("decision")
    rows = [
        ("ten_sources", len(source_manifest) == 10, "ten compact frozen source tables"),
        ("twenty_five_claims", evidence["claim_id"].tolist() == expected_ids, "ES01 through ES25 present once"),
        ("allowed_grades", evidence["evidence_grade"].isin(GRADES).all(), "all grades follow protocol"),
        ("point_inside_stored_range", bool(inside.all()), f"{len(ranged)} ranged claims"),
        ("one_no_interval_claim", evidence["evidence_grade"].eq("no_direct_uncertainty_interval").sum() == 1 and indexed.loc["ES15", "evidence_grade"] == "no_direct_uncertainty_interval", "ES15 retained without invented interval"),
        ("summary_counts", int(summary[summary["dimension"].eq("evidence_grade")]["count"].sum()) == 25, "grade counts sum to 25"),
        ("flags_complete", len(flags) == int((~evidence["evidence_grade"].eq("gate_supported_by_uncertainty")).sum()), "all non-fully-supported grades flagged"),
        ("clean_advance_failure_supported", indexed.loc["ES14", "evidence_grade"] == "gate_failure_supported_by_uncertainty", "clean +0.05 F1 gate ruled out by interval"),
        ("boundary_nuance_preserved", indexed.loc["ES19", "evidence_grade"] == "point_gate_only" and indexed.loc["ES19", "zero_relation"] == "positive", "positive direction but material gate not interval-supported"),
        ("nonadvance_unchanged", decision_index.loc["deployable_wearable_method_advances", "outcome"] == "do_not_advance" and decision_index.loc["block8_closeout", "outcome"] == "remain_closed", "Block 8 closeout and nonadvance retained"),
        ("source_paths_exclude_test", bool(no_test), "no current-test source path"),
    ]
    checks = pd.DataFrame(rows, columns=["check", "status", "detail"])
    checks["status"] = checks["status"].map({True: "pass", False: "fail"})
    return checks


def report(
    evidence: pd.DataFrame, summary: pd.DataFrame, decisions: pd.DataFrame
) -> str:
    grade_counts = summary[summary["dimension"].eq("evidence_grade")].set_index(
        "category"
    )["count"]
    point_only = int(grade_counts.get("point_gate_only", 0))
    supported = int(grade_counts.get("gate_supported_by_uncertainty", 0))
    supported_failures = int(
        grade_counts.get("gate_failure_supported_by_uncertainty", 0)
    )
    overlap_failures = int(grade_counts.get("point_failure_with_gate_overlap", 0))
    lines = "\n".join(
        f"- `{row.decision}`: {row.outcome} - {row.basis}"
        for row in decisions.itertuples(index=False)
    )
    return f"""# Block 8 Participant Evidence Strength v0.1

**Work date:** 2026-09-18
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Claims reviewed:** {len(evidence)}
**New fitting, scoring, resampling, or threshold selection:** None
**Current test accessed:** No

## Evidence Grades

- {supported} gates are supported by the full stored participant range.
- {point_only} gates pass only at the aggregate point estimate.
- {supported_failures} gate failures are supported by the full stored participant range.
- {overlap_failures} failed point gates have ranges that overlap the gate.
- One joint degradation-gap claim has no direct stored difference-in-differences interval.

## Block Decisions

{lines}

## Main Interpretation

The clean +0.05 F1 advancement gate for noise augmentation is ruled out by its participant interval, so wearable non-advancement is strengthened. The boundary-tolerance effect remains directionally positive, but its participant interval does not clear the original +0.10 materiality gate. Boundary uncertainty therefore remains a justified priority without being described as participant-supported material gain.

This analysis grades existing evidence only. It does not replace the original point decisions or reopen Block 8 development.
"""


def main() -> None:
    paths = source_paths()
    source_manifest = manifest(paths)
    evidence = evidence_strength(paths)
    summary = strength_summary(evidence)
    flags = interpretation_flags(evidence)
    decisions = block_decisions(evidence)
    checks = analysis_checks(source_manifest, evidence, summary, flags, decisions)
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Evidence-strength checks failed: {failed}")

    outputs = {
        "input_artifact_manifest_v0.1.tsv": source_manifest,
        "evidence_strength_v0.1.tsv": evidence,
        "evidence_strength_summary_v0.1.tsv": summary,
        "interpretation_flags_v0.1.tsv": flags,
        "block8_decisions_v0.1.tsv": decisions,
        "analysis_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
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
        output_dir() / "README.md", report(evidence, summary, decisions)
    )

    print(summary[summary["dimension"].eq("evidence_grade")].to_string(index=False))
    print(f"Analysis checks: {len(checks)}/{len(checks)} pass")


if __name__ == "__main__":
    main()
