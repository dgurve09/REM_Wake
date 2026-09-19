"""Independently reconstruct the Block 8 robustness synthesis."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: independent source configuration

EXPERIMENT_DIR = "2026-09-18_block8_robustness_synthesis_v0.1"
FOLDERS = {
    "single_channel": "2026-09-11_block8_single_channel_robustness_v0.1",
    "raw_noise": "2026-09-11_block8_raw_signal_noise_v0.1",
    "noise_augmentation": "2026-09-12_block8_noise_augmented_training_v0.1",
    "event_overlap": "2026-09-13_block8_cross_modality_event_overlap_v0.1",
    "alarm_fusion": "2026-09-13_block8_fixed_alarm_fusion_v0.1",
}
EXPECTED = {
    "single_channel": 26,
    "raw_noise": 29,
    "noise_augmentation": 38,
    "event_overlap": 19,
    "alarm_fusion": 20,
}


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def source(key: str, name: str) -> Path:
    return root() / "experiments" / FOLDERS[key] / name


def paths() -> dict[str, Path]:
    return {
        "single_comparisons": source(
            "single_channel", "primary_ablation_comparisons_v0.1.tsv"
        ),
        "single_in_run_checks": source("single_channel", "in_run_checks_v0.1.tsv"),
        "single_independent_checks": source(
            "single_channel", "output_integrity_checks_v0.1.tsv"
        ),
        "raw_decisions": source("raw_noise", "hypothesis_decisions_v0.1.tsv"),
        "raw_metrics": source("raw_noise", "validation_event_metrics_v0.1.tsv"),
        "raw_bootstrap": source("raw_noise", "paired_participant_bootstrap_v0.1.tsv"),
        "raw_in_run_checks": source("raw_noise", "in_run_checks_v0.1.tsv"),
        "raw_independent_checks": source(
            "raw_noise", "output_integrity_checks_v0.1.tsv"
        ),
        "augmentation_decisions": source(
            "noise_augmentation", "hypothesis_decisions_v0.1.tsv"
        ),
        "augmentation_metrics": source(
            "noise_augmentation", "validation_event_metrics_v0.1.tsv"
        ),
        "augmentation_bootstrap": source(
            "noise_augmentation", "paired_participant_bootstrap_v0.1.tsv"
        ),
        "augmentation_train_checks": source(
            "noise_augmentation", "train_phase_checks_v0.1.tsv"
        ),
        "augmentation_validation_checks": source(
            "noise_augmentation", "validation_phase_checks_v0.1.tsv"
        ),
        "augmentation_independent_checks": source(
            "noise_augmentation", "output_integrity_checks_v0.1.tsv"
        ),
        "overlap_decisions": source(
            "event_overlap", "hypothesis_decisions_v0.1.tsv"
        ),
        "overlap_bootstrap": source(
            "event_overlap", "participant_bootstrap_v0.1.tsv"
        ),
        "overlap_in_run_checks": source("event_overlap", "in_run_checks_v0.1.tsv"),
        "overlap_independent_checks": source(
            "event_overlap", "output_integrity_checks_v0.1.tsv"
        ),
        "fusion_decisions": source("alarm_fusion", "hypothesis_decisions_v0.1.tsv"),
        "fusion_metrics": source("alarm_fusion", "event_metrics_v0.1.tsv"),
        "fusion_bootstrap": source(
            "alarm_fusion", "paired_participant_bootstrap_v0.1.tsv"
        ),
        "fusion_in_run_checks": source("alarm_fusion", "in_run_checks_v0.1.tsv"),
        "fusion_independent_checks": source(
            "alarm_fusion", "output_integrity_checks_v0.1.tsv"
        ),
    }


CHECKS = {
    "single_channel": ["single_in_run_checks", "single_independent_checks"],
    "raw_noise": ["raw_in_run_checks", "raw_independent_checks"],
    "noise_augmentation": [
        "augmentation_train_checks",
        "augmentation_validation_checks",
        "augmentation_independent_checks",
    ],
    "event_overlap": ["overlap_in_run_checks", "overlap_independent_checks"],
    "alarm_fusion": ["fusion_in_run_checks", "fusion_independent_checks"],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"Expected Boolean value, found {value!r}")
    return text == "true"


def one(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    result = frame[frame[column].eq(value)]
    if len(result) != 1:
        raise ValueError(f"Expected one {column}={value} row")
    return result.iloc[0]


# Section 2: independent manifest and check census

def rebuild_manifest(source_paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_repository": path.relative_to(root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in source_paths.items()
        ]
    ).sort_values("artifact_role").reset_index(drop=True)


def rebuild_census(source_paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for experiment, roles in CHECKS.items():
        tables = [pd.read_csv(source_paths[role], sep="\t") for role in roles]
        recorded = sum(len(table) for table in tables)
        passing = sum(
            int(table["status"].astype(str).str.lower().eq("pass").sum())
            for table in tables
        )
        failing = recorded - passing
        rows.append(
            {
                "experiment": experiment,
                "check_tables": len(roles),
                "expected_checks": EXPECTED[experiment],
                "recorded_checks": recorded,
                "passing_checks": passing,
                "failing_checks": failing,
                "complete_and_passing": recorded == EXPECTED[experiment]
                and passing == recorded,
            }
        )
    frame = pd.DataFrame(rows)
    frame.loc[len(frame)] = {
        "experiment": "TOTAL",
        "check_tables": int(frame["check_tables"].sum()),
        "expected_checks": int(frame["expected_checks"].sum()),
        "recorded_checks": int(frame["recorded_checks"].sum()),
        "passing_checks": int(frame["passing_checks"].sum()),
        "failing_checks": int(frame["failing_checks"].sum()),
        "complete_and_passing": bool(frame["complete_and_passing"].all()),
    }
    return frame


# Section 3: independent evidence reconstruction

def raw_metric(frame: pd.DataFrame, condition: str) -> pd.Series:
    result = frame[
        frame["comparator"].eq(condition)
        & frame["membership"].eq("primary")
        & frame["tolerance_sec"].eq(15.0)
    ]
    if len(result) != 1:
        raise ValueError(f"Expected one raw metric for {condition}")
    return result.iloc[0]


def rebuild_evidence(source_paths: dict[str, Path]) -> pd.DataFrame:
    single = pd.read_csv(source_paths["single_comparisons"], sep="\t")
    raw_d = pd.read_csv(source_paths["raw_decisions"], sep="\t")
    raw_m = pd.read_csv(source_paths["raw_metrics"], sep="\t")
    aug = pd.read_csv(source_paths["augmentation_decisions"], sep="\t")
    overlap = pd.read_csv(source_paths["overlap_decisions"], sep="\t")
    overlap_boot = pd.read_csv(source_paths["overlap_bootstrap"], sep="\t")
    fusion = pd.read_csv(source_paths["fusion_decisions"], sep="\t")
    fusion_boot = pd.read_csv(source_paths["fusion_bootstrap"], sep="\t")

    ablation = one(single, "ablated_channel", "HB_1")
    mild = one(raw_d, "hypothesis", "H8.3_mild_noise_tolerance")
    clean = raw_metric(raw_m, "H2-CLEAN")
    severe = raw_metric(raw_m, "H2-BOTH-0DB")
    hb1 = raw_metric(raw_m, "H2-HB1-10DB")
    mitigation = one(aug, "hypothesis", "H8.7_hb1_failure_mitigation")
    clean_gate = one(aug, "hypothesis", "H8.9_meaningful_clean_advancement")
    psg_gap = one(overlap, "hypothesis", "H8.11_wearable_specific_recovery_gap")
    boundary = one(overlap, "hypothesis", "H8.13_boundary_tolerance_sensitivity")
    boundary_boot = one(overlap_boot, "metric", "boundary_union_recall_gain")
    or_gate = one(fusion, "hypothesis", "H8.14_two_modality_or_advancement")
    consensus = one(fusion, "hypothesis", "H8.15_consensus_advancement")
    f1_interval = fusion_boot[
        fusion_boot["comparison"].eq("P6-P2-H2-2OF3_minus_P6-D_at_15sec")
        & fusion_boot["metric"].eq("f1")
    ].iloc[0]
    far_interval = fusion_boot[
        fusion_boot["comparison"].eq("P6-P2-H2-2OF3_minus_P6-D_at_15sec")
        & fusion_boot["metric"].eq("false_alarms_per_hour")
    ].iloc[0]

    common = [
        (
            "B8-E1",
            "single_channel",
            "HB_1 neutralization causes a material clean F1 loss",
            "f1_difference",
            ablation.f1_difference,
            "material_drop_bound",
            -float(ablation.material_f1_drop_bound),
            as_bool(ablation.materially_adverse),
            "direct",
            "feature neutralization is not physical channel loss",
        ),
        (
            "B8-E2",
            "raw_noise",
            "Mild both-channel noise passes fixed material bounds",
            "f1_difference",
            mild.value_1,
            "far_difference_per_hour",
            mild.value_2,
            as_bool(mild.supported),
            "controlled_robustness",
            "white noise is not natural wearable artefact",
        ),
        (
            "B8-E3",
            "raw_noise",
            "Severe both-channel noise eliminates primary true detections",
            "true_positive",
            severe.true_positive,
            "f1",
            severe.f1,
            int(severe.true_positive) == 0,
            "direct",
            "0 dB Gaussian noise is a stress condition",
        ),
        (
            "B8-E4",
            "raw_noise",
            "Isolated HB_1 noise creates a high-alarm failure",
            "f1_difference",
            hb1.f1 - clean.f1,
            "far_difference_per_hour",
            hb1.false_alarms_per_hour - clean.false_alarms_per_hour,
            hb1.f1 < clean.f1
            and hb1.false_alarms_per_hour > clean.false_alarms_per_hour,
            "direct",
            "controlled isolated noise does not reproduce field failure",
        ),
        (
            "B8-E5",
            "noise_augmentation",
            "Train-only noise augmentation mitigates the HB_1 failure",
            "f1_difference",
            mitigation.value_1,
            "far_difference_per_hour",
            mitigation.value_2,
            as_bool(mitigation.supported),
            "degraded_condition_only",
            "other degradation conditions gained false alarms",
        ),
        (
            "B8-E6",
            "noise_augmentation",
            "Noise augmentation fails meaningful clean advancement",
            "clean_f1_difference",
            clean_gate.value_1,
            "clean_far_difference_per_hour",
            clean_gate.value_2,
            not as_bool(clean_gate.supported),
            "core_detector",
            "generic augmentation does not improve clean separability",
        ),
        (
            "B8-E7",
            "event_overlap",
            "Direct PSG recovers a material subset missed by wearable",
            "psg_only_fraction",
            psg_gap.observed,
            "fixed_gate",
            psg_gap.pass_threshold,
            as_bool(psg_gap.supported),
            "information_gap",
            "participant interval crosses the fixed gate",
        ),
        (
            "B8-E8",
            "event_overlap",
            "Boundary tolerance materially changes union recall",
            "recall_gain",
            boundary.observed,
            "bootstrap_lower_95",
            boundary_boot.lower_95,
            as_bool(boundary.supported) and float(boundary_boot.lower_95) > 0,
            "label_timing",
            "tolerance sensitivity does not improve precision",
        ),
        (
            "B8-E9",
            "alarm_fusion",
            "Fixed OR fusion worsens F1 and false-alarm burden",
            "f1_difference",
            or_gate.value_1,
            "far_difference_per_hour",
            or_gate.value_2,
            float(or_gate.value_1) < 0
            and float(or_gate.value_2) > 0
            and not as_bool(or_gate.supported),
            "nondeployable_fusion",
            "fusion includes laboratory PSG",
        ),
        (
            "B8-E10",
            "alarm_fusion",
            "Laboratory consensus suppresses FAR but F1 gain is inconclusive",
            "f1_interval_lower",
            f1_interval.lower_95,
            "far_interval_upper",
            far_interval.upper_95,
            as_bool(consensus.supported)
            and float(f1_interval.lower_95) <= 0 <= float(f1_interval.upper_95)
            and float(far_interval.upper_95) < 0,
            "mechanism_only",
            "requires PSG and F1 interval crosses zero",
        ),
    ]
    return pd.DataFrame(
        common,
        columns=[
            "evidence_id",
            "source_experiment",
            "statement",
            "value_1_name",
            "value_1",
            "value_2_name",
            "value_2",
            "supported",
            "wearable_relevance",
            "remaining_limitation",
        ],
    )


# Section 4: independent decisions and unresolved register

def rebuild_decisions(evidence: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    indexed = evidence.set_index("evidence_id")
    robustness = not any(
        as_bool(indexed.loc[item, "supported"])
        for item in ["B8-E1", "B8-E3", "B8-E4"]
    )
    wearable_advance = not as_bool(indexed.loc["B8-E6", "supported"])
    boundary = as_bool(indexed.loc["B8-E8", "supported"])
    total = census[census["experiment"].eq("TOTAL")].iloc[0]
    close = (
        len(evidence) == 10
        and evidence["supported"].map(as_bool).all()
        and as_bool(total.complete_and_passing)
        and int(total.recorded_checks) == 132
    )
    return pd.DataFrame(
        [
            ("C8.1_wearable_robustness_sufficient", robustness, "pass" if robustness else "fail", "material channel dependence and severe/channel-specific noise failures remain"),
            ("C8.2_deployable_wearable_method_advances", wearable_advance, "advance" if wearable_advance else "do_not_advance", "augmentation failed clean advancement and advancing fusion requires PSG"),
            ("C8.3_boundary_uncertainty_priority", boundary, "prioritize" if boundary else "do_not_prioritize", "tolerance changed union recall with participant interval above zero"),
            ("C8.4_block8_closeout_complete", close, "close" if close else "remain_open", "five experiments, ten evidence statements, and 132 integrity checks reviewed"),
        ],
        columns=["decision", "supported", "outcome", "basis"],
    )


def rebuild_unresolved() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("external_psg_compatibility", "unresolved", "Block 8 uses one paired dataset", "perform the scheduled Block 9 dataset compatibility audit", "not started before 2026-09-21"),
            ("stage_derived_boundary_timing", "unresolved_prioritized", "B8-E8 tolerance sensitivity", "predeclare interval-aware label and alarm-burden comparison", "planned Block 10 work; not performed"),
            ("wearable_only_temporal_specificity", "unresolved", "consensus FAR mechanism requires PSG", "test a bounded wearable temporal-representation hypothesis only with a new confirmation boundary", "conditional future work; not performed"),
            ("natural_wearable_artefacts", "unresolved", "controlled Gaussian noise is not a field artefact model", "identify externally observed or annotated artefact evidence before another robustness model", "no current experiment scheduled"),
            ("independent_wearable_confirmation", "unresolved", "current validation and test cohorts have both informed prior work", "require a new locked or external wearable cohort for confirmation", "not available in current Block 8 evidence"),
        ],
        columns=[
            "uncertainty",
            "status",
            "evidence_basis",
            "next_action",
            "timing_boundary",
        ],
    )


# Section 5: output comparison and validation report

def frames_match(expected: pd.DataFrame, actual: pd.DataFrame) -> bool:
    if list(expected.columns) != list(actual.columns) or expected.shape != actual.shape:
        return False
    for column in expected.columns:
        left = expected[column].reset_index(drop=True)
        right = actual[column].reset_index(drop=True)
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
    source_paths = paths()
    manifest = rebuild_manifest(source_paths)
    census = rebuild_census(source_paths)
    evidence = rebuild_evidence(source_paths)
    decisions = rebuild_decisions(evidence, census)
    unresolved = rebuild_unresolved()
    output = output_dir()

    recorded_manifest = pd.read_csv(output / "input_artifact_manifest_v0.1.tsv", sep="\t")
    recorded_census = pd.read_csv(output / "integrity_check_census_v0.1.tsv", sep="\t")
    recorded_evidence = pd.read_csv(output / "evidence_ledger_v0.1.tsv", sep="\t")
    recorded_decisions = pd.read_csv(output / "block8_closeout_decisions_v0.1.tsv", sep="\t")
    recorded_unresolved = pd.read_csv(output / "unresolved_uncertainties_v0.1.tsv", sep="\t")
    recorded_checks = pd.read_csv(output / "synthesis_checks_v0.1.tsv", sep="\t")

    no_test_path = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    no_binary = ~recorded_manifest["path_relative_to_repository"].str.contains(
        r"\.(?:edf|npz|joblib|pth|pt|tsv\.gz|csv\.gz)$", case=False, regex=True
    ).any()
    total = census[census["experiment"].eq("TOTAL")].iloc[0]
    synthesis_checks_valid = (
        len(recorded_checks) == 9
        and recorded_checks["status"].astype(str).str.lower().eq("pass").all()
    )
    checks = pd.DataFrame(
        [
            ("input_manifest_rehashed", frames_match(manifest, recorded_manifest), "23 frozen source artifacts"),
            ("integrity_census_reconstructed", frames_match(census, recorded_census), "five experiments plus total"),
            ("source_checks_all_pass", int(total.recorded_checks) == 132 and int(total.passing_checks) == 132 and int(total.failing_checks) == 0, "132/132 checks"),
            ("evidence_ledger_reconstructed", frames_match(evidence, recorded_evidence), "ten evidence statements"),
            ("block_decisions_reconstructed", frames_match(decisions, recorded_decisions), "four closeout decisions"),
            ("unresolved_register_reconstructed", frames_match(unresolved, recorded_unresolved), "five unresolved uncertainties"),
            ("synthesis_checks_valid", synthesis_checks_valid, "nine synthesis checks pass"),
            ("source_paths_exclude_test", bool(no_test_path), "no current-test path"),
            ("compact_sources_only", bool(no_binary), "no raw, model, feature-array, or compressed-score input"),
        ],
        columns=["check", "status", "detail"],
    )
    checks["status"] = checks["status"].map({True: "pass", False: "fail"})
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Independent synthesis validation failed: {failed}")

    verify_or_create_tsv(checks, output / "output_integrity_checks_v0.1.tsv")
    report = f"""# Block 8 Robustness Synthesis Output Validation

**Validation date:** 2026-09-18
**Method:** Independent source rehashing, integrity census, evidence reconstruction, and closeout-decision reconstruction

All {len(checks)}/{len(checks)} independent checks passed.

The validator opened only compact reviewed Block 8 tables. It did not open raw signals, feature arrays, models, full scores, probabilities, or current-test artifacts.
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
