"""Synthesize frozen Block 8 robustness evidence into a closeout decision."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: frozen synthesis configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-18_block8_robustness_synthesis_v0.1"
PROTOCOL_COMMIT = "64e8ca9"

EXPERIMENTS = {
    "single_channel": "2026-09-11_block8_single_channel_robustness_v0.1",
    "raw_noise": "2026-09-11_block8_raw_signal_noise_v0.1",
    "noise_augmentation": "2026-09-12_block8_noise_augmented_training_v0.1",
    "event_overlap": "2026-09-13_block8_cross_modality_event_overlap_v0.1",
    "alarm_fusion": "2026-09-13_block8_fixed_alarm_fusion_v0.1",
}

EXPECTED_CHECKS = {
    "single_channel": 26,
    "raw_noise": 29,
    "noise_augmentation": 38,
    "event_overlap": 19,
    "alarm_fusion": 20,
}


# Section 2: paths and source hashes

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def experiment_path(key: str, name: str) -> Path:
    return repo_root() / "experiments" / EXPERIMENTS[key] / name


def source_paths() -> dict[str, Path]:
    return {
        "single_comparisons": experiment_path(
            "single_channel", "primary_ablation_comparisons_v0.1.tsv"
        ),
        "single_in_run_checks": experiment_path(
            "single_channel", "in_run_checks_v0.1.tsv"
        ),
        "single_independent_checks": experiment_path(
            "single_channel", "output_integrity_checks_v0.1.tsv"
        ),
        "raw_decisions": experiment_path(
            "raw_noise", "hypothesis_decisions_v0.1.tsv"
        ),
        "raw_metrics": experiment_path(
            "raw_noise", "validation_event_metrics_v0.1.tsv"
        ),
        "raw_bootstrap": experiment_path(
            "raw_noise", "paired_participant_bootstrap_v0.1.tsv"
        ),
        "raw_in_run_checks": experiment_path(
            "raw_noise", "in_run_checks_v0.1.tsv"
        ),
        "raw_independent_checks": experiment_path(
            "raw_noise", "output_integrity_checks_v0.1.tsv"
        ),
        "augmentation_decisions": experiment_path(
            "noise_augmentation", "hypothesis_decisions_v0.1.tsv"
        ),
        "augmentation_metrics": experiment_path(
            "noise_augmentation", "validation_event_metrics_v0.1.tsv"
        ),
        "augmentation_bootstrap": experiment_path(
            "noise_augmentation", "paired_participant_bootstrap_v0.1.tsv"
        ),
        "augmentation_train_checks": experiment_path(
            "noise_augmentation", "train_phase_checks_v0.1.tsv"
        ),
        "augmentation_validation_checks": experiment_path(
            "noise_augmentation", "validation_phase_checks_v0.1.tsv"
        ),
        "augmentation_independent_checks": experiment_path(
            "noise_augmentation", "output_integrity_checks_v0.1.tsv"
        ),
        "overlap_decisions": experiment_path(
            "event_overlap", "hypothesis_decisions_v0.1.tsv"
        ),
        "overlap_bootstrap": experiment_path(
            "event_overlap", "participant_bootstrap_v0.1.tsv"
        ),
        "overlap_in_run_checks": experiment_path(
            "event_overlap", "in_run_checks_v0.1.tsv"
        ),
        "overlap_independent_checks": experiment_path(
            "event_overlap", "output_integrity_checks_v0.1.tsv"
        ),
        "fusion_decisions": experiment_path(
            "alarm_fusion", "hypothesis_decisions_v0.1.tsv"
        ),
        "fusion_metrics": experiment_path(
            "alarm_fusion", "event_metrics_v0.1.tsv"
        ),
        "fusion_bootstrap": experiment_path(
            "alarm_fusion", "paired_participant_bootstrap_v0.1.tsv"
        ),
        "fusion_in_run_checks": experiment_path(
            "alarm_fusion", "in_run_checks_v0.1.tsv"
        ),
        "fusion_independent_checks": experiment_path(
            "alarm_fusion", "output_integrity_checks_v0.1.tsv"
        ),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_repository": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in paths.items()
        ]
    ).sort_values("artifact_role").reset_index(drop=True)


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text not in {"true", "false"}:
        raise ValueError(f"Expected Boolean value, found {value!r}")
    return text == "true"


# Section 3: integrity-check census

CHECK_ROLES = {
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


def check_census(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for experiment, roles in CHECK_ROLES.items():
        total = 0
        passing = 0
        failing = 0
        for role in roles:
            frame = pd.read_csv(paths[role], sep="\t")
            total += len(frame)
            passing += int(frame["status"].astype(str).str.lower().eq("pass").sum())
            failing += int(
                (~frame["status"].astype(str).str.lower().eq("pass")).sum()
            )
        rows.append(
            {
                "experiment": experiment,
                "check_tables": len(roles),
                "expected_checks": EXPECTED_CHECKS[experiment],
                "recorded_checks": total,
                "passing_checks": passing,
                "failing_checks": failing,
                "complete_and_passing": total == EXPECTED_CHECKS[experiment]
                and passing == total
                and failing == 0,
            }
        )
    result = pd.DataFrame(rows)
    result.loc[len(result)] = {
        "experiment": "TOTAL",
        "check_tables": int(result["check_tables"].sum()),
        "expected_checks": int(result["expected_checks"].sum()),
        "recorded_checks": int(result["recorded_checks"].sum()),
        "passing_checks": int(result["passing_checks"].sum()),
        "failing_checks": int(result["failing_checks"].sum()),
        "complete_and_passing": bool(result["complete_and_passing"].all()),
    }
    return result


# Section 4: fixed evidence reconstruction

def selected_row(frame: pd.DataFrame, column: str, value: str) -> pd.Series:
    rows = frame[frame[column].eq(value)]
    if len(rows) != 1:
        raise ValueError(f"Expected one row where {column}={value}, found {len(rows)}")
    return rows.iloc[0]


def primary_metric(frame: pd.DataFrame, comparator: str) -> pd.Series:
    rows = frame[
        frame["comparator"].eq(comparator)
        & frame["membership"].eq("primary")
        & frame["tolerance_sec"].eq(15.0)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one primary metric for {comparator}")
    return rows.iloc[0]


def evidence_ledger(paths: dict[str, Path]) -> pd.DataFrame:
    single = pd.read_csv(paths["single_comparisons"], sep="\t")
    raw_decisions = pd.read_csv(paths["raw_decisions"], sep="\t")
    raw_metrics = pd.read_csv(paths["raw_metrics"], sep="\t")
    augmentation = pd.read_csv(paths["augmentation_decisions"], sep="\t")
    overlap = pd.read_csv(paths["overlap_decisions"], sep="\t")
    overlap_bootstrap = pd.read_csv(paths["overlap_bootstrap"], sep="\t")
    fusion = pd.read_csv(paths["fusion_decisions"], sep="\t")
    fusion_bootstrap = pd.read_csv(paths["fusion_bootstrap"], sep="\t")

    hb1_ablation = selected_row(single, "ablated_channel", "HB_1")
    mild_noise = selected_row(raw_decisions, "hypothesis", "H8.3_mild_noise_tolerance")
    severe_noise = primary_metric(raw_metrics, "H2-BOTH-0DB")
    clean_noise = primary_metric(raw_metrics, "H2-CLEAN")
    hb1_noise = primary_metric(raw_metrics, "H2-HB1-10DB")
    mitigation = selected_row(
        augmentation, "hypothesis", "H8.7_hb1_failure_mitigation"
    )
    clean_advance = selected_row(
        augmentation, "hypothesis", "H8.9_meaningful_clean_advancement"
    )
    psg_gap = selected_row(
        overlap, "hypothesis", "H8.11_wearable_specific_recovery_gap"
    )
    boundary = selected_row(
        overlap, "hypothesis", "H8.13_boundary_tolerance_sensitivity"
    )
    boundary_interval = selected_row(
        overlap_bootstrap, "metric", "boundary_union_recall_gain"
    )
    or_gate = selected_row(
        fusion, "hypothesis", "H8.14_two_modality_or_advancement"
    )
    consensus = selected_row(
        fusion, "hypothesis", "H8.15_consensus_advancement"
    )
    consensus_f1 = fusion_bootstrap[
        fusion_bootstrap["comparison"].eq(
            "P6-P2-H2-2OF3_minus_P6-D_at_15sec"
        )
        & fusion_bootstrap["metric"].eq("f1")
    ].iloc[0]
    consensus_far = fusion_bootstrap[
        fusion_bootstrap["comparison"].eq(
            "P6-P2-H2-2OF3_minus_P6-D_at_15sec"
        )
        & fusion_bootstrap["metric"].eq("false_alarms_per_hour")
    ].iloc[0]

    rows = [
        {
            "evidence_id": "B8-E1",
            "source_experiment": "single_channel",
            "statement": "HB_1 neutralization causes a material clean F1 loss",
            "value_1_name": "f1_difference",
            "value_1": hb1_ablation.f1_difference,
            "value_2_name": "material_drop_bound",
            "value_2": -float(hb1_ablation.material_f1_drop_bound),
            "supported": as_bool(hb1_ablation.materially_adverse),
            "wearable_relevance": "direct",
            "remaining_limitation": "feature neutralization is not physical channel loss",
        },
        {
            "evidence_id": "B8-E2",
            "source_experiment": "raw_noise",
            "statement": "Mild both-channel noise passes fixed material bounds",
            "value_1_name": "f1_difference",
            "value_1": mild_noise.value_1,
            "value_2_name": "far_difference_per_hour",
            "value_2": mild_noise.value_2,
            "supported": as_bool(mild_noise.supported),
            "wearable_relevance": "controlled_robustness",
            "remaining_limitation": "white noise is not natural wearable artefact",
        },
        {
            "evidence_id": "B8-E3",
            "source_experiment": "raw_noise",
            "statement": "Severe both-channel noise eliminates primary true detections",
            "value_1_name": "true_positive",
            "value_1": severe_noise.true_positive,
            "value_2_name": "f1",
            "value_2": severe_noise.f1,
            "supported": int(severe_noise.true_positive) == 0,
            "wearable_relevance": "direct",
            "remaining_limitation": "0 dB Gaussian noise is a stress condition",
        },
        {
            "evidence_id": "B8-E4",
            "source_experiment": "raw_noise",
            "statement": "Isolated HB_1 noise creates a high-alarm failure",
            "value_1_name": "f1_difference",
            "value_1": hb1_noise.f1 - clean_noise.f1,
            "value_2_name": "far_difference_per_hour",
            "value_2": hb1_noise.false_alarms_per_hour
            - clean_noise.false_alarms_per_hour,
            "supported": (hb1_noise.f1 < clean_noise.f1)
            and (
                hb1_noise.false_alarms_per_hour
                > clean_noise.false_alarms_per_hour
            ),
            "wearable_relevance": "direct",
            "remaining_limitation": "controlled isolated noise does not reproduce field failure",
        },
        {
            "evidence_id": "B8-E5",
            "source_experiment": "noise_augmentation",
            "statement": "Train-only noise augmentation mitigates the HB_1 failure",
            "value_1_name": "f1_difference",
            "value_1": mitigation.value_1,
            "value_2_name": "far_difference_per_hour",
            "value_2": mitigation.value_2,
            "supported": as_bool(mitigation.supported),
            "wearable_relevance": "degraded_condition_only",
            "remaining_limitation": "other degradation conditions gained false alarms",
        },
        {
            "evidence_id": "B8-E6",
            "source_experiment": "noise_augmentation",
            "statement": "Noise augmentation fails meaningful clean advancement",
            "value_1_name": "clean_f1_difference",
            "value_1": clean_advance.value_1,
            "value_2_name": "clean_far_difference_per_hour",
            "value_2": clean_advance.value_2,
            "supported": not as_bool(clean_advance.supported),
            "wearable_relevance": "core_detector",
            "remaining_limitation": "generic augmentation does not improve clean separability",
        },
        {
            "evidence_id": "B8-E7",
            "source_experiment": "event_overlap",
            "statement": "Direct PSG recovers a material subset missed by wearable",
            "value_1_name": "psg_only_fraction",
            "value_1": psg_gap.observed,
            "value_2_name": "fixed_gate",
            "value_2": psg_gap.pass_threshold,
            "supported": as_bool(psg_gap.supported),
            "wearable_relevance": "information_gap",
            "remaining_limitation": "participant interval crosses the fixed gate",
        },
        {
            "evidence_id": "B8-E8",
            "source_experiment": "event_overlap",
            "statement": "Boundary tolerance materially changes union recall",
            "value_1_name": "recall_gain",
            "value_1": boundary.observed,
            "value_2_name": "bootstrap_lower_95",
            "value_2": boundary_interval.lower_95,
            "supported": as_bool(boundary.supported)
            and float(boundary_interval.lower_95) > 0,
            "wearable_relevance": "label_timing",
            "remaining_limitation": "tolerance sensitivity does not improve precision",
        },
        {
            "evidence_id": "B8-E9",
            "source_experiment": "alarm_fusion",
            "statement": "Fixed OR fusion worsens F1 and false-alarm burden",
            "value_1_name": "f1_difference",
            "value_1": or_gate.value_1,
            "value_2_name": "far_difference_per_hour",
            "value_2": or_gate.value_2,
            "supported": (float(or_gate.value_1) < 0)
            and (float(or_gate.value_2) > 0)
            and not as_bool(or_gate.supported),
            "wearable_relevance": "nondeployable_fusion",
            "remaining_limitation": "fusion includes laboratory PSG",
        },
        {
            "evidence_id": "B8-E10",
            "source_experiment": "alarm_fusion",
            "statement": "Laboratory consensus suppresses FAR but F1 gain is inconclusive",
            "value_1_name": "f1_interval_lower",
            "value_1": consensus_f1.lower_95,
            "value_2_name": "far_interval_upper",
            "value_2": consensus_far.upper_95,
            "supported": as_bool(consensus.supported)
            and float(consensus_f1.lower_95) <= 0
            and float(consensus_f1.upper_95) >= 0
            and float(consensus_far.upper_95) < 0,
            "wearable_relevance": "mechanism_only",
            "remaining_limitation": "requires PSG and F1 interval crosses zero",
        },
    ]
    return pd.DataFrame(rows)


# Section 5: block decisions and unresolved uncertainties

def block_decisions(evidence: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    e = evidence.set_index("evidence_id")
    robustness_sufficient = not any(
        bool(e.loc[item, "supported"]) for item in ["B8-E1", "B8-E3", "B8-E4"]
    )
    wearable_method_advance = not bool(e.loc["B8-E6", "supported"])
    boundary_priority = bool(e.loc["B8-E8", "supported"])
    total = census[census["experiment"].eq("TOTAL")].iloc[0]
    block_complete = (
        len(evidence) == 10
        and evidence["supported"].all()
        and bool(total.complete_and_passing)
        and int(total.recorded_checks) == 132
    )
    rows = [
        (
            "C8.1_wearable_robustness_sufficient",
            robustness_sufficient,
            "pass" if robustness_sufficient else "fail",
            "material channel dependence and severe/channel-specific noise failures remain",
        ),
        (
            "C8.2_deployable_wearable_method_advances",
            wearable_method_advance,
            "advance" if wearable_method_advance else "do_not_advance",
            "augmentation failed clean advancement and advancing fusion requires PSG",
        ),
        (
            "C8.3_boundary_uncertainty_priority",
            boundary_priority,
            "prioritize" if boundary_priority else "do_not_prioritize",
            "tolerance changed union recall with participant interval above zero",
        ),
        (
            "C8.4_block8_closeout_complete",
            block_complete,
            "close" if block_complete else "remain_open",
            "five experiments, ten evidence statements, and 132 integrity checks reviewed",
        ),
    ]
    return pd.DataFrame(
        rows, columns=["decision", "supported", "outcome", "basis"]
    )


def unresolved_register() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "uncertainty": "external_psg_compatibility",
                "status": "unresolved",
                "evidence_basis": "Block 8 uses one paired dataset",
                "next_action": "perform the scheduled Block 9 dataset compatibility audit",
                "timing_boundary": "not started before 2026-09-21",
            },
            {
                "uncertainty": "stage_derived_boundary_timing",
                "status": "unresolved_prioritized",
                "evidence_basis": "B8-E8 tolerance sensitivity",
                "next_action": "predeclare interval-aware label and alarm-burden comparison",
                "timing_boundary": "planned Block 10 work; not performed",
            },
            {
                "uncertainty": "wearable_only_temporal_specificity",
                "status": "unresolved",
                "evidence_basis": "consensus FAR mechanism requires PSG",
                "next_action": "test a bounded wearable temporal-representation hypothesis only with a new confirmation boundary",
                "timing_boundary": "conditional future work; not performed",
            },
            {
                "uncertainty": "natural_wearable_artefacts",
                "status": "unresolved",
                "evidence_basis": "controlled Gaussian noise is not a field artefact model",
                "next_action": "identify externally observed or annotated artefact evidence before another robustness model",
                "timing_boundary": "no current experiment scheduled",
            },
            {
                "uncertainty": "independent_wearable_confirmation",
                "status": "unresolved",
                "evidence_basis": "current validation and test cohorts have both informed prior work",
                "next_action": "require a new locked or external wearable cohort for confirmation",
                "timing_boundary": "not available in current Block 8 evidence",
            },
        ]
    )


# Section 6: synthesis integrity and reviewed report

def synthesis_checks(
    manifest: pd.DataFrame,
    census: pd.DataFrame,
    evidence: pd.DataFrame,
    decisions: pd.DataFrame,
    unresolved: pd.DataFrame,
) -> pd.DataFrame:
    total = census[census["experiment"].eq("TOTAL")].iloc[0]
    no_test_path = ~manifest["path_relative_to_repository"].str.contains(
        r"(?:^|/)test(?:$|/)", case=False, regex=True
    ).any()
    prohibited_extension = manifest["path_relative_to_repository"].str.contains(
        r"\.(?:edf|npz|joblib|pth|pt|tsv\.gz|csv\.gz)$", case=False, regex=True
    ).any()
    rows = [
        ("five_experiments_present", len(census) == 6, "five experiments plus total row"),
        ("integrity_check_census", int(total.recorded_checks) == 132 and int(total.passing_checks) == 132 and int(total.failing_checks) == 0, "132/132 source checks pass"),
        ("ten_evidence_statements", len(evidence) == 10 and evidence["supported"].all(), "all ten source statements reconstructed"),
        ("four_block_decisions", len(decisions) == 4, "three technical decisions plus closeout decision"),
        ("wearable_nonadvance_preserved", decisions.set_index("decision").loc["C8.2_deployable_wearable_method_advances", "outcome"] == "do_not_advance", "no wearable detector advance"),
        ("block_closeout_not_success_claim", decisions.set_index("decision").loc["C8.4_block8_closeout_complete", "outcome"] == "close" and not bool(decisions.set_index("decision").loc["C8.2_deployable_wearable_method_advances", "supported"]), "closeout separated from performance success"),
        ("unresolved_register", len(unresolved) == 5 and unresolved["status"].str.startswith("unresolved").all(), "five unresolved uncertainties retained"),
        ("source_paths_exclude_test", bool(no_test_path), "no current-test path in source manifest"),
        ("compact_sources_only", not bool(prohibited_extension), "no raw, binary feature, model, or compressed score input"),
    ]
    result = pd.DataFrame(rows, columns=["check", "status", "detail"])
    result["status"] = result["status"].map({True: "pass", False: "fail"})
    return result


def readme(
    census: pd.DataFrame,
    decisions: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    total = census[census["experiment"].eq("TOTAL")].iloc[0]
    lines = "\n".join(
        f"- `{row.decision}`: {row.outcome} - {row.basis}"
        for row in decisions.itertuples(index=False)
    )
    return f"""# Block 8 Robustness Synthesis v0.1

**Work date:** 2026-09-18
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Source experiments:** Five frozen Block 8 experiments
**New fitting, scoring, or threshold selection:** None
**Current test accessed:** No

## Integrity Gate

All {int(total.passing_checks)}/{int(total.recorded_checks)} applicable source integrity checks passed across the five experiments.

## Block Decisions

{lines}

## Closeout

Block 8 can close because its planned robustness questions, failed approaches, mechanism findings, and unresolved limitations are recorded. The wearable detector does not advance: clean event performance remains inadequate, material channel and noise failures remain, generic augmentation failed its clean gate, and the only advancing consensus mechanism requires PSG.

All {(checks['status'] == 'pass').sum()}/{len(checks)} synthesis checks passed. No raw signal, feature array, model, full score table, or current-test artifact was opened.
"""


def main() -> None:
    paths = source_paths()
    manifest = source_manifest(paths)
    census = check_census(paths)
    evidence = evidence_ledger(paths)
    decisions = block_decisions(evidence, census)
    unresolved = unresolved_register()
    checks = synthesis_checks(manifest, census, evidence, decisions, unresolved)
    if not checks["status"].eq("pass").all():
        failed = checks.loc[checks["status"].eq("fail"), "check"].tolist()
        raise RuntimeError(f"Block 8 synthesis checks failed: {failed}")

    outputs = {
        "input_artifact_manifest_v0.1.tsv": manifest,
        "integrity_check_census_v0.1.tsv": census,
        "evidence_ledger_v0.1.tsv": evidence,
        "block8_closeout_decisions_v0.1.tsv": decisions,
        "unresolved_uncertainties_v0.1.tsv": unresolved,
        "synthesis_checks_v0.1.tsv": checks,
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
    verify_or_create_text(output_dir() / "README.md", readme(census, decisions, checks))

    for row in decisions.itertuples(index=False):
        print(f"{row.decision}: {row.outcome}")
    print(f"Source checks: {int(census.iloc[-1].passing_checks)}/132 pass")
    print(f"Synthesis checks: {len(checks)}/{len(checks)} pass")


if __name__ == "__main__":
    main()
