"""Assess the frozen Block 11 entry gate for a wearable streaming prototype."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: fixed reviewed sources

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-10-25_block11_streaming_gate_v0.1"


SOURCES = {
    "wearable_validation": "experiments/2026-09-06_block7_transfer_validation_v0.1/validation_event_metrics_v0.1.tsv",
    "wearable_test": "experiments/2026-09-06_block7_descriptive_test_v0.1/test_event_metrics_v0.1.tsv",
    "robustness_closeout": "experiments/2026-09-18_block8_robustness_synthesis_v0.1/block8_closeout_decisions_v0.1.tsv",
    "lstm_metrics": "experiments/2026-09-19_lstm_crf_train_oof_v0.1/train_oof_event_metrics_v0.1.tsv",
    "lstm_decision": "experiments/2026-09-19_lstm_crf_train_oof_v0.1/hypothesis_decisions_v0.1.tsv",
    "enriched_metrics": "experiments/2026-10-04_paired_enriched_feature_pruning_nested_v0.1/outer_event_metrics_v0.1.tsv",
    "enriched_decision": "experiments/2026-10-04_paired_enriched_feature_pruning_nested_v0.1/hypothesis_decisions_v0.1.tsv",
    "causal_metrics": "experiments/2026-10-25_causal_multiscale_change_features_v0.1/outer_event_metrics_v0.1.tsv",
    "causal_decision": "experiments/2026-10-25_causal_multiscale_change_features_v0.1/hypothesis_decision_v0.1.tsv",
    "psg_metrics": "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/heldout_event_metrics_v0.2.tsv",
    "psg_bootstrap": "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/paired_external_bootstrap_v0.2.tsv",
    "psg_decision": "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/hypothesis_decision_v0.2.tsv",
    "label_decisions": "experiments/2026-10-25_multidataset_hypnogram_transition_uncertainty_v0.2/hypothesis_decisions_v0.2.tsv",
}


def read(role: str) -> pd.DataFrame:
    return pd.read_csv(repo_root() / SOURCES[role], sep="\t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> pd.DataFrame:
    rows = []
    for role, relative in SOURCES.items():
        path = repo_root() / relative
        rows.append(
            {
                "source_role": role,
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


# Section 2: fixed primary evidence rows

def evidence_inventory() -> pd.DataFrame:
    validation = read("wearable_validation").query(
        "comparator == 'H2-D' and partition == 'validation' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    test = read("wearable_test").query(
        "analysis_subset == 'all_test' and comparator == 'H2-D' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    robustness = read("robustness_closeout").set_index("decision")
    lstm = read("lstm_metrics").query(
        "model == 'LC-1' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    lstm_decision = read("lstm_decision").iloc[0]
    enriched = read("enriched_metrics").query(
        "modality == 'H2' and pipeline == 'H2-F1-EN' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    enriched_decision = read("enriched_decision").query(
        "modality == 'H2' and hypothesis == 'H-F1-EN'"
    ).iloc[0]
    causal = read("causal_metrics").query("candidate == 'ADV-PEAK'").iloc[0]
    causal_decision = read("causal_decision").iloc[0]
    psg = read("psg_metrics").query(
        "heldout_dataset == 'POOLED_EXTERNAL' and candidate == 'ADV3-CHANGE'"
    ).iloc[0]
    psg_interval = read("psg_bootstrap").query("metric == 'f1_difference'").iloc[0]
    psg_decision = read("psg_decision").iloc[0]
    label_decisions = read("label_decisions")

    return pd.DataFrame(
        [
            {"evidence_id": "direct_wearable_validation", "modality": "wearable", "validation_status": "independent_validation", "precision": validation.precision, "recall": validation.recall, "f1": validation.f1, "false_alarms_per_hour": validation.false_alarms_per_hour, "decision": "not_advanced", "limitation": "low precision and high false-alarm burden", "source_role": "wearable_validation"},
            {"evidence_id": "direct_wearable_descriptive_test", "modality": "wearable", "validation_status": "descriptive_used_test", "precision": test.precision, "recall": test.recall, "f1": test.f1, "false_alarms_per_hour": test.false_alarms_per_hour, "decision": "not_advanced", "limitation": "descriptive test already used; no new confirmation", "source_role": "wearable_test"},
            {"evidence_id": "wearable_robustness", "modality": "wearable", "validation_status": "validation_synthesis", "precision": pd.NA, "recall": pd.NA, "f1": pd.NA, "false_alarms_per_hour": pd.NA, "decision": robustness.loc["C8.1_wearable_robustness_sufficient", "outcome"], "limitation": robustness.loc["C8.1_wearable_robustness_sufficient", "basis"], "source_role": "robustness_closeout"},
            {"evidence_id": "deployable_wearable_closeout", "modality": "wearable", "validation_status": "validation_synthesis", "precision": pd.NA, "recall": pd.NA, "f1": pd.NA, "false_alarms_per_hour": pd.NA, "decision": robustness.loc["C8.2_deployable_wearable_method_advances", "outcome"], "limitation": robustness.loc["C8.2_deployable_wearable_method_advances", "basis"], "source_role": "robustness_closeout"},
            {"evidence_id": "lstm_crf_candidate", "modality": "wearable", "validation_status": "train_oof_only", "precision": lstm.precision, "recall": lstm.recall, "f1": lstm.f1, "false_alarms_per_hour": lstm.false_alarms_per_hour, "decision": lstm_decision.decision, "limitation": "requires new confirmation", "source_role": "lstm_metrics"},
            {"evidence_id": "enriched_pruned_candidate", "modality": "wearable", "validation_status": "train_nested_oof_only", "precision": enriched.precision, "recall": enriched.recall, "f1": enriched.f1, "false_alarms_per_hour": enriched.false_alarms_per_hour, "decision": enriched_decision.decision, "limitation": "material participant support not reached", "source_role": "enriched_metrics"},
            {"evidence_id": "causal_peak_candidate", "modality": "wearable", "validation_status": "train_nested_oof_only", "precision": causal.precision, "recall": causal.recall, "f1": causal.f1, "false_alarms_per_hour": causal.false_alarms_per_hour, "decision": causal_decision.decision, "limitation": "causal method remains unconfirmed and low recall", "source_role": "causal_metrics"},
            {"evidence_id": "corrected_external_psg_candidate", "modality": "psg", "validation_status": "dataset_heldout_psg", "precision": psg.precision, "recall": psg.recall, "f1": psg.f1, "false_alarms_per_hour": psg.false_alarms_per_hour, "decision": psg_decision.decision, "limitation": f"not wearable; F1-difference interval {psg_interval.lower_95:.4f} to {psg_interval.upper_95:.4f}", "source_role": "psg_metrics"},
            {"evidence_id": "label_uncertainty", "modality": "hypnogram", "validation_status": "eleven_source_label_analysis", "precision": pd.NA, "recall": pd.NA, "f1": pd.NA, "false_alarms_per_hour": pd.NA, "decision": ";".join(label_decisions.decision), "limitation": "source, scorer, and repeated-night dependence remain material", "source_role": "label_decisions"},
        ]
    )


# Section 3: entry decision

def gate_table(evidence: pd.DataFrame) -> pd.DataFrame:
    indexed = evidence.set_index("evidence_id")
    rows = [
        {"condition": "wearable_candidate_independently_advanced", "passed": False, "basis": indexed.loc["deployable_wearable_closeout", "limitation"]},
        {"condition": "wearable_event_performance_interpretable_as_ready", "passed": False, "basis": f"validation F1 {indexed.loc['direct_wearable_validation', 'f1']:.4f}, precision {indexed.loc['direct_wearable_validation', 'precision']:.4f}; descriptive test F1 {indexed.loc['direct_wearable_descriptive_test', 'f1']:.4f}"},
        {"condition": "wearable_robustness_supported", "passed": False, "basis": indexed.loc["wearable_robustness", "limitation"]},
        {"condition": "causal_wearable_method_independently_confirmed", "passed": False, "basis": "causal peak candidate is train-development only with recall 0.1000"},
        {"condition": "boundary_can_be_reported_without_exact_timing_claim", "passed": True, "basis": "report scored-boundary intervals and material annotation uncertainty; do not claim exact physiological onset"},
    ]
    frame = pd.DataFrame(rows)
    frame["overall_decision"] = "go" if frame.passed.all() else "no_go"
    return frame


def main() -> None:
    manifest = source_manifest()
    evidence = evidence_inventory()
    gate = gate_table(evidence)
    checks = pd.DataFrame(
        [
            {"check": "source_files_hashed", "observed": len(manifest), "expected": 13, "passed": len(manifest) == 13},
            {"check": "evidence_rows", "observed": len(evidence), "expected": 9, "passed": len(evidence) == 9},
            {"check": "direct_wearable_validation_f1", "observed": float(evidence.set_index("evidence_id").loc["direct_wearable_validation", "f1"]), "expected": 0.11228070175438597, "passed": abs(float(evidence.set_index("evidence_id").loc["direct_wearable_validation", "f1"]) - 0.11228070175438597) < 1e-12},
            {"check": "wearable_robustness_failed", "observed": gate.set_index("condition").loc["wearable_robustness_supported", "passed"], "expected": False, "passed": not bool(gate.set_index("condition").loc["wearable_robustness_supported", "passed"])},
            {"check": "causal_candidate_unconfirmed", "observed": gate.set_index("condition").loc["causal_wearable_method_independently_confirmed", "passed"], "expected": False, "passed": not bool(gate.set_index("condition").loc["causal_wearable_method_independently_confirmed", "passed"])},
            {"check": "psg_not_counted_as_wearable", "observed": evidence.set_index("evidence_id").loc["corrected_external_psg_candidate", "modality"], "expected": "psg", "passed": evidence.set_index("evidence_id").loc["corrected_external_psg_candidate", "modality"] == "psg"},
            {"check": "streaming_gate", "observed": gate.overall_decision.iloc[0], "expected": "no_go", "passed": gate.overall_decision.iloc[0] == "no_go"},
            {"check": "prototype_or_benchmark_executed", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Block 11 streaming gate failed integrity checks")
    outputs = {
        "source_manifest_v0.1.tsv": manifest,
        "streaming_evidence_inventory_v0.1.tsv": evidence,
        "streaming_gate_conditions_v0.1.tsv": gate,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    print(evidence.to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
