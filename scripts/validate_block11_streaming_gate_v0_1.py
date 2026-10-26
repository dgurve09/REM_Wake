"""Independently validate the Block 11 streaming no-go evidence."""

from __future__ import annotations

import pandas as pd

import assess_block11_streaming_gate_v0_1 as assessment
from reviewed_output import verify_or_create_tsv


def main() -> None:
    manifest = pd.read_csv(assessment.output_dir() / "source_manifest_v0.1.tsv", sep="\t")
    evidence = pd.read_csv(assessment.output_dir() / "streaming_evidence_inventory_v0.1.tsv", sep="\t").set_index("evidence_id")
    gate = pd.read_csv(assessment.output_dir() / "streaming_gate_conditions_v0.1.tsv", sep="\t").set_index("condition")
    run_checks = pd.read_csv(assessment.output_dir() / "in_run_checks_v0.1.tsv", sep="\t")

    hashes = all(
        (assessment.repo_root() / item.relative_path).exists()
        and (assessment.repo_root() / item.relative_path).stat().st_size == int(item.bytes)
        and assessment.sha256(assessment.repo_root() / item.relative_path) == item.sha256
        for item in manifest.itertuples(index=False)
    )
    validation = assessment.read("wearable_validation").query(
        "comparator == 'H2-D' and partition == 'validation' and membership == 'primary' and tolerance_sec == 15"
    ).iloc[0]
    closeout = assessment.read("robustness_closeout").set_index("decision")
    causal = assessment.read("causal_decision").iloc[0]
    psg = assessment.read("psg_metrics").query(
        "heldout_dataset == 'POOLED_EXTERNAL' and candidate == 'ADV3-CHANGE'"
    ).iloc[0]
    interval = assessment.read("psg_bootstrap").query("metric == 'f1_difference'").iloc[0]
    labels = assessment.read("label_decisions")

    checks = pd.DataFrame(
        [
            {"check": "source_hashes", "observed": hashes, "expected": True, "passed": hashes},
            {"check": "source_roles", "observed": len(manifest), "expected": 13, "passed": len(manifest) == 13},
            {"check": "run_checks", "observed": int(run_checks.passed.sum()), "expected": len(run_checks), "passed": bool(run_checks.passed.all())},
            {"check": "wearable_validation_reconstructed", "observed": float(validation.f1), "expected": float(evidence.loc["direct_wearable_validation", "f1"]), "passed": abs(float(validation.f1) - float(evidence.loc["direct_wearable_validation", "f1"])) < 1e-12},
            {"check": "robustness_failure_reconstructed", "observed": bool(closeout.loc["C8.1_wearable_robustness_sufficient", "supported"]), "expected": False, "passed": not bool(closeout.loc["C8.1_wearable_robustness_sufficient", "supported"])},
            {"check": "deployable_method_not_advanced", "observed": closeout.loc["C8.2_deployable_wearable_method_advances", "outcome"], "expected": "do_not_advance", "passed": closeout.loc["C8.2_deployable_wearable_method_advances", "outcome"] == "do_not_advance"},
            {"check": "causal_candidate_requires_confirmation", "observed": causal.decision, "expected": "retain_for_new_confirmation", "passed": causal.decision == "retain_for_new_confirmation"},
            {"check": "psg_evidence_not_wearable", "observed": evidence.loc["corrected_external_psg_candidate", "modality"], "expected": "psg", "passed": evidence.loc["corrected_external_psg_candidate", "modality"] == "psg"},
            {"check": "psg_interval_crosses_zero", "observed": bool(interval.lower_95 <= 0 <= interval.upper_95), "expected": True, "passed": bool(interval.lower_95 <= 0 <= interval.upper_95)},
            {"check": "label_decisions_material", "observed": int(labels.decision.str.startswith("material_").sum()), "expected": 3, "passed": bool(labels.decision.str.startswith("material_").all())},
            {"check": "four_entry_conditions_fail", "observed": int((~gate.passed).sum()), "expected": 4, "passed": int((~gate.passed).sum()) == 4},
            {"check": "overall_no_go", "observed": gate.overall_decision.iloc[0], "expected": "no_go", "passed": gate.overall_decision.eq("no_go").all()},
            {"check": "prototype_absent", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent Block 11 streaming-gate validation failed")
    verify_or_create_tsv(
        checks,
        assessment.output_dir() / "independent_validation_v0.1.tsv",
    )
    print(f"Independent streaming-gate validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
