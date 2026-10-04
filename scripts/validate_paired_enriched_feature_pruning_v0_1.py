"""Independently validate paired enriched-feature experiment outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_paired_enriched_feature_pruning_v0_1 as experiment


def output_dir() -> Path:
    return experiment.repo_root() / "experiments" / experiment.EXPERIMENT_DIR


def same_frame(left: pd.DataFrame, right: pd.DataFrame, sort_columns: list[str]) -> bool:
    try:
        left_sorted = left.sort_values(sort_columns).reset_index(drop=True)
        right_sorted = right.sort_values(sort_columns).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            left_sorted,
            right_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=1e-10,
            atol=1e-12,
        )
        return True
    except AssertionError:
        return False


def validate() -> pd.DataFrame:
    output = output_dir()
    manifest = pd.read_csv(output / "external_artifact_manifest_v0.1.tsv", sep="\t")
    feature_summary = pd.read_csv(output / "feature_generation_summary_v0.1.tsv", sep="\t")
    schema = pd.read_csv(output / "feature_schema_v0.1.tsv", sep="\t")
    construction = pd.read_csv(output / "candidate_construction_v0.1.tsv", sep="\t")
    fit_summary = pd.read_csv(output / "model_fit_summary_v0.1.tsv", sep="\t")
    selections = pd.read_csv(output / "inner_threshold_selections_v0.1.tsv", sep="\t")
    support = pd.read_csv(output / "outer_support_v0.1.tsv", sep="\t")
    events = pd.read_csv(output / "outer_predicted_events_v0.1.tsv", sep="\t")
    stored_metrics = pd.read_csv(output / "outer_event_metrics_v0.1.tsv", sep="\t")
    stored_participants = pd.read_csv(output / "outer_event_participants_v0.1.tsv", sep="\t")
    stored_bootstrap = pd.read_csv(output / "paired_participant_bootstrap_v0.1.tsv", sep="\t")
    stored_decisions = pd.read_csv(output / "hypothesis_decisions_v0.1.tsv", sep="\t")
    stability = pd.read_csv(output / "outer_feature_selection_stability_v0.1.tsv", sep="\t")

    checks = []
    hashes_pass = True
    for row in manifest.itertuples(index=False):
        path = experiment.data_parent() / row.relative_path
        hashes_pass &= path.stat().st_size == int(row.bytes)
        hashes_pass &= experiment.sha256(path) == row.sha256
    checks.append(("external_hashes", hashes_pass, f"{len(manifest)} files"))

    feature_pass = len(feature_summary) == 164
    maximum_difference = 0.0
    for row in feature_summary.itertuples(index=False):
        path = experiment.cache_path(row.subject, row.modality)
        with np.load(path, allow_pickle=False) as values:
            onsets = values["onset"].astype(float)
            features = values["features"].astype(float)
            names = values["feature_names"].astype(str).tolist()
        with np.load(experiment.block7_feature_path(row.subject, row.modality), allow_pickle=False) as values:
            reviewed_onsets = values["onset"].astype(float)
            reviewed_base = values["features"].astype(float)
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(features[:, experiment.base_indices()] - reviewed_base))),
        )
        feature_pass &= np.array_equal(onsets, reviewed_onsets)
        feature_pass &= features.shape == (len(onsets), experiment.RICH_FEATURES_PER_EPOCH)
        feature_pass &= names == experiment.rich_feature_names(
            experiment.MODALITIES[row.modality]["channels"]
        )
        feature_pass &= experiment.feature_bounds_pass(features)
    feature_pass &= maximum_difference <= 1e-5
    checks.append(("feature_reconstruction", feature_pass, f"max F0 difference={maximum_difference:.3g}"))

    schema_with_flag = schema.assign(f0_flag=experiment.truth(schema["is_f0_feature"]))
    schema_pass = (
        len(schema) == 2 * experiment.CONTEXT_EPOCHS * experiment.RICH_FEATURES_PER_EPOCH
        and schema.groupby("modality")["feature_index"].nunique().eq(360).all()
        and schema_with_flag.groupby("modality")["f0_flag"].sum().eq(80).all()
    )
    checks.append(("feature_schema", schema_pass, "360 enriched and 80 F0 values per modality"))

    retained = construction[experiment.truth(construction["retained"])]
    candidate_pass = len(retained) == 2743 and int(retained["label"].sum()) == 180
    checks.append(("candidate_accounting", candidate_pass, "2743 retained; 180 positive"))

    fit_pass = len(fit_summary) == 230 and experiment.truth(fit_summary["converged"]).all()
    checks.append(("fit_accounting", fit_pass, f"{len(fit_summary)} converged fits"))

    selection_pass = (
        len(selections) == 30
        and set(selections["candidate"]) == set(experiment.PIPELINES)
        and set(selections[selections["candidate"].eq("F1-EN")]["C"]).issubset(set(experiment.EN_C_VALUES))
    )
    checks.append(("nested_selections", selection_pass, "30 outer selections"))

    assignments = experiment.train_assignments()
    references = experiment.reference_events(assignments)
    reconstructed = experiment.evaluate_all(events, support, references)
    metric_pass = same_frame(
        reconstructed["metrics"],
        stored_metrics,
        ["pipeline", "membership", "tolerance_sec"],
    )
    participant_pass = same_frame(
        reconstructed["participants"],
        stored_participants,
        ["pipeline", "membership", "tolerance_sec", "pid"],
    )
    checks.append(("event_reconstruction", metric_pass and participant_pass, "metrics and participant rows"))

    reconstructed_bootstrap = experiment.paired_bootstrap(reconstructed["participants"])
    bootstrap_pass = same_frame(
        reconstructed_bootstrap,
        stored_bootstrap,
        ["modality", "comparison", "metric"],
    )
    checks.append(("bootstrap_reconstruction", bootstrap_pass, "12 paired interval rows"))

    reconstructed_decisions = experiment.decision_table(
        reconstructed["metrics"], reconstructed_bootstrap
    )
    decision_pass = same_frame(
        reconstructed_decisions,
        stored_decisions,
        ["modality", "hypothesis"],
    )
    checks.append(("decision_reconstruction", decision_pass, "four advancement decisions"))

    stability_pass = (
        stability.groupby(["modality", "candidate"])["outer_folds"].max().eq(5).all()
        and stability["retained_folds"].between(0, 5).all()
        and stability["nonzero_folds"].between(0, 5).all()
    )
    checks.append(("selection_stability", stability_pass, f"{len(stability)} feature rows"))

    result = pd.DataFrame(
        [
            {"check": name, "status": "pass" if passed else "fail", "detail": detail}
            for name, passed, detail in checks
        ]
    )
    if not result["status"].eq("pass").all():
        raise ValueError("Validation failed:\n" + result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate()
    print(result.to_string(index=False))
    if args.report:
        value = result.to_csv(sep="\t", index=False, lineterminator="\n")
        if args.report.exists() and args.report.read_text(encoding="utf-8") != value:
            raise RuntimeError("Validation report changed")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
