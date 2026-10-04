"""Independently validate reviewed quality-balanced LSTM-CRF outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from run_spectral_unet_train_nested_v0_1 import (
    data_parent,
    local_event_inputs,
    reference_events,
    repo_root,
    train_assignments,
    truth,
)
from stage_first_event_evaluation_v0_1 import evaluate_events


EXPERIMENT = "2026-10-03_quality_balanced_lstm_crf_nested_v0.1"
DERIVED = "quality_balanced_lstm_crf_nested_v0.1"
PIPELINES = ["LC-1-NESTED", "LC-QB1"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close_frame(left: pd.DataFrame, right: pd.DataFrame, keys: list[str], values: list[str]) -> bool:
    compared = left[keys + values].merge(
        right[keys + values], on=keys, suffixes=("_stored", "_new"), validate="one_to_one"
    )
    if len(compared) != len(left) or len(compared) != len(right):
        return False
    return all(np.allclose(compared[f"{column}_stored"], compared[f"{column}_new"],
                           rtol=0, atol=1e-12, equal_nan=True) for column in values)


def validate() -> pd.DataFrame:
    output = repo_root() / "experiments" / EXPERIMENT
    derived = data_parent() / "derived" / DERIVED
    assignments = train_assignments()
    references = reference_events(assignments)
    scores = pd.read_csv(derived / "scores/paired_outer_scores_v0.1.tsv.gz", sep="\t")
    events = pd.read_csv(output / "outer_predicted_events_v0.1.tsv", sep="\t")
    support = pd.read_csv(output / "outer_support_v0.1.tsv", sep="\t")
    stored_metrics = pd.read_csv(output / "outer_event_metrics_v0.1.tsv", sep="\t")
    stored_matches = pd.read_csv(output / "outer_event_matches_v0.1.tsv", sep="\t")
    construction = pd.read_csv(output / "candidate_construction_v0.1.tsv", sep="\t")
    selections = pd.read_csv(output / "inner_threshold_selections_v0.1.tsv", sep="\t")
    fits = pd.read_csv(output / "model_fit_summary_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(output / "external_artifact_manifest_v0.1.tsv", sep="\t")
    stored_tiers = pd.read_csv(output / "quality_tier_metrics_v0.1.tsv", sep="\t")

    checks = []
    checks.append(("train_membership", len(assignments) == 82 and assignments.pid.nunique() == 64,
                   f"{len(assignments)} recordings; {assignments.pid.nunique()} pid"))
    retained = construction[truth(construction.retained)]
    tier_counts = retained[retained.label == 1].membership_tier.value_counts()
    candidate_pass = (len(retained) == 2743 and retained.label.sum() == 180 and
                      tier_counts.get("primary_clean", 0) == 53 and
                      tier_counts.get("primary_mad_flagged", 0) == 127)
    checks.append(("candidate_counts", candidate_pass,
                   f"{len(retained)} total; {retained.label.sum()} positive; "
                   f"{tier_counts.get('primary_clean', 0)} clean; "
                   f"{tier_counts.get('primary_mad_flagged', 0)} flagged"))
    score_pass = (set(scores.pipeline) == set(PIPELINES) and
                  scores.groupby("pipeline").size().eq(75539).all() and
                  np.isfinite(scores.probability).all() and
                  scores.probability.between(0, 1).all())
    checks.append(("outer_score_integrity", score_pass,
                   "; ".join(f"{key}={value}" for key, value in scores.groupby('pipeline').size().items())))
    checks.append(("threshold_selections", len(selections) == 10 and
                   selections.groupby("pipeline").outer_fold.nunique().eq(5).all(),
                   f"{len(selections)} selections"))
    checks.append(("fit_accounting", len(fits) == 50 and
                   fits.groupby("pipeline").size().eq(25).all(), f"{len(fits)} fits"))

    recomputed_metrics, recomputed_matches = [], []
    for pipeline in PIPELINES:
        predictions = events[events.pipeline == pipeline]
        for membership in ["primary", "expanded"]:
            eligible, ignored = local_event_inputs(references, membership)
            for tolerance in [15.0, 45.0]:
                _, _, matches, summary = evaluate_events(
                    eligible, predictions[["subject", "pid", "event_time_sec"]], ignored,
                    support[["subject", "pid", "supported_hours"]], tolerance,
                )
                config = {"pipeline": pipeline, "partition": "train_nested_oof",
                          "membership": membership, "tolerance_sec": tolerance}
                recomputed_metrics.append({**config, **summary})
                if len(matches):
                    for key, value in reversed(list(config.items())):
                        matches.insert(0, key, value)
                    recomputed_matches.append(matches)
    recomputed_metrics = pd.DataFrame(recomputed_metrics)
    metric_keys = ["pipeline", "partition", "membership", "tolerance_sec"]
    metric_values = ["reference_events", "predicted_events", "true_positive", "false_positive",
                     "false_negative", "ignored_predictions", "supported_hours", "precision",
                     "recall", "f1", "false_alarms_per_hour"]
    metric_pass = close_frame(stored_metrics, recomputed_metrics, metric_keys, metric_values)
    checks.append(("event_metrics_recomputed", metric_pass, "stored tables match event-level recomputation"))

    recomputed_matches = pd.concat(recomputed_matches, ignore_index=True)
    primary = references[truth(references.primary_analysis_eligible)].copy()
    tier_rows = []
    for pipeline in PIPELINES:
        matched_rows = recomputed_matches[(recomputed_matches.pipeline == pipeline) &
                                          (recomputed_matches.membership == "primary") &
                                          (recomputed_matches.tolerance_sec == 15.0) &
                                          (recomputed_matches.match_type == "eligible")]
        matched = set(zip(matched_rows.subject, matched_rows.reference_time_sec))
        local = primary.copy()
        local["detected"] = [int((row.subject, row.event_time_sec) in matched)
                             for row in local.itertuples(index=False)]
        for tier, group in local.groupby("membership_tier"):
            tier_rows.append({"pipeline": pipeline, "membership_tier": tier,
                              "reference_events": len(group), "detected_events": int(group.detected.sum()),
                              "recall": group.detected.mean()})
    recomputed_tiers = pd.DataFrame(tier_rows)
    tier_pass = close_frame(stored_tiers, recomputed_tiers,
                            ["pipeline", "membership_tier"],
                            ["reference_events", "detected_events", "recall"])
    checks.append(("quality_tier_metrics_recomputed", tier_pass,
                   "53 clean and 127 MAD-flagged reference events"))

    manifest_pass = True
    for row in manifest.itertuples(index=False):
        path = data_parent() / row.relative_path
        manifest_pass &= path.is_file() and path.stat().st_size == int(row.bytes) and sha256(path) == row.sha256
    checks.append(("external_manifest", manifest_pass, f"{len(manifest)} files checked"))

    prior = pd.read_csv(data_parent() / "derived/deep_temporal_nested_cv_v0.1/scores/nested_blstm_crf_outer_scores_v0.1.tsv.gz", sep="\t")
    control = scores[scores.pipeline == "LC-1-NESTED"]
    keys = ["outer_fold", "subject", "pid", "candidate_time_sec"]
    compared = control.merge(prior[keys + ["probability"]], on=keys,
                             suffixes=("_current", "_prior"), validate="one_to_one")
    error = float(np.max(np.abs(compared.probability_current - compared.probability_prior)))
    checks.append(("control_reproduces_prior_nested_scores", error <= 1e-12,
                   f"maximum absolute probability difference={error:.3g}"))

    result = pd.DataFrame([{"check": name, "status": "pass" if passed else "fail", "detail": detail}
                           for name, passed, detail in checks])
    if not result.status.eq("pass").all():
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
