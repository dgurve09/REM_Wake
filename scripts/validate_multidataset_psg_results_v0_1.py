"""Independently reconstruct and validate the pooled PSG experiment results."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
import run_multidataset_psg_external_generalization_v0_1 as experiment
from reviewed_output import verify_or_create_tsv


# Section 1: reviewed and external artifact helpers

def reviewed(name: str) -> pd.DataFrame:
    return pd.read_csv(experiment.output_dir() / name, sep="\t")


def assert_frame(actual: pd.DataFrame, expected: pd.DataFrame, sort_by: list[str]) -> None:
    actual = actual.sort_values(sort_by).reset_index(drop=True)
    expected = expected.sort_values(sort_by).reset_index(drop=True)
    actual = actual[expected.columns]
    pd.testing.assert_frame_equal(
        actual,
        expected,
        check_dtype=False,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def validate_manifest(manifest: pd.DataFrame) -> tuple[bool, bool]:
    file_checks = []
    payload_checks = []
    for item in manifest.itertuples(index=False):
        path = features.data_parent() / item.relative_path
        file_checks.append(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and features.sha256(path) == item.sha256
        )
        if item.artifact_type == "model":
            with gzip.open(path, "rt", encoding="utf-8") as source:
                payload = json.load(source)
            expected_features = 120 if item.candidate == "PSD3-BASE" else 444
            payload_checks.append(
                len(payload["feature_names"]) == expected_features
                and len(payload["coefficient"]) == expected_features
                and len(payload["scaler_mean"]) == expected_features
                and len(payload["scaler_scale"]) == expected_features
                and np.isfinite(payload["coefficient"]).all()
                and np.isfinite(payload["scaler_mean"]).all()
                and np.isfinite(payload["scaler_scale"]).all()
                and bool(payload["converged"])
            )
    return all(file_checks), all(payload_checks)


# Section 2: independent reconstruction

def load_scores() -> pd.DataFrame:
    frames = []
    for heldout in experiment.DATASETS:
        for candidate in experiment.CANDIDATES:
            frames.append(pd.read_csv(experiment.score_path(heldout, candidate), sep="\t"))
    return pd.concat(frames, ignore_index=True)


def reconstruct_alarms(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for heldout in experiment.DATASETS:
        for candidate in experiment.CANDIDATES:
            local = scores[
                scores.heldout_dataset.eq(heldout)
                & scores.candidate.eq(candidate)
            ]
            threshold = float(
                selections[
                    selections.heldout_dataset.eq(heldout)
                    & selections.candidate.eq(candidate)
                ].iloc[0].threshold
            )
            alarms = experiment.predictions(experiment.peak_scores(local), threshold)
            alarms.insert(0, "heldout_dataset", heldout)
            alarms.insert(1, "candidate", candidate)
            alarms["threshold"] = threshold
            frames.append(alarms)
    return pd.concat(frames, ignore_index=True)


def validate_thresholds(curves: pd.DataFrame, selections: pd.DataFrame) -> bool:
    passed = []
    for (heldout, candidate), local in curves.groupby(["heldout_dataset", "candidate"]):
        eligible = local[local.false_alarms_per_hour.le(experiment.ALARM_BUDGET + 1e-12)]
        selected = eligible.sort_values(
            ["f1", "recall", "false_alarms_per_hour", "threshold"],
            ascending=[False, False, True, False],
            kind="stable",
        ).iloc[0]
        saved = selections[
            selections.heldout_dataset.eq(heldout)
            & selections.candidate.eq(candidate)
        ].iloc[0]
        passed.append(
            np.isclose(selected.threshold, saved.threshold, rtol=0, atol=1e-15)
            and np.isclose(selected.f1, saved.f1, rtol=0, atol=1e-15)
            and selected.false_alarms_per_hour <= experiment.ALARM_BUDGET + 1e-12
        )
    return all(passed)


# Section 3: result-level validation

def main() -> None:
    manifest = reviewed("external_artifact_manifest_v0.1.tsv")
    fits = reviewed("model_fit_summary_v0.1.tsv")
    curves = reviewed("inner_threshold_curves_v0.1.tsv")
    selections = reviewed("inner_threshold_selections_v0.1.tsv")
    support = reviewed("outer_support_v0.1.tsv")
    saved_alarms = reviewed("outer_predicted_events_v0.1.tsv")
    saved_metrics = reviewed("heldout_event_metrics_v0.1.tsv")
    saved_participants = reviewed("heldout_event_participants_v0.1.tsv")
    saved_ranking = reviewed("score_ranking_diagnostics_v0.1.tsv")
    saved_sensitivity = reviewed("scorer_sensitivity_v0.1.tsv")
    saved_bootstrap = reviewed("paired_external_bootstrap_v0.1.tsv")
    saved_gate = reviewed("hypothesis_decision_v0.1.tsv")

    files_pass, payloads_pass = validate_manifest(manifest)
    scores = load_scores()
    reconstructed_alarms = reconstruct_alarms(scores, selections)
    assert_frame(
        reconstructed_alarms,
        saved_alarms,
        ["heldout_dataset", "candidate", "recording", "event_time_sec"],
    )

    recordings = experiment.load_recordings()
    recomputed = experiment.result_tables(
        {"scores": scores, "support": support, "alarms": reconstructed_alarms},
        recordings,
    )
    assert_frame(
        recomputed["metrics"], saved_metrics,
        ["heldout_dataset", "candidate"],
    )
    assert_frame(
        recomputed["participants"], saved_participants,
        ["heldout_dataset", "candidate", "dataset", "participant_group"],
    )
    assert_frame(
        recomputed["ranking"], saved_ranking,
        ["heldout_dataset", "candidate"],
    )
    assert_frame(
        recomputed["sensitivity"], saved_sensitivity,
        ["heldout_dataset", "candidate", "label_variant"],
    )
    recomputed_bootstrap = experiment.bootstrap(recomputed["participants"])
    assert_frame(recomputed_bootstrap, saved_bootstrap, ["comparison", "metric"])
    recomputed_gate = experiment.decision(recomputed["metrics"], fits)
    assert_frame(recomputed_gate, saved_gate, ["hypothesis"])

    checks = pd.DataFrame(
        [
            {"check": "manifest_rows", "observed": len(manifest), "expected": 42, "passed": len(manifest) == 42},
            {"check": "external_artifact_hashes", "observed": files_pass, "expected": True, "passed": files_pass},
            {"check": "model_payload_dimensions_and_values", "observed": payloads_pass, "expected": True, "passed": payloads_pass},
            {"check": "model_fit_rows", "observed": len(fits), "expected": 36, "passed": len(fits) == 36},
            {"check": "thresholds_reselected_from_inner_curves", "observed": validate_thresholds(curves, selections), "expected": True, "passed": validate_thresholds(curves, selections)},
            {"check": "outer_score_rows", "observed": len(scores), "expected": 750636, "passed": len(scores) == 750636},
            {"check": "outer_alarm_rows_reconstructed", "observed": len(reconstructed_alarms), "expected": len(saved_alarms), "passed": True},
            {"check": "event_metrics_recomputed", "observed": len(recomputed["metrics"]), "expected": 8, "passed": True},
            {"check": "participant_metrics_recomputed", "observed": len(recomputed["participants"]), "expected": 260, "passed": len(recomputed["participants"]) == 260},
            {"check": "ranking_metrics_recomputed", "observed": len(recomputed["ranking"]), "expected": 6, "passed": True},
            {"check": "scorer_sensitivity_recomputed", "observed": len(recomputed["sensitivity"]), "expected": 10, "passed": len(recomputed["sensitivity"]) == 10},
            {"check": "participant_bootstrap_recomputed", "observed": len(recomputed_bootstrap), "expected": 3, "passed": True},
            {"check": "hypothesis_gate_recomputed", "observed": bool(recomputed_gate.point_gate_pass.iloc[0]), "expected": True, "passed": bool(recomputed_gate.point_gate_pass.iloc[0])},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent multi-dataset result validation failed")
    verify_or_create_tsv(
        checks,
        experiment.output_dir() / "independent_result_validation_v0.1.tsv",
    )
    print(f"Independent result validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(manifest)} external artifact hashes.")


if __name__ == "__main__":
    main()
