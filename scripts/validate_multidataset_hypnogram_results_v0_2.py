"""Independently reconstruct the corrected hypnogram transition results."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_2 as common
import run_multidataset_hypnogram_transition_uncertainty_v0_1 as analysis
from reviewed_output import verify_or_create_tsv


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(common.output_dir() / name, sep="\t")


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


def main() -> None:
    analysis.common = common
    saved_recordings = load("recording_transition_measures_v0.2.tsv")
    saved_events = load("transition_events_v0.2.tsv")
    saved_datasets = load("dataset_transition_summary_v0.2.tsv")
    saved_groups = load("cohort_group_summary_v0.2.tsv")
    saved_scorer_pairs = load("scorer_pair_records_v0.2.tsv")
    saved_scorer_summary = load("scorer_uncertainty_summary_v0.2.tsv")
    saved_repeated_participants = load("repeated_participant_measures_v0.2.tsv")
    saved_repeated_summary = load("repeated_night_summary_v0.2.tsv")
    saved_sensitivity = load("full_record_sensitivity_v0.2.tsv")
    saved_decisions = load("hypothesis_decisions_v0.2.tsv")
    saved_checks = load("in_run_checks_v0.2.tsv")
    input_checks = load("pre_analysis_correction_audit_v0.2.tsv")
    input_validation = load("independent_input_validation_v0.2.tsv")

    records = common.load_primary_records()
    recordings, events = analysis.recording_measures(records)
    datasets, groups = analysis.dataset_summaries(recordings, events)
    scorer_pairs, scorer_summary = analysis.scorer_uncertainty(records)
    repeated_participants, repeated_summary = analysis.repeated_nights(recordings)
    sensitivity = analysis.full_record_sensitivity(recordings)
    decisions = analysis.decisions(datasets, scorer_summary, repeated_summary)

    assert_frame(recordings, saved_recordings, ["dataset", "recording"])
    assert_frame(events, saved_events, ["event_id"])
    assert_frame(datasets, saved_datasets, ["dataset"])
    assert_frame(groups, saved_groups, ["dataset", "cohort_group"])
    assert_frame(scorer_pairs, saved_scorer_pairs, ["dataset", "recording", "scorer_left", "scorer_right"])
    assert_frame(scorer_summary, saved_scorer_summary, ["dataset"])
    assert_frame(repeated_participants, saved_repeated_participants, ["dataset", "participant_group"])
    assert_frame(repeated_summary, saved_repeated_summary, ["dataset"])
    assert_frame(sensitivity, saved_sensitivity, ["dataset"])
    assert_frame(decisions, saved_decisions, ["hypothesis"])

    record_map = {item["recording"]: item for item in records}
    adjacency = all(
        record_map[item.recording]["stages"][int(item.event_boundary_epoch) - 1] == "REM"
        and record_map[item.recording]["stages"][int(item.event_boundary_epoch)] == "W"
        for item in events.itertuples(index=False)
    )
    gx_records = [item for item in records if item["dataset"] == "gx36m"]
    gx_unknown = sum(int((item["stages"] == "UNKNOWN").sum()) for item in gx_records)
    raw_unknown = sum(
        int((value == "UNKNOWN").sum())
        for item in gx_records
        for value in item["scorers"].values()
    )
    checks = pd.DataFrame(
        [
            {"check": "input_audit_pass", "observed": int(input_checks.passed.astype(bool).sum()), "expected": len(input_checks), "passed": input_checks.passed.astype(bool).all()},
            {"check": "input_validation_pass", "observed": int(input_validation.passed.astype(bool).sum()), "expected": len(input_validation), "passed": input_validation.passed.astype(bool).all()},
            {"check": "in_run_checks_pass", "observed": int(saved_checks.passed.astype(bool).sum()), "expected": len(saved_checks), "passed": saved_checks.passed.astype(bool).all()},
            {"check": "recording_rows_recomputed", "observed": len(recordings), "expected": 1993, "passed": len(recordings) == 1993},
            {"check": "event_rows_recomputed", "observed": len(events), "expected": 4046, "passed": len(events) == 4046},
            {"check": "event_adjacency_verified", "observed": adjacency, "expected": True, "passed": adjacency},
            {"check": "dataset_rows_recomputed", "observed": len(datasets), "expected": 11, "passed": len(datasets) == 11},
            {"check": "scorer_pair_rows_recomputed", "observed": len(scorer_pairs), "expected": 1100, "passed": len(scorer_pairs) == 1100},
            {"check": "repeated_participant_rows_recomputed", "observed": len(repeated_participants), "expected": 145, "passed": len(repeated_participants) == 145},
            {"check": "gx36m_raw_unknown_labels", "observed": raw_unknown, "expected": 818, "passed": raw_unknown == 818},
            {"check": "gx36m_consensus_unknown_epochs", "observed": gx_unknown, "expected": 1479, "passed": gx_unknown == 1479},
            {"check": "decision_rows_recomputed", "observed": len(decisions), "expected": 3, "passed": len(decisions) == 3},
            {"check": "material_decisions", "observed": int(decisions.decision.str.startswith("material_").sum()), "expected": 3, "passed": decisions.decision.str.startswith("material_").all()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent corrected hypnogram result validation failed")
    verify_or_create_tsv(
        checks,
        common.output_dir() / "independent_result_validation_v0.2.tsv",
    )
    print(f"Independent corrected result validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Recomputed {len(events):,} events, {len(scorer_pairs):,} scorer pairs, and {len(repeated_participants)} repeated participants.")


if __name__ == "__main__":
    main()
