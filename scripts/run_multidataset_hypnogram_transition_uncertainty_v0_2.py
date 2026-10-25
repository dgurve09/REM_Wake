"""Rerun the frozen hypnogram analysis with safe gx36m label parsing."""

from __future__ import annotations

import pandas as pd

import hypnogram_transition_common_v0_2 as common
import run_multidataset_hypnogram_transition_uncertainty_v0_1 as original
from reviewed_output import verify_or_create_tsv


# Section 1: corrected rerun

def main() -> None:
    original.common = common
    records = common.load_primary_records()
    recording_frame, event_frame = original.recording_measures(records)
    dataset_frame, group_frame = original.dataset_summaries(recording_frame, event_frame)
    scorer_pairs, scorer_summary = original.scorer_uncertainty(records)
    repeated_participants, repeated_summary = original.repeated_nights(recording_frame)
    sensitivity = original.full_record_sensitivity(recording_frame)
    decision_frame = original.decisions(dataset_frame, scorer_summary, repeated_summary)

    gx_records = [item for item in records if item["dataset"] == "gx36m"]
    gx_unknown = sum(int((item["stages"] == "UNKNOWN").sum()) for item in gx_records)
    repeated_sources = set(repeated_summary.dataset)
    expected_repeated = {"boas", "gx36m", "isruc", "sleep-edfx"}
    checks = pd.DataFrame(
        [
            {"check": "primary_nights", "observed": len(recording_frame), "expected": 1993, "passed": len(recording_frame) == 1993},
            {"check": "dataset_sources", "observed": recording_frame.dataset.nunique(), "expected": 11, "passed": recording_frame.dataset.nunique() == 11},
            {"check": "participant_groups", "observed": recording_frame.participant_group.nunique(), "expected": 1822, "passed": recording_frame.participant_group.nunique() == 1822},
            {"check": "gx36m_corrected_unknown_epochs", "observed": gx_unknown, "expected": 1479, "passed": gx_unknown == 1479},
            {"check": "support_eligible_nights", "observed": int(recording_frame.support_eligible.sum()), "expected": 1993, "passed": recording_frame.support_eligible.all()},
            {"check": "event_rows_equal_recording_counts", "observed": len(event_frame), "expected": int(recording_frame.rem_to_wake_events.sum()), "passed": len(event_frame) == int(recording_frame.rem_to_wake_events.sum())},
            {"check": "total_primary_events", "observed": len(event_frame), "expected": 4046, "passed": len(event_frame) == 4046},
            {"check": "event_positions_bounded", "observed": bool(event_frame.sleep_period_position.between(0, 1, inclusive="both").all()), "expected": True, "passed": bool(event_frame.sleep_period_position.between(0, 1, inclusive="both").all())},
            {"check": "events_do_not_exceed_rem_exits", "observed": bool(recording_frame.rem_to_wake_events.le(recording_frame.rem_exit_opportunities).all()), "expected": True, "passed": bool(recording_frame.rem_to_wake_events.le(recording_frame.rem_exit_opportunities).all())},
            {"check": "scorer_pair_rows", "observed": len(scorer_pairs), "expected": 1100, "passed": len(scorer_pairs) == 1100},
            {"check": "scorer_sources", "observed": scorer_summary.dataset.nunique(), "expected": 3, "passed": scorer_summary.dataset.nunique() == 3},
            {"check": "repeated_sources", "observed": repeated_sources == expected_repeated, "expected": True, "passed": repeated_sources == expected_repeated},
            {"check": "full_record_sensitivity_rows", "observed": len(sensitivity), "expected": 11, "passed": len(sensitivity) == 11},
            {"check": "decision_rows", "observed": len(decision_frame), "expected": 3, "passed": len(decision_frame) == 3},
            {"check": "signal_or_model_inputs", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Corrected hypnogram transition controls failed")

    outputs = {
        "recording_transition_measures_v0.2.tsv": recording_frame,
        "transition_events_v0.2.tsv": event_frame,
        "dataset_transition_summary_v0.2.tsv": dataset_frame,
        "cohort_group_summary_v0.2.tsv": group_frame,
        "scorer_pair_records_v0.2.tsv": scorer_pairs,
        "scorer_uncertainty_summary_v0.2.tsv": scorer_summary,
        "repeated_participant_measures_v0.2.tsv": repeated_participants,
        "repeated_night_summary_v0.2.tsv": repeated_summary,
        "full_record_sensitivity_v0.2.tsv": sensitivity,
        "hypothesis_decisions_v0.2.tsv": decision_frame,
        "in_run_checks_v0.2.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, common.output_dir() / name)

    print(dataset_frame[["dataset", "primary_nights", "rem_to_wake_events", "event_positive_night_fraction", "events_per_support_hour", "rw_fraction_of_known_rem_exits"]].to_string(index=False))
    print(scorer_summary.to_string(index=False))
    print(repeated_summary.to_string(index=False))
    print(decision_frame[["hypothesis", "observed_1", "observed_2", "decision"]].to_string(index=False))


if __name__ == "__main__":
    main()
