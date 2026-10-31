"""Validate retained stage-first failure diagnostics without rerunning the analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


def table(name: str) -> pd.DataFrame:
    return pd.read_csv(common.output_dir() / name, sep="\t")


def close(left: pd.Series, right: pd.Series) -> bool:
    return bool(np.allclose(left, right, atol=1e-12, rtol=0.0, equal_nan=True))


def main() -> None:
    fragmentation = table("failure_sequence_fragmentation_v0.1.tsv")
    references = table("failure_reference_pattern_summary_v0.1.tsv")
    patterns = table("failure_reference_pattern_counts_v0.1.tsv")
    offsets = table("failure_alarm_offset_diagnostics_v0.1.tsv")
    run_checks = table("failure_diagnostic_checks_v0.1.tsv")
    primary = table("recording_stage_input_summary_v0.1.tsv")

    checks = []

    def add(name: str, observed, expected, passed: bool) -> None:
        checks.append(
            {"check": name, "observed": observed, "expected": expected, "passed": bool(passed)}
        )

    add("run_checks_pass", run_checks.passed.all(), True, run_checks.passed.all())
    add("three_fragmentation_rows", len(fragmentation), 3, len(fragmentation) == 3)
    add("six_alarm_rows", len(offsets), 6, len(offsets) == 6)

    reference_expected = (
        primary.groupby("dataset", as_index=False).primary_events.sum().sort_values("dataset")
    )
    reference_observed = references[["dataset", "reference_events"]].sort_values("dataset")
    reference_ok = (
        reference_observed.dataset.tolist() == reference_expected.dataset.tolist()
        and reference_observed.reference_events.tolist() == reference_expected.primary_events.tolist()
    )
    add("references_match_input_audit", reference_ok, True, reference_ok)

    pattern_total = patterns.groupby("dataset").reference_events.sum().sort_index()
    reference_total = references.set_index("dataset").reference_events.sort_index()
    add("pattern_counts_match_references", pattern_total.equals(reference_total), True, pattern_total.equals(reference_total))

    exact = (
        patterns[patterns.predicted_pattern.eq("REM|REM|W|W")]
        .set_index("dataset")
        .reference_events.reindex(reference_total.index, fill_value=0)
    )
    exact_expected = references.set_index("dataset").exact_confirm2_references.sort_index()
    add("exact_pattern_counts_recompute", exact.equals(exact_expected), True, exact.equals(exact_expected))

    hard = patterns.assign(
        hard=patterns.predicted_pattern.str.split("|").apply(
            lambda value: len(value) == 4 and value[1] == "REM" and value[2] == "W"
        )
    )
    hard = hard[hard.hard].groupby("dataset").reference_events.sum().reindex(reference_total.index, fill_value=0)
    hard_expected = references.set_index("dataset").hard_change_at_reference.sort_index()
    add("hard_pattern_counts_recompute", hard.equals(hard_expected), True, hard.equals(hard_expected))

    add(
        "reference_fractions_recompute",
        True,
        True,
        close(references.exact_confirm2_fraction, references.exact_confirm2_references / references.reference_events)
        and close(
            references.hard_change_at_reference_fraction,
            references.hard_change_at_reference / references.reference_events,
        ),
    )
    add(
        "transition_rates_recompute",
        True,
        True,
        close(
            fragmentation.human_transitions_per_hour,
            fragmentation.human_all_stage_transitions / fragmentation.supported_hours,
        )
        and close(
            fragmentation.predicted_transitions_per_hour,
            fragmentation.predicted_all_stage_transitions / fragmentation.supported_hours,
        )
        and close(
            fragmentation.predicted_to_human_transition_ratio,
            fragmentation.predicted_all_stage_transitions / fragmentation.human_all_stage_transitions,
        ),
    )
    add(
        "rem_bout_rates_recompute",
        True,
        True,
        close(
            fragmentation.human_rem_bouts_per_hour,
            fragmentation.human_rem_bouts / fragmentation.supported_hours,
        )
        and close(
            fragmentation.predicted_rem_bouts_per_hour,
            fragmentation.predicted_rem_bouts / fragmentation.supported_hours,
        )
        and close(
            fragmentation.predicted_to_human_rem_bout_ratio,
            fragmentation.predicted_rem_bouts / fragmentation.human_rem_bouts,
        ),
    )
    add(
        "offset_coverage_recompute",
        True,
        True,
        close(
            offsets.coverage_fraction,
            offsets.references_with_alarm_within_120_sec / offsets.reference_events,
        ),
    )
    finite_offsets = offsets[
        ["nearest_offset_median_sec", "nearest_offset_q25_sec", "nearest_offset_q75_sec"]
    ].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    offset_bounds = np.isnan(finite_offsets) | (np.abs(finite_offsets) <= 120.0 + 1e-12)
    add("reported_offsets_within_window", offset_bounds.all(), True, offset_bounds.all())
    add(
        "no_new_candidate",
        sorted(offsets.candidate.unique().tolist()),
        sorted(common.CANDIDATES),
        set(offsets.candidate) == set(common.CANDIDATES),
    )

    result = pd.DataFrame(checks)
    if not result.passed.all():
        print(result.to_string(index=False))
        raise ValueError("Independent stage-first failure validation failed")
    verify_or_create_tsv(
        result,
        common.output_dir() / "independent_failure_diagnostic_validation_v0.1.tsv",
    )
    print(f"Independent stage-first failure validation passed {len(result)}/{len(result)} checks.")


if __name__ == "__main__":
    main()
