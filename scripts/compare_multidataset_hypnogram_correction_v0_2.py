"""Compare corrected hypnogram results with the preserved v0.1 result."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_1 as old_common
import hypnogram_transition_common_v0_2 as new_common
from reviewed_output import verify_or_create_tsv


def read(folder, name: str) -> pd.DataFrame:
    return pd.read_csv(folder() / name, sep="\t")


def comparison_row(category: str, measure: str, old, new) -> dict:
    old_number = pd.to_numeric(pd.Series([old]), errors="coerce").iloc[0]
    new_number = pd.to_numeric(pd.Series([new]), errors="coerce").iloc[0]
    numeric = np.isfinite(old_number) and np.isfinite(new_number)
    changed = not np.isclose(old_number, new_number, rtol=0, atol=1e-12) if numeric else str(old) != str(new)
    return {
        "category": category,
        "measure": measure,
        "v0.1": old,
        "v0.2": new,
        "v0.2_minus_v0.1": float(new_number - old_number) if numeric else "",
        "changed": changed,
    }


def main() -> None:
    old_datasets = read(old_common.output_dir, "dataset_transition_summary_v0.1.tsv").set_index("dataset")
    new_datasets = read(new_common.output_dir, "dataset_transition_summary_v0.2.tsv").set_index("dataset")
    old_scorers = read(old_common.output_dir, "scorer_uncertainty_summary_v0.1.tsv").set_index("dataset")
    new_scorers = read(new_common.output_dir, "scorer_uncertainty_summary_v0.2.tsv").set_index("dataset")
    old_repeated = read(old_common.output_dir, "repeated_night_summary_v0.1.tsv").set_index("dataset")
    new_repeated = read(new_common.output_dir, "repeated_night_summary_v0.2.tsv").set_index("dataset")
    old_decisions = read(old_common.output_dir, "hypothesis_decisions_v0.1.tsv").set_index("hypothesis")
    new_decisions = read(new_common.output_dir, "hypothesis_decisions_v0.2.tsv").set_index("hypothesis")
    correction = read(new_common.output_dir, "gx_recording_label_correction_v0.2.tsv")

    rows = [
        comparison_row("label", "gx_primary_unknown_epochs", correction.unsafe_unknown_epochs.sum(), correction.corrected_unknown_epochs.sum()),
        comparison_row("label", "gx_primary_events", correction.unsafe_events.sum(), correction.corrected_events.sum()),
        comparison_row("dataset", "all_primary_events", old_datasets.rem_to_wake_events.sum(), new_datasets.rem_to_wake_events.sum()),
    ]
    dataset_measures = [
        "support_hours",
        "rem_hours",
        "rem_to_wake_events",
        "event_positive_night_fraction",
        "events_per_support_hour",
        "known_rem_exit_opportunities",
        "rw_fraction_of_known_rem_exits",
    ]
    for measure in dataset_measures:
        rows.append(comparison_row("gx_dataset", measure, old_datasets.loc["gx36m", measure], new_datasets.loc["gx36m", measure]))

    scorer_measures = [
        "informative_event_pairs",
        "event_presence_agreement_fraction",
        "median_stage_disagreement_fraction",
        "median_absolute_event_count_difference",
        "median_exact_event_f1",
        "median_tolerant_30_sec_event_f1",
    ]
    for measure in scorer_measures:
        rows.append(comparison_row("gx_scorer", measure, old_scorers.loc["gx36m", measure], new_scorers.loc["gx36m", measure]))

    repeated_measures = [
        "discordant_event_presence_fraction",
        "median_within_participant_event_count_range",
        "median_within_participant_event_rate_range",
    ]
    for measure in repeated_measures:
        rows.append(comparison_row("gx_repeated", measure, old_repeated.loc["gx36m", measure], new_repeated.loc["gx36m", measure]))

    for hypothesis in old_decisions.index:
        rows.append(comparison_row("decision", f"{hypothesis}:observed_1", old_decisions.loc[hypothesis, "observed_1"], new_decisions.loc[hypothesis, "observed_1"]))
        rows.append(comparison_row("decision", f"{hypothesis}:observed_2", old_decisions.loc[hypothesis, "observed_2"], new_decisions.loc[hypothesis, "observed_2"]))
        rows.append(comparison_row("decision", f"{hypothesis}:decision", old_decisions.loc[hypothesis, "decision"], new_decisions.loc[hypothesis, "decision"]))
    comparison = pd.DataFrame(rows)

    non_gx_dataset_equal = old_datasets.drop(index="gx36m").equals(new_datasets.drop(index="gx36m"))
    non_gx_scorer_equal = old_scorers.drop(index="gx36m").equals(new_scorers.drop(index="gx36m"))
    non_gx_repeated_equal = old_repeated.drop(index="gx36m").equals(new_repeated.drop(index="gx36m"))
    decisions_equal = old_decisions.decision.equals(new_decisions.decision)
    checks = pd.DataFrame(
        [
            {"check": "non_gx_dataset_rows_unchanged", "observed": non_gx_dataset_equal, "expected": True, "passed": non_gx_dataset_equal},
            {"check": "non_gx_scorer_rows_unchanged", "observed": non_gx_scorer_equal, "expected": True, "passed": non_gx_scorer_equal},
            {"check": "non_gx_repeated_rows_unchanged", "observed": non_gx_repeated_equal, "expected": True, "passed": non_gx_repeated_equal},
            {"check": "gx_primary_event_count_unchanged", "observed": int(new_datasets.loc["gx36m", "rem_to_wake_events"]), "expected": 85, "passed": int(new_datasets.loc["gx36m", "rem_to_wake_events"]) == 85},
            {"check": "all_primary_event_count_unchanged", "observed": int(new_datasets.rem_to_wake_events.sum()), "expected": 4046, "passed": int(new_datasets.rem_to_wake_events.sum()) == 4046},
            {"check": "gx_scorer_summary_changed", "observed": not old_scorers.loc["gx36m"].equals(new_scorers.loc["gx36m"]), "expected": True, "passed": not old_scorers.loc["gx36m"].equals(new_scorers.loc["gx36m"])},
            {"check": "high_level_decisions_unchanged", "observed": decisions_equal, "expected": True, "passed": decisions_equal},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Hypnogram version comparison failed")
    verify_or_create_tsv(comparison, new_common.output_dir() / "v0.1_v0.2_comparison_v0.2.tsv")
    verify_or_create_tsv(checks, new_common.output_dir() / "version_comparison_checks_v0.2.tsv")
    print(f"Version comparison passed {len(checks)}/{len(checks)} checks.")
    print(comparison[comparison.changed].to_string(index=False))


if __name__ == "__main__":
    main()
