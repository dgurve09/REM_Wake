"""Combine the three frozen stage-first comparators into a Block 5 closeout."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VERSION = "v0.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments/2026-08-15_stage_first_comparison_v0.1"


def load_stage_metrics() -> pd.DataFrame:
    experiments = repo_root() / "experiments"
    fixed = pd.read_csv(
        experiments / "2026-08-15_stage_first_fixed_comparator_v0.1/stage_metrics_v0.1.tsv",
        sep="\t",
    ).rename(columns={"comparator_version": "model_version"})
    transparent = pd.concat(
        [
            pd.read_csv(
                experiments
                / "2026-08-15_stage_first_feature_baseline_v0.1/train_validation_stage_metrics_v0.1.tsv",
                sep="\t",
            ),
            pd.read_csv(
                experiments
                / "2026-08-15_stage_first_feature_baseline_v0.1/test_stage_metrics_v0.1.tsv",
                sep="\t",
            ),
        ],
        ignore_index=True,
    )
    result = pd.concat([fixed, transparent], ignore_index=True)
    result.insert(
        2,
        "provenance_role",
        result["comparator"].map(
            {
                "SF-A": "fixed_descriptive_unknown_training_provenance",
                "SF-B": "transparent_participant_grouped_epoch_ablation",
                "SF-C": "transparent_participant_grouped_primary",
            }
        ),
    )
    return result.sort_values(["partition", "comparator"])


def load_event_metrics() -> pd.DataFrame:
    experiments = repo_root() / "experiments"
    fixed = pd.read_csv(
        experiments / "2026-08-15_stage_first_fixed_comparator_v0.1/event_metrics_v0.1.tsv",
        sep="\t",
    ).rename(columns={"comparator_version": "model_version"})
    transparent = pd.concat(
        [
            pd.read_csv(
                experiments
                / "2026-08-15_stage_first_feature_baseline_v0.1/train_validation_event_metrics_v0.1.tsv",
                sep="\t",
            ),
            pd.read_csv(
                experiments
                / "2026-08-15_stage_first_feature_baseline_v0.1/test_event_metrics_v0.1.tsv",
                sep="\t",
            ),
        ],
        ignore_index=True,
    )
    result = pd.concat([fixed, transparent], ignore_index=True)
    result.insert(
        2,
        "provenance_role",
        result["comparator"].map(
            {
                "SF-A": "fixed_descriptive_unknown_training_provenance",
                "SF-B": "transparent_participant_grouped_epoch_ablation",
                "SF-C": "transparent_participant_grouped_primary",
            }
        ),
    )
    return result.sort_values(["partition", "membership", "tolerance_sec", "comparator"])


def write_readme(stages: pd.DataFrame, events: pd.DataFrame) -> None:
    test_stages = stages[stages["partition"] == "test"].set_index("comparator")
    primary = events[
        (events["partition"] == "test")
        & (events["membership"] == "primary")
        & (events["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    table = []
    for comparator in ["SF-A", "SF-B", "SF-C"]:
        stage = test_stages.loc[comparator]
        event = primary.loc[comparator]
        table.append(
            f"| {comparator} | {stage.macro_f1:.4f} | {event.supported_hours:.3f} | {event.precision:.4f} | {event.recall:.4f} | {event.f1:.4f} | {event.false_alarms_per_hour:.4f} |"
        )
    text = f"""# Stage-First Comparator Closeout v0.1

**Created:** 2026-08-15
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Primary event definition:** primary quality membership, +/-15-second matching

## Frozen Test Comparison

| Comparator | Stage macro F1 | Supported hours | Event precision | Event recall | Event F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

## Answer to the Block 5 Question

The transparent stage-first result is inadequate as a REM-to-Wake alarm. SF-C obtains test stage macro F1 {test_stages.loc['SF-C', 'macro_f1']:.4f}, but its event precision is only {primary.loc['SF-C', 'precision']:.4f} and it produces {primary.loc['SF-C', 'false_alarms_per_hour']:.4f} false alarms/hour. Five-epoch context improves over the epoch-only ablation, but does not resolve the event-specific failure.

SF-A is substantially stronger, with test event F1 {primary.loc['SF-A', 'f1']:.4f}, but it has unknown training provenance and cannot be assumed independent of BOAS. It remains a fixed descriptive reference, not the primary held-out estimate and not a model-selection target.

Valid prediction coverage differs slightly across comparators: SF-A, SF-B, and SF-C contribute {primary.loc['SF-A', 'supported_hours']:.3f}, {primary.loc['SF-B', 'supported_hours']:.3f}, and {primary.loc['SF-C', 'supported_hours']:.3f} supported test hours, respectively. Event metrics and false alarms/hour therefore use comparator-specific exposure and should not be interpreted as an identical-coverage ranking.

## Remaining Uncertainty

Block 5 does not establish that direct boundary detection will work. It establishes a measured failure mode for a reproducible stage-first approach: modest stage errors create many spurious adjacent REM-to-Wake boundaries. Block 6 must test whether a direct event objective can reduce false alarms while preserving useful recall under the same split, labels, quality tiers, and uncertainty tolerances.

An exploratory failure analysis using frozen result tables found a specific mechanism. On test, SF-C produced 2.1784 times the human all-stage transition rate, 799 predicted REM bouts versus 156 human bouts, and a median REM-bout duration of 60 seconds versus 600 seconds. All 20 test participants produced at least one false positive. These post-result diagnostics support sequence fragmentation as the main stage-first limitation but do not replace the primary metrics.

A second exploratory paired diagnostic found SF-C stage macro F1 improved for all 20 test participants, but event F1 improved for 10 and was unchanged for 10. The paired-bootstrap event-F1 difference was +0.0431 with a 95% interval of +0.0199 to +0.0676; the false-alarm-rate difference interval crossed zero. Six of seven additional SF-C matches admitted at +/-45 seconds were one epoch early, and 59.20% of predicted REM bouts lasted only 30 or 60 seconds. These findings explain context's partial benefit without establishing an adequate alarm rate.

## Decision

Proceed to the prespecified simple direct feature baseline. Do not add a CNN until the direct feature result identifies a specific limitation requiring a richer representation. Preserve the SF-C 500-iteration convergence warning when comparing methods.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    stages = load_stage_metrics()
    events = load_event_metrics()
    stages.to_csv(destination / "stage_first_stage_metrics_v0.1.tsv", sep="\t", index=False)
    events.to_csv(destination / "stage_first_event_metrics_v0.1.tsv", sep="\t", index=False)
    write_readme(stages, events)
    print(stages[stages["partition"] == "test"].to_string(index=False))
    print(
        events[
            (events["partition"] == "test")
            & (events["membership"] == "primary")
            & (events["tolerance_sec"] == 15.0)
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
