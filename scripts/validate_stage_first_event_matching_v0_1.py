"""Validate event matching using synthetic edge cases before model evaluation."""

from pathlib import Path

import pandas as pd

from stage_first_event_evaluation_v0_1 import evaluate_events, optimal_matches


VERSION = "v0.1"
EXPERIMENT_DIR = "2026-08-15_event_matching_validation_v0.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def event_frame(subject: str, pid: int, times: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"subject": subject, "pid": pid, "event_time_sec": time} for time in times],
        columns=["subject", "pid", "event_time_sec"],
    )


def run_case(
    name: str,
    references: list[float],
    predictions: list[float],
    ignored: list[float],
    tolerance: float,
    expected: tuple[int, int, int, int],
) -> tuple[dict, pd.DataFrame]:
    support = pd.DataFrame([{"subject": "sub-1", "pid": 1, "supported_hours": 1.0}])
    recordings, _, matches, summary = evaluate_events(
        event_frame("sub-1", 1, references),
        event_frame("sub-1", 1, predictions),
        event_frame("sub-1", 1, ignored),
        support,
        tolerance,
    )
    actual = (
        summary["true_positive"],
        summary["false_positive"],
        summary["false_negative"],
        summary["ignored_predictions"],
    )
    row = {
        "validation_version": VERSION,
        "case": name,
        "tolerance_sec": tolerance,
        "reference_times_sec": ";".join(str(value) for value in references) or "none",
        "prediction_times_sec": ";".join(str(value) for value in predictions) or "none",
        "ignored_times_sec": ";".join(str(value) for value in ignored) or "none",
        "expected_tp": expected[0],
        "expected_fp": expected[1],
        "expected_fn": expected[2],
        "expected_ignored": expected[3],
        "actual_tp": actual[0],
        "actual_fp": actual[1],
        "actual_fn": actual[2],
        "actual_ignored": actual[3],
        "passed": actual == expected,
    }
    matches = matches.copy()
    if len(matches):
        matches.insert(0, "case", name)
    return row, matches


def main() -> None:
    cases = [
        ("exact_match", [60], [60], [], 15, (1, 0, 0, 0)),
        ("uncertainty_edge", [60], [75], [], 15, (1, 0, 0, 0)),
        ("outside_tolerance", [60], [90], [], 15, (0, 1, 1, 0)),
        ("duplicate_prediction", [60], [60, 70], [], 15, (1, 1, 0, 0)),
        ("one_ignored_duplicate", [], [120, 130], [120], 15, (0, 1, 0, 1)),
        ("eligible_precedes_ignore", [120], [120], [120], 15, (1, 0, 0, 0)),
        ("no_prediction", [60], [], [], 15, (0, 0, 1, 0)),
        ("no_reference", [], [60], [], 15, (0, 1, 0, 0)),
    ]
    rows = []
    match_frames = []
    for case in cases:
        row, matches = run_case(*case)
        rows.append(row)
        if len(matches):
            match_frames.append(matches)

    ambiguous = optimal_matches([0, 30], [29], 30)
    ambiguous_pass = len(ambiguous) == 1 and ambiguous[0][0] == 1 and ambiguous[0][2] == 1
    rows.append(
        {
            "validation_version": VERSION,
            "case": "minimum_error_tiebreak",
            "tolerance_sec": 30,
            "reference_times_sec": "0;30",
            "prediction_times_sec": "29",
            "ignored_times_sec": "none",
            "expected_tp": 1,
            "expected_fp": 0,
            "expected_fn": 1,
            "expected_ignored": 0,
            "actual_tp": len(ambiguous),
            "actual_fp": 0,
            "actual_fn": 2 - len(ambiguous),
            "actual_ignored": 0,
            "passed": ambiguous_pass,
        }
    )

    support = pd.DataFrame(
        [
            {"subject": "sub-1", "pid": 1, "supported_hours": 1.0},
            {"subject": "sub-2", "pid": 2, "supported_hours": 1.0},
        ]
    )
    _, _, _, cross_summary = evaluate_events(
        event_frame("sub-1", 1, [60]),
        event_frame("sub-2", 2, [60]),
        event_frame("sub-1", 1, []),
        support,
        15,
    )
    cross_pass = (
        cross_summary["true_positive"] == 0
        and cross_summary["false_positive"] == 1
        and cross_summary["false_negative"] == 1
    )
    rows.append(
        {
            "validation_version": VERSION,
            "case": "cross_recording_isolation",
            "tolerance_sec": 15,
            "reference_times_sec": "sub-1:60",
            "prediction_times_sec": "sub-2:60",
            "ignored_times_sec": "none",
            "expected_tp": 0,
            "expected_fp": 1,
            "expected_fn": 1,
            "expected_ignored": 0,
            "actual_tp": cross_summary["true_positive"],
            "actual_fp": cross_summary["false_positive"],
            "actual_fn": cross_summary["false_negative"],
            "actual_ignored": cross_summary["ignored_predictions"],
            "passed": cross_pass,
        }
    )

    results = pd.DataFrame(rows)
    all_matches = (
        pd.concat(match_frames, ignore_index=True)
        if match_frames
        else pd.DataFrame()
    )
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination / "synthetic_event_matching_cases_v0.1.tsv", sep="\t", index=False)
    all_matches.to_csv(destination / "synthetic_event_matches_v0.1.tsv", sep="\t", index=False)
    passed = int(results["passed"].sum())
    text = f"""# Event Matching Validation v0.1

**Created:** 2026-08-15
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Model predictions used:** No

## Result

Synthetic cases passed: {passed} of {len(results)}.

The cases cover exact and boundary matches, outside-tolerance predictions, duplicate predictions, one-to-one quality ignores, eligible-reference precedence, empty prediction/reference sets, minimum-error tie breaking, and cross-recording isolation.

## Decision

{'Use the validated matcher for stage-first event evaluation.' if passed == len(results) else 'Do not evaluate model events until the failed matcher cases are resolved.'}
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")
    print(results.to_string(index=False))
    if passed != len(results):
        raise RuntimeError("Synthetic event-matching validation failed")


if __name__ == "__main__":
    main()
