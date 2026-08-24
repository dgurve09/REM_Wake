"""Recompute frozen test summaries without drift-review test recordings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FLAGGED_TEST_RECORDINGS = {"sub-32", "sub-50"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_path() -> Path:
    return (
        repo_root()
        / "experiments"
        / "2026-08-23_boas_alignment_drift_audit_v0.1"
        / "flagged_test_recording_sensitivity_v0.1.tsv"
    )


def source_paths() -> dict[str, Path]:
    root = repo_root() / "experiments"
    return {
        "SF-C": root
        / "2026-08-15_stage_first_feature_baseline_v0.1"
        / "test_event_recordings_v0.1.tsv",
        "DE-B": root
        / "2026-08-22_direct_event_baseline_v0.1"
        / "test_event_recordings_v0.1.tsv",
    }


def metrics(data: pd.DataFrame) -> dict:
    true_positive = int(data["true_positive"].sum())
    false_positive = int(data["false_positive"].sum())
    false_negative = int(data["false_negative"].sum())
    supported_hours = float(data["supported_hours"].sum())
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "recordings": data["subject"].nunique(),
        "reference_events": true_positive + false_negative,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "supported_hours": supported_hours,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_hour": false_positive / supported_hours,
    }


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite changed reviewed output: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    rows = []
    for comparator, path in source_paths().items():
        data = pd.read_csv(path, sep="\t")
        selected = data[
            (data["comparator"] == comparator)
            & (data["partition"] == "test")
            & (data["membership"] == "primary")
            & (data["tolerance_sec"] == 15.0)
        ].copy()
        if FLAGGED_TEST_RECORDINGS - set(selected["subject"]):
            raise RuntimeError(f"Missing flagged recording for {comparator}")
        retained = selected[~selected["subject"].isin(FLAGGED_TEST_RECORDINGS)]
        rows.append(
            {
                "comparator": comparator,
                "analysis": "frozen_primary_without_drift_review_test_recordings",
                "excluded_recordings": ";".join(sorted(FLAGGED_TEST_RECORDINGS)),
                **metrics(retained),
            }
        )

    result = pd.DataFrame(rows)
    write_once(output_path(), result.to_csv(sep="\t", index=False, lineterminator="\n"))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
