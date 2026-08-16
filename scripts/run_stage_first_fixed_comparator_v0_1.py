"""Evaluate the fixed BOAS headband stage_ai sequence against PSG stage_hum."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

from stage_first_event_evaluation_v0_1 import evaluate_events, participant_bootstrap


VERSION = "v0.1"
COMPARATOR = "SF-A"
VALID_STAGES = [0, 1, 2, 3, 4]
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
PARTITIONS = ["train", "validation", "test"]
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
EXPERIMENT_DIR = "2026-08-15_stage_first_fixed_comparator_v0.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def dataset_root() -> Path:
    default = repo_root().parent / "REM_W_data"
    return Path(os.environ.get("REM_W_DATA_ROOT", default)) / "boas_ds005555_v1.1.1"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def subject_assignments(root: Path) -> pd.DataFrame:
    split = pd.read_csv(
        root / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    rows = []
    for item in split.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append(
                {"subject": subject, "pid": int(item.pid), "partition": item.partition}
            )
    result = pd.DataFrame(rows)
    if len(result) != 128 or result["subject"].nunique() != 128:
        raise ValueError("Expected 128 unique subject assignments")
    return result


def load_recording(data: Path, subject: str, pid: int, partition: str) -> tuple:
    eeg = data / subject / "eeg"
    psg_path = eeg / f"{subject}_task-Sleep_acq-psg_events.tsv"
    headband_path = eeg / f"{subject}_task-Sleep_acq-headband_events.tsv"
    psg = pd.read_csv(
        psg_path,
        sep="\t",
        usecols=["onset", "duration", "begsample", "endsample", "stage_hum"],
    )
    headband = pd.read_csv(
        headband_path,
        sep="\t",
        usecols=["onset", "duration", "begsample", "endsample", "stage_ai"],
    )
    merged = psg.merge(
        headband,
        on="onset",
        how="outer",
        suffixes=("_psg", "_headband"),
        indicator=True,
        validate="one_to_one",
    ).sort_values("onset")
    aligned = bool(
        len(psg) == len(headband)
        and (merged["_merge"] == "both").all()
        and np.isclose(merged["duration_psg"], 30.0).all()
        and np.isclose(merged["duration_headband"], 30.0).all()
        and (merged["begsample_psg"] == merged["begsample_headband"]).all()
        and (merged["endsample_psg"] == merged["endsample_headband"]).all()
    )
    merged["subject"] = subject
    merged["pid"] = pid
    merged["partition"] = partition
    human_codes = set(merged["stage_hum"].dropna().astype(int).unique())
    ai_codes = set(merged["stage_ai"].dropna().astype(int).unique())
    valid_epoch = merged["stage_hum"].isin(VALID_STAGES) & merged["stage_ai"].isin(
        VALID_STAGES
    )
    stage_rows = merged.loc[
        valid_epoch,
        ["subject", "pid", "partition", "onset", "stage_hum", "stage_ai"],
    ].copy()
    stage_rows[["stage_hum", "stage_ai"]] = stage_rows[
        ["stage_hum", "stage_ai"]
    ].astype(int)

    predicted_events = []
    supported_boundaries = 0
    previous = None
    for row in merged.itertuples(index=False):
        current_valid = row.stage_hum in VALID_STAGES and row.stage_ai in VALID_STAGES
        if previous is not None:
            previous_valid = (
                previous.stage_hum in VALID_STAGES and previous.stage_ai in VALID_STAGES
            )
            contiguous = np.isclose(float(row.onset) - float(previous.onset), 30.0)
            if current_valid and previous_valid and contiguous:
                supported_boundaries += 1
                if int(previous.stage_ai) == 4 and int(row.stage_ai) == 0:
                    predicted_events.append(
                        {
                            "subject": subject,
                            "pid": pid,
                            "partition": partition,
                            "event_time_sec": float(row.onset),
                        }
                    )
        previous = row

    readiness = {
        "comparator_version": VERSION,
        "subject": subject,
        "pid": pid,
        "partition": partition,
        "psg_rows": len(psg),
        "headband_rows": len(headband),
        "aligned_rows": int((merged["_merge"] == "both").sum()),
        "event_table_alignment_pass": aligned,
        "human_stage_codes": ";".join(str(value) for value in sorted(human_codes)),
        "headband_ai_stage_codes": ";".join(str(value) for value in sorted(ai_codes)),
        "valid_stage_comparison_epochs": len(stage_rows),
        "supported_boundaries": supported_boundaries,
        "predicted_rem_to_wake_events": len(predicted_events),
    }
    support = {
        "subject": subject,
        "pid": pid,
        "partition": partition,
        "supported_hours": supported_boundaries * 30.0 / 3600.0,
    }
    return readiness, stage_rows, predicted_events, support


def stage_metrics(stage_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []
    confusion_rows = []
    for partition in PARTITIONS:
        group = stage_rows[stage_rows["partition"] == partition]
        truth = group["stage_hum"].to_numpy(dtype=int)
        prediction = group["stage_ai"].to_numpy(dtype=int)
        summary_rows.append(
            {
                "comparator": COMPARATOR,
                "comparator_version": VERSION,
                "partition": partition,
                "epochs": len(group),
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, labels=VALID_STAGES, average="macro"),
                "cohen_kappa": cohen_kappa_score(truth, prediction, labels=VALID_STAGES),
            }
        )
        report = classification_report(
            truth,
            prediction,
            labels=VALID_STAGES,
            target_names=[STAGE_NAMES[value] for value in VALID_STAGES],
            output_dict=True,
            zero_division=0,
        )
        for stage in VALID_STAGES:
            values = report[STAGE_NAMES[stage]]
            class_rows.append(
                {
                    "comparator": COMPARATOR,
                    "comparator_version": VERSION,
                    "partition": partition,
                    "stage_code": stage,
                    "stage": STAGE_NAMES[stage],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
        matrix = confusion_matrix(truth, prediction, labels=VALID_STAGES)
        for truth_index, truth_stage in enumerate(VALID_STAGES):
            for prediction_index, prediction_stage in enumerate(VALID_STAGES):
                confusion_rows.append(
                    {
                        "comparator": COMPARATOR,
                        "comparator_version": VERSION,
                        "partition": partition,
                        "true_stage_code": truth_stage,
                        "true_stage": STAGE_NAMES[truth_stage],
                        "predicted_stage_code": prediction_stage,
                        "predicted_stage": STAGE_NAMES[prediction_stage],
                        "epochs": int(matrix[truth_index, prediction_index]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def reference_events(root: Path) -> pd.DataFrame:
    membership = pd.read_csv(
        root / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        root / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    rows = membership.merge(quality, on="transition_id", validate="one_to_one")
    rows = rows[rows["is_primary_label"].astype(str).str.lower() == "true"].copy()
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    return rows


def event_results(
    references: pd.DataFrame,
    predictions: pd.DataFrame,
    support: pd.DataFrame,
) -> tuple:
    summary_rows = []
    bootstrap_frames = []
    recording_frames = []
    participant_frames = []
    match_frames = []
    for partition in PARTITIONS:
        partition_support = support[support["partition"] == partition][
            ["subject", "pid", "supported_hours"]
        ]
        partition_predictions = predictions[predictions["partition"] == partition][
            ["subject", "pid", "event_time_sec"]
        ]
        partition_references = references[references["partition"] == partition]
        for membership in MEMBERSHIPS:
            eligible_column = (
                "primary_analysis_eligible"
                if membership == "primary"
                else "expanded_quality_analysis_eligible"
            )
            eligibility = (
                partition_references[eligible_column]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )
            eligible = partition_references[
                eligibility
            ][["subject", "pid", "event_time_sec"]]
            ignored = partition_references[
                ~eligibility
            ][["subject", "pid", "event_time_sec"]]
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible,
                    partition_predictions,
                    ignored,
                    partition_support,
                    tolerance,
                )
                configuration = {
                    "comparator": COMPARATOR,
                    "comparator_version": VERSION,
                    "partition": partition,
                    "membership": membership,
                    "tolerance_sec": tolerance,
                }
                summary_rows.append({**configuration, **summary})
                bootstrap = participant_bootstrap(participants)
                for key, value in reversed(list(configuration.items())):
                    bootstrap.insert(0, key, value)
                bootstrap_frames.append(bootstrap)
                for frame, collection in [
                    (recordings, recording_frames),
                    (participants, participant_frames),
                    (matches, match_frames),
                ]:
                    if len(frame):
                        frame = frame.copy()
                        for key, value in reversed(list(configuration.items())):
                            if key not in frame.columns:
                                frame.insert(0, key, value)
                        collection.append(frame)
    return (
        pd.DataFrame(summary_rows),
        pd.concat(bootstrap_frames, ignore_index=True),
        pd.concat(recording_frames, ignore_index=True),
        pd.concat(participant_frames, ignore_index=True),
        pd.concat(match_frames, ignore_index=True),
    )


def write_readme(destination: Path, readiness: pd.DataFrame, stages: pd.DataFrame, events: pd.DataFrame) -> None:
    stage_lines = []
    for row in stages.itertuples(index=False):
        stage_lines.append(
            f"| {row.partition.title()} | {row.epochs:,} | {row.macro_f1:.4f} | {row.balanced_accuracy:.4f} | {row.cohen_kappa:.4f} |"
        )
    primary = events[
        (events["membership"] == "primary") & (events["tolerance_sec"] == 15.0)
    ]
    event_lines = []
    for row in primary.itertuples(index=False):
        event_lines.append(
            f"| {row.partition.title()} | {row.reference_events} | {row.predicted_events} | {row.true_positive} | {row.false_positive} | {row.false_negative} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        )
    text = f"""# Fixed Stage-First Comparator v0.1

**Created:** 2026-08-15
**Comparator:** BOAS headband `stage_ai` (`SF-A`)
**Ground truth:** PSG human consensus `stage_hum`
**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`
**Model trained in this experiment:** No

## Readiness

- Event-table pairs checked: {len(readiness)}
- Event-table alignment passes: {int(readiness['event_table_alignment_pass'].sum())} of {len(readiness)}
- Valid stage-comparison epochs: {int(readiness['valid_stage_comparison_epochs'].sum()):,}
- Predicted REM-to-Wake events: {int(readiness['predicted_rem_to_wake_events'].sum()):,}

## Stage Diagnostics

| Partition | Epochs | Macro F1 | Balanced accuracy | Cohen kappa |
|---|---:|---:|---:|---:|
{chr(10).join(stage_lines)}

## Primary Event Result (+/-15 seconds)

| Partition | Reference | Predicted | TP | FP | FN | Precision | Recall | F1 | False alarms/hour |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(event_lines)}

## Execution Record

The initial execution stopped before any metric file was written because the wrapper attempted to add a `tolerance_sec` column that was already returned by the evaluator. The output-assembly guard was corrected and the frozen scientific configuration was not changed. This was an implementation failure, not evidence about the research hypothesis.

## Interpretation Boundary

The fixed headband stage sequence is a useful stage-first comparator, but its training provenance and independence from BOAS are not established by the dataset files. It is not human ground truth and cannot replace the participant-independent transparent baselines.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    data = dataset_root()
    assignments = subject_assignments(root).sort_values(
        "subject", key=lambda values: values.map(subject_number)
    )
    readiness_rows = []
    stage_frames = []
    predicted_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        readiness, stages, predictions, support = load_recording(
            data, item.subject, int(item.pid), item.partition
        )
        readiness_rows.append(readiness)
        stage_frames.append(stages)
        predicted_rows.extend(predictions)
        support_rows.append(support)
    readiness = pd.DataFrame(readiness_rows)
    stages = pd.concat(stage_frames, ignore_index=True)
    predictions = pd.DataFrame(
        predicted_rows,
        columns=["subject", "pid", "partition", "event_time_sec"],
    )
    support = pd.DataFrame(support_rows)
    if not readiness["event_table_alignment_pass"].all():
        raise RuntimeError("At least one PSG/headband event-table pair failed alignment")
    if set(stages["stage_hum"].unique()) != set(VALID_STAGES):
        raise RuntimeError("Human stage comparison does not contain all five stages")
    if set(stages["stage_ai"].unique()) != set(VALID_STAGES):
        raise RuntimeError("Headband stage comparison does not contain all five stages")

    stage_summary, stage_classes, stage_confusion = stage_metrics(stages)
    references = reference_events(root)
    event_summary, bootstrap, recording_metrics, participant_metrics, matches = event_results(
        references, predictions, support
    )

    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    readiness.to_csv(destination / "recording_event_table_readiness_v0.1.tsv", sep="\t", index=False)
    stage_summary.to_csv(destination / "stage_metrics_v0.1.tsv", sep="\t", index=False)
    stage_classes.to_csv(destination / "stage_class_metrics_v0.1.tsv", sep="\t", index=False)
    stage_confusion.to_csv(destination / "stage_confusion_matrix_v0.1.tsv", sep="\t", index=False)
    predictions.to_csv(destination / "predicted_rem_to_wake_events_v0.1.tsv", sep="\t", index=False)
    support.to_csv(destination / "event_support_by_recording_v0.1.tsv", sep="\t", index=False)
    event_summary.to_csv(destination / "event_metrics_v0.1.tsv", sep="\t", index=False)
    bootstrap.to_csv(destination / "event_metrics_bootstrap_v0.1.tsv", sep="\t", index=False)
    recording_metrics.to_csv(destination / "event_recording_metrics_v0.1.tsv", sep="\t", index=False)
    participant_metrics.to_csv(destination / "event_participant_metrics_v0.1.tsv", sep="\t", index=False)
    matches.to_csv(destination / "event_matches_v0.1.tsv", sep="\t", index=False)
    write_readme(destination, readiness, stage_summary, event_summary)
    print(stage_summary.to_string(index=False))
    print(event_summary.to_string(index=False))
    print(f"Wrote fixed stage-first comparator to {destination}")


if __name__ == "__main__":
    main()
