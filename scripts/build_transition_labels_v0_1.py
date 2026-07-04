"""Build deterministic REM/Wake transition labels v0.1.

This script combines the E0 transition inventory with full PSG-to-headband
alignment checks. It creates a reviewed label artifact only; it does not train
or evaluate a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_VERSION = "v0.1"
DATASET = "BOAS ds005555 snapshot 1.1.1"
LABEL_SOURCE = "PSG stage_hum"
WINDOW_SEC = 240.0
PRIMARY_TRANSITION = "REM_to_Wake"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def out_dir() -> Path:
    return repo_root() / "labels" / "transition_labels_v0.1"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    return str(value).strip()


def build_labels() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = repo_root()
    candidates = read_tsv(
        root
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "candidate_transition_events.tsv"
    )
    label_quality = read_tsv(
        root
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "label_quality_summary.tsv"
    )
    sample_alignment = read_tsv(
        root
        / "experiments"
        / "2026-07-04_boas_full_signal_alignment"
        / "transition_window_sample_alignment.tsv"
    )
    subject_alignment = read_tsv(
        root
        / "experiments"
        / "2026-07-04_boas_full_signal_alignment"
        / "subject_alignment_summary.tsv"
    )

    label_quality = label_quality[
        [
            "subject",
            "stage_hum_missing_epochs",
            "stage_hum_disconnection_epochs",
            "non_30_sec_epoch_count",
            "onset_gap_issue_count",
            "unlabeled_tail_sec",
        ]
    ]
    subject_alignment = subject_alignment[
        [
            "subject",
            "timeline_alignment_flag",
            "pulse_windows",
            "pulse_windows_usable",
            "pulse_usable_near_zero_lag_2s",
            "eeg_envelope_windows",
            "eeg_envelope_usable",
            "eeg_envelope_near_zero_lag_2s",
        ]
    ]
    sample_alignment = sample_alignment[
        [
            "transition_id",
            "headband_start_sample",
            "headband_stop_sample",
            "psg_start_sample",
            "psg_stop_sample",
            "window_sample_count_difference",
            "boundary_sample_difference",
            "sample_alignment_flag",
        ]
    ]

    labels = candidates.merge(sample_alignment, on="transition_id", how="left")
    labels = labels.merge(label_quality, on="subject", how="left", suffixes=("", "_recording"))
    labels = labels.merge(subject_alignment, on="subject", how="left")

    labels["label_version"] = LABEL_VERSION
    labels["dataset"] = DATASET
    labels["label_source"] = LABEL_SOURCE
    labels["nominal_boundary_sec"] = labels["boundary_onset_sec"]
    labels["uncertainty_width_sec"] = (
        labels["uncertainty_end_sec"] - labels["uncertainty_start_sec"]
    )
    labels["headband_boundary_sample_zero_based"] = (
        labels["nominal_boundary_sec"] * 256.0
    ).round().astype("int64")
    labels["psg_boundary_sample_zero_based"] = labels["headband_boundary_sample_zero_based"]
    labels["event_boundary_sample_zero_based"] = labels["next_begsample"].astype("int64") - 1
    labels["is_primary_label"] = labels["transition_type"] == PRIMARY_TRANSITION

    labels["quality_flags"] = labels.apply(quality_flags, axis=1)
    labels["label_decision"] = labels["quality_flags"].apply(
        lambda text: "include" if text == "pass" else "review"
    )

    ordered_columns = [
        "label_version",
        "transition_id",
        "subject",
        "participant_id",
        "pid",
        "dataset",
        "label_source",
        "transition_type",
        "is_primary_label",
        "stage_from",
        "stage_from_label",
        "stage_to",
        "stage_to_label",
        "previous_epoch_index_zero_based",
        "next_epoch_index_zero_based",
        "previous_onset_sec",
        "nominal_boundary_sec",
        "uncertainty_start_sec",
        "uncertainty_end_sec",
        "uncertainty_width_sec",
        "headband_boundary_sample_zero_based",
        "psg_boundary_sample_zero_based",
        "event_boundary_sample_zero_based",
        "headband_start_sample",
        "headband_stop_sample",
        "psg_start_sample",
        "psg_stop_sample",
        "inspection_window_sec",
        "timeline_alignment_flag",
        "sample_alignment_flag",
        "window_sample_count_difference",
        "boundary_sample_difference",
        "stage_hum_missing_epochs",
        "stage_hum_disconnection_epochs",
        "psg_disconnection_epochs_in_window",
        "has_psg_disconnection_in_window",
        "non_30_sec_epoch_count",
        "onset_gap_issue_count",
        "unlabeled_tail_sec",
        "pulse_windows",
        "pulse_windows_usable",
        "pulse_usable_near_zero_lag_2s",
        "eeg_envelope_windows",
        "eeg_envelope_usable",
        "eeg_envelope_near_zero_lag_2s",
        "quality_flags",
        "label_decision",
        "relative_events_path",
    ]
    labels = labels[ordered_columns].sort_values(["subject", "nominal_boundary_sec"])

    summary = label_summary(labels)
    pid_distribution = pid_label_distribution(labels)
    return labels, summary, pid_distribution


def quality_flags(row: pd.Series) -> str:
    flags = []
    if row["timeline_alignment_flag"] != "pass":
        flags.append("timeline_alignment_review")
    if row["sample_alignment_flag"] != "pass":
        flags.append("sample_alignment_review")
    if int(row["window_sample_count_difference"]) != 0:
        flags.append("window_sample_count_mismatch")
    if int(row["boundary_sample_difference"]) != 0:
        flags.append("boundary_sample_mismatch")
    if int(row["stage_hum_missing_epochs"]) != 0:
        flags.append("missing_stage_hum")
    if int(row["non_30_sec_epoch_count"]) != 0:
        flags.append("non_30_sec_epoch")
    if int(row["onset_gap_issue_count"]) != 0:
        flags.append("onset_gap_issue")
    if bool_text(row["has_psg_disconnection_in_window"]) == "True":
        flags.append("psg_disconnection_in_window")
    return "pass" if not flags else ";".join(flags)


def label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for transition_type, group in labels.groupby("transition_type", sort=True):
        rows.append(
            {
                "transition_type": transition_type,
                "rows": int(len(group)),
                "unique_subjects": int(group["subject"].nunique()),
                "unique_pid": int(group["pid"].nunique()),
                "include_rows": int((group["label_decision"] == "include").sum()),
                "review_rows": int((group["label_decision"] == "review").sum()),
                "sample_alignment_pass": int((group["sample_alignment_flag"] == "pass").sum()),
                "timeline_alignment_pass_subjects": int(
                    group[group["timeline_alignment_flag"] == "pass"]["subject"].nunique()
                ),
                "primary_rows": int(group["is_primary_label"].sum()),
            }
        )
    rows.append(
        {
            "transition_type": "ALL",
            "rows": int(len(labels)),
            "unique_subjects": int(labels["subject"].nunique()),
            "unique_pid": int(labels["pid"].nunique()),
            "include_rows": int((labels["label_decision"] == "include").sum()),
            "review_rows": int((labels["label_decision"] == "review").sum()),
            "sample_alignment_pass": int((labels["sample_alignment_flag"] == "pass").sum()),
            "timeline_alignment_pass_subjects": int(
                labels[labels["timeline_alignment_flag"] == "pass"]["subject"].nunique()
            ),
            "primary_rows": int(labels["is_primary_label"].sum()),
        }
    )
    return pd.DataFrame(rows)


def pid_label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, group in labels.groupby("pid", sort=True):
        primary = group[group["transition_type"] == PRIMARY_TRANSITION]
        secondary = group[group["transition_type"] != PRIMARY_TRANSITION]
        rows.append(
            {
                "pid": int(pid),
                "subjects": ";".join(sorted(group["subject"].unique())),
                "recordings_with_labels": int(group["subject"].nunique()),
                "total_transition_labels": int(len(group)),
                "rem_to_wake_labels": int(len(primary)),
                "wake_to_rem_labels": int(len(secondary)),
                "include_labels": int((group["label_decision"] == "include").sum()),
                "review_labels": int((group["label_decision"] == "review").sum()),
                "has_primary_label": bool(len(primary) > 0),
                "has_secondary_label": bool(len(secondary) > 0),
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    labels: pd.DataFrame,
    summary: pd.DataFrame,
    pid_distribution: pd.DataFrame,
    destination: Path,
) -> None:
    primary = labels[labels["is_primary_label"]]
    secondary = labels[~labels["is_primary_label"]]
    review_rows = labels[labels["label_decision"] == "review"]

    text = f"""# Transition Labels v0.1

**Created:** 2026-07-04
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Reference labels:** PSG `stage_hum`
**Alignment evidence:** `experiments/2026-07-04_boas_full_signal_alignment/`
**Model training performed:** No

## 1. Purpose

This artifact converts the E0 REM/Wake transition inventory into a versioned deterministic label table for later wearable EEG preprocessing.

The uncertainty addressed here is whether PSG-derived REM/Wake transition labels can be represented reproducibly with explicit 30-second label uncertainty and sample-level headband mapping.

## 2. Method

- Start from `candidate_transition_events.tsv` created during the E0 inventory.
- Preserve the nominal boundary and the `+/-15` second uncertainty interval from label specification `v0.1`.
- Add zero-based PSG/headband sample indices using the full signal-alignment validation.
- Keep primary REM-to-Wake labels separate from secondary Wake-to-REM labels.
- Add quality flags from event-label checks and PSG/headband alignment checks.
- Do not create train, validation, or test splits in this artifact.

## 3. Result

| Item | Value |
|---|---:|
| Total transition-label rows | {len(labels)} |
| Primary REM-to-Wake rows | {len(primary)} |
| Secondary Wake-to-REM rows | {len(secondary)} |
| Unique recordings | {labels['subject'].nunique()} |
| Unique `pid` values | {labels['pid'].nunique()} |
| Rows marked include | {(labels['label_decision'] == 'include').sum()} |
| Rows marked review | {len(review_rows)} |
| Rows with sample-alignment pass | {(labels['sample_alignment_flag'] == 'pass').sum()} |

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_labels_v0.1.tsv` | Versioned deterministic REM/Wake transition-label table |
| `transition_label_summary_v0.1.tsv` | Count summary by transition type |
| `pid_transition_distribution_v0.1.tsv` | Participant-level label distribution for grouped split planning |
| `grouped_split_policy_draft_v0.1.md` | Draft rules for later leakage-safe train/validation/test splitting |

## 5. Limitations

- The labels are derived from 30-second sleep-stage epochs, not exact physiological transition onsets.
- BOAS is not a dedicated narcolepsy or sleep-paralysis cohort.
- This table does not include negative/background windows yet.
- This table does not define train, validation, or test splits.
- Model training remains blocked until the label/preprocessing gate.

## 6. Decision

Use `transition_labels_v0.1.tsv` as the first reviewed positive-event label artifact for deterministic preprocessing and split-policy design.

Next work should define background-window rules and recording/window-level signal-quality flags before model work.
"""
    (destination / "README.md").write_text(text, encoding="utf-8")


def write_split_policy(pid_distribution: pd.DataFrame, destination: Path) -> None:
    repeated_pid = pid_distribution[pid_distribution["recordings_with_labels"] > 1]
    primary_pid = pid_distribution[pid_distribution["has_primary_label"]]

    text = f"""# Grouped Split Policy Draft v0.1

**Created:** 2026-07-04
**Applies to:** `transition_labels_v0.1.tsv`
**Status:** Draft policy; no train/validation/test split assigned yet
**Model training performed:** No

## 1. Purpose

This draft defines split constraints for later model evaluation without creating a split prematurely.

The uncertainty addressed here is leakage risk: BOAS contains repeated recordings for some `pid` values, so splitting by recording alone could place the same participant in more than one evaluation partition.

## 2. Hard Rules

- Split by `pid`, not by recording folder.
- Never place recordings from the same `pid` in more than one partition.
- Keep primary REM-to-Wake labels separate from secondary Wake-to-REM labels during stratification summaries.
- Do not create final train/validation/test assignments until background-window rules and signal-quality flags are complete.
- Record the random seed and exact label-table version when a split is eventually created.

## 3. Current Label Distribution

| Item | Value |
|---|---:|
| `pid` values with transition labels | {len(pid_distribution)} |
| `pid` values with primary REM-to-Wake labels | {len(primary_pid)} |
| `pid` values with repeated labeled recordings | {len(repeated_pid)} |
| Total transition labels | {int(pid_distribution['total_transition_labels'].sum())} |
| Primary REM-to-Wake labels | {int(pid_distribution['rem_to_wake_labels'].sum())} |
| Secondary Wake-to-REM labels | {int(pid_distribution['wake_to_rem_labels'].sum())} |

## 4. Later Split Design Requirements

Before creating a split, summarize candidate partitions by:

- `pid` count;
- recording count;
- primary REM-to-Wake count;
- secondary Wake-to-REM count;
- label-quality flags;
- signal-quality flags;
- recording duration distribution if needed.

## 5. Decision

Do not assign final splits yet. Use this draft and `pid_transition_distribution_v0.1.tsv` to design a leakage-safe split after background-window and signal-quality rules are defined.
"""
    (destination / "grouped_split_policy_draft_v0.1.md").write_text(text, encoding="utf-8")


def main() -> None:
    destination = out_dir()
    destination.mkdir(parents=True, exist_ok=True)
    labels, summary, pid_distribution = build_labels()
    labels.to_csv(destination / "transition_labels_v0.1.tsv", sep="\t", index=False)
    summary.to_csv(destination / "transition_label_summary_v0.1.tsv", sep="\t", index=False)
    pid_distribution.to_csv(destination / "pid_transition_distribution_v0.1.tsv", sep="\t", index=False)
    write_readme(labels, summary, pid_distribution, destination)
    write_split_policy(pid_distribution, destination)

    print("Transition labels v0.1")
    print(f"Rows: {len(labels)}")
    print(f"Primary REM-to-Wake rows: {int(labels['is_primary_label'].sum())}")
    print(f"PID values with labels: {len(pid_distribution)}")
    print(f"Rows marked include: {int((labels['label_decision'] == 'include').sum())}")
    print(f"Rows marked review: {int((labels['label_decision'] == 'review').sum())}")
    print(f"Outputs: {destination}")


if __name__ == "__main__":
    main()
