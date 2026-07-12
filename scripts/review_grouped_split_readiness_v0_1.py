"""Review participant-grouped split readiness v0.1.

This script summarizes positive transition labels, background review windows,
and quality decisions by BOAS `pid`. It does not create train/validation/test
assignments and does not train or evaluate a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_VERSION = "v0.1"
PRIMARY_TRANSITION = "REM_to_Wake"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "labels" / "split_readiness_v0.1"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def subject_list(values: pd.Series) -> str:
    subjects = sorted(set(str(value) for value in values if pd.notna(value)))
    return ";".join(subjects)


def build_pid_readiness(
    transitions: pd.DataFrame,
    transition_quality: pd.DataFrame,
    background_quality: pd.DataFrame,
) -> pd.DataFrame:
    transition_joined = transitions.merge(
        transition_quality[["transition_id", "window_decision"]],
        on="transition_id",
        how="left",
    )

    rows = []
    all_pids = sorted(
        set(int(value) for value in transition_joined["pid"].unique())
        | set(int(value) for value in background_quality["pid"].unique())
    )

    for pid in all_pids:
        p_transitions = transition_joined[transition_joined["pid"] == pid]
        p_background = background_quality[background_quality["pid"] == pid]
        primary = p_transitions[p_transitions["transition_type"] == PRIMARY_TRANSITION]
        secondary = p_transitions[p_transitions["transition_type"] != PRIMARY_TRANSITION]
        strict_background = p_background[
            p_background["background_tier"] == "strict_same_stage_window"
        ]
        nontarget_background = p_background[
            p_background["background_tier"] == "nontarget_window_no_remwake_nearby"
        ]

        transition_subjects = set(str(value) for value in p_transitions["subject"].unique())
        background_subjects = set(str(value) for value in p_background["subject"].unique())
        all_subjects = sorted(transition_subjects | background_subjects)

        has_primary = len(primary) > 0
        has_background = len(p_background) > 0
        if has_primary and has_background:
            readiness = "positive_and_background_available"
        elif has_primary:
            readiness = "positive_only_review_needed"
        elif has_background:
            readiness = "background_only"
        else:
            readiness = "no_windows"

        rows.append(
            {
                "label_version": LABEL_VERSION,
                "pid": pid,
                "subjects": ";".join(all_subjects),
                "subject_count": len(all_subjects),
                "transition_subject_count": len(transition_subjects),
                "background_subject_count": len(background_subjects),
                "primary_rem_to_wake_labels": int(len(primary)),
                "secondary_wake_to_rem_labels": int(len(secondary)),
                "transition_windows_include": int(
                    (p_transitions["window_decision"] == "include_for_preprocessing").sum()
                ),
                "transition_windows_review": int(
                    (p_transitions["window_decision"] == "review_before_modeling").sum()
                ),
                "background_review_windows": int(len(p_background)),
                "strict_background_review_windows": int(len(strict_background)),
                "nontarget_background_review_windows": int(len(nontarget_background)),
                "background_windows_include": int(
                    (p_background["window_decision"] == "include_for_preprocessing").sum()
                ),
                "background_windows_review": int(
                    (p_background["window_decision"] == "review_before_modeling").sum()
                ),
                "has_repeated_recordings": len(all_subjects) > 1,
                "split_readiness_status": readiness,
            }
        )
    return pd.DataFrame(rows)


def build_summary(pid_readiness: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "pid_values_with_any_window", "value": len(pid_readiness)},
        {
            "metric": "pid_values_with_primary_rem_to_wake",
            "value": int((pid_readiness["primary_rem_to_wake_labels"] > 0).sum()),
        },
        {
            "metric": "pid_values_with_background_review_windows",
            "value": int((pid_readiness["background_review_windows"] > 0).sum()),
        },
        {
            "metric": "pid_values_with_positive_and_background",
            "value": int(
                (
                    pid_readiness["split_readiness_status"]
                    == "positive_and_background_available"
                ).sum()
            ),
        },
        {
            "metric": "pid_values_background_only",
            "value": int((pid_readiness["split_readiness_status"] == "background_only").sum()),
        },
        {
            "metric": "pid_values_with_repeated_recordings",
            "value": int(pid_readiness["has_repeated_recordings"].sum()),
        },
        {
            "metric": "primary_rem_to_wake_labels",
            "value": int(pid_readiness["primary_rem_to_wake_labels"].sum()),
        },
        {
            "metric": "secondary_wake_to_rem_labels",
            "value": int(pid_readiness["secondary_wake_to_rem_labels"].sum()),
        },
        {
            "metric": "background_review_windows",
            "value": int(pid_readiness["background_review_windows"].sum()),
        },
        {
            "metric": "quality_review_windows",
            "value": int(
                pid_readiness["transition_windows_review"].sum()
                + pid_readiness["background_windows_review"].sum()
            ),
        },
    ]
    return pd.DataFrame(rows)


def write_readme(destination: Path, summary: pd.DataFrame) -> None:
    metrics = dict(zip(summary["metric"], summary["value"]))
    text = f"""# Split Readiness v0.1

**Created:** 2026-07-09
**Applies to:** `transition_labels_v0.1`, `background_windows_v0.1`, and `signal_quality_flags_v0.1`
**Status:** Readiness review only; no train/validation/test split assigned
**Model training performed:** No

## 1. Purpose

This artifact reviews whether participant-grouped split design is feasible after transition labels, background rules, and quality flags are available.

The uncertainty addressed here is leakage risk and group balance: BOAS has repeated recordings for some `pid` values, so future evaluation must group by `pid`, not by recording folder.

## 2. Result

| Item | Value |
|---|---:|
| `pid` values with any reviewed window | {metrics['pid_values_with_any_window']} |
| `pid` values with primary REM-to-Wake labels | {metrics['pid_values_with_primary_rem_to_wake']} |
| `pid` values with background review windows | {metrics['pid_values_with_background_review_windows']} |
| `pid` values with both primary positives and background windows | {metrics['pid_values_with_positive_and_background']} |
| Background-only `pid` values | {metrics['pid_values_background_only']} |
| `pid` values with repeated recordings | {metrics['pid_values_with_repeated_recordings']} |
| Primary REM-to-Wake labels | {metrics['primary_rem_to_wake_labels']} |
| Secondary Wake-to-REM labels | {metrics['secondary_wake_to_rem_labels']} |
| Background review windows | {metrics['background_review_windows']} |
| Windows requiring quality review | {metrics['quality_review_windows']} |

The repeated-recording count here is based on all reviewed windows after adding background candidates. The earlier transition-label draft counted repeated `pid` values only among recordings with transition labels.

## 3. Outputs

| File | Purpose |
|---|---|
| `pid_split_readiness_v0.1.tsv` | Participant-level counts for future grouped split design |
| `split_readiness_summary_v0.1.tsv` | Overall readiness count summary |

## 4. Decision

Do not assign final splits yet. The next split-design step should choose a deterministic policy that preserves all recordings from each `pid` in one partition and reports event/background balance before model work.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    transitions = read_tsv(root / "labels" / "transition_labels_v0.1" / "transition_labels_v0.1.tsv")
    transition_quality = read_tsv(
        root
        / "labels"
        / "signal_quality_flags_v0.1"
        / "transition_window_quality_flags_v0.1.tsv"
    )
    background_quality = read_tsv(
        root
        / "labels"
        / "signal_quality_flags_v0.1"
        / "background_window_quality_flags_v0.1.tsv"
    )

    pid_readiness = build_pid_readiness(transitions, transition_quality, background_quality)
    summary = build_summary(pid_readiness)

    pid_readiness.to_csv(destination / "pid_split_readiness_v0.1.tsv", sep="\t", index=False)
    summary.to_csv(destination / "split_readiness_summary_v0.1.tsv", sep="\t", index=False)
    write_readme(destination, summary)

    print("Split readiness v0.1")
    print(summary.to_string(index=False))
    print(f"Outputs: {destination}")


if __name__ == "__main__":
    main()
