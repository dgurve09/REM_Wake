"""Add explicit full-window signal coverage to quality decisions v0.3."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


QUALITY_VERSION = "v0.3"
PREVIOUS_VERSION = "v0.2"
EXPECTED_WINDOW_SAMPLES = 61_440


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "labels" / "signal_quality_flags_v0.3"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def merge_flag(existing: object, new_flag: str) -> str:
    flags = [] if str(existing) in {"pass", "", "nan"} else str(existing).split(";")
    if new_flag != "pass":
        flags.append(new_flag)
    return ";".join(sorted(set(flags))) if flags else "pass"


def add_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["previous_quality_version"] = PREVIOUS_VERSION
    frame["previous_preprocessing_decision"] = frame["preprocessing_decision"]
    frame["actual_window_samples"] = (
        frame["window_stop_sample"].astype("int64")
        - frame["window_start_sample"].astype("int64")
    )
    frame["expected_window_samples"] = EXPECTED_WINDOW_SAMPLES
    frame["full_window_coverage"] = (
        frame["actual_window_samples"] == EXPECTED_WINDOW_SAMPLES
    )
    frame["coverage_critical_flag"] = frame["full_window_coverage"].map(
        {True: "pass", False: "coverage:incomplete_240s_signal_coverage"}
    )
    frame["combined_critical_flags"] = frame.apply(
        lambda row: merge_flag(
            row["combined_critical_flags"], row["coverage_critical_flag"]
        ),
        axis=1,
    )
    frame["preprocessing_decision"] = frame.apply(
        lambda row: (
            "exclude_critical"
            if row["combined_critical_flags"] != "pass"
            else row["previous_preprocessing_decision"]
        ),
        axis=1,
    )
    frame["new_critical_exclusion_v0.3"] = (
        (frame["previous_preprocessing_decision"] != "exclude_critical")
        & (frame["preprocessing_decision"] == "exclude_critical")
    )
    frame["quality_version"] = QUALITY_VERSION
    return frame


def artifact_summary(transitions: pd.DataFrame, backgrounds: pd.DataFrame) -> pd.DataFrame:
    transition_groups = transitions.copy()
    transition_groups["artifact"] = transition_groups["is_primary_label"].map(
        lambda value: (
            "primary_rem_to_wake"
            if str(value).lower() == "true"
            else "secondary_wake_to_rem"
        )
    )
    background_groups = backgrounds.copy()
    background_groups["artifact"] = "background_review"
    rows = []
    for frame in [transition_groups, background_groups]:
        for (artifact, decision), group in frame.groupby(
            ["artifact", "preprocessing_decision"]
        ):
            rows.append(
                {
                    "quality_version": QUALITY_VERSION,
                    "artifact": artifact,
                    "preprocessing_decision": decision,
                    "rows": int(len(group)),
                    "unique_subjects": int(group["subject"].nunique()),
                    "unique_pid": int(group["pid"].nunique()),
                    "new_critical_exclusions_v0.3": int(
                        group["new_critical_exclusion_v0.3"].sum()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["artifact", "preprocessing_decision"])


def recording_summary(
    transitions: pd.DataFrame, backgrounds: pd.DataFrame
) -> pd.DataFrame:
    subjects = sorted(
        set(transitions["subject"]) | set(backgrounds["subject"]),
        key=lambda value: int(value.replace("sub-", "")),
    )
    rows = []
    decisions = [
        "include",
        "include_mad_sensitivity",
        "review_targeted",
        "exclude_critical",
    ]
    for subject in subjects:
        transition_rows = transitions[transitions["subject"] == subject]
        background_rows = backgrounds[backgrounds["subject"] == subject]
        source = transition_rows if len(transition_rows) else background_rows
        row = {
            "quality_version": QUALITY_VERSION,
            "subject": subject,
            "participant_id": source["participant_id"].iloc[0],
            "pid": int(source["pid"].iloc[0]),
        }
        for decision in decisions:
            row[f"transition_{decision}"] = int(
                (transition_rows["preprocessing_decision"] == decision).sum()
            )
            row[f"background_{decision}"] = int(
                (background_rows["preprocessing_decision"] == decision).sum()
            )
        row["incomplete_transition_windows"] = int(
            (~transition_rows["full_window_coverage"]).sum()
        )
        row["new_critical_exclusions_v0.3"] = int(
            transition_rows["new_critical_exclusion_v0.3"].sum()
            + background_rows["new_critical_exclusion_v0.3"].sum()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def count(summary: pd.DataFrame, artifact: str, decision: str) -> int:
    selected = summary[
        (summary["artifact"] == artifact)
        & (summary["preprocessing_decision"] == decision)
    ]
    return int(selected["rows"].sum())


def write_readme(
    destination: Path,
    transitions: pd.DataFrame,
    backgrounds: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    primary = transitions[
        transitions["is_primary_label"].astype(str).str.lower() == "true"
    ]
    retained_primary = primary[primary["preprocessing_decision"] != "exclude_critical"]
    incomplete = transitions[~transitions["full_window_coverage"]]
    text = f"""# Signal Quality Flags v0.3

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/signal_quality_flag_spec_v0.3.md`
**Model training performed:** No

## 1. Trigger

Minimal preprocessing v0.1 found two retained train windows with incomplete 240-second signal coverage. The earlier rule verified equal PSG/headband lengths but did not verify the required absolute length.

## 2. Result

| Artifact | Include | Include with 10-MAD sensitivity flag | Targeted review | Critical exclusion |
|---|---:|---:|---:|---:|
| Primary REM-to-Wake | {count(summary, 'primary_rem_to_wake', 'include')} | {count(summary, 'primary_rem_to_wake', 'include_mad_sensitivity')} | {count(summary, 'primary_rem_to_wake', 'review_targeted')} | {count(summary, 'primary_rem_to_wake', 'exclude_critical')} |
| Secondary Wake-to-REM | {count(summary, 'secondary_wake_to_rem', 'include')} | {count(summary, 'secondary_wake_to_rem', 'include_mad_sensitivity')} | {count(summary, 'secondary_wake_to_rem', 'review_targeted')} | {count(summary, 'secondary_wake_to_rem', 'exclude_critical')} |
| Background review | {count(summary, 'background_review', 'include')} | {count(summary, 'background_review', 'include_mad_sensitivity')} | {count(summary, 'background_review', 'review_targeted')} | {count(summary, 'background_review', 'exclude_critical')} |

- Transition windows with incomplete 240-second coverage: {len(incomplete)}
- Newly excluded windows relative to v0.2: {int(transitions['new_critical_exclusion_v0.3'].sum() + backgrounds['new_critical_exclusion_v0.3'].sum())}
- Primary REM-to-Wake windows retained: {len(retained_primary)}
- `pid` values retaining at least one primary REM-to-Wake window: {retained_primary['pid'].nunique()}

## 3. Failure and Resolution

The previous check was insufficient because equal device-window lengths can still be equally short. Version 0.3 requires exactly 61,440 input samples for every 240-second window. Incomplete windows are excluded rather than padded or redefined.

Six incomplete transition windows were already critically excluded by amplitude/flatline evidence. Two additional train transitions are newly excluded. The frozen participant assignment is unchanged because inspected train participants must not move into validation or test.

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_window_quality_flags_v0.3.tsv` | Transition decisions with explicit coverage evidence |
| `background_window_quality_flags_v0.3.tsv` | Background decisions with explicit coverage evidence |
| `recording_signal_quality_summary_v0.3.tsv` | Recording-level counts and coverage failures |
| `quality_flag_summary_v0.3.tsv` | Counts by artifact and current decision |

## 5. Decision

Use v0.3 for preprocessing and later model input construction. Model training remains blocked pending successful preprocessing rerun and the final label/preprocessing gate.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    previous = root / "labels" / "signal_quality_flags_v0.2"
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    transitions = add_coverage(
        read_tsv(previous / "transition_window_quality_flags_v0.2.tsv")
    )
    backgrounds = add_coverage(
        read_tsv(previous / "background_window_quality_flags_v0.2.tsv")
    )
    if len(transitions) != 476 or len(backgrounds) != 4302:
        raise ValueError("Quality row counts changed during v0.3 integration")
    if int((~transitions["full_window_coverage"]).sum()) != 8:
        raise ValueError("Expected eight incomplete transition windows")
    if int((~backgrounds["full_window_coverage"]).sum()) != 0:
        raise ValueError("Unexpected incomplete background window")

    summary = artifact_summary(transitions, backgrounds)
    recordings = recording_summary(transitions, backgrounds)
    transitions.to_csv(
        destination / "transition_window_quality_flags_v0.3.tsv", sep="\t", index=False
    )
    backgrounds.to_csv(
        destination / "background_window_quality_flags_v0.3.tsv", sep="\t", index=False
    )
    recordings.to_csv(
        destination / "recording_signal_quality_summary_v0.3.tsv", sep="\t", index=False
    )
    summary.to_csv(destination / "quality_flag_summary_v0.3.tsv", sep="\t", index=False)
    write_readme(destination, transitions, backgrounds, summary)

    print(summary.to_string(index=False))
    print(f"Wrote coverage-aware quality artifact to {destination}")


if __name__ == "__main__":
    main()
