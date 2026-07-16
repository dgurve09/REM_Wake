"""Integrate structural and headband amplitude signal-quality evidence.

The script creates versioned preprocessing decisions without deleting failed
windows, assigning participant splits, preprocessing EEG, or training a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


QUALITY_VERSION = "v0.2"
STRUCTURAL_VERSION = "v0.1"
AMPLITUDE_VERSION = "v0.1"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "labels" / "signal_quality_flags_v0.2"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def prefixed_flags(value: object, prefix: str) -> str:
    text = str(value).strip()
    if not text or text == "pass" or text == "nan":
        return "pass"
    return ";".join(f"{prefix}{flag}" for flag in text.split(";") if flag)


def merge_flags(*values: str) -> str:
    flags = sorted(
        {
            flag
            for value in values
            if value and value != "pass"
            for flag in value.split(";")
            if flag
        }
    )
    return ";".join(flags) if flags else "pass"


def combined_decision(row: pd.Series) -> str:
    if row["combined_critical_flags"] != "pass":
        return "exclude_critical"
    if row["amplitude_review_priority"] == "targeted_review":
        return "review_targeted"
    if row["amplitude_review_priority"] == "mad_only_review":
        return "include_mad_sensitivity"
    return "include"


def load_amplitude(root: Path) -> pd.DataFrame:
    path = (
        root
        / "experiments"
        / "2026-07-11_boas_headband_window_quality"
        / "headband_window_signal_decisions_v0.1.tsv"
    )
    amplitude = read_tsv(path)
    if amplitude["window_id"].duplicated().any():
        raise ValueError("Amplitude artifact contains duplicate window_id values")
    return amplitude


def add_combined_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["structural_critical_flags"] = frame["critical_quality_flags"].map(
        lambda value: prefixed_flags(value, "structural:")
    )
    frame["amplitude_critical_flags"] = frame["critical_flags"].map(
        lambda value: prefixed_flags(value, "amplitude:")
    )
    frame["amplitude_review_flags"] = frame["review_flags"].map(
        lambda value: prefixed_flags(value, "amplitude:")
    )
    frame["combined_critical_flags"] = frame.apply(
        lambda row: merge_flags(
            row["structural_critical_flags"], row["amplitude_critical_flags"]
        ),
        axis=1,
    )
    frame["combined_review_flags"] = frame["amplitude_review_flags"]
    frame["preprocessing_decision"] = frame.apply(combined_decision, axis=1)
    return frame


def build_transition_flags(root: Path, amplitude: pd.DataFrame) -> pd.DataFrame:
    structural = read_tsv(
        root
        / "labels"
        / "signal_quality_flags_v0.1"
        / "transition_window_quality_flags_v0.1.tsv"
    )
    amp = amplitude[amplitude["window_source"] == "transition"].copy()
    amp["transition_id"] = amp["window_id"].str.removeprefix("T").astype(int)

    merged = structural.merge(
        amp[
            [
                "transition_id",
                "subject",
                "pid",
                "window_signal_decision",
                "review_priority",
                "critical_flags",
                "review_flags",
            ]
        ],
        on="transition_id",
        how="inner",
        suffixes=("", "_amplitude"),
        validate="one_to_one",
    )
    if len(merged) != 476:
        raise ValueError(f"Expected 476 transition rows after integration, found {len(merged)}")
    if not (merged["subject"] == merged["subject_amplitude"]).all():
        raise ValueError("Transition subject mismatch between quality artifacts")
    if not (merged["pid"] == merged["pid_amplitude"]).all():
        raise ValueError("Transition pid mismatch between quality artifacts")

    merged = merged.rename(
        columns={
            "window_signal_decision": "amplitude_window_decision",
            "review_priority": "amplitude_review_priority",
        }
    )
    merged = add_combined_columns(merged)
    merged.insert(0, "quality_version", QUALITY_VERSION)
    merged["structural_quality_version"] = STRUCTURAL_VERSION
    merged["amplitude_quality_version"] = AMPLITUDE_VERSION

    columns = [
        "quality_version",
        "structural_quality_version",
        "amplitude_quality_version",
        "transition_id",
        "subject",
        "participant_id",
        "pid",
        "transition_type",
        "is_primary_label",
        "nominal_boundary_sec",
        "window_start_sample",
        "window_stop_sample",
        "eeg_envelope_proxy_status",
        "amplitude_window_decision",
        "amplitude_review_priority",
        "structural_critical_flags",
        "amplitude_critical_flags",
        "amplitude_review_flags",
        "combined_critical_flags",
        "combined_review_flags",
        "preprocessing_decision",
    ]
    return merged[columns].sort_values(["subject", "nominal_boundary_sec"])


def build_background_flags(root: Path, amplitude: pd.DataFrame) -> pd.DataFrame:
    structural = read_tsv(
        root
        / "labels"
        / "signal_quality_flags_v0.1"
        / "background_window_quality_flags_v0.1.tsv"
    )
    amp = amplitude[amplitude["window_source"] == "background_review"].copy()
    amp["background_review_id"] = amp["window_id"].str.removeprefix("B").astype(int)

    merged = structural.merge(
        amp[
            [
                "background_review_id",
                "subject",
                "pid",
                "window_signal_decision",
                "review_priority",
                "critical_flags",
                "review_flags",
            ]
        ],
        on="background_review_id",
        how="inner",
        suffixes=("", "_amplitude"),
        validate="one_to_one",
    )
    if len(merged) != 4302:
        raise ValueError(f"Expected 4,302 background rows after integration, found {len(merged)}")
    if not (merged["subject"] == merged["subject_amplitude"]).all():
        raise ValueError("Background subject mismatch between quality artifacts")
    if not (merged["pid"] == merged["pid_amplitude"]).all():
        raise ValueError("Background pid mismatch between quality artifacts")

    merged = merged.rename(
        columns={
            "window_signal_decision": "amplitude_window_decision",
            "review_priority": "amplitude_review_priority",
        }
    )
    merged = add_combined_columns(merged)
    merged.insert(0, "quality_version", QUALITY_VERSION)
    merged["structural_quality_version"] = STRUCTURAL_VERSION
    merged["amplitude_quality_version"] = AMPLITUDE_VERSION

    columns = [
        "quality_version",
        "structural_quality_version",
        "amplitude_quality_version",
        "background_review_id",
        "background_pool_id",
        "subject",
        "participant_id",
        "pid",
        "background_tier",
        "center_pair",
        "center_sec",
        "window_start_sample",
        "window_stop_sample",
        "min_distance_to_remwake_boundary_sec",
        "amplitude_window_decision",
        "amplitude_review_priority",
        "structural_critical_flags",
        "amplitude_critical_flags",
        "amplitude_review_flags",
        "combined_critical_flags",
        "combined_review_flags",
        "preprocessing_decision",
    ]
    return merged[columns].sort_values(["subject", "center_sec"])


def build_summary(
    transitions: pd.DataFrame, backgrounds: pd.DataFrame
) -> pd.DataFrame:
    transition_groups = transitions.copy()
    transition_groups["artifact"] = transition_groups["is_primary_label"].map(
        lambda value: "primary_rem_to_wake" if str(value).lower() == "true" else "secondary_wake_to_rem"
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
                }
            )
    return pd.DataFrame(rows).sort_values(["artifact", "preprocessing_decision"])


def build_recording_summary(
    recording_v01: pd.DataFrame,
    transitions: pd.DataFrame,
    backgrounds: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    decisions = [
        "include",
        "include_mad_sensitivity",
        "review_targeted",
        "exclude_critical",
    ]
    for _, recording in recording_v01.iterrows():
        subject = recording["subject"]
        transition_rows = transitions[transitions["subject"] == subject]
        background_rows = backgrounds[backgrounds["subject"] == subject]
        row = {
            "quality_version": QUALITY_VERSION,
            "subject": subject,
            "participant_id": recording["participant_id"],
            "pid": int(recording["pid"]),
            "structural_recording_decision_v0.1": recording["recording_decision"],
            "structural_context_flags_v0.1": recording["context_quality_flags"],
        }
        for decision in decisions:
            row[f"transition_{decision}"] = int(
                (transition_rows["preprocessing_decision"] == decision).sum()
            )
            row[f"background_{decision}"] = int(
                (background_rows["preprocessing_decision"] == decision).sum()
            )

        if row["transition_exclude_critical"] or row["background_exclude_critical"]:
            recording_decision = "retain_with_window_exclusions"
        elif row["transition_review_targeted"] or row["background_review_targeted"]:
            recording_decision = "retain_with_targeted_review"
        else:
            recording_decision = "retain"
        row["recording_decision_v0.2"] = recording_decision
        rows.append(row)
    return pd.DataFrame(rows).sort_values("subject")


def count(
    summary: pd.DataFrame, artifact: str, decision: str
) -> int:
    selected = summary[
        (summary["artifact"] == artifact)
        & (summary["preprocessing_decision"] == decision)
    ]
    return int(selected["rows"].sum())


def write_readme(
    destination: Path,
    summary: pd.DataFrame,
    transitions: pd.DataFrame,
    recordings: pd.DataFrame,
) -> None:
    retained_primary = transitions[
        transitions["is_primary_label"].astype(str).str.lower() == "true"
    ]
    retained_primary = retained_primary[
        retained_primary["preprocessing_decision"] != "exclude_critical"
    ]
    text = f"""# Signal Quality Flags v0.2

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/signal_quality_flag_spec_v0.2.md`
**Model training performed:** No

## 1. Purpose

This artifact integrates structural quality flags v0.1 with the predeclared full headband amplitude/continuity assessment. It provides one current preprocessing decision per reviewed transition and background window without deleting excluded or inconclusive cases.

## 2. Result

| Artifact | Include | Include with 10-MAD sensitivity flag | Targeted review | Critical exclusion |
|---|---:|---:|---:|---:|
| Primary REM-to-Wake | {count(summary, 'primary_rem_to_wake', 'include')} | {count(summary, 'primary_rem_to_wake', 'include_mad_sensitivity')} | {count(summary, 'primary_rem_to_wake', 'review_targeted')} | {count(summary, 'primary_rem_to_wake', 'exclude_critical')} |
| Secondary Wake-to-REM | {count(summary, 'secondary_wake_to_rem', 'include')} | {count(summary, 'secondary_wake_to_rem', 'include_mad_sensitivity')} | {count(summary, 'secondary_wake_to_rem', 'review_targeted')} | {count(summary, 'secondary_wake_to_rem', 'exclude_critical')} |
| Background review | {count(summary, 'background_review', 'include')} | {count(summary, 'background_review', 'include_mad_sensitivity')} | {count(summary, 'background_review', 'review_targeted')} | {count(summary, 'background_review', 'exclude_critical')} |

- Primary REM-to-Wake windows retained before targeted-review sensitivity decisions: {len(retained_primary)}
- `pid` values retaining at least one primary REM-to-Wake window: {retained_primary['pid'].nunique()}
- Recordings retained with window-level critical exclusions: {(recordings['recording_decision_v0.2'] == 'retain_with_window_exclusions').sum()}

## 3. Interpretation

The v0.1 statement that every reviewed window could be included is superseded for current preprocessing decisions. Critical amplitude failures now exclude 15 primary transition windows and 20 background review windows. The feasibility conclusion remains unchanged because 350 primary windows remain across all 88 contributing `pid` groups.

The 10-MAD-only outcome remains visible but does not trigger exclusion. Targeted amplitude, jump, and endpoint flags remain separate so later sensitivity analysis can test their effect rather than treating uncertain windows as automatically clean or unusable.

## 4. Outputs

| File | Purpose |
|---|---|
| `transition_window_quality_flags_v0.2.tsv` | Integrated transition-window decisions |
| `background_window_quality_flags_v0.2.tsv` | Integrated background-window decisions |
| `recording_signal_quality_summary_v0.2.tsv` | Recording-level counts and retained status |
| `quality_flag_summary_v0.2.tsv` | Counts by artifact and combined decision |

## 5. Decision

Use v0.2, not v0.1 alone, for grouped split design and the label/preprocessing gate. No final split is assigned in this artifact, and model training remains blocked.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    amplitude = load_amplitude(root)
    transitions = build_transition_flags(root, amplitude)
    backgrounds = build_background_flags(root, amplitude)
    recording_v01 = read_tsv(
        root
        / "labels"
        / "signal_quality_flags_v0.1"
        / "recording_signal_quality_flags_v0.1.tsv"
    )
    recordings = build_recording_summary(recording_v01, transitions, backgrounds)
    summary = build_summary(transitions, backgrounds)

    transitions.to_csv(
        destination / "transition_window_quality_flags_v0.2.tsv", sep="\t", index=False
    )
    backgrounds.to_csv(
        destination / "background_window_quality_flags_v0.2.tsv", sep="\t", index=False
    )
    recordings.to_csv(
        destination / "recording_signal_quality_summary_v0.2.tsv", sep="\t", index=False
    )
    summary.to_csv(destination / "quality_flag_summary_v0.2.tsv", sep="\t", index=False)
    write_readme(destination, summary, transitions, recordings)

    print(summary.to_string(index=False))
    print(f"Wrote integrated quality artifact to {destination}")


if __name__ == "__main__":
    main()
