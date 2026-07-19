"""Build deterministic primary and quality-sensitivity membership tables.

The script uses existing quality and split artifacts only. It does not open raw
signals, preprocess data, sample training negatives, or train a model.
"""

from pathlib import Path

import pandas as pd


MEMBERSHIP_VERSION = "v0.1"
QUALITY_VERSION = "v0.3"
SPLIT_VERSION = "v0.1"

MEMBERSHIP_MAP = {
    "include": ("primary_clean", True, True),
    "include_mad_sensitivity": ("primary_mad_flagged", True, True),
    "review_targeted": ("quality_sensitivity_only", False, True),
    "exclude_critical": ("excluded_critical", False, False),
}

PARTITION_ORDER = ["train", "validation", "test"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "labels" / "quality_analysis_membership_v0.1"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def load_assignments(root: Path) -> pd.DataFrame:
    path = root / "splits" / "grouped_pid_split_v0.1" / "pid_split_assignments_v0.1.tsv"
    assignments = read_tsv(path)[["pid", "partition"]].copy()
    if len(assignments) != 100 or assignments["pid"].nunique() != 100:
        raise ValueError("Expected one split assignment for each of 100 pid values")
    if set(assignments["partition"]) != set(PARTITION_ORDER):
        raise ValueError("Unexpected partition names")
    return assignments


def add_membership(frame: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    frame = frame.merge(assignments, on="pid", how="left", validate="many_to_one")
    if frame["partition"].isna().any():
        raise ValueError("At least one quality row has no frozen split assignment")

    decisions = set(frame["preprocessing_decision"])
    if decisions != set(MEMBERSHIP_MAP):
        raise ValueError(f"Unexpected quality decisions: {sorted(decisions)}")

    frame.insert(0, "membership_version", MEMBERSHIP_VERSION)
    frame["membership_tier"] = frame["preprocessing_decision"].map(
        lambda value: MEMBERSHIP_MAP[value][0]
    )
    frame["primary_analysis_eligible"] = frame["preprocessing_decision"].map(
        lambda value: MEMBERSHIP_MAP[value][1]
    )
    frame["expanded_quality_analysis_eligible"] = frame[
        "preprocessing_decision"
    ].map(lambda value: MEMBERSHIP_MAP[value][2])
    return frame


def artifact_name(frame: pd.DataFrame, transition: bool) -> pd.Series:
    if not transition:
        return pd.Series("background_review", index=frame.index)
    return frame["is_primary_label"].map(
        lambda value: (
            "primary_rem_to_wake"
            if str(value).lower() == "true"
            else "secondary_wake_to_rem"
        )
    )


def membership_summary(
    transitions: pd.DataFrame, backgrounds: pd.DataFrame
) -> pd.DataFrame:
    transition_rows = transitions.copy()
    transition_rows["artifact"] = artifact_name(transition_rows, transition=True)
    background_rows = backgrounds.copy()
    background_rows["artifact"] = artifact_name(background_rows, transition=False)
    combined = pd.concat([transition_rows, background_rows], ignore_index=True)

    rows = []
    group_columns = [
        "artifact",
        "partition",
        "preprocessing_decision",
        "membership_tier",
    ]
    for keys, group in combined.groupby(group_columns, observed=True):
        artifact, partition, decision, tier = keys
        rows.append(
            {
                "membership_version": MEMBERSHIP_VERSION,
                "quality_version": QUALITY_VERSION,
                "split_version": SPLIT_VERSION,
                "artifact": artifact,
                "partition": partition,
                "preprocessing_decision": decision,
                "membership_tier": tier,
                "rows": len(group),
                "unique_subjects": group["subject"].nunique(),
                "unique_pid": group["pid"].nunique(),
                "primary_analysis_rows": int(group["primary_analysis_eligible"].sum()),
                "expanded_quality_analysis_rows": int(
                    group["expanded_quality_analysis_eligible"].sum()
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["partition"] = pd.Categorical(
        summary["partition"], categories=PARTITION_ORDER, ordered=True
    )
    return summary.sort_values(
        ["artifact", "partition", "preprocessing_decision"]
    ).reset_index(drop=True)


def analysis_balance(frame: pd.DataFrame, artifact: str) -> pd.DataFrame:
    rows = []
    sets = {
        "primary_analysis": frame["primary_analysis_eligible"],
        "expanded_quality_analysis": frame["expanded_quality_analysis_eligible"],
        "targeted_review_stratum": frame["membership_tier"]
        == "quality_sensitivity_only",
        "critical_exclusion_stratum": frame["membership_tier"]
        == "excluded_critical",
    }
    for partition in PARTITION_ORDER:
        partition_rows = frame[frame["partition"] == partition]
        for analysis_set, mask in sets.items():
            selected = partition_rows[mask.loc[partition_rows.index]]
            rows.append(
                {
                    "membership_version": MEMBERSHIP_VERSION,
                    "artifact": artifact,
                    "partition": partition,
                    "analysis_set": analysis_set,
                    "rows": len(selected),
                    "unique_subjects": selected["subject"].nunique(),
                    "unique_pid": selected["pid"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def primary_pid_coverage(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (partition, pid), group in primary.groupby(["partition", "pid"]):
        primary_events = int(group["primary_analysis_eligible"].sum())
        expanded_events = int(group["expanded_quality_analysis_eligible"].sum())
        rows.append(
            {
                "membership_version": MEMBERSHIP_VERSION,
                "partition": partition,
                "pid": int(pid),
                "subjects": ";".join(sorted(group["subject"].unique())),
                "primary_analysis_events": primary_events,
                "expanded_quality_analysis_events": expanded_events,
                "targeted_review_events": int(
                    (group["membership_tier"] == "quality_sensitivity_only").sum()
                ),
                "critical_exclusion_events": int(
                    (group["membership_tier"] == "excluded_critical").sum()
                ),
                "primary_positive_pid": primary_events > 0,
                "expanded_positive_pid": expanded_events > 0,
                "targeted_only_positive_pid": (
                    primary_events == 0 and expanded_events > 0
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage["partition"] = pd.Categorical(
        coverage["partition"], categories=PARTITION_ORDER, ordered=True
    )
    return coverage.sort_values(["partition", "pid"]).reset_index(drop=True)


def validate(
    transitions: pd.DataFrame,
    backgrounds: pd.DataFrame,
    primary_balance: pd.DataFrame,
    pid_coverage: pd.DataFrame,
) -> None:
    if len(transitions) != 476 or transitions["transition_id"].nunique() != 476:
        raise ValueError("Transition membership does not preserve 476 unique rows")
    if len(backgrounds) != 4302 or backgrounds["background_review_id"].nunique() != 4302:
        raise ValueError("Background membership does not preserve 4,302 unique rows")

    primary = transitions[transitions["is_primary_label"].astype(str).str.lower() == "true"]
    expected_counts = {
        "include": 76,
        "include_mad_sensitivity": 200,
        "review_targeted": 72,
        "exclude_critical": 17,
    }
    if primary["preprocessing_decision"].value_counts().to_dict() != expected_counts:
        raise ValueError("Primary REM-to-Wake v0.3 decision counts changed")

    expected_primary = {
        "train": (180, 47),
        "validation": (37, 10),
        "test": (59, 15),
    }
    expected_expanded = {
        "train": (227, 56),
        "validation": (51, 14),
        "test": (70, 18),
    }
    for partition in PARTITION_ORDER:
        for analysis_set, expected in [
            ("primary_analysis", expected_primary[partition]),
            ("expanded_quality_analysis", expected_expanded[partition]),
        ]:
            row = primary_balance[
                (primary_balance["partition"] == partition)
                & (primary_balance["analysis_set"] == analysis_set)
            ].iloc[0]
            actual = (int(row["rows"]), int(row["unique_pid"]))
            if actual != expected:
                raise ValueError(
                    f"Unexpected {partition} {analysis_set} balance: {actual}"
                )

    if transitions.groupby("pid")["partition"].nunique().max() != 1:
        raise ValueError("Participant leakage detected in transition membership")
    if backgrounds.groupby("pid")["partition"].nunique().max() != 1:
        raise ValueError("Participant leakage detected in background membership")
    if int(pid_coverage["targeted_only_positive_pid"].sum()) != 16:
        raise ValueError("Unexpected targeted-only primary participant count")


def write_readme(destination: Path, primary_balance: pd.DataFrame) -> None:
    def result(partition: str, analysis_set: str, column: str) -> int:
        row = primary_balance[
            (primary_balance["partition"] == partition)
            & (primary_balance["analysis_set"] == analysis_set)
        ]
        return int(row.iloc[0][column])

    text = f"""# Quality Analysis Membership v0.1

**Created:** 2026-07-18
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/labels/quality_analysis_membership_spec_v0.1.md`
**Input quality artifact:** `signal_quality_flags_v0.3`
**Frozen split:** `grouped_pid_split_v0.1`
**Model training performed:** No
**Raw signal data read:** No

## Result

Targeted-review windows are excluded from the primary analysis but retained in an expanded quality-sensitivity analysis. Critical windows are excluded from both. The nonspecific 10-MAD-only tier remains primary eligible and separately identifiable.

| Partition | Primary REM-to-Wake events | Primary positive `pid` | Expanded events | Expanded positive `pid` |
|---|---:|---:|---:|---:|
| Train | {result('train', 'primary_analysis', 'rows')} | {result('train', 'primary_analysis', 'unique_pid')} | {result('train', 'expanded_quality_analysis', 'rows')} | {result('train', 'expanded_quality_analysis', 'unique_pid')} |
| Validation | {result('validation', 'primary_analysis', 'rows')} | {result('validation', 'primary_analysis', 'unique_pid')} | {result('validation', 'expanded_quality_analysis', 'rows')} | {result('validation', 'expanded_quality_analysis', 'unique_pid')} |
| Test | {result('test', 'primary_analysis', 'rows')} | {result('test', 'primary_analysis', 'unique_pid')} | {result('test', 'expanded_quality_analysis', 'rows')} | {result('test', 'expanded_quality_analysis', 'unique_pid')} |
| Total | {sum(result(p, 'primary_analysis', 'rows') for p in PARTITION_ORDER)} | {sum(result(p, 'primary_analysis', 'unique_pid') for p in PARTITION_ORDER)} | {sum(result(p, 'expanded_quality_analysis', 'rows') for p in PARTITION_ORDER)} | {sum(result(p, 'expanded_quality_analysis', 'unique_pid') for p in PARTITION_ORDER)} |

The primary set contains 276 events across 72 participant groups. The expanded quality-sensitivity set contains 348 events across 88 participant groups. The 72 targeted-review primary events therefore account for 16 participant groups with no clean or 10-MAD-only primary event.

The smaller primary validation set, 37 events across 10 positive participant groups, must be reported as a precision limitation. The expanded analysis tests whether conclusions depend on the conservative targeted-review exclusion.

## Outputs

| File | Purpose |
|---|---|
| `transition_analysis_membership_v0.1.tsv` | Row-level membership for all transition labels |
| `background_analysis_membership_v0.1.tsv` | Row-level membership for the background review pool |
| `membership_summary_v0.1.tsv` | Counts by artifact, split, quality decision, and tier |
| `primary_event_balance_v0.1.tsv` | Primary REM-to-Wake counts for each analysis set |
| `primary_pid_coverage_v0.1.tsv` | Per-participant primary, expanded, and targeted-only event coverage |
| `background_balance_v0.1.tsv` | Background review-pool counts for each analysis set |

## Decision

Use the primary analysis membership for the first baseline. Run the expanded quality-sensitivity membership as a prespecified robustness comparison. Do not use model performance to alter these tiers or the frozen participant split.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    quality = root / "labels" / "signal_quality_flags_v0.3"
    assignments = load_assignments(root)
    transitions = add_membership(
        read_tsv(quality / "transition_window_quality_flags_v0.3.tsv"), assignments
    )
    backgrounds = add_membership(
        read_tsv(quality / "background_window_quality_flags_v0.3.tsv"), assignments
    )

    summary = membership_summary(transitions, backgrounds)
    primary = transitions[
        transitions["is_primary_label"].astype(str).str.lower() == "true"
    ].copy()
    primary_balance = analysis_balance(primary, "primary_rem_to_wake")
    pid_coverage = primary_pid_coverage(primary)
    background_balance = analysis_balance(backgrounds, "background_review")
    validate(transitions, backgrounds, primary_balance, pid_coverage)

    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    transition_columns = [
        "membership_version",
        "transition_id",
        "subject",
        "participant_id",
        "pid",
        "transition_type",
        "is_primary_label",
        "partition",
        "preprocessing_decision",
        "membership_tier",
        "primary_analysis_eligible",
        "expanded_quality_analysis_eligible",
    ]
    background_columns = [
        "membership_version",
        "background_review_id",
        "subject",
        "participant_id",
        "pid",
        "background_tier",
        "partition",
        "preprocessing_decision",
        "membership_tier",
        "primary_analysis_eligible",
        "expanded_quality_analysis_eligible",
    ]
    transitions[transition_columns].to_csv(
        destination / "transition_analysis_membership_v0.1.tsv", sep="\t", index=False
    )
    backgrounds[background_columns].to_csv(
        destination / "background_analysis_membership_v0.1.tsv", sep="\t", index=False
    )
    summary.to_csv(destination / "membership_summary_v0.1.tsv", sep="\t", index=False)
    primary_balance.to_csv(
        destination / "primary_event_balance_v0.1.tsv", sep="\t", index=False
    )
    pid_coverage.to_csv(
        destination / "primary_pid_coverage_v0.1.tsv", sep="\t", index=False
    )
    background_balance.to_csv(
        destination / "background_balance_v0.1.tsv", sep="\t", index=False
    )
    write_readme(destination, primary_balance)

    print(primary_balance.to_string(index=False))
    print(f"Wrote quality analysis membership to {destination}")


if __name__ == "__main__":
    main()
