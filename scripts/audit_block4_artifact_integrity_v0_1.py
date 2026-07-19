"""Audit one-to-one linkage across the frozen Block 4 artifacts.

This script reads versioned tables already in the repository. It does not read
raw signals, alter frozen artifacts, construct model inputs, or train a model.
"""

import hashlib
from pathlib import Path

import pandas as pd


AUDIT_VERSION = "v0.1"
EXPECTED_SPLIT_HASH = "52450EDA07795D198E2722D4D804E71D0E17A8A4B62BA5AF93AE811B211D83A7"
EXPERIMENT_DIR = "2026-07-18_block4_artifact_integrity_v0.1"

MEMBERSHIP_MAP = {
    "include": ("primary_clean", True, True),
    "include_mad_sensitivity": ("primary_mad_flagged", True, True),
    "review_targeted": ("quality_sensitivity_only", False, True),
    "exclude_critical": ("excluded_critical", False, False),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def sha256(path: Path) -> str:
    content = path.read_bytes()
    if path.suffix.lower() in {".md", ".tsv"}:
        text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        content = text.encode("utf-8")
    return hashlib.sha256(content).hexdigest().upper()


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"Cannot interpret boolean value: {value}")


def mismatch_count(left: pd.Series, right: pd.Series) -> int:
    return int((left.astype(str) != right.astype(str)).sum())


def add_check(
    checks: list[dict],
    name: str,
    passed: bool,
    observed: object,
    expected: object,
) -> None:
    checks.append(
        {
            "audit_version": AUDIT_VERSION,
            "check": name,
            "passed": bool(passed),
            "observed": observed,
            "expected": expected,
        }
    )


def load_subject_assignments(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assignment in assignments.itertuples(index=False):
        for subject in str(assignment.subjects).split(";"):
            rows.append(
                {
                    "subject": subject,
                    "pid": int(assignment.pid),
                    "partition": assignment.partition,
                }
            )
    return pd.DataFrame(rows)


def audit_transition_chain(
    checks: list[dict],
    labels: pd.DataFrame,
    quality: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    for name, frame in [
        ("label", labels),
        ("quality", quality),
        ("membership", membership),
    ]:
        add_check(
            checks,
            f"transition_{name}_rows",
            len(frame) == 476,
            len(frame),
            476,
        )
        add_check(
            checks,
            f"transition_{name}_unique_ids",
            frame["transition_id"].nunique() == 476,
            frame["transition_id"].nunique(),
            476,
        )

    label_quality = labels.merge(
        quality, on="transition_id", suffixes=("_label", "_quality"), validate="one_to_one"
    )
    add_check(
        checks,
        "transition_label_quality_id_set",
        len(label_quality) == 476,
        len(label_quality),
        476,
    )
    for field in [
        "subject",
        "participant_id",
        "pid",
        "transition_type",
        "is_primary_label",
    ]:
        mismatches = mismatch_count(
            label_quality[f"{field}_label"], label_quality[f"{field}_quality"]
        )
        add_check(
            checks,
            f"transition_label_quality_{field}",
            mismatches == 0,
            mismatches,
            0,
        )
    start_mismatch = int(
        (
            label_quality["headband_start_sample"].astype("int64")
            != label_quality["window_start_sample"].astype("int64")
        ).sum()
    )
    stop_mismatch = int(
        (
            label_quality["headband_stop_sample"].astype("int64")
            != label_quality["window_stop_sample"].astype("int64")
        ).sum()
    )
    add_check(checks, "transition_start_sample_linkage", start_mismatch == 0, start_mismatch, 0)
    add_check(checks, "transition_stop_sample_linkage", stop_mismatch == 0, stop_mismatch, 0)

    quality_membership = quality.merge(
        membership,
        on="transition_id",
        suffixes=("_quality", "_membership"),
        validate="one_to_one",
    )
    add_check(
        checks,
        "transition_quality_membership_id_set",
        len(quality_membership) == 476,
        len(quality_membership),
        476,
    )
    for field in [
        "subject",
        "participant_id",
        "pid",
        "transition_type",
        "is_primary_label",
        "preprocessing_decision",
    ]:
        mismatches = mismatch_count(
            quality_membership[f"{field}_quality"],
            quality_membership[f"{field}_membership"],
        )
        add_check(
            checks,
            f"transition_quality_membership_{field}",
            mismatches == 0,
            mismatches,
            0,
        )


def audit_background_chain(
    checks: list[dict],
    source: pd.DataFrame,
    quality: pd.DataFrame,
    membership: pd.DataFrame,
) -> None:
    for name, frame in [
        ("source", source),
        ("quality", quality),
        ("membership", membership),
    ]:
        add_check(
            checks,
            f"background_{name}_rows",
            len(frame) == 4302,
            len(frame),
            4302,
        )
        add_check(
            checks,
            f"background_{name}_unique_ids",
            frame["background_review_id"].nunique() == 4302,
            frame["background_review_id"].nunique(),
            4302,
        )

    source_quality = source.merge(
        quality,
        on="background_review_id",
        suffixes=("_source", "_quality"),
        validate="one_to_one",
    )
    add_check(
        checks,
        "background_source_quality_id_set",
        len(source_quality) == 4302,
        len(source_quality),
        4302,
    )
    for field in ["subject", "participant_id", "pid", "background_tier"]:
        mismatches = mismatch_count(
            source_quality[f"{field}_source"], source_quality[f"{field}_quality"]
        )
        add_check(
            checks,
            f"background_source_quality_{field}",
            mismatches == 0,
            mismatches,
            0,
        )
    start_mismatch = int(
        (
            source_quality["headband_start_sample"].astype("int64")
            != source_quality["window_start_sample"].astype("int64")
        ).sum()
    )
    stop_mismatch = int(
        (
            source_quality["headband_stop_sample"].astype("int64")
            != source_quality["window_stop_sample"].astype("int64")
        ).sum()
    )
    add_check(checks, "background_start_sample_linkage", start_mismatch == 0, start_mismatch, 0)
    add_check(checks, "background_stop_sample_linkage", stop_mismatch == 0, stop_mismatch, 0)

    quality_membership = quality.merge(
        membership,
        on="background_review_id",
        suffixes=("_quality", "_membership"),
        validate="one_to_one",
    )
    add_check(
        checks,
        "background_quality_membership_id_set",
        len(quality_membership) == 4302,
        len(quality_membership),
        4302,
    )
    for field in [
        "subject",
        "participant_id",
        "pid",
        "background_tier",
        "preprocessing_decision",
    ]:
        mismatches = mismatch_count(
            quality_membership[f"{field}_quality"],
            quality_membership[f"{field}_membership"],
        )
        add_check(
            checks,
            f"background_quality_membership_{field}",
            mismatches == 0,
            mismatches,
            0,
        )


def audit_split_and_membership(
    checks: list[dict],
    split_path: Path,
    assignments: pd.DataFrame,
    subject_assignments: pd.DataFrame,
    transition_membership: pd.DataFrame,
    background_membership: pd.DataFrame,
) -> None:
    add_check(checks, "split_pid_rows", len(assignments) == 100, len(assignments), 100)
    add_check(
        checks,
        "split_unique_pid",
        assignments["pid"].nunique() == 100,
        assignments["pid"].nunique(),
        100,
    )
    leakage = int((subject_assignments.groupby("pid")["partition"].nunique() > 1).sum())
    add_check(checks, "split_participant_leakage", leakage == 0, leakage, 0)
    add_check(
        checks,
        "split_recording_assignments",
        len(subject_assignments) == 128 and subject_assignments["subject"].nunique() == 128,
        len(subject_assignments),
        128,
    )
    split_hash = sha256(split_path)
    add_check(
        checks,
        "split_assignment_sha256",
        split_hash == EXPECTED_SPLIT_HASH,
        split_hash,
        EXPECTED_SPLIT_HASH,
    )

    partition_map = assignments.set_index("pid")["partition"]
    for name, frame in [
        ("transition", transition_membership),
        ("background", background_membership),
    ]:
        expected_partition = frame["pid"].map(partition_map)
        mismatches = mismatch_count(frame["partition"], expected_partition)
        add_check(
            checks,
            f"{name}_membership_partition",
            mismatches == 0,
            mismatches,
            0,
        )

        tier_mismatch = 0
        primary_mismatch = 0
        expanded_mismatch = 0
        for row in frame.itertuples(index=False):
            tier, primary, expanded = MEMBERSHIP_MAP[row.preprocessing_decision]
            tier_mismatch += row.membership_tier != tier
            primary_mismatch += bool_value(row.primary_analysis_eligible) != primary
            expanded_mismatch += (
                bool_value(row.expanded_quality_analysis_eligible) != expanded
            )
        add_check(
            checks,
            f"{name}_membership_tier_mapping",
            tier_mismatch == 0,
            tier_mismatch,
            0,
        )
        add_check(
            checks,
            f"{name}_primary_eligibility_mapping",
            primary_mismatch == 0,
            primary_mismatch,
            0,
        )
        add_check(
            checks,
            f"{name}_expanded_eligibility_mapping",
            expanded_mismatch == 0,
            expanded_mismatch,
            0,
        )


def expected_train_windows(
    transition_quality: pd.DataFrame,
    background_quality: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    train_pid = set(assignments.loc[assignments["partition"] == "train", "pid"])
    transitions = transition_quality[
        transition_quality["pid"].isin(train_pid)
        & (transition_quality["preprocessing_decision"] != "exclude_critical")
    ].copy()
    transitions["window_source"] = "transition"
    transitions["window_id"] = transitions["transition_id"].map(
        lambda value: f"T{int(value):04d}"
    )
    transitions["label_class"] = transitions["transition_type"]

    backgrounds = background_quality[
        background_quality["pid"].isin(train_pid)
        & (background_quality["preprocessing_decision"] != "exclude_critical")
    ].copy()
    backgrounds["window_source"] = "background_review"
    backgrounds["window_id"] = backgrounds["background_review_id"].map(
        lambda value: f"B{int(value):05d}"
    )
    backgrounds["label_class"] = backgrounds["background_tier"]

    columns = [
        "window_source",
        "window_id",
        "subject",
        "participant_id",
        "pid",
        "label_class",
        "preprocessing_decision",
    ]
    return pd.concat([transitions[columns], backgrounds[columns]], ignore_index=True)


def audit_preprocessing(
    checks: list[dict],
    expected: pd.DataFrame,
    windows: pd.DataFrame,
    recordings: pd.DataFrame,
    subject_assignments: pd.DataFrame,
) -> None:
    add_check(checks, "preprocessing_expected_train_windows", len(expected) == 3063, len(expected), 3063)
    add_check(checks, "preprocessing_actual_train_windows", len(windows) == 3063, len(windows), 3063)
    actual_unique = windows[["window_source", "window_id"]].drop_duplicates()
    add_check(
        checks,
        "preprocessing_unique_window_keys",
        len(actual_unique) == 3063,
        len(actual_unique),
        3063,
    )
    linked = expected.merge(
        windows,
        on=["window_source", "window_id"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
        validate="one_to_one",
    )
    missing_or_extra = int((linked["_merge"] != "both").sum())
    add_check(
        checks,
        "preprocessing_exact_window_set",
        missing_or_extra == 0,
        missing_or_extra,
        0,
    )
    linked = linked[linked["_merge"] == "both"]
    for expected_field, actual_field, name in [
        ("subject_expected", "subject_actual", "subject"),
        ("participant_id_expected", "participant_id_actual", "participant_id"),
        ("pid_expected", "pid_actual", "pid"),
        ("label_class_expected", "label_class_actual", "label_class"),
        (
            "preprocessing_decision",
            "quality_decision",
            "quality_decision",
        ),
    ]:
        mismatches = mismatch_count(linked[expected_field], linked[actual_field])
        add_check(
            checks,
            f"preprocessing_window_{name}",
            mismatches == 0,
            mismatches,
            0,
        )

    train_pid = set(subject_assignments.loc[subject_assignments["partition"] == "train", "pid"])
    nontrain_windows = int((~windows["pid"].isin(train_pid)).sum())
    add_check(checks, "preprocessing_nontrain_windows", nontrain_windows == 0, nontrain_windows, 0)
    critical_windows = int((windows["quality_decision"] == "exclude_critical").sum())
    add_check(checks, "preprocessing_critical_windows", critical_windows == 0, critical_windows, 0)
    input_failures = int((windows["input_samples_per_channel"] != 61440).sum())
    output_failures = int((windows["output_samples_per_channel"] != 30720).sum())
    decision_failures = int((windows["window_check_decision"] != "pass").sum())
    add_check(checks, "preprocessing_input_geometry", input_failures == 0, input_failures, 0)
    add_check(checks, "preprocessing_output_geometry", output_failures == 0, output_failures, 0)
    add_check(checks, "preprocessing_window_decisions", decision_failures == 0, decision_failures, 0)
    version_failures = int(
        (
            (windows["preprocessing_version"] != "v0.2")
            | (windows["quality_version"] != "v0.3")
            | (windows["split_version"] != "v0.1")
        ).sum()
    )
    add_check(checks, "preprocessing_window_versions", version_failures == 0, version_failures, 0)

    expected_recordings = subject_assignments[
        subject_assignments["partition"] == "train"
    ][["subject", "pid"]]
    linked_recordings = expected_recordings.merge(
        recordings,
        on="subject",
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True,
        validate="one_to_one",
    )
    recording_set_failures = int((linked_recordings["_merge"] != "both").sum())
    add_check(
        checks,
        "preprocessing_exact_recording_set",
        recording_set_failures == 0 and len(recordings) == 82,
        f"rows={len(recordings)}; set_failures={recording_set_failures}",
        "rows=82; set_failures=0",
    )
    linked_recordings = linked_recordings[linked_recordings["_merge"] == "both"]
    pid_mismatch = mismatch_count(
        linked_recordings["pid_expected"], linked_recordings["pid_actual"]
    )
    recording_failures = int((recordings["recording_check_decision"] != "pass").sum())
    add_check(checks, "preprocessing_recording_pid", pid_mismatch == 0, pid_mismatch, 0)
    add_check(
        checks,
        "preprocessing_recording_decisions",
        recording_failures == 0,
        recording_failures,
        0,
    )


def build_manifest(root: Path) -> pd.DataFrame:
    artifacts = [
        ("transition_label_spec", "docs/labels/transition_label_spec_v0.1.md"),
        ("transition_labels", "labels/transition_labels_v0.1/transition_labels_v0.1.tsv"),
        ("background_spec", "docs/labels/background_window_spec_v0.1.md"),
        ("background_review", "labels/background_windows_v0.1/background_review_windows_v0.1.tsv"),
        ("quality_spec", "docs/labels/signal_quality_flag_spec_v0.3.md"),
        ("transition_quality", "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv"),
        ("background_quality", "labels/signal_quality_flags_v0.3/background_window_quality_flags_v0.3.tsv"),
        ("split_spec", "docs/splits/grouped_pid_split_spec_v0.1.md"),
        ("split_assignments", "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv"),
        ("membership_spec", "docs/labels/quality_analysis_membership_spec_v0.1.md"),
        ("transition_membership", "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv"),
        ("background_membership", "labels/quality_analysis_membership_v0.1/background_analysis_membership_v0.1.tsv"),
        ("preprocessing_spec", "docs/preprocessing/minimal_wearable_eeg_preprocessing_spec_v0.2.md"),
        ("preprocessing_filter", "experiments/2026-07-15_minimal_preprocessing_v0.2/filter_response_v0.2.tsv"),
        ("preprocessing_synthetic", "experiments/2026-07-15_minimal_preprocessing_v0.2/synthetic_frequency_checks_v0.2.tsv"),
        ("preprocessing_recordings", "experiments/2026-07-15_minimal_preprocessing_v0.2/train_recording_preprocessing_checks_v0.2.tsv"),
        ("preprocessing_scaler", "experiments/2026-07-15_minimal_preprocessing_v0.2/train_robust_scaler_v0.2.tsv"),
        ("preprocessing_windows", "experiments/2026-07-15_minimal_preprocessing_v0.2/train_window_preprocessing_checks_v0.2.tsv"),
        ("preprocessing_summary", "experiments/2026-07-15_minimal_preprocessing_v0.2/train_window_preprocessing_summary_v0.2.tsv"),
    ]
    rows = []
    for role, relative_path in artifacts:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        row_count = "not_applicable"
        if path.suffix.lower() == ".tsv":
            with path.open("r", encoding="utf-8") as handle:
                row_count = max(sum(1 for _ in handle) - 1, 0)
        rows.append(
            {
                "audit_version": AUDIT_VERSION,
                "artifact_role": role,
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "data_rows": row_count,
                "sha256_lf_normalized": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_readme(
    destination: Path,
    checks: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    passed = int(checks["passed"].sum())
    total = len(checks)
    text = f"""# Block 4 Artifact Integrity v0.1

**Created:** 2026-07-18
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Protocol:** `docs/feasibility/block4_artifact_integrity_protocol_v0.1.md`
**Model training performed:** No
**Raw signal data read:** No

## Result

Integrity decision: **{'pass' if passed == total else 'fail'}**.

- Checks passed: {passed} of {total}
- Frozen files hashed: {len(manifest)}
- Transition rows linked label-to-quality-to-membership: 476
- Background rows linked source-to-quality-to-membership: 4,302
- Noncritical train windows linked quality-to-preprocessing: 3,063
- Train recordings linked split-to-preprocessing: 82
- Validation/test windows found in preprocessing output: 0
- LF-normalized split assignment SHA-256: `{EXPECTED_SPLIT_HASH}`

## Method

The audit compared row identities and invariant fields rather than relying on aggregate totals. It independently rebuilt the expected noncritical train-window set from quality v0.3 and the frozen split, then required exact equality with preprocessing v0.2.

## Outputs

| File | Purpose |
|---|---|
| `artifact_integrity_checks_v0.1.tsv` | Pass/fail result and observed value for each linkage check |
| `frozen_artifact_manifest_v0.1.tsv` | Relative path, size, row count, and LF-normalized SHA-256 for each frozen file |

## Decision

{'Retain the July 18 label/preprocessing gate pass. The frozen artifacts are internally traceable and no split contamination was found.' if passed == total else 'Do not retain the gate pass until every failed linkage is resolved with a versioned artifact.'}

This result establishes internal consistency only. It does not estimate detector performance or establish clinical validity.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    transition_labels = read_tsv(
        root / "labels/transition_labels_v0.1/transition_labels_v0.1.tsv"
    )
    background_source = read_tsv(
        root / "labels/background_windows_v0.1/background_review_windows_v0.1.tsv"
    )
    transition_quality = read_tsv(
        root / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv"
    )
    background_quality = read_tsv(
        root / "labels/signal_quality_flags_v0.3/background_window_quality_flags_v0.3.tsv"
    )
    transition_membership = read_tsv(
        root / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv"
    )
    background_membership = read_tsv(
        root / "labels/quality_analysis_membership_v0.1/background_analysis_membership_v0.1.tsv"
    )
    split_path = root / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv"
    assignments = read_tsv(split_path)
    subject_assignments = load_subject_assignments(assignments)
    preprocessing_windows = read_tsv(
        root / "experiments/2026-07-15_minimal_preprocessing_v0.2/train_window_preprocessing_checks_v0.2.tsv"
    )
    preprocessing_recordings = read_tsv(
        root / "experiments/2026-07-15_minimal_preprocessing_v0.2/train_recording_preprocessing_checks_v0.2.tsv"
    )

    checks = []
    audit_transition_chain(
        checks, transition_labels, transition_quality, transition_membership
    )
    audit_background_chain(
        checks, background_source, background_quality, background_membership
    )
    audit_split_and_membership(
        checks,
        split_path,
        assignments,
        subject_assignments,
        transition_membership,
        background_membership,
    )
    expected_windows = expected_train_windows(
        transition_quality, background_quality, assignments
    )
    audit_preprocessing(
        checks,
        expected_windows,
        preprocessing_windows,
        preprocessing_recordings,
        subject_assignments,
    )
    checks = pd.DataFrame(checks)
    manifest = build_manifest(root)

    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    checks.to_csv(
        destination / "artifact_integrity_checks_v0.1.tsv", sep="\t", index=False
    )
    manifest.to_csv(
        destination / "frozen_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    write_readme(destination, checks, manifest)

    print(checks.to_string(index=False))
    print(f"Wrote Block 4 artifact-integrity audit to {destination}")
    if not checks["passed"].all():
        raise RuntimeError("Block 4 artifact-integrity audit failed")


if __name__ == "__main__":
    main()
