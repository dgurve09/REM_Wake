"""Build preprocessing signal-quality flags v0.1.

This script combines label structure checks, sample-alignment checks, and
signal-alignment proxy summaries. It creates review flags only; it does not
train or evaluate a model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LABEL_VERSION = "v0.1"
WINDOW_SEC = 240.0
SAMPLE_RATE_HZ = 256.0
EXPECTED_WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE_HZ)
REM_WAKE_EXCLUSION_SEC = 135.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return repo_root() / "labels" / "signal_quality_flags_v0.1"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def bool_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    return str(value).strip()


def int_value(value: object) -> int:
    if pd.isna(value) or value == "":
        return 0
    return int(float(value))


def float_value(value: object) -> float:
    if pd.isna(value) or value == "":
        return 0.0
    return float(value)


def flag_text(flags: list[str]) -> str:
    return "pass" if not flags else ";".join(flags)


def proxy_status(row: pd.Series, prefix: str) -> str:
    total = int_value(row[f"{prefix}_windows"])
    usable_column = f"{prefix}_windows_usable"
    if usable_column not in row.index:
        usable_column = f"{prefix}_usable"
    usable = int_value(row[usable_column])
    near_column = f"{prefix}_usable_near_zero_lag_2s"
    if near_column not in row.index:
        near_column = f"{prefix}_near_zero_lag_2s"
    near = int_value(row[near_column])
    if total == 0:
        return "not_available"
    if usable == 0:
        return "unusable"
    if usable == near:
        return "all_usable_near_zero_lag"
    if near > 0:
        return "mixed_lag"
    return "usable_lag_review"


def eeg_window_proxy_status(row: pd.Series) -> str:
    availability = str(row.get("availability", "")).strip()
    usable = bool_text(row.get("usable_for_alignment", "False")) == "True"
    if availability != "available":
        return "not_available"
    if not usable:
        return "unusable"
    lag = float_value(row.get("best_lag_sec", 0.0))
    if abs(lag) <= 2.0:
        return "near_zero_lag"
    return "lag_review"


def build_recording_flags(
    recording_inventory: pd.DataFrame,
    label_quality: pd.DataFrame,
    subject_alignment: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    merged = recording_inventory.merge(
        label_quality,
        on=["subject", "participant_id", "pid"],
        suffixes=("", "_label"),
    ).merge(subject_alignment, on=["subject", "pid"])

    for _, row in merged.iterrows():
        critical = []
        context = []

        if row["timeline_alignment_flag"] != "pass":
            critical.append("timeline_alignment_review")
        if int_value(row["transition_windows"]) != int_value(row["transition_windows_pass"]):
            critical.append("transition_sample_alignment_review")
        if int_value(row["stage_hum_missing_epochs"]) != 0:
            critical.append("missing_stage_hum")
        if int_value(row["non_30_sec_epoch_count"]) != 0:
            critical.append("non_30_sec_epoch")
        if int_value(row["onset_gap_issue_count"]) != 0:
            critical.append("onset_gap_issue")
        if float_value(row["sampling_mismatch_hz"]) != 0.0:
            critical.append("sampling_mismatch")
        if float_value(row["duration_mismatch_sec"]) != 0.0:
            critical.append("duration_mismatch")
        if float_value(row["unlabeled_tail_sec"]) >= 30.0:
            critical.append("unlabeled_tail_at_least_one_epoch")

        if int_value(row["stage_hum_disconnection_epochs"]) > 0:
            context.append("recording_contains_psg_disconnection")
        if int_value(row["candidate_windows_with_disconnection"]) > 0:
            critical.append("candidate_window_disconnection")

        pulse_status = proxy_status(row, "pulse")
        eeg_status = proxy_status(row, "eeg_envelope")
        if pulse_status != "all_usable_near_zero_lag":
            context.append(f"pulse_proxy_{pulse_status}")
        if eeg_status != "all_usable_near_zero_lag":
            context.append(f"eeg_envelope_proxy_{eeg_status}")

        decision = "include_for_preprocessing" if not critical else "review_before_modeling"
        rows.append(
            {
                "label_version": LABEL_VERSION,
                "subject": row["subject"],
                "participant_id": row["participant_id"],
                "pid": int(row["pid"]),
                "psg_event_rows": int_value(row["psg_event_rows"]),
                "timeline_alignment_flag": row["timeline_alignment_flag"],
                "transition_windows": int_value(row["transition_windows"]),
                "transition_windows_pass": int_value(row["transition_windows_pass"]),
                "stage_hum_disconnection_epochs": int_value(row["stage_hum_disconnection_epochs"]),
                "unlabeled_tail_sec": float_value(row["unlabeled_tail_sec"]),
                "pulse_proxy_status": pulse_status,
                "eeg_envelope_proxy_status": eeg_status,
                "critical_quality_flags": flag_text(critical),
                "context_quality_flags": flag_text(context),
                "recording_decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values("subject")


def build_transition_window_flags(
    labels: pd.DataFrame,
    eeg_proxy: pd.DataFrame,
    recording_flags: pd.DataFrame,
) -> pd.DataFrame:
    eeg_proxy = eeg_proxy[eeg_proxy["transition_id"].notna()].copy()
    eeg_proxy["transition_id"] = pd.to_numeric(eeg_proxy["transition_id"], errors="coerce")
    eeg_proxy = eeg_proxy.dropna(subset=["transition_id"])
    eeg_proxy["transition_id"] = eeg_proxy["transition_id"].astype("int64")

    merged = labels.merge(
        eeg_proxy[
            [
                "transition_id",
                "availability",
                "best_lag_sec",
                "best_abs_corr",
                "usable_for_alignment",
            ]
        ],
        on="transition_id",
        how="left",
    ).merge(
        recording_flags[["subject", "critical_quality_flags", "recording_decision"]],
        on="subject",
        how="left",
        suffixes=("", "_recording"),
    )

    rows = []
    for _, row in merged.iterrows():
        critical = []
        if row["label_decision"] != "include":
            critical.append("label_review")
        if row["timeline_alignment_flag"] != "pass":
            critical.append("timeline_alignment_review")
        if row["sample_alignment_flag"] != "pass":
            critical.append("sample_alignment_review")
        if int_value(row["window_sample_count_difference"]) != 0:
            critical.append("window_sample_count_mismatch")
        if int_value(row["boundary_sample_difference"]) != 0:
            critical.append("boundary_sample_mismatch")
        if bool_text(row["has_psg_disconnection_in_window"]) == "True":
            critical.append("psg_disconnection_in_window")
        if row["recording_decision"] != "include_for_preprocessing":
            critical.append("recording_quality_review")

        eeg_status = eeg_window_proxy_status(row)
        decision = "include_for_preprocessing" if not critical else "review_before_modeling"
        rows.append(
            {
                "label_version": LABEL_VERSION,
                "window_source": "transition_label",
                "transition_id": int(row["transition_id"]),
                "subject": row["subject"],
                "participant_id": row["participant_id"],
                "pid": int(row["pid"]),
                "transition_type": row["transition_type"],
                "is_primary_label": bool_text(row["is_primary_label"]),
                "nominal_boundary_sec": float_value(row["nominal_boundary_sec"]),
                "window_start_sample": int_value(row["headband_start_sample"]),
                "window_stop_sample": int_value(row["headband_stop_sample"]),
                "eeg_envelope_proxy_status": eeg_status,
                "eeg_envelope_best_lag_sec": row.get("best_lag_sec", ""),
                "eeg_envelope_best_abs_corr": row.get("best_abs_corr", ""),
                "critical_quality_flags": flag_text(critical),
                "window_decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "nominal_boundary_sec"])


def build_background_window_flags(
    background: pd.DataFrame,
    recording_flags: pd.DataFrame,
) -> pd.DataFrame:
    merged = background.merge(
        recording_flags[["subject", "critical_quality_flags", "recording_decision"]],
        on="subject",
        how="left",
        suffixes=("", "_recording"),
    )

    rows = []
    for _, row in merged.iterrows():
        critical = []
        sample_count = int_value(row["headband_stop_sample"]) - int_value(row["headband_start_sample"])
        psg_sample_count = int_value(row["psg_stop_sample"]) - int_value(row["psg_start_sample"])

        if row["background_decision"] != "candidate":
            critical.append("background_not_candidate")
        if sample_count != EXPECTED_WINDOW_SAMPLES:
            critical.append("headband_window_sample_count_mismatch")
        if psg_sample_count != EXPECTED_WINDOW_SAMPLES:
            critical.append("psg_window_sample_count_mismatch")
        if sample_count != psg_sample_count:
            critical.append("psg_headband_window_sample_mismatch")
        if float_value(row["min_distance_to_remwake_boundary_sec"]) < REM_WAKE_EXCLUSION_SEC:
            critical.append("remwake_uncertainty_overlap")
        if row["recording_decision"] != "include_for_preprocessing":
            critical.append("recording_quality_review")

        decision = "include_for_preprocessing" if not critical else "review_before_modeling"
        rows.append(
            {
                "label_version": LABEL_VERSION,
                "window_source": "background_review_candidate",
                "background_review_id": int(row["background_review_id"]),
                "background_pool_id": int(row["background_pool_id"]),
                "subject": row["subject"],
                "participant_id": row["participant_id"],
                "pid": int(row["pid"]),
                "background_tier": row["background_tier"],
                "center_pair": row["center_pair"],
                "center_sec": float_value(row["center_sec"]),
                "window_start_sample": int_value(row["headband_start_sample"]),
                "window_stop_sample": int_value(row["headband_stop_sample"]),
                "min_distance_to_remwake_boundary_sec": float_value(
                    row["min_distance_to_remwake_boundary_sec"]
                ),
                "critical_quality_flags": flag_text(critical),
                "window_decision": decision,
            }
        )
    return pd.DataFrame(rows).sort_values(["subject", "center_sec"])


def build_summary(
    recording_flags: pd.DataFrame,
    transition_flags: pd.DataFrame,
    background_flags: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for name, frame, decision_col in [
        ("recordings", recording_flags, "recording_decision"),
        ("transition_windows", transition_flags, "window_decision"),
        ("background_review_windows", background_flags, "window_decision"),
    ]:
        rows.append(
            {
                "artifact": name,
                "rows": len(frame),
                "include_for_preprocessing": int(
                    (frame[decision_col] == "include_for_preprocessing").sum()
                ),
                "review_before_modeling": int(
                    (frame[decision_col] == "review_before_modeling").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def write_readme(destination: Path, summary: pd.DataFrame) -> None:
    rows = {row["artifact"]: row for _, row in summary.iterrows()}
    text = f"""# Signal Quality Flags v0.1

**Created:** 2026-07-09
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Transition labels:** `labels/transition_labels_v0.1/`
**Background windows:** `labels/background_windows_v0.1/`
**Model training performed:** No

## 1. Purpose

This artifact creates recording-level and window-level quality flags for the label/preprocessing gate.

The uncertainty addressed here is whether the transition and background windows can be carried forward with explicit critical flags and signal-proxy notes before any train/validation/test split or model work.

## 2. Method

- Critical flags use PSG label structure, PSG/headband sample mapping, and window geometry.
- Pulse and EEG-envelope lag summaries are retained as signal-proxy notes.
- Proxy notes do not automatically exclude windows because they are not ground-truth synchronization markers.
- Background review windows inherit recording-level critical flags and are checked for 240-second sample geometry and REM/Wake uncertainty separation.

## 3. Result

| Artifact | Rows | Include for preprocessing | Review before modeling |
|---|---:|---:|---:|
| Recordings | {rows['recordings']['rows']} | {rows['recordings']['include_for_preprocessing']} | {rows['recordings']['review_before_modeling']} |
| Transition windows | {rows['transition_windows']['rows']} | {rows['transition_windows']['include_for_preprocessing']} | {rows['transition_windows']['review_before_modeling']} |
| Background review windows | {rows['background_review_windows']['rows']} | {rows['background_review_windows']['include_for_preprocessing']} | {rows['background_review_windows']['review_before_modeling']} |

## 4. Outputs

| File | Purpose |
|---|---|
| `recording_signal_quality_flags_v0.1.tsv` | Recording-level critical flags and signal-proxy status |
| `transition_window_quality_flags_v0.1.tsv` | Window-level flags for REM/Wake transition labels |
| `background_window_quality_flags_v0.1.tsv` | Window-level flags for deterministic background review candidates |
| `quality_flag_summary_v0.1.tsv` | Count summary by artifact |

## 5. Limitations

- These flags do not yet compute full amplitude artifact metrics from every headband channel.
- EEG-envelope and pulse lag statuses are review proxies, not exclusion rules by themselves.
- Final split creation remains blocked until the background sampling policy and these flags are reviewed together.

## 6. Decision

Use this artifact for the label/preprocessing gate. Model training remains blocked.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    recording_inventory = read_tsv(
        root
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "recording_transition_inventory.tsv"
    )
    label_quality = read_tsv(
        root
        / "experiments"
        / "2026-06-29_to_2026-07-05_boas_e0_transition_inventory"
        / "label_quality_summary.tsv"
    )
    subject_alignment = read_tsv(
        root / "experiments" / "2026-07-04_boas_full_signal_alignment" / "subject_alignment_summary.tsv"
    )
    labels = read_tsv(root / "labels" / "transition_labels_v0.1" / "transition_labels_v0.1.tsv")
    eeg_proxy = read_tsv(
        root / "experiments" / "2026-07-04_boas_full_signal_alignment" / "eeg_envelope_alignment.tsv"
    )
    background = read_tsv(
        root / "labels" / "background_windows_v0.1" / "background_review_windows_v0.1.tsv"
    )

    recording_flags = build_recording_flags(recording_inventory, label_quality, subject_alignment)
    transition_flags = build_transition_window_flags(labels, eeg_proxy, recording_flags)
    background_flags = build_background_window_flags(background, recording_flags)
    summary = build_summary(recording_flags, transition_flags, background_flags)

    recording_flags.to_csv(
        destination / "recording_signal_quality_flags_v0.1.tsv", sep="\t", index=False
    )
    transition_flags.to_csv(
        destination / "transition_window_quality_flags_v0.1.tsv", sep="\t", index=False
    )
    background_flags.to_csv(
        destination / "background_window_quality_flags_v0.1.tsv", sep="\t", index=False
    )
    summary.to_csv(destination / "quality_flag_summary_v0.1.tsv", sep="\t", index=False)
    write_readme(destination, summary)

    print("Signal quality flags v0.1")
    print(summary.to_string(index=False))
    print(f"Outputs: {destination}")


if __name__ == "__main__":
    main()
