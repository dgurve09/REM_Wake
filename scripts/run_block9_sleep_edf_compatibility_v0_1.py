"""Run the frozen Block 9 Sleep-EDF compatibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
from pathlib import Path

import edfio
import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT = "2026-10-04_block9_sleep_edf_compatibility_v0.1"
PROTOCOL_COMMIT = "a3c81fe"
DATASET_VERSION = "1.0.0"
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0"
PILOT_PSG = [
    "sleep-cassette/SC4001E0-PSG.edf",
    "sleep-telemetry/ST7011J0-PSG.edf",
]
PILOT_HYPNOGRAM = [
    "sleep-cassette/SC4001EC-Hypnogram.edf",
    "sleep-telemetry/ST7011JP-Hypnogram.edf",
]
REQUIRED_DOWNLOADS = [
    "RECORDS",
    "SC-subjects.xls",
    "ST-subjects.xls",
    *PILOT_PSG,
    *PILOT_HYPNOGRAM,
]
EXPECTED_EEG = {"EEG Fpz-Cz", "EEG Pz-Oz"}
ALLOWED_LABELS = {
    "Sleep stage W",
    "Sleep stage R",
    "Sleep stage 1",
    "Sleep stage 2",
    "Sleep stage 3",
    "Sleep stage 4",
    "Sleep stage M",
    "Sleep stage ?",
}


# Section 2: paths and immutable writers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return repo_root().parent / "REM_W_data" / "sleep_edfx_v1.0.0"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(), text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_or_create_text(path: Path, value: str) -> None:
    expected = value.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


# Section 3: official manifest and participant grouping

def checksum_annex() -> dict[str, str]:
    rows = {}
    for line in (data_root() / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if match:
            rows[match.group(2)] = match.group(1)
    return rows


def file_inventory(annex: dict[str, str]) -> pd.DataFrame:
    rows = []
    for relative in REQUIRED_DOWNLOADS:
        path = data_root() / relative
        observed = sha256(path)
        rows.append({
            "relative_path": relative,
            "source_url": f"{BASE_URL}/{relative}",
            "bytes": path.stat().st_size,
            "expected_sha256": annex.get(relative, ""),
            "observed_sha256": observed,
            "hash_match": observed == annex.get(relative),
            "retained_in_git": False,
        })
    return pd.DataFrame(rows)


def record_identity(path: str) -> dict:
    filename = Path(path).name
    if filename.startswith("SC"):
        match = re.fullmatch(r"SC4(\d{2})(\d)[A-Z]0-PSG\.edf", filename)
        study = "sleep-cassette"
    else:
        match = re.fullmatch(r"ST7(\d{2})(\d)J0-PSG\.edf", filename)
        study = "sleep-telemetry"
    if match is None:
        raise ValueError(f"Unrecognized PSG filename: {filename}")
    return {
        "study": study,
        "participant_id": f"{study}_{match.group(1)}",
        "subject_number": int(match.group(1)),
        "night": int(match.group(2)),
        "pair_key": f"{study}/{filename[:7]}",
    }


def manifest_pairing(annex: dict[str, str]) -> pd.DataFrame:
    records = [value.strip() for value in (data_root() / "RECORDS").read_text().splitlines() if value.strip()]
    hypnograms = [path for path in annex if path.endswith("-Hypnogram.edf")]
    rows = []
    for psg_path in sorted(records):
        identity = record_identity(psg_path)
        matches = [path for path in hypnograms if f"{Path(path).parent.as_posix()}/{Path(path).name[:7]}" == identity["pair_key"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one hypnogram for {psg_path}; found {len(matches)}")
        rows.append({
            **identity,
            "psg_path": psg_path,
            "hypnogram_path": matches[0],
            "psg_sha256": annex.get(psg_path, ""),
            "hypnogram_sha256": annex.get(matches[0], ""),
            "complete_checksum_pair": bool(annex.get(psg_path) and annex.get(matches[0])),
        })
    return pd.DataFrame(rows)


def grouping_summary(pairing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for study, group in pairing.groupby("study", sort=True):
        nights = group.groupby("participant_id").size()
        rows.append({
            "study": study,
            "recordings": len(group),
            "participants": group.participant_id.nunique(),
            "participants_one_night": int((nights == 1).sum()),
            "participants_two_nights": int((nights == 2).sum()),
            "minimum_nights": int(nights.min()),
            "maximum_nights": int(nights.max()),
        })
    total_nights = pairing.groupby("participant_id").size()
    rows.append({
        "study": "all",
        "recordings": len(pairing),
        "participants": pairing.participant_id.nunique(),
        "participants_one_night": int((total_nights == 1).sum()),
        "participants_two_nights": int((total_nights == 2).sum()),
        "minimum_nights": int(total_nights.min()),
        "maximum_nights": int(total_nights.max()),
    })
    return pd.DataFrame(rows)


def spreadsheet_schema() -> pd.DataFrame:
    rows = []
    for filename in ["SC-subjects.xls", "ST-subjects.xls"]:
        workbook = pd.ExcelFile(data_root() / filename)
        for sheet in workbook.sheet_names:
            frame = pd.read_excel(data_root() / filename, sheet_name=sheet)
            rows.append({
                "file": filename,
                "sheet": sheet,
                "rows": len(frame),
                "columns": len(frame.columns),
                "column_names": ";".join(str(value) for value in frame.columns),
            })
    return pd.DataFrame(rows)


# Section 4: pilot EDF and annotation inspection

def pilot_channel_table() -> pd.DataFrame:
    rows = []
    for relative in PILOT_PSG:
        edf = edfio.read_edf(data_root() / relative, lazy_load_data=True)
        identity = record_identity(relative)
        for signal in edf.signals:
            sample_count = int(signal.samples_per_data_record * edf.num_data_records)
            rows.append({
                **identity,
                "psg_path": relative,
                "recording_duration_sec": float(edf.duration),
                "data_record_duration_sec": float(edf.data_record_duration),
                "num_data_records": int(edf.num_data_records),
                "channel": signal.label,
                "sampling_frequency_hz": float(signal.sampling_frequency),
                "physical_dimension": signal.physical_dimension,
                "sample_count": sample_count,
                "signal_duration_sec": sample_count / float(signal.sampling_frequency),
            })
    return pd.DataFrame(rows)


def annotation_tables(channels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detail_rows, transition_rows = [], []
    psg_duration = channels.groupby("psg_path").recording_duration_sec.first().to_dict()
    for relative in PILOT_HYPNOGRAM:
        study = Path(relative).parent.name
        psg_path = next(path for path in PILOT_PSG if Path(path).parent.name == study)
        edf = edfio.read_edf(data_root() / relative, lazy_load_data=True)
        annotations = sorted(edf.annotations, key=lambda item: (item.onset, item.duration, item.text))
        previous_end = None
        for index, item in enumerate(annotations):
            onset = float(item.onset)
            duration = float(item.duration)
            end = onset + duration
            detail_rows.append({
                "study": study,
                "hypnogram_path": relative,
                "annotation_index": index,
                "onset_sec": onset,
                "duration_sec": duration,
                "end_sec": end,
                "label": item.text,
                "duration_multiple_30": bool(np.isclose(duration / 30.0, round(duration / 30.0))),
                "nonoverlap_with_previous": previous_end is None or onset >= previous_end - 1e-9,
                "within_psg_support": onset >= 0 and end <= psg_duration[psg_path] + 1e-9,
            })
            previous_end = end if previous_end is None else max(previous_end, end)
        for left, right in zip(annotations[:-1], annotations[1:]):
            if left.text == "Sleep stage R" and right.text == "Sleep stage W":
                transition_rows.append({
                    "study": study,
                    "hypnogram_path": relative,
                    "transition_type": "REM_to_Wake",
                    "boundary_sec": float(right.onset),
                    "left_label": left.text,
                    "right_label": right.text,
                    "contiguous": bool(np.isclose(left.onset + left.duration, right.onset)),
                    "boundary_on_30_sec_grid": bool(np.isclose(right.onset / 30.0, round(right.onset / 30.0))),
                })
    detail = pd.DataFrame(detail_rows)
    summary = (
        detail.groupby(["study", "hypnogram_path", "label"], as_index=False)
        .agg(
            annotations=("annotation_index", "size"),
            total_duration_sec=("duration_sec", "sum"),
            minimum_duration_sec=("duration_sec", "min"),
            maximum_duration_sec=("duration_sec", "max"),
            all_duration_multiple_30=("duration_multiple_30", "all"),
            all_nonoverlap=("nonoverlap_with_previous", "all"),
            all_within_psg_support=("within_psg_support", "all"),
        )
    )
    return detail, summary, pd.DataFrame(transition_rows)


# Section 5: fixed compatibility decision

def compatibility_matrix(pairing: pd.DataFrame, inventory: pd.DataFrame,
                         channels: pd.DataFrame, annotations: pd.DataFrame,
                         transitions: pd.DataFrame) -> pd.DataFrame:
    eeg = channels[channels.channel.str.startswith("EEG ")]
    eeg_sets = eeg.groupby("study").channel.apply(set)
    labels = set(annotations.label)
    checks = [
        ("dataset_identity", inventory.hash_match.all(), "All locally inspected files match the official SHA-256 annex", "required"),
        ("manifest_pairing", len(pairing) == 197 and pairing.complete_checksum_pair.all(), "197 PSG paths pair one-to-one with 197 checksum-listed hypnograms", "required"),
        ("participant_grouping", pairing.participant_id.nunique() == 100, "Filename rules yield 100 participant groups across two studies", "required"),
        ("wake_and_rem_labels", {"Sleep stage W", "Sleep stage R"}.issubset(labels), "Both pilot hypnograms contain explicit W and R labels", "required"),
        ("known_label_vocabulary", labels.issubset(ALLOWED_LABELS), "Pilot labels are within the official R&K vocabulary", "required"),
        ("annotation_timing", annotations.duration_multiple_30.all() and annotations.nonoverlap_with_previous.all() and annotations.within_psg_support.all(), "Durations use the 30-second grid and do not overlap, but one excluded '?' interval extends beyond PSG support", "required"),
        ("rem_to_wake_derivation", len(transitions) > 0 and transitions.contiguous.all() and transitions.boundary_on_30_sec_grid.all(), "Contiguous R-to-W events can be derived on the 30-second grid", "required"),
        ("sampling_and_units", set(eeg.sampling_frequency_hz) == {100.0} and set(eeg.physical_dimension.str.lower()) <= {"uv", "µv"}, "Pilot EEG is 100 Hz and expressed in microvolts", "adaptable"),
        ("fixed_eeg_availability", all(value == EXPECTED_EEG for value in eeg_sets), "Both studies provide Fpz-Cz and Pz-Oz in the pilots", "required"),
        ("boas_feature_semantics", False, "Fpz-Cz and Pz-Oz are anterior-posterior bipolar derivations, not the ordered left/right F3-M1 and F4-M1 BOAS pair", "required"),
        ("wearable_confirmation", False, "Sleep-EDF has no simultaneous BOAS HB_1/HB_2 headband recording", "scope"),
    ]
    return pd.DataFrame([
        {"criterion": name, "status": "pass" if passed else "fail", "evidence": evidence,
         "role": role, "blocks_direct_generalization": role == "required" and not passed}
        for name, passed, evidence, role in checks
    ])


def decision_table(matrix: pd.DataFrame) -> pd.DataFrame:
    direct_pass = not matrix.blocks_direct_generalization.any()
    labels_pass = matrix[matrix.criterion.isin([
        "manifest_pairing", "participant_grouping", "wake_and_rem_labels",
        "known_label_vocabulary", "annotation_timing", "rem_to_wake_derivation",
    ])].status.eq("pass").all()
    if direct_pass:
        outcome = "direct_generalization_pass"
    elif labels_pass:
        outcome = "limited_compatibility_direct_generalization_no_go"
    else:
        outcome = "complete_no_go"
    return pd.DataFrame([{
        "dataset": "Sleep-EDF Expanded",
        "version": DATASET_VERSION,
        "outcome": outcome,
        "direct_external_model_evaluation_authorized": direct_pass,
        "full_dataset_download_authorized": direct_pass,
        "label_timing_study_authorized": labels_pass,
        "wearable_confirmation_possible": False,
        "blocking_reason": "channel_derivation_changes_frozen_feature_meaning" if labels_pass and not direct_pass else "annotation_support_and_channel_derivation_incompatibility" if not direct_pass else "",
        "next_step": "begin_Block_10_interval_aware_temporal_localization" if labels_pass and not direct_pass else "write_external_evaluation_protocol" if direct_pass else "close_external_path",
    }])


# Section 6: complete audit

def run(result_code_commit: str) -> None:
    git_commit = current_git_commit()
    if not git_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")
    annex = checksum_annex()
    inventory = file_inventory(annex)
    pairing = manifest_pairing(annex)
    grouping = grouping_summary(pairing)
    spreadsheets = spreadsheet_schema()
    channels = pilot_channel_table()
    annotation_detail, annotation_summary, transitions = annotation_tables(channels)
    matrix = compatibility_matrix(pairing, inventory, channels, annotation_detail, transitions)
    decision = decision_table(matrix)

    if not inventory.hash_match.all():
        raise ValueError("Official file hash verification failed")
    if decision.iloc[0].outcome == "direct_generalization_pass":
        raise ValueError("Unexpected direct pass requires manual protocol review")

    output = output_dir()
    tables = [
        (inventory, "official_file_inventory_v0.1.tsv"),
        (pairing, "manifest_pairing_v0.1.tsv"),
        (grouping, "participant_grouping_summary_v0.1.tsv"),
        (spreadsheets, "subject_spreadsheet_schema_v0.1.tsv"),
        (channels, "pilot_channel_timing_v0.1.tsv"),
        (annotation_summary, "pilot_annotation_summary_v0.1.tsv"),
        (transitions, "pilot_rem_to_wake_events_v0.1.tsv"),
        (matrix, "compatibility_matrix_v0.1.tsv"),
        (decision, "compatibility_decision_v0.1.tsv"),
    ]
    for frame, filename in tables:
        verify_or_create_tsv(frame.reset_index(drop=True), output / filename)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "edfio": edfio.__version__,
        "git_commit": git_commit,
        "protocol_commit": PROTOCOL_COMMIT,
        "dataset_version": DATASET_VERSION,
    }
    verify_or_create_text(output / "software_versions_v0.1.json",
                          json.dumps(versions, indent=2, sort_keys=True) + "\n")

    outcome = decision.iloc[0].outcome
    readme = f"""# Block 9 Sleep-EDF Compatibility Audit v0.1

Sleep-EDF Expanded version 1.0.0 passed identity, participant-grouping, REM/Wake-label, 30-second-grid, nonoverlap, and pilot signal-readability checks. The manifest contains {len(pairing)} PSG/hypnogram pairs from {pairing.participant_id.nunique()} participant groups. One excluded `Sleep stage ?` interval begins at the PSG end and extends beyond signal support, so the protocol's complete annotation-support criterion failed.

The direct external model gate did not pass. Sleep-EDF provides bipolar `Fpz-Cz` and `Pz-Oz`, whereas the frozen BOAS reduced-PSG model uses ordered left/right `F3-M1` and `F4-M1` inputs. Substituting the Sleep-EDF derivations would change both electrode location and feature meaning. It would be technically executable but scientifically uninterpretable as direct model generalization.

**Decision:** `{outcome}`.

No model was fitted, no full-dataset download was authorized, and no BOAS validation or current-test artifact was accessed. The pilot demonstrates that R-to-W boundaries can be derived, but the complete Block 9 gate does not authorize further Sleep-EDF analysis. It is not a wearable confirmation cohort.

Official dataset: https://physionet.org/content/sleep-edfx/1.0.0/

Dataset DOI: https://doi.org/10.13026/C2X676

Original paper: https://doi.org/10.1109/10.867928
"""
    verify_or_create_text(output / "README.md", readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    run(args.result_code_commit)


if __name__ == "__main__":
    main()
