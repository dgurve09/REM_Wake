"""Audit fixed PSG and wearable channel sets before Block 7 feature extraction."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mne
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: fixed audit configuration

DATASET = "ds005555"
SNAPSHOT = "1.1.1"
EXPERIMENT_DIR = "2026-08-25_block7_channel_compatibility_v0.1"
PROTOCOL_COMMIT = "1f6797f"
RESULT_CODE_COMMIT = "d73a2784395a"
EXPECTED_RECORDINGS = 128
EXPECTED_REFERENCE = "M1 (left mastoid)"
EXPECTED_SFREQ = 256.0
PSG6 = ["PSG_F3", "PSG_F4", "PSG_C3", "PSG_C4", "PSG_O1", "PSG_O2"]
PSG2 = ["PSG_F3", "PSG_F4"]
HB2 = ["HB_1", "HB_2"]


# Section 2: repository and data paths

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    parent = Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))
    return parent / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def subject_number(subject: str) -> int:
    return int(subject.replace("sub-", ""))


def subject_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
    )
    rows = []
    for item in split.itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append(
                {"subject": subject, "pid": int(item.pid), "partition": item.partition}
            )
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_RECORDINGS or result["subject"].nunique() != len(result):
        raise ValueError("Frozen split does not contain 128 unique recordings")
    return result.sort_values("subject", key=lambda values: values.map(subject_number))


def acquisition_paths(subject: str, acquisition: str) -> dict[str, Path]:
    prefix = data_root() / subject / "eeg" / f"{subject}_task-Sleep_acq-{acquisition}"
    return {
        "channels": Path(f"{prefix}_channels.tsv"),
        "json": Path(f"{prefix}_eeg.json"),
        "edf": Path(f"{prefix}_eeg.edf"),
    }


# Section 3: one-acquisition inspection

def load_channels(path: Path) -> pd.DataFrame:
    required_columns = ["name", "type", "units", "sampling_frequency"]
    table = pd.read_csv(path, sep="\t", dtype=str)
    if table.columns.tolist() != required_columns:
        raise ValueError(f"Unexpected channel columns in {path.name}: {table.columns.tolist()}")
    if table["name"].duplicated().any():
        raise ValueError(f"Duplicate channel name in {path.name}")
    table["sampling_frequency"] = table["sampling_frequency"].astype(float)
    return table


def inspect_acquisition(subject: str, acquisition: str) -> dict:
    paths = acquisition_paths(subject, acquisition)
    missing = [kind for kind, path in paths.items() if not path.exists()]
    if missing:
        return {
            "missing_files": ";".join(missing),
            "sidecar": pd.DataFrame(),
            "header_sidecar_match": False,
            "reference": "",
            "json_sfreq_hz": float("nan"),
            "json_duration_sec": float("nan"),
            "header_sfreq_hz": float("nan"),
            "header_duration_sec": float("nan"),
            "channel_signature": "",
        }

    sidecar = load_channels(paths["channels"])
    metadata = json.loads(paths["json"].read_text(encoding="utf-8"))
    raw = mne.io.read_raw_edf(paths["edf"], preload=False, verbose="ERROR")
    header_duration = raw.n_times / float(raw.info["sfreq"])
    return {
        "missing_files": "none",
        "sidecar": sidecar,
        "header_sidecar_match": raw.ch_names == sidecar["name"].tolist(),
        "reference": str(metadata.get("EEGReference", "")),
        "json_sfreq_hz": float(metadata.get("SamplingFrequency", float("nan"))),
        "json_duration_sec": float(metadata.get("RecordingDuration", float("nan"))),
        "header_sfreq_hz": float(raw.info["sfreq"]),
        "header_duration_sec": header_duration,
        "channel_signature": ",".join(sidecar["name"].tolist()),
    }


def required_channels_pass(sidecar: pd.DataFrame, required: list[str]) -> bool:
    if sidecar.empty:
        return False
    selected = sidecar[sidecar["name"].isin(required)].copy()
    return bool(
        len(selected) == len(required)
        and set(selected["name"]) == set(required)
        and selected["units"].eq("uV").all()
        and selected["sampling_frequency"].eq(EXPECTED_SFREQ).all()
    )


# Section 4: cohort audit

def audit_recordings() -> tuple[pd.DataFrame, pd.DataFrame]:
    eligibility_rows = []
    availability_rows = []
    for assignment in subject_assignments().itertuples(index=False):
        inspected = {
            acquisition: inspect_acquisition(assignment.subject, acquisition)
            for acquisition in ["psg", "headband"]
        }
        psg = inspected["psg"]
        headband = inspected["headband"]

        for acquisition, values in inspected.items():
            for channel in values["sidecar"].itertuples(index=False):
                availability_rows.append(
                    {
                        "subject": assignment.subject,
                        "acquisition": acquisition,
                        "channel": channel.name,
                        "type": channel.type,
                        "units": channel.units,
                        "sampling_frequency_hz": float(channel.sampling_frequency),
                    }
                )

        psg6_pass = required_channels_pass(psg["sidecar"], PSG6)
        psg2_pass = required_channels_pass(psg["sidecar"], PSG2)
        hb2_pass = required_channels_pass(headband["sidecar"], HB2)
        reference_match = (
            psg["reference"] == headband["reference"] == EXPECTED_REFERENCE
        )
        sampling_match = all(
            value == EXPECTED_SFREQ
            for value in [
                psg["json_sfreq_hz"],
                headband["json_sfreq_hz"],
                psg["header_sfreq_hz"],
                headband["header_sfreq_hz"],
            ]
        )
        duration_match = (
            abs(psg["json_duration_sec"] - headband["json_duration_sec"]) <= 1e-9
            and abs(psg["header_duration_sec"] - headband["header_duration_sec"])
            <= 1.0 / EXPECTED_SFREQ
            and abs(psg["json_duration_sec"] - psg["header_duration_sec"])
            <= 1.0 / EXPECTED_SFREQ
            and abs(headband["json_duration_sec"] - headband["header_duration_sec"])
            <= 1.0 / EXPECTED_SFREQ
        )
        required_files = psg["missing_files"] == headband["missing_files"] == "none"
        header_match = psg["header_sidecar_match"] and headband["header_sidecar_match"]
        pair_pass = all(
            [
                required_files,
                header_match,
                psg6_pass,
                psg2_pass,
                hb2_pass,
                reference_match,
                sampling_match,
                duration_match,
            ]
        )
        failed_checks = [
            name
            for name, passed in [
                ("required_files", required_files),
                ("header_sidecar", header_match),
                ("psg6", psg6_pass),
                ("psg2", psg2_pass),
                ("hb2", hb2_pass),
                ("reference", reference_match),
                ("sampling", sampling_match),
                ("duration", duration_match),
            ]
            if not passed
        ]
        eligibility_rows.append(
            {
                "subject": assignment.subject,
                "pid": int(assignment.pid),
                "partition": assignment.partition,
                "psg_channel_count": len(psg["sidecar"]),
                "headband_channel_count": len(headband["sidecar"]),
                "psg_header_sidecar_match": psg["header_sidecar_match"],
                "headband_header_sidecar_match": headband["header_sidecar_match"],
                "psg6_eligible": psg6_pass,
                "psg2_eligible": psg2_pass,
                "hb2_eligible": hb2_pass,
                "reference_match": reference_match,
                "sampling_match": sampling_match,
                "duration_match": duration_match,
                "pair_pass": pair_pass,
                "failed_checks": ";".join(failed_checks) if failed_checks else "none",
                "psg_channel_signature": psg["channel_signature"],
                "headband_channel_signature": headband["channel_signature"],
            }
        )
        print(f"{assignment.subject}: {'pass' if pair_pass else 'fail'}")

    eligibility = pd.DataFrame(eligibility_rows)
    availability = pd.DataFrame(availability_rows)
    return eligibility, availability


# Section 5: reviewed summaries

def channel_availability_summary(availability: pd.DataFrame) -> pd.DataFrame:
    summary = (
        availability.groupby(
            ["acquisition", "channel", "type", "units", "sampling_frequency_hz"],
            as_index=False,
        )["subject"]
        .nunique()
        .rename(columns={"subject": "recordings_present"})
    )
    summary["recording_proportion"] = summary["recordings_present"] / EXPECTED_RECORDINGS
    return summary.sort_values(["acquisition", "channel"]).reset_index(drop=True)


def configuration_summary(eligibility: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for acquisition in ["psg", "headband"]:
        column = f"{acquisition}_channel_signature"
        frame = (
            eligibility.groupby(column, as_index=False)
            .size()
            .rename(columns={column: "channel_signature", "size": "recordings"})
        )
        frame.insert(0, "acquisition", acquisition)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["acquisition", "recordings", "channel_signature"],
        ascending=[True, False, True],
    )


def verify_or_create_text(path: Path, expected: str) -> None:
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(
                f"Reviewed output differs from recomputation; create a new version: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


# Section 6: execution and decision

def main() -> None:
    if not data_root().exists():
        raise FileNotFoundError(f"BOAS data root not found: {data_root()}")
    eligibility, availability_detail = audit_recordings()
    availability = channel_availability_summary(availability_detail)
    configurations = configuration_summary(eligibility)

    passed = int(eligibility["pair_pass"].sum())
    psg6 = int(eligibility["psg6_eligible"].sum())
    psg2 = int(eligibility["psg2_eligible"].sum())
    hb2 = int(eligibility["hb2_eligible"].sum())
    psg_configurations = int(
        configurations[configurations["acquisition"] == "psg"]["channel_signature"].nunique()
    )
    headband_configurations = int(
        configurations[configurations["acquisition"] == "headband"]["channel_signature"].nunique()
    )

    if passed == EXPECTED_RECORDINGS:
        decision = "pass"
        interpretation = (
            "The common PSG-6, reduced PSG-2, and wearable HB-2 channel sets cover "
            "the complete frozen cohort. Optional sensor heterogeneity does not require exclusions."
        )
    elif min(psg6, psg2, hb2) > 0:
        decision = "revise"
        interpretation = (
            "At least one required compatibility check failed. Revise the channel protocol "
            "before feature extraction without consulting model performance."
        )
    else:
        decision = "no-go"
        interpretation = (
            "No proposed fixed PSG/wearable channel set covers a scientifically interpretable cohort."
        )

    out = output_dir()
    verify_or_create_tsv(
        eligibility,
        out / "recording_channel_eligibility_v0.1.tsv",
    )
    verify_or_create_tsv(
        availability,
        out / "channel_availability_v0.1.tsv",
    )
    verify_or_create_tsv(
        configurations,
        out / "channel_configuration_summary_v0.1.tsv",
    )
    readme = f"""# Block 7 Channel Compatibility Audit v0.1

**Work initiated:** 2026-08-25
**Audit executed:** 2026-08-26
**Protocol commit:** `{PROTOCOL_COMMIT}`
**Code commit:** `{RESULT_CODE_COMMIT}`
**Dataset:** BOAS OpenNeuro `{DATASET}`, snapshot `{SNAPSHOT}`
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Frozen recording pairs | {len(eligibility)} |
| Complete pair-level checks passed | {passed} |
| `PSG-6` eligible recordings | {psg6} |
| `PSG-2` eligible recordings | {psg2} |
| `HB-2` eligible recordings | {hb2} |
| Distinct PSG channel configurations | {psg_configurations} |
| Distinct headband channel configurations | {headband_configurations} |
| Gate decision | **{decision}** |

{interpretation}

The full common PSG comparator is six-channel EEG, not every clinical PSG sensor. Optional EOG, respiratory, pulse, oxygen-saturation, and wearable motion channels vary across recordings and remain excluded from the fixed Block 7 input sets. The reduced `F3/F4` to `HB_1/HB_2` mapping preserves laterality and feature dimension, but the electrode locations are not equivalent; subsequent transfer results combine device and location shift.

Raw EDF files remain outside Git. The retained tables contain public BOAS identifiers and technical channel metadata only.
"""
    verify_or_create_text(out / "README.md", readme)

    print(f"Passed pairs: {passed}/{len(eligibility)}")
    print(f"PSG configurations: {psg_configurations}")
    print(f"Headband configurations: {headband_configurations}")
    print(f"Decision: {decision}")
    if decision != "pass":
        raise SystemExit("Block 7 channel compatibility gate did not pass")


if __name__ == "__main__":
    main()
