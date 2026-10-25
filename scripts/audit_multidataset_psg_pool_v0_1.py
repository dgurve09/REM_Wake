"""Audit the local datasets for a common-channel PSG transition study."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import edfio
import numpy as np
import pandas as pd

from reviewed_output import verify_or_create_tsv


# Section 1: fixed local sources and channel mapping

COMMON_CHANNELS = {
    "BOAS": {"F4": "PSG_F4", "C4": "PSG_C4", "O2": "PSG_O2"},
    "gx36m": {"F4": "F4-A1", "C4": "C4-A1", "O2": "O2-A1"},
    "ISRUC-3": {"F4": "F4-A1", "C4": "C4-A1", "O2": "O2-A1"},
}
EXPECTED_SFREQ = {"BOAS": 256.0, "gx36m": 500.0, "ISRUC-3": 200.0}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parents[1]


def hypno_root() -> Path:
    return project_root() / "Project_hypno"


def boas_root() -> Path:
    return repo_root().parent / "REM_W_data" / "boas_ds005555_v1.1.1"


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-10-25_multidataset_psg_pool_audit_v0.1"


def relative_source(path: Path) -> str:
    return path.relative_to(project_root()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Section 2: small EDF header reader for archived gx36m files

def text_fields(raw: bytes, width: int, count: int) -> list[str]:
    return [
        raw[index * width : (index + 1) * width].decode("ascii").strip()
        for index in range(count)
    ]


def archived_edf_header(source: io.BufferedReader) -> dict:
    fixed = source.read(256)
    if len(fixed) != 256:
        raise ValueError("Incomplete EDF fixed header")
    records = int(fixed[236:244].decode("ascii").strip())
    record_duration = float(fixed[244:252].decode("ascii").strip())
    signal_count = int(fixed[252:256].decode("ascii").strip())
    variable = source.read(256 * signal_count)
    if len(variable) != 256 * signal_count:
        raise ValueError("Incomplete EDF signal header")
    labels = text_fields(variable[: 16 * signal_count], 16, signal_count)
    samples_start = 216 * signal_count
    samples = [
        int(value)
        for value in text_fields(
            variable[samples_start : samples_start + 8 * signal_count],
            8,
            signal_count,
        )
    ]
    frequencies = {
        label: sample_count / record_duration
        for label, sample_count in zip(labels, samples)
    }
    return {
        "channels": labels,
        "sampling_frequencies": frequencies,
        "duration_sec": records * record_duration,
    }


# Section 3: label consensus and event counts

def rem_to_wake_count(stages: np.ndarray) -> int:
    return int(((stages[:-1] == "REM") & (stages[1:] == "W")).sum())


def gx_consensus(paths: list[Path]) -> tuple[np.ndarray, float]:
    scorings = [np.loadtxt(path, dtype=int, ndmin=1) for path in paths]
    length = min(map(len, scorings))
    values = np.vstack([scoring[:length] for scoring in scorings])
    counts = np.stack([(values == stage).sum(axis=0) for stage in range(5)])
    consensus = counts.argmax(axis=0)
    stage_names = np.asarray(["W", "N1", "N2", "N3", "REM"])
    disagreement = float(np.mean(counts.max(axis=0) < len(scorings)))
    return stage_names[consensus], disagreement


def isruc_consensus(paths: list[Path]) -> tuple[np.ndarray, float]:
    scorings = [pd.read_csv(path).stage.astype(str).to_numpy() for path in paths]
    length = min(map(len, scorings))
    left, right = scorings[0][:length], scorings[1][:length]
    agreement = left == right
    consensus = np.where(agreement, left, "UNKNOWN")
    return consensus, float(np.mean(~agreement))


# Section 4: recording-level eligibility

def boas_recordings() -> list[dict]:
    participants = pd.read_csv(boas_root() / "participants.tsv", sep="\t")
    rows = []
    for item in participants.itertuples(index=False):
        subject = item.participant_id
        path = boas_root() / subject / "eeg" / f"{subject}_task-Sleep_acq-psg_eeg.edf"
        hypnogram = hypno_root() / "data" / "boas" / "csv_hypno" / f"{subject}.csv"
        edf = edfio.read_edf(path, lazy_load_data=True)
        channels = [signal.label for signal in edf.signals]
        frequencies = {signal.label: float(signal.sampling_frequency) for signal in edf.signals}
        stages = pd.read_csv(hypnogram).stage.astype(str).to_numpy()
        expected = COMMON_CHANNELS["BOAS"]
        channel_pass = all(channel in channels for channel in expected.values())
        sampling_pass = all(
            frequencies[channel] == EXPECTED_SFREQ["BOAS"] for channel in expected.values()
        )
        label_duration = len(stages) * 30.0
        rows.append(
            {
                "dataset": "BOAS",
                "recording": subject,
                "participant_group": f"BOAS-pid-{int(item.pid)}",
                "night": subject,
                "label_rule": "publisher_human_consensus",
                "scorers": 3,
                "epochs": len(stages),
                "scorer_disagreement_fraction": np.nan,
                "rem_to_wake_events": rem_to_wake_count(stages),
                "signal_duration_sec": float(edf.duration),
                "label_duration_sec": label_duration,
                "trailing_unscored_sec": float(edf.duration) - label_duration,
                "common_channels_present": channel_pass,
                "sampling_frequency_pass": sampling_pass,
                "label_within_signal_support": label_duration <= float(edf.duration) + 1e-6,
                "signal_source": relative_source(path),
                "label_source": relative_source(hypnogram),
            }
        )
    return rows


def gx_recordings() -> tuple[list[dict], list[Path]]:
    source_root = hypno_root() / "data" / "gx36m"
    staging_root = source_root / "staging"
    rows, archives = [], []
    for archive in sorted(source_root.glob("SLP*.zip")):
        archives.append(archive)
        subject = archive.stem
        with zipfile.ZipFile(archive) as package:
            edf_members = sorted(name for name in package.namelist() if name.lower().endswith(".edf"))
            for member in edf_members:
                key = Path(member).stem
                scoring_paths = sorted(staging_root.glob(f"{key}_Expert*.csv"))
                if len(scoring_paths) != 3:
                    raise ValueError(f"Expected three gx36m scorings for {key}")
                stages, disagreement = gx_consensus(scoring_paths)
                with package.open(member) as stream:
                    header = archived_edf_header(stream)
                expected = COMMON_CHANNELS["gx36m"]
                channel_pass = all(channel in header["channels"] for channel in expected.values())
                sampling_pass = all(
                    header["sampling_frequencies"][channel] == EXPECTED_SFREQ["gx36m"]
                    for channel in expected.values()
                )
                label_duration = len(stages) * 30.0
                rows.append(
                    {
                        "dataset": "gx36m",
                        "recording": key,
                        "participant_group": f"gx36m-{subject}",
                        "night": key.rsplit("_", 1)[-1],
                        "label_rule": "three_scorer_epoch_majority",
                        "scorers": 3,
                        "epochs": len(stages),
                        "scorer_disagreement_fraction": disagreement,
                        "rem_to_wake_events": rem_to_wake_count(stages),
                        "signal_duration_sec": float(header["duration_sec"]),
                        "label_duration_sec": label_duration,
                        "trailing_unscored_sec": float(header["duration_sec"]) - label_duration,
                        "common_channels_present": channel_pass,
                        "sampling_frequency_pass": sampling_pass,
                        "label_within_signal_support": (
                            label_duration <= float(header["duration_sec"]) + 1e-6
                        ),
                        "signal_source": f"{relative_source(archive)}::{member}",
                        "label_source": ";".join(relative_source(path) for path in scoring_paths),
                    }
                )
    return rows, archives


def isruc_recordings() -> tuple[list[dict], list[Path]]:
    source_root = hypno_root() / "data" / "isruc"
    rows, signals = [], []
    for number in range(1, 11):
        path = source_root / "ISURC_3" / "x" / f"{number}.rec"
        signals.append(path)
        scoring_paths = sorted((source_root / "csv_hypno").glob(f"ISURC_3__{number}__{number}_*.csv"))
        if len(scoring_paths) != 2:
            raise ValueError(f"Expected two ISRUC scorings for recording {number}")
        stages, disagreement = isruc_consensus(scoring_paths)
        edf = edfio.read_edf(path, lazy_load_data=True)
        channels = [signal.label for signal in edf.signals]
        frequencies = {signal.label: float(signal.sampling_frequency) for signal in edf.signals}
        expected = COMMON_CHANNELS["ISRUC-3"]
        channel_pass = all(channel in channels for channel in expected.values())
        sampling_pass = all(
            frequencies[channel] == EXPECTED_SFREQ["ISRUC-3"] for channel in expected.values()
        )
        label_duration = len(stages) * 30.0
        rows.append(
            {
                "dataset": "ISRUC-3",
                "recording": f"ISRUC-3-{number}",
                "participant_group": f"ISRUC-3-{number}",
                "night": "1",
                "label_rule": "two_scorer_unanimous_else_unknown",
                "scorers": 2,
                "epochs": len(stages),
                "scorer_disagreement_fraction": disagreement,
                "rem_to_wake_events": rem_to_wake_count(stages),
                "signal_duration_sec": float(edf.duration),
                "label_duration_sec": label_duration,
                "trailing_unscored_sec": float(edf.duration) - label_duration,
                "common_channels_present": channel_pass,
                "sampling_frequency_pass": sampling_pass,
                "label_within_signal_support": label_duration <= float(edf.duration) + 1e-6,
                "signal_source": relative_source(path),
                "label_source": ";".join(relative_source(item) for item in scoring_paths),
            }
        )
    return rows, signals


# Section 5: local corpus roles and reviewed outputs

def dataset_inventory() -> pd.DataFrame:
    metadata = pd.read_csv(hypno_root() / "metadata_all.csv")
    counts = metadata.groupby("dataset").size().to_dict()
    roles = {
        "boas": ("full_raw_elsewhere", "paired_psg_and_wearable", "signal_model_and_labels"),
        "challenge-2018": ("labels_only", "arousal_label_mat", "label_statistics"),
        "dod": ("labels_only", "five_scorer_hypnograms", "scorer_uncertainty"),
        "isruc": ("partial_raw", "ten_group3_raw_recordings", "signal_model_and_scorer_uncertainty"),
        "sleep-edfx": ("labels_only", "hypnogram_edf", "label_statistics"),
        "hmc-sleep-staging": ("labels_only", "scoring_edf", "label_statistics"),
        "cap-sleep": ("labels_only", "text_hypnograms", "pathology_stratified_labels"),
        "isleeps": ("labels_only", "spreadsheet_hypnograms", "clinical_label_stratification"),
        "sleep-accel": ("labels_only", "labels_without_eeg", "label_statistics"),
        "dreams": ("labels_only", "text_hypnograms", "label_statistics"),
    }
    rows = [
        {
            "dataset": dataset,
            "harmonized_scoring_rows": int(counts.get(dataset, 0)),
            "local_signal_status": roles[dataset][0],
            "local_container_interpretation": roles[dataset][1],
            "valid_project_role": roles[dataset][2],
        }
        for dataset in roles
    ]
    rows.append(
        {
            "dataset": "gx36m",
            "harmonized_scoring_rows": 174,
            "local_signal_status": "full_raw_archives",
            "local_container_interpretation": "58_psg_nights_three_scorers",
            "valid_project_role": "signal_model_multinight_scorer_uncertainty",
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    gx_rows, gx_archives = gx_recordings()
    isruc_rows, isruc_signals = isruc_recordings()
    recordings = pd.DataFrame(boas_recordings() + gx_rows + isruc_rows)
    recordings = recordings.sort_values(["dataset", "participant_group", "recording"]).reset_index(drop=True)

    summary = recordings.groupby("dataset", as_index=False).agg(
        recordings=("recording", "size"),
        participant_groups=("participant_group", "nunique"),
        epochs=("epochs", "sum"),
        rem_to_wake_events=("rem_to_wake_events", "sum"),
        recordings_with_event=("rem_to_wake_events", lambda value: int((value > 0).sum())),
        mean_scorer_disagreement_fraction=("scorer_disagreement_fraction", "mean"),
        maximum_trailing_unscored_sec=("trailing_unscored_sec", "max"),
        common_channel_pass=("common_channels_present", "all"),
        sampling_pass=("sampling_frequency_pass", "all"),
        label_support_pass=("label_within_signal_support", "all"),
    )
    total = {
        "dataset": "TOTAL",
        "recordings": len(recordings),
        "participant_groups": recordings.participant_group.nunique(),
        "epochs": int(recordings.epochs.sum()),
        "rem_to_wake_events": int(recordings.rem_to_wake_events.sum()),
        "recordings_with_event": int(recordings.rem_to_wake_events.gt(0).sum()),
        "mean_scorer_disagreement_fraction": float(recordings.scorer_disagreement_fraction.mean()),
        "maximum_trailing_unscored_sec": float(recordings.trailing_unscored_sec.max()),
        "common_channel_pass": bool(recordings.common_channels_present.all()),
        "sampling_pass": bool(recordings.sampling_frequency_pass.all()),
        "label_support_pass": bool(recordings.label_within_signal_support.all()),
    }
    summary = pd.concat([summary, pd.DataFrame([total])], ignore_index=True)

    channel_rows = []
    for dataset, mapping in COMMON_CHANNELS.items():
        reference = "M1" if dataset == "BOAS" else "A1"
        for canonical, source in mapping.items():
            channel_rows.append(
                {
                    "dataset": dataset,
                    "canonical_channel": canonical,
                    "source_channel": source,
                    "reference": reference,
                    "sampling_frequency_hz": EXPECTED_SFREQ[dataset],
                    "target_sampling_frequency_hz": 128.0,
                }
            )
    channels = pd.DataFrame(channel_rows)

    source_files = gx_archives + isruc_signals
    manifest = pd.DataFrame(
        [
            {
                "dataset": "gx36m" if path.suffix.lower() == ".zip" else "ISRUC-3",
                "relative_path": relative_source(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in source_files
        ]
    )

    inventory = dataset_inventory()
    checks = pd.DataFrame(
        [
            {"check": "inventory_sources", "observed": len(inventory), "expected": 11, "passed": len(inventory) == 11},
            {"check": "signal_datasets", "observed": recordings.dataset.nunique(), "expected": 3, "passed": recordings.dataset.nunique() == 3},
            {"check": "eligible_recordings", "observed": len(recordings), "expected": 196, "passed": len(recordings) == 196},
            {"check": "participant_groups", "observed": recordings.participant_group.nunique(), "expected": 130, "passed": recordings.participant_group.nunique() == 130},
            {"check": "boas_recordings", "observed": int((recordings.dataset == "BOAS").sum()), "expected": 128, "passed": int((recordings.dataset == "BOAS").sum()) == 128},
            {"check": "gx36m_recordings", "observed": int((recordings.dataset == "gx36m").sum()), "expected": 58, "passed": int((recordings.dataset == "gx36m").sum()) == 58},
            {"check": "isruc3_recordings", "observed": int((recordings.dataset == "ISRUC-3").sum()), "expected": 10, "passed": int((recordings.dataset == "ISRUC-3").sum()) == 10},
            {"check": "common_channels", "observed": bool(recordings.common_channels_present.all()), "expected": True, "passed": bool(recordings.common_channels_present.all())},
            {"check": "sampling_headers", "observed": bool(recordings.sampling_frequency_pass.all()), "expected": True, "passed": bool(recordings.sampling_frequency_pass.all())},
            {"check": "label_signal_support", "observed": bool(recordings.label_within_signal_support.all()), "expected": True, "passed": bool(recordings.label_within_signal_support.all())},
            {"check": "rem_to_wake_events", "observed": int(recordings.rem_to_wake_events.sum()), "expected": 521, "passed": int(recordings.rem_to_wake_events.sum()) == 521},
            {"check": "external_source_hashes", "observed": len(manifest), "expected": 30, "passed": len(manifest) == 30 and manifest.sha256.str.len().eq(64).all()},
            {"check": "model_fits_performed", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Multi-dataset PSG pool audit failed")

    verify_or_create_tsv(inventory, output_dir() / "local_dataset_inventory_v0.1.tsv")
    verify_or_create_tsv(recordings, output_dir() / "recording_eligibility_v0.1.tsv")
    verify_or_create_tsv(summary, output_dir() / "dataset_summary_v0.1.tsv")
    verify_or_create_tsv(channels, output_dir() / "common_channel_mapping_v0.1.tsv")
    verify_or_create_tsv(manifest, output_dir() / "source_container_manifest_v0.1.tsv")
    verify_or_create_tsv(checks, output_dir() / "audit_checks_v0.1.tsv")
    print(f"Multi-dataset PSG audit passed {len(checks)}/{len(checks)} checks.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
