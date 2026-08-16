"""Fit and evaluate frozen spectral stage-first baselines SF-B and SF-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import warnings
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from stage_first_event_evaluation_v0_1 import evaluate_events, participant_bootstrap


# Section 1: frozen configuration

VERSION = "v0.1"
PREPROCESSING_VERSION = "v0.2"
EXPERIMENT_DIR = "2026-08-15_stage_first_feature_baseline_v0.1"
DERIVED_DIR = "stage_first_feature_baseline_v0.1"
CHANNELS = ["HB_1", "HB_2"]
VALID_STAGES = [0, 1, 2, 3, 4]
STAGE_NAMES = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}
COMPARATORS = {"SF-B": "epoch_only", "SF-C": "five_epoch_context"}
MEMBERSHIPS = ["primary", "expanded"]
TOLERANCES = [15.0, 45.0]
INPUT_SFREQ = 256.0
OUTPUT_SFREQ = 128.0
EPOCH_SEC = 30.0
EPOCH_SAMPLES = int(EPOCH_SEC * OUTPUT_SFREQ)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
FREEZE_MARKER = "FROZEN_FOR_SINGLE_TEST_EVALUATION"


# Section 2: paths and frozen inputs

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def dataset_root() -> Path:
    return data_parent() / "boas_ds005555_v1.1.1"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


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
    if len(result) != 128 or result["subject"].nunique() != 128:
        raise ValueError("Expected 128 uniquely assigned BOAS recordings")
    return result.sort_values("subject", key=lambda value: value.map(subject_number))


def scaler_parameters() -> dict[str, dict[str, float]]:
    path = (
        repo_root()
        / "experiments/2026-07-15_minimal_preprocessing_v0.2/train_robust_scaler_v0.2.tsv"
    )
    scaler = pd.read_csv(path, sep="\t").set_index("channel")
    if set(scaler.index) != set(CHANNELS):
        raise ValueError("Frozen preprocessing scaler does not contain both headband channels")
    return scaler[["median_uv", "robust_scale_uv"]].to_dict("index")


# Section 3: deterministic feature extraction

def feature_names() -> list[str]:
    return [f"{channel}_{band}_log10_mean_psd" for channel in CHANNELS for band in BANDS]


def recording_paths(subject: str) -> tuple[Path, Path]:
    eeg = dataset_root() / subject / "eeg"
    return (
        eeg / f"{subject}_task-Sleep_acq-headband_eeg.edf",
        eeg / f"{subject}_task-Sleep_acq-psg_events.tsv",
    )


def recording_feature_path(subject: str) -> Path:
    return derived_dir() / "recording_features" / f"{subject}_features_v0.1.npz"


def extract_recording_features(subject: str, overwrite: bool = False) -> Path:
    destination = recording_feature_path(subject)
    if destination.exists() and not overwrite:
        return destination

    edf_path, event_path = recording_paths(subject)
    events = pd.read_csv(event_path, sep="\t", usecols=["onset", "duration", "stage_hum"])
    events = events.sort_values("onset").reset_index(drop=True)
    valid = events[events["stage_hum"].isin(VALID_STAGES)].copy()
    if not np.isclose(events["duration"].astype(float), EPOCH_SEC).all():
        raise ValueError(f"{subject} contains a non-30-second scoring epoch")

    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    if float(raw.info["sfreq"]) != INPUT_SFREQ:
        raise ValueError(f"{subject} sampling frequency is not {INPUT_SFREQ:g} Hz")
    missing = [channel for channel in CHANNELS if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{subject} missing channels {missing}")
    signal_uv = raw.get_data(picks=CHANNELS) * 1e6
    if not np.isfinite(signal_uv).all():
        raise ValueError(f"{subject} contains nonfinite signal samples")

    sos = butter(4, [0.3, 35.0], btype="bandpass", fs=INPUT_SFREQ, output="sos")
    filtered = sosfiltfilt(sos, signal_uv, axis=1)
    resampled = resample_poly(filtered, up=1, down=2, axis=1)
    parameters = scaler_parameters()
    for channel_index, channel in enumerate(CHANNELS):
        center = float(parameters[channel]["median_uv"])
        scale = float(parameters[channel]["robust_scale_uv"])
        resampled[channel_index] = (resampled[channel_index] - center) / scale

    onset = valid["onset"].to_numpy(dtype=float)
    stage = valid["stage_hum"].to_numpy(dtype=np.int8)
    epochs = []
    retained_onset = []
    retained_stage = []
    for epoch_onset, epoch_stage in zip(onset, stage):
        start = int(round(epoch_onset * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        if start < 0 or stop > resampled.shape[1]:
            continue
        epoch = resampled[:, start:stop]
        if epoch.shape[1] == EPOCH_SAMPLES and np.isfinite(epoch).all():
            epochs.append(epoch)
            retained_onset.append(epoch_onset)
            retained_stage.append(epoch_stage)
    if not epochs:
        raise ValueError(f"{subject} has no usable scored epochs")

    epoch_array = np.stack(epochs)
    frequencies, density = welch(
        epoch_array,
        fs=OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    columns = []
    for channel_index in range(len(CHANNELS)):
        for low, high in BANDS.values():
            frequency_mask = (frequencies >= low) & (frequencies < high)
            mean_power = density[:, channel_index, :][:, frequency_mask].mean(axis=1)
            columns.append(np.log10(np.maximum(mean_power, np.finfo(float).eps)))
    features = np.column_stack(columns).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"{subject} produced nonfinite spectral features")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        onset=np.asarray(retained_onset, dtype=np.float64),
        stage=np.asarray(retained_stage, dtype=np.int8),
        features=features,
        feature_names=np.asarray(feature_names()),
    )
    return destination


def load_recording_rows(assignment, comparator: str) -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(recording_feature_path(assignment.subject), allow_pickle=False) as values:
        onset = values["onset"]
        stage = values["stage"].astype(int)
        features = values["features"].astype(float)
    if comparator == "SF-B":
        selected_onset = onset
        selected_stage = stage
        selected_features = features
    elif comparator == "SF-C":
        indices = []
        contexts = []
        for index in range(2, len(onset) - 2):
            local_onset = onset[index - 2 : index + 3]
            if np.allclose(np.diff(local_onset), EPOCH_SEC):
                indices.append(index)
                contexts.append(features[index - 2 : index + 3].reshape(-1))
        selected_onset = onset[indices]
        selected_stage = stage[indices]
        selected_features = np.asarray(contexts, dtype=float)
    else:
        raise ValueError(f"Unknown comparator {comparator}")

    metadata = pd.DataFrame(
        {
            "subject": assignment.subject,
            "pid": int(assignment.pid),
            "partition": assignment.partition,
            "onset": selected_onset,
            "stage_hum": selected_stage,
        }
    )
    return selected_features, metadata


def load_partition_arrays(assignments: pd.DataFrame, comparator: str) -> tuple[np.ndarray, pd.DataFrame]:
    feature_parts = []
    metadata_parts = []
    for item in assignments.itertuples(index=False):
        features, metadata = load_recording_rows(item, comparator)
        feature_parts.append(features)
        metadata_parts.append(metadata)
    return np.vstack(feature_parts), pd.concat(metadata_parts, ignore_index=True)


# Section 4: model fitting and stage diagnostics

def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            random_state=20260815,
        ),
    )


def fit_models(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_dir = derived_dir() / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    train_assignments = assignments[assignments["partition"] == "train"]
    for comparator in COMPARATORS:
        train_features, train_metadata = load_partition_arrays(train_assignments, comparator)
        model = build_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(train_features, train_metadata["stage_hum"].to_numpy(dtype=int))
        convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
        model_path = model_dir / f"{comparator.lower()}_model_v0.1.joblib"
        joblib.dump(model, model_path)
        logistic = model.named_steps["logisticregression"]
        rows.append(
            {
                "comparator": comparator,
                "model_version": VERSION,
                "train_epochs": len(train_metadata),
                "input_features": train_features.shape[1],
                "classes": ";".join(map(str, logistic.classes_)),
                "maximum_iterations_used": int(np.max(logistic.n_iter_)),
                "convergence_warning_count": len(convergence),
                "fit_decision": "pass" if not convergence else "convergence_warning_retained",
            }
        )
    return pd.DataFrame(rows)


def predict_partitions(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparator in COMPARATORS:
        model_path = derived_dir() / "models" / f"{comparator.lower()}_model_v0.1.joblib"
        model = joblib.load(model_path)
        features, metadata = load_partition_arrays(assignments, comparator)
        metadata = metadata.copy()
        metadata.insert(0, "comparator", comparator)
        metadata.insert(1, "model_version", VERSION)
        metadata["stage_pred"] = model.predict(features).astype(int)
        rows.append(metadata)
    return pd.concat(rows, ignore_index=True)


def stage_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []
    confusion_rows = []
    for (comparator, partition), group in predictions.groupby(["comparator", "partition"]):
        truth = group["stage_hum"].to_numpy(dtype=int)
        prediction = group["stage_pred"].to_numpy(dtype=int)
        summary_rows.append(
            {
                "comparator": comparator,
                "model_version": VERSION,
                "partition": partition,
                "epochs": len(group),
                "accuracy": accuracy_score(truth, prediction),
                "balanced_accuracy": balanced_accuracy_score(truth, prediction),
                "macro_f1": f1_score(truth, prediction, labels=VALID_STAGES, average="macro"),
                "cohen_kappa": cohen_kappa_score(truth, prediction, labels=VALID_STAGES),
            }
        )
        report = classification_report(
            truth,
            prediction,
            labels=VALID_STAGES,
            target_names=[STAGE_NAMES[stage] for stage in VALID_STAGES],
            output_dict=True,
            zero_division=0,
        )
        matrix = confusion_matrix(truth, prediction, labels=VALID_STAGES)
        for stage in VALID_STAGES:
            values = report[STAGE_NAMES[stage]]
            class_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": partition,
                    "stage_code": stage,
                    "stage": STAGE_NAMES[stage],
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
        for true_index, true_stage in enumerate(VALID_STAGES):
            for predicted_index, predicted_stage in enumerate(VALID_STAGES):
                confusion_rows.append(
                    {
                        "comparator": comparator,
                        "model_version": VERSION,
                        "partition": partition,
                        "true_stage_code": true_stage,
                        "true_stage": STAGE_NAMES[true_stage],
                        "predicted_stage_code": predicted_stage,
                        "predicted_stage": STAGE_NAMES[predicted_stage],
                        "epochs": int(matrix[true_index, predicted_index]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


# Section 5: event derivation and frozen matching

def derive_events_and_support(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_rows = []
    support_rows = []
    for (comparator, partition, subject, pid), group in predictions.groupby(
        ["comparator", "partition", "subject", "pid"]
    ):
        group = group.sort_values("onset")
        onset = group["onset"].to_numpy(dtype=float)
        stage = group["stage_pred"].to_numpy(dtype=int)
        contiguous = np.isclose(np.diff(onset), EPOCH_SEC)
        event_indices = np.flatnonzero(contiguous & (stage[:-1] == 4) & (stage[1:] == 0)) + 1
        for index in event_indices:
            event_rows.append(
                {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": partition,
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(onset[index]),
                }
            )
        support_rows.append(
            {
                "comparator": comparator,
                "model_version": VERSION,
                "partition": partition,
                "subject": subject,
                "pid": int(pid),
                "supported_boundaries": int(contiguous.sum()),
                "supported_hours": float(contiguous.sum() * EPOCH_SEC / 3600.0),
            }
        )
    events = pd.DataFrame(
        event_rows,
        columns=["comparator", "model_version", "partition", "subject", "pid", "event_time_sec"],
    )
    return events, pd.DataFrame(support_rows)


def reference_events() -> pd.DataFrame:
    membership = pd.read_csv(
        repo_root() / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        repo_root() / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    rows = membership.merge(quality, on="transition_id", validate="one_to_one")
    rows = rows[rows["is_primary_label"].astype(str).str.lower().eq("true")].copy()
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    return rows


def evaluate_all_events(events: pd.DataFrame, support: pd.DataFrame) -> dict[str, pd.DataFrame]:
    references = reference_events()
    summaries = []
    bootstraps = []
    recordings_all = []
    participants_all = []
    matches_all = []
    configurations = events[["comparator", "partition"]].drop_duplicates()
    support_configurations = support[["comparator", "partition"]].drop_duplicates()
    configurations = pd.concat([configurations, support_configurations]).drop_duplicates()
    for configuration_row in configurations.itertuples(index=False):
        comparator = configuration_row.comparator
        partition = configuration_row.partition
        local_support = support[
            (support["comparator"] == comparator) & (support["partition"] == partition)
        ][["subject", "pid", "supported_hours"]]
        local_predictions = events[
            (events["comparator"] == comparator) & (events["partition"] == partition)
        ][["subject", "pid", "event_time_sec"]]
        local_references = references[references["partition"] == partition]
        for membership in MEMBERSHIPS:
            eligible_column = (
                "primary_analysis_eligible"
                if membership == "primary"
                else "expanded_quality_analysis_eligible"
            )
            eligibility = local_references[eligible_column].astype(str).str.lower().eq("true")
            eligible = local_references[eligibility][["subject", "pid", "event_time_sec"]]
            ignored = local_references[~eligibility][["subject", "pid", "event_time_sec"]]
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(
                    eligible, local_predictions, ignored, local_support, tolerance
                )
                config = {
                    "comparator": comparator,
                    "model_version": VERSION,
                    "partition": partition,
                    "membership": membership,
                    "tolerance_sec": tolerance,
                }
                summaries.append({**config, **summary})
                bootstrap = participant_bootstrap(participants)
                for key, value in reversed(list(config.items())):
                    bootstrap.insert(0, key, value)
                bootstraps.append(bootstrap)
                for frame, collection in [
                    (recordings, recordings_all),
                    (participants, participants_all),
                    (matches, matches_all),
                ]:
                    if len(frame):
                        frame = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in frame.columns:
                                frame.insert(0, key, value)
                        collection.append(frame)
    return {
        "event_metrics": pd.DataFrame(summaries),
        "event_bootstrap": pd.concat(bootstraps, ignore_index=True),
        "event_recordings": pd.concat(recordings_all, ignore_index=True),
        "event_participants": pd.concat(participants_all, ignore_index=True),
        "event_matches": pd.concat(matches_all, ignore_index=True) if matches_all else pd.DataFrame(),
    }


# Section 6: artifact records and phase gates

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    paths = [recording_feature_path(subject) for subject in assignments["subject"]]
    paths.extend(sorted((derived_dir() / "models").glob("*.joblib")))
    for path in sorted(set(paths)):
        rows.append(
            {
                "artifact_role": "fitted_model" if path.suffix == ".joblib" else "recording_features",
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return pd.DataFrame(rows)


def write_tables(prefix: str, predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    destination = output_dir()
    stages, classes, confusion = stage_metrics(predictions)
    events, support = derive_events_and_support(predictions)
    event_outputs = evaluate_all_events(events, support)
    predictions.to_csv(destination / f"{prefix}_stage_predictions_v0.1.tsv", sep="\t", index=False)
    stages.to_csv(destination / f"{prefix}_stage_metrics_v0.1.tsv", sep="\t", index=False)
    classes.to_csv(destination / f"{prefix}_stage_class_metrics_v0.1.tsv", sep="\t", index=False)
    confusion.to_csv(destination / f"{prefix}_stage_confusion_matrix_v0.1.tsv", sep="\t", index=False)
    events.to_csv(destination / f"{prefix}_predicted_rem_to_wake_events_v0.1.tsv", sep="\t", index=False)
    support.to_csv(destination / f"{prefix}_event_support_v0.1.tsv", sep="\t", index=False)
    for name, frame in event_outputs.items():
        frame.to_csv(destination / f"{prefix}_{name}_v0.1.tsv", sep="\t", index=False)
    return stages, event_outputs["event_metrics"]


def write_validation_decision(stages: pd.DataFrame, events: pd.DataFrame) -> None:
    validation_stages = stages[stages["partition"] == "validation"].set_index("comparator")
    validation_events = events[
        (events["partition"] == "validation")
        & (events["membership"] == "primary")
        & (events["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    sf_b_stage = float(validation_stages.loc["SF-B", "macro_f1"])
    sf_c_stage = float(validation_stages.loc["SF-C", "macro_f1"])
    sf_b_event = float(validation_events.loc["SF-B", "f1"])
    sf_c_event = float(validation_events.loc["SF-C", "f1"])
    context_stage = sf_c_stage > sf_b_stage
    context_event = sf_c_event > sf_b_event
    text = f"""# Validation Decision and Test Freeze v0.1

**Created:** 2026-08-15
**Marker:** `{FREEZE_MARKER}`

The frozen train/validation run completed without changing features, model settings, split, quality membership, event derivation, or matching tolerance.

| Comparator | Validation macro F1 | Validation primary event F1 (+/-15 s) |
|---|---:|---:|
| SF-B epoch-only | {sf_b_stage:.6f} | {sf_b_event:.6f} |
| SF-C five-epoch context | {sf_c_stage:.6f} | {sf_c_event:.6f} |

Temporal context improved validation macro F1: **{context_stage}**. Temporal context improved validation event F1: **{context_event}**. The prespecified H5.3 is supported only if the relevant metric improved; both outcomes are retained.

## Frozen Test Decision

Evaluate both SF-B and SF-C once on the untouched test recordings. SF-C remains the prespecified primary stage-first comparator and SF-B remains its ablation regardless of the validation ordering. No configuration or threshold will be revised after seeing test results.
"""
    output_dir().joinpath("validation_decision_and_test_freeze_v0.1.md").write_text(
        text, encoding="utf-8"
    )


def write_readme() -> None:
    validation_stage = pd.read_csv(output_dir() / "train_validation_stage_metrics_v0.1.tsv", sep="\t")
    validation_event = pd.read_csv(output_dir() / "train_validation_event_metrics_v0.1.tsv", sep="\t")
    fit = pd.read_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t")
    test_stage_path = output_dir() / "test_stage_metrics_v0.1.tsv"
    sections = [
        "# Transparent Stage-First Feature Baseline v0.1",
        "",
        "**Created:** 2026-08-15",
        "**Protocol:** `docs/evaluation/stage_first_baseline_protocol_v0.1.md`",
        "**Models:** SF-B epoch-only logistic regression; SF-C five-epoch-context logistic regression",
        "",
        "## Validation Result",
        "",
        "| Comparator | Macro F1 | Balanced accuracy | Cohen kappa | Primary event F1 (+/-15 s) |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparator in COMPARATORS:
        stage_row = validation_stage[
            (validation_stage["comparator"] == comparator)
            & (validation_stage["partition"] == "validation")
        ].iloc[0]
        event_row = validation_event[
            (validation_event["comparator"] == comparator)
            & (validation_event["partition"] == "validation")
            & (validation_event["membership"] == "primary")
            & (validation_event["tolerance_sec"] == 15.0)
        ].iloc[0]
        sections.append(
            f"| {comparator} | {stage_row.macro_f1:.4f} | {stage_row.balanced_accuracy:.4f} | {stage_row.cohen_kappa:.4f} | {event_row.f1:.4f} |"
        )
    sections.extend(
        [
            "",
            "## Fit Record",
            "",
            "| Comparator | Train epochs | Features | Iterations used | Convergence warnings | Decision |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in fit.itertuples(index=False):
        sections.append(
            f"| {row.comparator} | {row.train_epochs:,} | {row.input_features} | {row.maximum_iterations_used} | {row.convergence_warning_count} | {row.fit_decision} |"
        )
    if test_stage_path.exists():
        test_stage = pd.read_csv(test_stage_path, sep="\t")
        test_event = pd.read_csv(output_dir() / "test_event_metrics_v0.1.tsv", sep="\t")
        sections.extend(
            [
                "",
                "## Frozen Test Result",
                "",
                "| Comparator | Macro F1 | Balanced accuracy | Cohen kappa | Primary event F1 (+/-15 s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for comparator in COMPARATORS:
            stage_row = test_stage[test_stage["comparator"] == comparator].iloc[0]
            event_row = test_event[
                (test_event["comparator"] == comparator)
                & (test_event["membership"] == "primary")
                & (test_event["tolerance_sec"] == 15.0)
            ].iloc[0]
            sections.append(
                f"| {comparator} | {stage_row.macro_f1:.4f} | {stage_row.balanced_accuracy:.4f} | {stage_row.cohen_kappa:.4f} | {event_row.f1:.4f} |"
            )
    sections.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "This is a transparent stage-first comparator, not the proposed direct boundary detector. Poor event performance despite useful stage metrics would identify a concrete limitation of deriving boundary alarms from independently classified 30-second stages.",
            "",
            "SF-C reached the frozen 500-iteration ceiling and its convergence warning is retained. The model was not refitted or altered after validation; this limits claims about an optimum but does not erase the observed fixed-comparator result.",
            "",
            "Feature arrays and fitted models are stored outside Git. Their paths and SHA-256 hashes are recorded in `external_artifact_manifest_v0.1.tsv`.",
        ]
    )
    output_dir().joinpath("README.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


# Section 7: separated train/validation and test execution

def train_validation_phase(overwrite_features: bool) -> None:
    assignments = subject_assignments()
    active = assignments[assignments["partition"].isin(["train", "validation"])].copy()
    if (assignments["partition"] == "test").sum() == 0:
        raise ValueError("Frozen split has no test recordings")
    output_dir().mkdir(parents=True, exist_ok=True)
    derived_dir().mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(active.itertuples(index=False), start=1):
        extract_recording_features(item.subject, overwrite=overwrite_features)
        print(f"Feature extraction {item.subject} ({index}/{len(active)})")
    fit = fit_models(active)
    fit.to_csv(output_dir() / "model_fit_summary_v0.1.tsv", sep="\t", index=False)
    predictions = predict_partitions(active)
    stages, events = write_tables("train_validation", predictions)
    write_validation_decision(stages, events)
    artifact_manifest(active).to_csv(
        output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    environment = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "mne": mne.__version__,
        "sklearn": __import__("sklearn").__version__,
        "scipy": __import__("scipy").__version__,
    }
    output_dir().joinpath("software_versions_v0.1.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    write_readme()
    print(stages.to_string(index=False))
    print(events[(events["membership"] == "primary") & (events["tolerance_sec"] == 15.0)].to_string(index=False))


def test_phase(overwrite_features: bool) -> None:
    freeze_path = output_dir() / "validation_decision_and_test_freeze_v0.1.md"
    if not freeze_path.exists() or FREEZE_MARKER not in freeze_path.read_text(encoding="utf-8"):
        raise RuntimeError("Test phase blocked: validation decision and freeze marker are missing")
    assignments = subject_assignments()
    active = assignments[assignments["partition"] == "test"].copy()
    for index, item in enumerate(active.itertuples(index=False), start=1):
        extract_recording_features(item.subject, overwrite=overwrite_features)
        print(f"Test feature extraction {item.subject} ({index}/{len(active)})")
    predictions = predict_partitions(active)
    stages, events = write_tables("test", predictions)
    artifact_manifest(assignments).to_csv(
        output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t", index=False
    )
    output_dir().joinpath("test_access_record_v0.1.md").write_text(
        "# Test Access Record v0.1\n\n"
        "**First signal access by this baseline:** 2026-08-15\n"
        "**Access count under protocol:** 1\n\n"
        "The frozen SF-B and SF-C models were applied without refitting or configuration changes. "
        "The reported test result was retained regardless of outcome.\n",
        encoding="utf-8",
    )
    write_readme()
    print(stages.to_string(index=False))
    print(events[(events["membership"] == "primary") & (events["tolerance_sec"] == 15.0)].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["train-validation", "test"])
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "train-validation":
        train_validation_phase(args.overwrite_features)
    else:
        test_phase(args.overwrite_features)


if __name__ == "__main__":
    main()
