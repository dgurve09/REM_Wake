"""Run the frozen Block 8 train-only noise-augmentation experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from reviewed_output import verify_or_create_tsv
from run_block7_transfer_validation_v0_1 import (
    context_matrix,
    feature_path,
    labeled_candidates,
    local_event_inputs,
    model_arrays,
    model_path as source_model_path,
    normalize_signal,
    reference_events,
    scaler_maps,
    sha256,
    subject_assignments,
    time_key,
    verify_or_create_model,
)
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values
from validate_block7_feature_generation_v0_1 import (
    EPOCH_SAMPLES,
    HB2,
    OUTPUT_SFREQ,
    epoch_features,
    filter_resample,
    filter_sos,
    read_uv,
    valid_events,
    verify_or_create_npz,
)


# Section 1: frozen experiment configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-12_block8_noise_augmented_training_v0.1"
RESULT_DERIVED_DIR = "block8_noise_augmented_training_v0.1"
VALIDATION_NOISE_DERIVED_DIR = "block8_raw_signal_noise_v0.1"
PROTOCOL_COMMIT = "540cd87"
BASE_SEED = 20260912
FOLDS = 5
BOOTSTRAP_RESAMPLES = 2000
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
EPOCH_SEC = 30.0
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
ORIGINAL_MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
ORIGINAL_THRESHOLD = 0.96
SNR_TOLERANCE_DB = 0.02
FEATURE_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-10

CLEAN_F1_LOSS_BOUND = 0.03
CLEAN_FAR_INCREASE_BOUND = 0.50
HB1_F1_IMPROVEMENT = 0.03
HB1_FAR_REDUCTION = 0.50
MEANINGFUL_CLEAN_F1_GAIN = 0.05

TRAIN_CONDITIONS = {
    "TRAIN-CLEAN": {},
    "TRAIN-BOTH-20DB": {"HB_1": 20.0, "HB_2": 20.0},
    "TRAIN-BOTH-10DB": {"HB_1": 10.0, "HB_2": 10.0},
    "TRAIN-HB1-10DB": {"HB_1": 10.0},
    "TRAIN-HB2-10DB": {"HB_2": 10.0},
}
TRAIN_FOLDERS = {
    "TRAIN-BOTH-20DB": "both_20db",
    "TRAIN-BOTH-10DB": "both_10db",
    "TRAIN-HB1-10DB": "hb1_10db",
    "TRAIN-HB2-10DB": "hb2_10db",
}
VALIDATION_CONDITIONS = [
    "H2-CLEAN",
    "H2-BOTH-20DB",
    "H2-BOTH-10DB",
    "H2-BOTH-0DB",
    "H2-HB1-10DB",
    "H2-HB2-10DB",
]
VALIDATION_FOLDERS = {
    "H2-BOTH-20DB": "both_20db",
    "H2-BOTH-10DB": "both_10db",
    "H2-BOTH-0DB": "both_0db",
    "H2-HB1-10DB": "hb1_10db",
    "H2-HB2-10DB": "hb2_10db",
}


# Section 2: paths and immutable output helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def result_dir() -> Path:
    return data_parent() / "derived" / RESULT_DERIVED_DIR


def generated_train_feature_path(subject: str, condition: str) -> Path:
    folder = TRAIN_FOLDERS[condition]
    return result_dir() / "recording_features" / "train" / folder / f"{subject}_features_v0.1.npz"


def validation_feature_path(subject: str, condition: str) -> Path:
    if condition == "H2-CLEAN":
        return feature_path(subject, "validation", "HB-2")
    folder = VALIDATION_FOLDERS[condition]
    return data_parent() / "derived" / VALIDATION_NOISE_DERIVED_DIR / "recording_features" / folder / f"{subject}_features_v0.1.npz"


def augmented_model_path() -> Path:
    return result_dir() / "models" / "h2_na_model_v0.1.joblib"


def oof_score_path() -> Path:
    return result_dir() / "candidate_scores" / "train_oof_clean_scores_v0.1.tsv.gz"


def prior_validation_score_path() -> Path:
    return data_parent() / "derived" / VALIDATION_NOISE_DERIVED_DIR / "candidate_scores" / "validation_noise_scores_v0.1.tsv.gz"


def validation_score_path() -> Path:
    return result_dir() / "candidate_scores" / "validation_model_condition_scores_v0.1.tsv.gz"


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def verify_or_create_gzip_tsv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as stream:
            frame.to_csv(stream, sep="\t", index=False, lineterminator="\n")
    if path.exists():
        if sha256(path) != sha256(temporary):
            temporary.unlink()
            raise RuntimeError(f"External artifact changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


# Section 3: train-only raw-signal noise features

def valid_sample_mask(events: pd.DataFrame, samples: int) -> np.ndarray:
    mask = np.zeros(samples, dtype=bool)
    for item in events.itertuples(index=False):
        start = int(round(float(item.onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        if start >= 0 and stop <= samples:
            mask[start:stop] = True
    if not mask.any():
        raise ValueError("No valid scored samples for SNR calibration")
    return mask


def noise_seed(subject: str, channel: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{subject}|{channel}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def load_feature_array(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["stage"].astype(np.int8),
            values["features"].astype(np.float32),
            values["feature_names"].astype(str).tolist(),
        )


def train_feature_path(subject: str, condition: str) -> Path:
    if condition == "TRAIN-CLEAN":
        return feature_path(subject, "train", "HB-2")
    return generated_train_feature_path(subject, condition)


def generate_train_noise_features(
    train_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, hb_scaler, _ = scaler_maps()
    sos = filter_sos()
    generated_rows = []
    calibration_rows = []
    basis_rows = []
    clean_rows = []

    for index, item in enumerate(train_assignments.itertuples(index=False), start=1):
        print(f"Train noise feature generation {index}/{len(train_assignments)}: {item.subject}", flush=True)
        events = valid_events(item.subject)
        raw = read_uv(item.subject, "headband", HB2)
        clean_filtered = filter_resample(raw, sos)
        mask = valid_sample_mask(events, clean_filtered.shape[1])

        noise_raw = np.empty_like(raw)
        for channel_index, channel in enumerate(HB2):
            seed = noise_seed(item.subject, channel)
            noise_raw[channel_index] = np.random.default_rng(seed).standard_normal(raw.shape[1])
            basis_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "channel": channel,
                    "seed": seed,
                    "raw_samples": raw.shape[1],
                    "noise_basis_sha256": array_sha256(noise_raw[channel_index]),
                }
            )
        noise_filtered = filter_resample(noise_raw, sos)

        clean = epoch_features(normalize_signal(clean_filtered, HB2, hb_scaler), HB2, events)
        source_path = feature_path(item.subject, "train", "HB-2")
        source = load_feature_array(source_path)
        timing_match = np.array_equal(clean[0], source[0]) and np.array_equal(clean[1], source[1])
        schema_match = clean[3] == source[3]
        difference = float(np.max(np.abs(clean[2] - source[2])))
        clean_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "timing_and_stage_match": timing_match,
                "feature_schema_match": schema_match,
                "maximum_absolute_feature_difference": difference,
                "tolerance": FEATURE_TOLERANCE,
                "reproduction_pass": timing_match and schema_match and difference <= FEATURE_TOLERANCE,
                "source_feature_sha256": sha256(source_path),
            }
        )

        scales = {}
        for channel_index, channel in enumerate(HB2):
            clean_rms = float(np.sqrt(np.mean(np.square(clean_filtered[channel_index, mask]))))
            noise_rms = float(np.sqrt(np.mean(np.square(noise_filtered[channel_index, mask]))))
            if clean_rms <= 0 or noise_rms <= 0:
                raise ValueError(f"Invalid calibration RMS: {item.subject}, {channel}")
            for target_snr in [20.0, 10.0]:
                scales[(channel, target_snr)] = clean_rms / (10 ** (target_snr / 20.0) * noise_rms)

        for condition, perturbations in TRAIN_CONDITIONS.items():
            if condition == "TRAIN-CLEAN":
                continue
            degraded = clean_filtered.copy()
            for channel_index, channel in enumerate(HB2):
                if channel not in perturbations:
                    continue
                target_snr = float(perturbations[channel])
                scale = scales[(channel, target_snr)]
                added = scale * noise_filtered[channel_index]
                degraded[channel_index] += added
                clean_rms = float(np.sqrt(np.mean(np.square(clean_filtered[channel_index, mask]))))
                added_rms = float(np.sqrt(np.mean(np.square(added[mask]))))
                achieved = 20.0 * np.log10(clean_rms / added_rms)
                calibration_rows.append(
                    {
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "condition": condition,
                        "channel": channel,
                        "target_snr_db": target_snr,
                        "achieved_snr_db": achieved,
                        "absolute_snr_error_db": abs(achieved - target_snr),
                        "noise_scale": scale,
                        "valid_samples": int(mask.sum()),
                        "calibration_pass": abs(achieved - target_snr) <= SNR_TOLERANCE_DB,
                    }
                )
            values = epoch_features(normalize_signal(degraded, HB2, hb_scaler), HB2, events)
            path = generated_train_feature_path(item.subject, condition)
            verify_or_create_npz(path, *values)
            centers, _ = context_matrix(values[0], values[2])
            generated_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "condition": condition,
                    "epochs": len(values[0]),
                    "context_rows": len(centers),
                    "feature_dimensions": values[2].shape[1],
                    "all_features_finite": bool(np.isfinite(values[2]).all()),
                    "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return pd.DataFrame(generated_rows), pd.DataFrame(calibration_rows), pd.DataFrame(basis_rows), pd.DataFrame(clean_rows)


# Section 4: augmented labeled rows and model fitting

def build_augmented_labeled_matrix(
    train_candidates: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices = []
    retained_rows = []
    dropped_rows = []
    for condition in TRAIN_CONDITIONS:
        for subject, group in train_candidates.groupby("subject", sort=True):
            onsets, _, features, _ = load_feature_array(train_feature_path(subject, condition))
            centers, contexts = context_matrix(onsets, features)
            lookup = {time_key(value): index for index, value in enumerate(centers)}
            for item in group.itertuples(index=False):
                row = {**item._asdict(), "augmentation_condition": condition}
                location = lookup.get(time_key(item.candidate_time_sec))
                if location is None:
                    row["drop_reason"] = "missing_required_context"
                    dropped_rows.append(row)
                else:
                    matrices.append(contexts[location])
                    retained_rows.append(row)
    columns = list(train_candidates.columns) + ["augmentation_condition"]
    retained = pd.DataFrame(retained_rows, columns=columns)
    dropped = pd.DataFrame(dropped_rows, columns=columns + ["drop_reason"])
    if not matrices:
        raise ValueError("No augmented labeled rows were retained")

    grouped = retained.groupby(["augmentation_condition", "label", "source_tier"], as_index=False).size()
    grouped = grouped.rename(columns={"size": "retained_rows"})
    dropped_summary = dropped.groupby(["augmentation_condition", "label", "source_tier"], as_index=False).size()
    dropped_summary = dropped_summary.rename(columns={"size": "dropped_rows"})
    construction = grouped.merge(dropped_summary, on=["augmentation_condition", "label", "source_tier"], how="left")
    construction["dropped_rows"] = construction["dropped_rows"].fillna(0).astype(int)

    identity = retained.groupby(["sample_id", "subject", "pid", "label", "source_tier"], as_index=False).agg(
        augmentation_conditions=("augmentation_condition", "nunique")
    )
    identity["complete_five_condition_multiplicity"] = identity["augmentation_conditions"].eq(len(TRAIN_CONDITIONS))
    return np.vstack(matrices), retained, construction, identity


def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=500,
            tol=1e-4,
            random_state=BASE_SEED,
        ),
    )


def fit_model(matrix: np.ndarray, labels: np.ndarray) -> tuple[object, int]:
    model = build_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, labels)
    count = sum(issubclass(item.category, ConvergenceWarning) for item in caught)
    return model, count


def score_clean_subjects(assignments: pd.DataFrame, model, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        onsets, _, features, _ = load_feature_array(feature_path(item.subject, "train", "HB-2"))
        centers, matrix = context_matrix(onsets, features)
        probability = model.predict_proba(matrix)[:, 1]
        score_rows.append(
            pd.DataFrame(
                {
                    "model": "H2-NA-OOF",
                    "condition": "TRAIN-CLEAN",
                    "partition": "train",
                    "fold": fold,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": centers,
                    "probability": probability,
                }
            )
        )
        support_rows.append(
            {
                "partition": "train",
                "fold": fold,
                "subject": item.subject,
                "pid": int(item.pid),
                "supported_boundaries": len(centers),
                "supported_hours": len(centers) * EPOCH_SEC / 3600.0,
            }
        )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
    group_columns = ["model", "condition", "partition", "subject", "pid"]
    for keys, group in marked.groupby(group_columns, sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        starts = [0]
        starts.extend((np.flatnonzero(np.diff(group["candidate_time_sec"].to_numpy(dtype=float)) > EPOCH_SEC + 1e-6) + 1).tolist())
        stops = starts[1:] + [len(group)]
        for start, stop in zip(starts, stops):
            run = group.iloc[start:stop]
            maximum = float(run["probability"].max())
            best = run[np.isclose(run["probability"], maximum)].sort_values("candidate_time_sec").iloc[0]
            rows.append(
                {
                    **dict(zip(group_columns, keys)),
                    "event_time_sec": float(best.candidate_time_sec),
                    "probability": float(best.probability),
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    columns = group_columns + ["event_time_sec", "probability", "threshold", "run_candidates"]
    return pd.DataFrame(rows, columns=columns)


def select_oof_threshold(
    scores: pd.DataFrame, support: pd.DataFrame, train_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, ignored = local_event_inputs(train_reference, "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    rows = []
    for threshold in THRESHOLDS:
        alarms = collapse_alarms(scores, float(threshold))
        _, _, _, summary = evaluate_events(
            eligible,
            alarms[["subject", "pid", "event_time_sec"]],
            ignored,
            local_support,
            15.0,
        )
        rows.append(
            {
                "model": "H2-NA",
                "partition": "train_oof",
                "membership": "primary",
                "tolerance_sec": 15.0,
                "threshold": float(threshold),
                **summary,
            }
        )
    curve = pd.DataFrame(rows)
    selected = curve.sort_values(
        ["f1", "false_alarms_per_hour", "recall", "threshold"],
        ascending=[False, True, False, False],
        kind="stable",
    ).iloc[[0]].copy()
    selected["selection_rule"] = "max_f1_then_min_far_then_max_recall_then_max_threshold"
    return curve, selected


def train_phase(result_code_commit: str) -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    all_assignments = subject_assignments()
    train_assignments = all_assignments[all_assignments["partition"] == "train"].copy()
    candidates = labeled_candidates(train_assignments)
    generated, calibration, basis, clean = generate_train_noise_features(train_assignments)
    if not truth(clean["reproduction_pass"]).all() or not truth(calibration["calibration_pass"]).all():
        raise ValueError("Train clean reproduction or SNR calibration failed")

    matrix, metadata, construction, identity = build_augmented_labeled_matrix(candidates)
    labels = metadata["label"].to_numpy(dtype=int)
    groups = metadata["pid"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=FOLDS)
    oof_score_rows = []
    oof_support_rows = []
    fold_rows = []
    fit_rows = []
    for fold, (fit_indices, heldout_indices) in enumerate(splitter.split(matrix, labels, groups), start=1):
        fit_pids = set(groups[fit_indices])
        heldout_pids = set(groups[heldout_indices])
        if fit_pids & heldout_pids:
            raise ValueError("Participant leakage in train out-of-fold split")
        model, convergence = fit_model(matrix[fit_indices], labels[fit_indices])
        heldout_assignments = train_assignments[train_assignments["pid"].isin(heldout_pids)]
        scores, support = score_clean_subjects(heldout_assignments, model, fold)
        oof_score_rows.append(scores)
        oof_support_rows.append(support)
        for pid in sorted(heldout_pids):
            fold_rows.append({"pid": int(pid), "fold": fold, "role": "heldout_threshold_selection"})
        fit_rows.append(
            {
                "fit": f"fold_{fold}",
                "fit_pid": len(fit_pids),
                "heldout_pid": len(heldout_pids),
                "fit_rows": len(fit_indices),
                "fit_positive": int(labels[fit_indices].sum()),
                "fit_negative": int((labels[fit_indices] == 0).sum()),
                "convergence_warning_count": convergence,
                "maximum_iterations_used": int(model.named_steps["logisticregression"].n_iter_.max()),
            }
        )
    oof_scores = pd.concat(oof_score_rows, ignore_index=True)
    oof_support = pd.concat(oof_support_rows, ignore_index=True)
    fold_assignments = pd.DataFrame(fold_rows)
    curve, selected = select_oof_threshold(oof_scores, oof_support, reference_events(train_assignments))
    selected_threshold = float(selected.iloc[0].threshold)

    final_model, convergence = fit_model(matrix, labels)
    verify_or_create_model(final_model, augmented_model_path())
    final_hash = sha256(augmented_model_path())
    fit_rows.append(
        {
            "fit": "final_all_train",
            "fit_pid": metadata["pid"].nunique(),
            "heldout_pid": 0,
            "fit_rows": len(labels),
            "fit_positive": int(labels.sum()),
            "fit_negative": int((labels == 0).sum()),
            "convergence_warning_count": convergence,
            "maximum_iterations_used": int(final_model.named_steps["logisticregression"].n_iter_.max()),
        }
    )
    fit_record = pd.DataFrame(fit_rows)
    freeze = pd.DataFrame(
        [
            {
                "model": "H2-NA",
                "result_code_commit": result_code_commit,
                "threshold": selected_threshold,
                "threshold_source": "five_fold_grouped_train_oof_clean_full_night",
                "model_sha256": final_hash,
                "train_recordings": len(train_assignments),
                "train_pid": train_assignments["pid"].nunique(),
                "augmented_labeled_rows": len(metadata),
                "validation_accessed": False,
                "test_accessed": False,
            }
        ]
    )
    verify_or_create_gzip_tsv(oof_scores, oof_score_path())

    manifest_paths = [("source_train_clean_feature", feature_path(item.subject, "train", "HB-2")) for item in train_assignments.itertuples(index=False)]
    manifest_paths.extend(("generated_train_noise_feature", data_parent() / item.path_relative_to_data_parent) for item in generated.itertuples(index=False))
    manifest_paths.extend([("train_oof_scores", oof_score_path()), ("augmented_model", augmented_model_path())])
    manifest = pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in manifest_paths
        ]
    ).sort_values(["artifact_role", "path_relative_to_data_parent"])

    fold_counts = fold_assignments.groupby("pid").size()
    checks = pd.DataFrame(
        [
            {"check": "train_membership", "status": "pass" if len(train_assignments) == 82 and train_assignments["pid"].nunique() == 64 else "fail", "detail": "82 recordings; 64 pid groups"},
            {"check": "clean_feature_reproduction", "status": "pass" if truth(clean["reproduction_pass"]).all() else "fail", "detail": f"maximum={clean['maximum_absolute_feature_difference'].max():.12g}"},
            {"check": "noise_basis", "status": "pass" if len(basis) == 164 and basis[["subject", "channel"]].drop_duplicates().shape[0] == 164 else "fail", "detail": "82 recordings x two channels"},
            {"check": "snr_calibration", "status": "pass" if len(calibration) == 492 and truth(calibration["calibration_pass"]).all() else "fail", "detail": f"maximum error={calibration['absolute_snr_error_db'].max():.12g} dB"},
            {"check": "generated_train_features", "status": "pass" if len(generated) == 328 and truth(generated["all_features_finite"]).all() else "fail", "detail": "82 recordings x four noise conditions"},
            {"check": "five_condition_labeled_multiplicity", "status": "pass" if len(identity) > 0 and truth(identity["complete_five_condition_multiplicity"]).all() else "fail", "detail": f"base rows={len(identity)}; augmented rows={len(metadata)}"},
            {"check": "grouped_oof_assignment", "status": "pass" if len(fold_assignments) == 64 and fold_counts.eq(1).all() else "fail", "detail": "each train pid held out exactly once"},
            {"check": "complete_oof_scores", "status": "pass" if oof_scores["pid"].nunique() == 64 and oof_scores["subject"].nunique() == 82 else "fail", "detail": f"rows={len(oof_scores)}"},
            {"check": "threshold_frozen", "status": "pass" if len(curve) == 99 and selected_threshold in set(THRESHOLDS) else "fail", "detail": f"threshold={selected_threshold:.2f}"},
            {"check": "final_model_frozen", "status": "pass" if len(final_hash) == 64 else "fail", "detail": final_hash},
            {"check": "no_convergence_warning", "status": "pass" if fit_record["convergence_warning_count"].eq(0).all() else "fail", "detail": f"six fits; max iterations={fit_record['maximum_iterations_used'].max()}"},
            {"check": "train_external_manifest", "status": "pass" if len(manifest) == 412 and manifest["sha256"].str.len().eq(64).all() else "fail", "detail": "82 source features, 328 generated features, model, OOF scores"},
            {"check": "validation_and_test_closed", "status": "pass" if not manifest["path_relative_to_data_parent"].str.contains("/recording_features/validation/|/test/|test_", case=False, regex=True).any() else "fail", "detail": "train artifacts only"},
        ]
    )

    reviewed = {
        "train_clean_feature_reproduction_v0.1.tsv": clean,
        "train_noise_basis_manifest_v0.1.tsv": basis,
        "train_noise_calibration_v0.1.tsv": calibration,
        "generated_train_feature_manifest_v0.1.tsv": generated,
        "augmented_labeled_construction_v0.1.tsv": construction,
        "augmented_labeled_identity_v0.1.tsv": identity,
        "train_oof_fold_assignments_v0.1.tsv": fold_assignments,
        "train_oof_support_v0.1.tsv": oof_support,
        "train_oof_threshold_curve_v0.1.tsv": curve,
        "train_oof_threshold_selection_v0.1.tsv": selected,
        "model_fit_record_v0.1.tsv": fit_record,
        "train_model_freeze_v0.1.tsv": freeze,
        "train_external_artifact_manifest_v0.1.tsv": manifest,
        "train_phase_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)
    software = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    verify_or_create_text(output_dir() / "software_versions_v0.1.json", json.dumps(software, indent=2, sort_keys=True) + "\n")
    freeze_text = "\n".join(
        [
            "# H2-NA Train Freeze",
            "",
            "**Date:** 2026-09-12",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            f"**Selected train-OOF threshold:** `{selected_threshold:.2f}`",
            f"**Final model SHA-256:** `{final_hash}`",
            "**Validation accessed:** No",
            "**Test accessed:** No",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} train-phase checks passed. The model and threshold are frozen before validation scoring.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "TRAIN_FREEZE.md", freeze_text)
    print(selected.to_string(index=False))
    print(fit_record.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one train-phase check failed")


# Section 5: frozen validation scoring and event evaluation

def verify_train_freeze(result_code_commit: str) -> tuple[object, float, pd.DataFrame]:
    freeze = pd.read_csv(output_dir() / "train_model_freeze_v0.1.tsv", sep="\t")
    if len(freeze) != 1 or str(freeze.iloc[0].result_code_commit) != result_code_commit:
        raise ValueError("Train freeze does not match the committed result code")
    model_hash = sha256(augmented_model_path())
    if model_hash != freeze.iloc[0].model_sha256:
        raise ValueError("Frozen augmented model hash changed")
    if sha256(source_model_path("H2-D")) != ORIGINAL_MODEL_SHA256:
        raise ValueError("Original H2-D model hash changed")
    return joblib.load(augmented_model_path()), float(freeze.iloc[0].threshold), freeze


def score_augmented_validation(validation_assignments: pd.DataFrame, model) -> pd.DataFrame:
    rows = []
    for condition in VALIDATION_CONDITIONS:
        for item in validation_assignments.itertuples(index=False):
            onsets, _, features, _ = load_feature_array(validation_feature_path(item.subject, condition))
            centers, matrix = context_matrix(onsets, features)
            probability = model.predict_proba(matrix)[:, 1]
            rows.append(
                pd.DataFrame(
                    {
                        "model": "H2-NA",
                        "condition": condition,
                        "partition": "validation",
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": centers,
                        "probability": probability,
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def combined_validation_scores(validation_assignments: pd.DataFrame, augmented_model) -> pd.DataFrame:
    prior = pd.read_csv(prior_validation_score_path(), sep="\t")
    original = prior.rename(columns={"comparator": "condition"}).copy()
    original["model"] = "H2-D"
    original = original[["model", "condition", "partition", "subject", "pid", "candidate_time_sec", "probability"]]
    augmented = score_augmented_validation(validation_assignments, augmented_model)
    result = pd.concat([original, augmented], ignore_index=True)
    if set(result["partition"]) != {"validation"}:
        raise ValueError("Non-validation score row found")
    return result


def validation_support(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scores.groupby(["model", "condition", "subject", "pid"], sort=True):
        rows.append(
            {
                "model": keys[0],
                "condition": keys[1],
                "partition": "validation",
                "subject": keys[2],
                "pid": int(keys[3]),
                "supported_boundaries": len(group),
                "supported_hours": len(group) * EPOCH_SEC / 3600.0,
            }
        )
    return pd.DataFrame(rows)


def evaluate_validation(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
    threshold_na: float,
) -> dict[str, pd.DataFrame]:
    thresholds = {"H2-D": ORIGINAL_THRESHOLD, "H2-NA": threshold_na}
    event_rows = []
    summary_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for model in ["H2-D", "H2-NA"]:
        for condition in VALIDATION_CONDITIONS:
            local_scores = scores[(scores["model"] == model) & (scores["condition"] == condition)]
            local_support = support[(support["model"] == model) & (support["condition"] == condition)][["subject", "pid", "supported_hours"]]
            predictions = collapse_alarms(local_scores, thresholds[model])
            event_rows.append(predictions)
            for membership in MEMBERSHIPS:
                eligible, ignored = local_event_inputs(references, membership)
                for tolerance in TOLERANCES:
                    recordings, participants, matches, summary = evaluate_events(
                        eligible,
                        predictions[["subject", "pid", "event_time_sec"]],
                        ignored,
                        local_support,
                        tolerance,
                    )
                    config = {
                        "model": model,
                        "condition": condition,
                        "partition": "validation",
                        "membership": membership,
                        "tolerance_sec": tolerance,
                        "threshold": thresholds[model],
                    }
                    summary_rows.append({**config, **summary})
                    for frame, collection in [(recordings, recording_rows), (participants, participant_rows), (matches, match_rows)]:
                        if len(frame):
                            local = frame.copy()
                            for key, value in reversed(list(config.items())):
                                if key not in local.columns:
                                    local.insert(0, key, value)
                            collection.append(local)
    return {
        "events": pd.concat(event_rows, ignore_index=True),
        "metrics": pd.DataFrame(summary_rows),
        "recordings": pd.concat(recording_rows, ignore_index=True),
        "participants": pd.concat(participant_rows, ignore_index=True),
        "matches": pd.concat(match_rows, ignore_index=True),
    }


# Section 6: paired participant uncertainty and decisions

def aggregate_participants(frame: pd.DataFrame) -> dict:
    return metric_values(
        int(frame["true_positive"].sum()),
        int(frame["false_positive"].sum()),
        int(frame["false_negative"].sum()),
        float(frame["supported_hours"].sum()),
    )


def bootstrap_comparison(left: pd.DataFrame, right: pd.DataFrame, comparison: str, kind: str) -> list[dict]:
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    paired = left[columns].merge(right[columns], on="pid", suffixes=("_left", "_right"), validate="one_to_one")
    if len(paired) != 16:
        raise ValueError(f"Incomplete paired validation participants: {comparison}")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        values = {}
        for side in ["left", "right"]:
            values[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()),
                int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()),
                float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "event_f1_difference": values["left"]["f1"] - values["right"]["f1"],
                "false_alarms_per_hour_difference": values["left"]["false_alarms_per_hour"] - values["right"]["false_alarms_per_hour"],
            }
        )
    sample_frame = pd.DataFrame(samples)
    point_left = aggregate_participants(left)
    point_right = aggregate_participants(right)
    points = {
        "event_f1_difference": point_left["f1"] - point_right["f1"],
        "false_alarms_per_hour_difference": point_left["false_alarms_per_hour"] - point_right["false_alarms_per_hour"],
    }
    return [
        {
            "comparison_kind": kind,
            "comparison": comparison,
            "metric": metric,
            "point_difference": value,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BASE_SEED,
            "lower_95": float(sample_frame[metric].quantile(0.025)),
            "median": float(sample_frame[metric].quantile(0.5)),
            "upper_95": float(sample_frame[metric].quantile(0.975)),
        }
        for metric, value in points.items()
    ]


def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[(participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)]
    rows = []
    for condition in VALIDATION_CONDITIONS:
        left = primary[(primary["model"] == "H2-NA") & (primary["condition"] == condition)]
        right = primary[(primary["model"] == "H2-D") & (primary["condition"] == condition)]
        rows.extend(bootstrap_comparison(left, right, f"H2-NA_minus_H2-D__{condition}", "between_model"))
    for model in ["H2-D", "H2-NA"]:
        left = primary[(primary["model"] == model) & (primary["condition"] == "H2-HB1-10DB")]
        right = primary[(primary["model"] == model) & (primary["condition"] == "H2-CLEAN")]
        rows.extend(bootstrap_comparison(left, right, f"{model}__HB1-10DB_minus_CLEAN", "within_model_degradation"))
    return pd.DataFrame(rows)


def condition_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index(["model", "condition"])
    rows = []
    for condition in VALIDATION_CONDITIONS:
        original = primary.loc[("H2-D", condition)]
        augmented = primary.loc[("H2-NA", condition)]
        rows.append(
            {
                "condition": condition,
                "h2_d_f1": original.f1,
                "h2_na_f1": augmented.f1,
                "f1_difference": augmented.f1 - original.f1,
                "h2_d_false_alarms_per_hour": original.false_alarms_per_hour,
                "h2_na_false_alarms_per_hour": augmented.false_alarms_per_hour,
                "false_alarms_per_hour_difference": augmented.false_alarms_per_hour - original.false_alarms_per_hour,
            }
        )
    return pd.DataFrame(rows)


def hypothesis_decisions(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index(["model", "condition"])
    original_clean = primary.loc[("H2-D", "H2-CLEAN")]
    augmented_clean = primary.loc[("H2-NA", "H2-CLEAN")]
    original_hb1 = primary.loc[("H2-D", "H2-HB1-10DB")]
    augmented_hb1 = primary.loc[("H2-NA", "H2-HB1-10DB")]

    clean_f1_difference = float(augmented_clean.f1 - original_clean.f1)
    clean_far_difference = float(augmented_clean.false_alarms_per_hour - original_clean.false_alarms_per_hour)
    h86 = clean_f1_difference > -CLEAN_F1_LOSS_BOUND and clean_far_difference < CLEAN_FAR_INCREASE_BOUND

    hb1_f1_difference = float(augmented_hb1.f1 - original_hb1.f1)
    hb1_far_difference = float(augmented_hb1.false_alarms_per_hour - original_hb1.false_alarms_per_hour)
    h87 = hb1_f1_difference >= HB1_F1_IMPROVEMENT and hb1_far_difference <= -HB1_FAR_REDUCTION

    original_f1_gap = float(original_hb1.f1 - original_clean.f1)
    augmented_f1_gap = float(augmented_hb1.f1 - augmented_clean.f1)
    f1_gap_improvement = augmented_f1_gap - original_f1_gap
    original_far_gap = float(original_hb1.false_alarms_per_hour - original_clean.false_alarms_per_hour)
    augmented_far_gap = float(augmented_hb1.false_alarms_per_hour - augmented_clean.false_alarms_per_hour)
    far_gap_reduction = original_far_gap - augmented_far_gap
    h88 = f1_gap_improvement >= HB1_F1_IMPROVEMENT and far_gap_reduction >= HB1_FAR_REDUCTION

    h89 = clean_f1_difference >= MEANINGFUL_CLEAN_F1_GAIN and clean_far_difference <= 0.0
    robustness_advance = h86 and h87 and h88
    core_advance = robustness_advance and h89
    return pd.DataFrame(
        [
            {"hypothesis": "H8.6_clean_preservation", "value_1": clean_f1_difference, "value_2": clean_far_difference, "supported": h86, "decision": "pass" if h86 else "fail"},
            {"hypothesis": "H8.7_hb1_failure_mitigation", "value_1": hb1_f1_difference, "value_2": hb1_far_difference, "supported": h87, "decision": "pass" if h87 else "fail"},
            {"hypothesis": "H8.8_degradation_gap_reduction", "value_1": f1_gap_improvement, "value_2": far_gap_reduction, "supported": h88, "decision": "pass" if h88 else "fail"},
            {"hypothesis": "H8.9_meaningful_clean_advancement", "value_1": clean_f1_difference, "value_2": clean_far_difference, "supported": h89, "decision": "pass" if h89 else "fail"},
            {"hypothesis": "overall_robustness_method_advance", "value_1": float(robustness_advance), "value_2": np.nan, "supported": robustness_advance, "decision": "advance" if robustness_advance else "stop"},
            {"hypothesis": "overall_core_detector_advance", "value_1": float(core_advance), "value_2": np.nan, "supported": core_advance, "decision": "advance" if core_advance else "do_not_replace"},
        ]
    )


def validation_phase(result_code_commit: str) -> None:
    augmented_model, threshold_na, freeze = verify_train_freeze(result_code_commit)
    all_assignments = subject_assignments()
    validation_assignments = all_assignments[all_assignments["partition"] == "validation"].copy()
    scores = combined_validation_scores(validation_assignments, augmented_model)
    support = validation_support(scores)
    outputs = evaluate_validation(scores, support, reference_events(validation_assignments), threshold_na)
    bootstrap = paired_bootstrap(outputs["participants"])
    comparisons = condition_comparisons(outputs["metrics"])
    hypotheses = hypothesis_decisions(outputs["metrics"])
    verify_or_create_gzip_tsv(scores, validation_score_path())

    manifest_paths = [("source_train_clean_feature", feature_path(item.subject, "train", "HB-2")) for item in all_assignments[all_assignments["partition"] == "train"].itertuples(index=False)]
    train_manifest = pd.read_csv(output_dir() / "generated_train_feature_manifest_v0.1.tsv", sep="\t")
    manifest_paths.extend(("generated_train_noise_feature", data_parent() / item.path_relative_to_data_parent) for item in train_manifest.itertuples(index=False))
    manifest_paths.extend(("source_validation_clean_feature", validation_feature_path(item.subject, "H2-CLEAN")) for item in validation_assignments.itertuples(index=False))
    for condition in VALIDATION_CONDITIONS[1:]:
        manifest_paths.extend(("source_validation_noise_feature", validation_feature_path(item.subject, condition)) for item in validation_assignments.itertuples(index=False))
    manifest_paths.extend(
        [
            ("original_h2_model", source_model_path("H2-D")),
            ("augmented_h2_model", augmented_model_path()),
            ("source_validation_noise_scores", prior_validation_score_path()),
            ("train_oof_scores", oof_score_path()),
            ("validation_combined_scores", validation_score_path()),
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "artifact_role": role,
                "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in manifest_paths
        ]
    ).sort_values(["artifact_role", "path_relative_to_data_parent"])

    model_condition_counts = scores.groupby(["model", "condition"])["subject"].nunique()
    support_counts = support.groupby(["subject"])["supported_boundaries"].nunique()
    checks = pd.DataFrame(
        [
            {"check": "train_freeze_verified", "status": "pass", "detail": f"threshold={threshold_na:.2f}; model={freeze.iloc[0].model_sha256}"},
            {"check": "original_model_verified", "status": "pass" if sha256(source_model_path("H2-D")) == ORIGINAL_MODEL_SHA256 else "fail", "detail": ORIGINAL_MODEL_SHA256},
            {"check": "validation_membership", "status": "pass" if len(validation_assignments) == 20 and validation_assignments["pid"].nunique() == 16 else "fail", "detail": "20 recordings; 16 pid groups"},
            {"check": "complete_model_condition_scores", "status": "pass" if len(model_condition_counts) == 12 and model_condition_counts.eq(20).all() else "fail", "detail": f"rows={len(scores)}"},
            {"check": "identical_temporal_support", "status": "pass" if len(support) == 240 and support_counts.eq(1).all() else "fail", "detail": "two models x six conditions x 20 recordings"},
            {"check": "finite_probabilities", "status": "pass" if np.isfinite(scores["probability"]).all() and scores["probability"].between(0, 1).all() else "fail", "detail": "all probabilities finite and bounded"},
            {"check": "complete_event_outputs", "status": "pass" if len(outputs["metrics"]) == 48 else "fail", "detail": "2 models x 6 conditions x 2 memberships x 2 tolerances"},
            {"check": "paired_participant_bootstrap", "status": "pass" if len(bootstrap) == 16 and bootstrap["resamples"].eq(BOOTSTRAP_RESAMPLES).all() else "fail", "detail": "six between-model and two within-model comparisons"},
            {"check": "four_fixed_hypotheses", "status": "pass" if len(hypotheses) == 6 else "fail", "detail": "H8.6-H8.9 plus two overall decisions"},
            {"check": "external_manifest", "status": "pass" if len(manifest) == 535 and manifest["sha256"].str.len().eq(64).all() else "fail", "detail": "train, validation, models, and scores"},
            {"check": "test_closed", "status": "pass" if not manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any() else "fail", "detail": "no test artifact path"},
        ]
    )

    reviewed = {
        "validation_support_v0.1.tsv": support,
        "validation_predicted_events_v0.1.tsv": outputs["events"],
        "validation_event_metrics_v0.1.tsv": outputs["metrics"],
        "validation_event_recordings_v0.1.tsv": outputs["recordings"],
        "validation_event_participants_v0.1.tsv": outputs["participants"],
        "validation_event_matches_v0.1.tsv": outputs["matches"],
        "validation_condition_comparisons_v0.1.tsv": comparisons,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": hypotheses,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "validation_phase_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)

    primary = outputs["metrics"][(outputs["metrics"]["membership"] == "primary") & (outputs["metrics"]["tolerance_sec"] == 15.0)]
    metric_rows = [f"| {item.model} | {item.condition} | {item.threshold:.2f} | {item.precision:.4f} | {item.recall:.4f} | {item.f1:.4f} | {item.false_alarms_per_hour:.4f} |" for item in primary.itertuples(index=False)]
    decision_rows = [f"| {item.hypothesis} | {item.value_1:+.4f} | {item.value_2:+.4f} | {item.decision} |" for item in hypotheses.itertuples(index=False)]
    text = "\n".join(
        [
            "# Block 8 Train-Only Noise Augmentation v0.1",
            "",
            "**Work date:** 2026-09-12",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            f"**H2-NA train-OOF threshold:** `{threshold_na:.2f}`",
            f"**H2-NA model SHA-256:** `{freeze.iloc[0].model_sha256}`",
            "**Validation role:** Reused development partition",
            "**Test data accessed:** No",
            "",
            "## Primary Validation Results",
            "",
            "| Model | Condition | Threshold | Precision | Recall | F1 | False alarms/hour |",
            "|---|---|---:|---:|---:|---:|---:|",
            *metric_rows,
            "",
            "## Frozen Decisions",
            "",
            "| Hypothesis | Value 1 | Value 2 | Decision |",
            "|---|---:|---:|---|",
            *decision_rows,
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} validation-phase checks passed. This experiment evaluates one fixed Gaussian-noise augmentation mechanism on a reused validation cohort. It does not establish clinical performance, natural artefact robustness, or independent generalization.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)
    print(primary[["model", "condition", "threshold", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(hypotheses.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one validation-phase check failed")


# Section 7: command entry point

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "validation"], required=True)
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")
    if args.phase == "train":
        train_phase(args.result_code_commit)
    else:
        validation_phase(args.result_code_commit)


if __name__ == "__main__":
    main()
