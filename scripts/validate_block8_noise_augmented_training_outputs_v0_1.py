"""Independently validate Block 8 noise-augmented training outputs."""

from __future__ import annotations

import hashlib
import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
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
)


# Section 1: independent fixed configuration

EXPERIMENT_DIR = "2026-09-12_block8_noise_augmented_training_v0.1"
RESULT_DERIVED_DIR = "block8_noise_augmented_training_v0.1"
VALIDATION_NOISE_DERIVED_DIR = "block8_raw_signal_noise_v0.1"
BASE_SEED = 20260912
FOLDS = 5
BOOTSTRAP_RESAMPLES = 2000
THRESHOLDS = np.arange(1, 100, dtype=float) / 100.0
EPOCH_SEC = 30.0
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
ORIGINAL_MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
ORIGINAL_THRESHOLD = 0.96
FEATURE_TOLERANCE = 1e-6
SNR_TOLERANCE_DB = 0.02
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


# Section 2: paths and comparison helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def result_dir() -> Path:
    return data_parent() / "derived" / RESULT_DERIVED_DIR


def train_feature_path(subject: str, condition: str) -> Path:
    if condition == "TRAIN-CLEAN":
        return feature_path(subject, "train", "HB-2")
    return result_dir() / "recording_features" / "train" / TRAIN_FOLDERS[condition] / f"{subject}_features_v0.1.npz"


def validation_feature_path(subject: str, condition: str) -> Path:
    if condition == "H2-CLEAN":
        return feature_path(subject, "validation", "HB-2")
    return data_parent() / "derived" / VALIDATION_NOISE_DERIVED_DIR / "recording_features" / VALIDATION_FOLDERS[condition] / f"{subject}_features_v0.1.npz"


def augmented_model_path() -> Path:
    return result_dir() / "models" / "h2_na_model_v0.1.joblib"


def oof_score_path() -> Path:
    return result_dir() / "candidate_scores" / "train_oof_clean_scores_v0.1.tsv.gz"


def prior_validation_score_path() -> Path:
    return data_parent() / "derived" / VALIDATION_NOISE_DERIVED_DIR / "candidate_scores" / "validation_noise_scores_v0.1.tsv.gz"


def validation_score_path() -> Path:
    return result_dir() / "candidate_scores" / "validation_model_condition_scores_v0.1.tsv.gz"


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def record(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def frames_match(left: pd.DataFrame, right: pd.DataFrame, sort_by: list[str]) -> bool:
    try:
        left = left.sort_values(sort_by).reset_index(drop=True)
        right = right.sort_values(sort_by).reset_index(drop=True)
        assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)
        return True
    except (AssertionError, ValueError, KeyError):
        return False


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed validation output changed: {path}")
        return
    path.write_text(expected, encoding="utf-8")


def load_feature(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["stage"].astype(np.int8),
            values["features"].astype(np.float32),
            values["feature_names"].astype(str).tolist(),
        )


# Section 3: reconstruct all train raw-signal perturbations

def valid_sample_mask(events: pd.DataFrame, samples: int) -> np.ndarray:
    mask = np.zeros(samples, dtype=bool)
    for item in events.itertuples(index=False):
        start = int(round(float(item.onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        if start >= 0 and stop <= samples:
            mask[start:stop] = True
    return mask


def noise_seed(subject: str, channel: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{subject}|{channel}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def reconstruct_train_noise(rows: list[dict], train_assignments: pd.DataFrame) -> None:
    _, hb_scaler, _ = scaler_maps()
    sos = filter_sos()
    basis_rows = []
    calibration_rows = []
    clean_rows = []
    generated_rows = []
    arrays_match = True
    maximum_array_difference = 0.0

    for index, item in enumerate(train_assignments.itertuples(index=False), start=1):
        print(f"Independent train-noise reconstruction {index}/{len(train_assignments)}: {item.subject}", flush=True)
        events = valid_events(item.subject)
        raw = read_uv(item.subject, "headband", HB2)
        clean_filtered = filter_resample(raw, sos)
        mask = valid_sample_mask(events, clean_filtered.shape[1])
        if not mask.any():
            raise ValueError(f"No valid calibration samples: {item.subject}")

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
        source = load_feature(source_path)
        timing_match = np.array_equal(clean[0], source[0]) and np.array_equal(clean[1], source[1])
        schema_match = clean[3] == source[3]
        clean_difference = float(np.max(np.abs(clean[2] - source[2])))
        clean_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "timing_and_stage_match": timing_match,
                "feature_schema_match": schema_match,
                "maximum_absolute_feature_difference": clean_difference,
                "tolerance": FEATURE_TOLERANCE,
                "reproduction_pass": timing_match and schema_match and clean_difference <= FEATURE_TOLERANCE,
                "source_feature_sha256": sha256(source_path),
            }
        )

        scales = {}
        for channel_index, channel in enumerate(HB2):
            clean_rms = float(np.sqrt(np.mean(np.square(clean_filtered[channel_index, mask]))))
            noise_rms = float(np.sqrt(np.mean(np.square(noise_filtered[channel_index, mask]))))
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
                added = scales[(channel, target_snr)] * noise_filtered[channel_index]
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
                        "noise_scale": scales[(channel, target_snr)],
                        "valid_samples": int(mask.sum()),
                        "calibration_pass": abs(achieved - target_snr) <= SNR_TOLERANCE_DB,
                    }
                )
            expected = epoch_features(normalize_signal(degraded, HB2, hb_scaler), HB2, events)
            path = train_feature_path(item.subject, condition)
            stored = load_feature(path)
            difference = float(np.max(np.abs(expected[2] - stored[2])))
            maximum_array_difference = max(maximum_array_difference, difference)
            arrays_match &= (
                np.array_equal(expected[0], stored[0])
                and np.array_equal(expected[1], stored[1])
                and expected[3] == stored[3]
                and difference <= FEATURE_TOLERANCE
            )
            centers, _ = context_matrix(stored[0], stored[2])
            generated_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "condition": condition,
                    "epochs": len(stored[0]),
                    "context_rows": len(centers),
                    "feature_dimensions": stored[2].shape[1],
                    "all_features_finite": bool(np.isfinite(stored[2]).all()),
                    "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    comparisons = [
        (pd.DataFrame(basis_rows), "train_noise_basis_manifest_v0.1.tsv", ["subject", "channel"]),
        (pd.DataFrame(calibration_rows), "train_noise_calibration_v0.1.tsv", ["condition", "subject", "channel"]),
        (pd.DataFrame(clean_rows), "train_clean_feature_reproduction_v0.1.tsv", ["subject"]),
        (pd.DataFrame(generated_rows), "generated_train_feature_manifest_v0.1.tsv", ["condition", "subject"]),
    ]
    controls_match = all(
        frames_match(frame, pd.read_csv(output_dir() / name, sep="\t"), sort_by)
        for frame, name, sort_by in comparisons
    )
    record(rows, "train_noise_arrays_reconstructed", arrays_match, f"328 arrays; maximum difference={maximum_array_difference:.12g}")
    record(rows, "train_noise_controls_recomputed", controls_match, "basis, calibration, clean reproduction, and feature manifest")


# Section 4: independently rebuild train matrices and threshold freeze

def build_augmented_matrix(candidates: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrices = []
    metadata_rows = []
    dropped_rows = []
    for condition in TRAIN_CONDITIONS:
        for subject, group in candidates.groupby("subject", sort=True):
            onsets, _, features, _ = load_feature(train_feature_path(subject, condition))
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
                    metadata_rows.append(row)
    columns = list(candidates.columns) + ["augmentation_condition"]
    metadata = pd.DataFrame(metadata_rows, columns=columns)
    dropped = pd.DataFrame(dropped_rows, columns=columns + ["drop_reason"])
    construction = metadata.groupby(["augmentation_condition", "label", "source_tier"], as_index=False).size().rename(columns={"size": "retained_rows"})
    drop_counts = dropped.groupby(["augmentation_condition", "label", "source_tier"], as_index=False).size().rename(columns={"size": "dropped_rows"})
    construction = construction.merge(drop_counts, on=["augmentation_condition", "label", "source_tier"], how="left")
    construction["dropped_rows"] = construction["dropped_rows"].fillna(0).astype(int)
    identity = metadata.groupby(["sample_id", "subject", "pid", "label", "source_tier"], as_index=False).agg(augmentation_conditions=("augmentation_condition", "nunique"))
    identity["complete_five_condition_multiplicity"] = identity["augmentation_conditions"].eq(5)
    return np.vstack(matrices), metadata, construction, identity


def build_model():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, tol=1e-4, random_state=BASE_SEED),
    )


def fit_model(matrix: np.ndarray, labels: np.ndarray) -> tuple[object, int]:
    model = build_model()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(matrix, labels)
    return model, sum(issubclass(item.category, ConvergenceWarning) for item in caught)


def score_clean(assignments: pd.DataFrame, model, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for item in assignments.itertuples(index=False):
        onsets, _, features, _ = load_feature(feature_path(item.subject, "train", "HB-2"))
        centers, matrix = context_matrix(onsets, features)
        probability = model.predict_proba(matrix)[:, 1]
        score_rows.append(pd.DataFrame({"model": "H2-NA-OOF", "condition": "TRAIN-CLEAN", "partition": "train", "fold": fold, "subject": item.subject, "pid": int(item.pid), "candidate_time_sec": centers, "probability": probability}))
        support_rows.append({"partition": "train", "fold": fold, "subject": item.subject, "pid": int(item.pid), "supported_boundaries": len(centers), "supported_hours": len(centers) * EPOCH_SEC / 3600.0})
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
    columns = ["model", "condition", "partition", "subject", "pid"]
    for keys, group in marked.groupby(columns, sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        starts = [0]
        starts.extend((np.flatnonzero(np.diff(group["candidate_time_sec"].to_numpy(dtype=float)) > EPOCH_SEC + 1e-6) + 1).tolist())
        stops = starts[1:] + [len(group)]
        for start, stop in zip(starts, stops):
            run = group.iloc[start:stop]
            maximum = float(run["probability"].max())
            best = run[np.isclose(run["probability"], maximum)].sort_values("candidate_time_sec").iloc[0]
            rows.append({**dict(zip(columns, keys)), "event_time_sec": float(best.candidate_time_sec), "probability": float(best.probability), "threshold": float(threshold), "run_candidates": len(run)})
    return pd.DataFrame(rows, columns=columns + ["event_time_sec", "probability", "threshold", "run_candidates"])


def threshold_curve(scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible, ignored = local_event_inputs(references, "primary")
    local_support = support[["subject", "pid", "supported_hours"]]
    rows = []
    for threshold in THRESHOLDS:
        alarms = collapse_alarms(scores, float(threshold))
        _, _, _, summary = evaluate_events(eligible, alarms[["subject", "pid", "event_time_sec"]], ignored, local_support, 15.0)
        rows.append({"model": "H2-NA", "partition": "train_oof", "membership": "primary", "tolerance_sec": 15.0, "threshold": float(threshold), **summary})
    curve = pd.DataFrame(rows)
    selected = curve.sort_values(["f1", "false_alarms_per_hour", "recall", "threshold"], ascending=[False, True, False, False], kind="stable").iloc[[0]].copy()
    selected["selection_rule"] = "max_f1_then_min_far_then_max_recall_then_max_threshold"
    return curve, selected


def reconstruct_train_fit(rows: list[dict], train_assignments: pd.DataFrame) -> tuple[object, float]:
    candidates = labeled_candidates(train_assignments)
    matrix, metadata, construction, identity = build_augmented_matrix(candidates)
    construction_ok = frames_match(construction, pd.read_csv(output_dir() / "augmented_labeled_construction_v0.1.tsv", sep="\t"), ["augmentation_condition", "label", "source_tier"])
    identity_ok = frames_match(identity, pd.read_csv(output_dir() / "augmented_labeled_identity_v0.1.tsv", sep="\t"), ["sample_id"])
    record(rows, "augmented_labeled_rows_recomputed", construction_ok and identity_ok, f"base rows={len(identity)}; augmented rows={len(metadata)}")

    labels = metadata["label"].to_numpy(dtype=int)
    groups = metadata["pid"].to_numpy(dtype=int)
    splitter = GroupKFold(n_splits=FOLDS)
    score_rows = []
    support_rows = []
    fold_rows = []
    fit_rows = []
    for fold, (fit_indices, heldout_indices) in enumerate(splitter.split(matrix, labels, groups), start=1):
        fit_pids = set(groups[fit_indices])
        heldout_pids = set(groups[heldout_indices])
        model, convergence = fit_model(matrix[fit_indices], labels[fit_indices])
        heldout = train_assignments[train_assignments["pid"].isin(heldout_pids)]
        scores, support = score_clean(heldout, model, fold)
        score_rows.append(scores)
        support_rows.append(support)
        fold_rows.extend({"pid": int(pid), "fold": fold, "role": "heldout_threshold_selection"} for pid in sorted(heldout_pids))
        fit_rows.append({"fit": f"fold_{fold}", "fit_pid": len(fit_pids), "heldout_pid": len(heldout_pids), "fit_rows": len(fit_indices), "fit_positive": int(labels[fit_indices].sum()), "fit_negative": int((labels[fit_indices] == 0).sum()), "convergence_warning_count": convergence, "maximum_iterations_used": int(model.named_steps["logisticregression"].n_iter_.max())})

    oof_scores = pd.concat(score_rows, ignore_index=True)
    oof_support = pd.concat(support_rows, ignore_index=True)
    fold_assignments = pd.DataFrame(fold_rows)
    curve, selected = threshold_curve(oof_scores, oof_support, reference_events(train_assignments))
    oof_ok = frames_match(oof_scores, pd.read_csv(oof_score_path(), sep="\t"), ["fold", "subject", "candidate_time_sec"])
    support_ok = frames_match(oof_support, pd.read_csv(output_dir() / "train_oof_support_v0.1.tsv", sep="\t"), ["fold", "subject"])
    fold_ok = frames_match(fold_assignments, pd.read_csv(output_dir() / "train_oof_fold_assignments_v0.1.tsv", sep="\t"), ["pid"])
    curve_ok = frames_match(curve, pd.read_csv(output_dir() / "train_oof_threshold_curve_v0.1.tsv", sep="\t"), ["threshold"])
    selection_ok = frames_match(selected, pd.read_csv(output_dir() / "train_oof_threshold_selection_v0.1.tsv", sep="\t"), ["threshold"])
    record(rows, "train_oof_threshold_recomputed", oof_ok and support_ok and fold_ok and curve_ok and selection_ok, f"threshold={float(selected.iloc[0].threshold):.2f}; rows={len(oof_scores)}")

    final_model, convergence = fit_model(matrix, labels)
    fit_rows.append({"fit": "final_all_train", "fit_pid": metadata["pid"].nunique(), "heldout_pid": 0, "fit_rows": len(labels), "fit_positive": int(labels.sum()), "fit_negative": int((labels == 0).sum()), "convergence_warning_count": convergence, "maximum_iterations_used": int(final_model.named_steps["logisticregression"].n_iter_.max())})
    fit_ok = frames_match(pd.DataFrame(fit_rows), pd.read_csv(output_dir() / "model_fit_record_v0.1.tsv", sep="\t"), ["fit"])
    stored_model = joblib.load(augmented_model_path())
    model_ok = all(np.allclose(left, right, rtol=0.0, atol=1e-12) for left, right in zip(model_arrays(final_model), model_arrays(stored_model)))
    freeze = pd.read_csv(output_dir() / "train_model_freeze_v0.1.tsv", sep="\t")
    freeze_ok = float(freeze.iloc[0].threshold) == float(selected.iloc[0].threshold) and freeze.iloc[0].model_sha256 == sha256(augmented_model_path())
    record(rows, "final_model_and_freeze_recomputed", fit_ok and model_ok and freeze_ok, str(freeze.iloc[0].model_sha256))
    return final_model, float(selected.iloc[0].threshold)


# Section 5: independently rebuild validation scores and events

def validation_scores(assignments: pd.DataFrame, model) -> pd.DataFrame:
    prior = pd.read_csv(prior_validation_score_path(), sep="\t").rename(columns={"comparator": "condition"})
    prior["model"] = "H2-D"
    frames = [prior[["model", "condition", "partition", "subject", "pid", "candidate_time_sec", "probability"]]]
    for condition in VALIDATION_CONDITIONS:
        for item in assignments.itertuples(index=False):
            onsets, _, features, _ = load_feature(validation_feature_path(item.subject, condition))
            centers, matrix = context_matrix(onsets, features)
            frames.append(pd.DataFrame({"model": "H2-NA", "condition": condition, "partition": "validation", "subject": item.subject, "pid": int(item.pid), "candidate_time_sec": centers, "probability": model.predict_proba(matrix)[:, 1]}))
    return pd.concat(frames, ignore_index=True)


def support_table(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in scores.groupby(["model", "condition", "subject", "pid"], sort=True):
        rows.append({"model": keys[0], "condition": keys[1], "partition": "validation", "subject": keys[2], "pid": int(keys[3]), "supported_boundaries": len(group), "supported_hours": len(group) * EPOCH_SEC / 3600.0})
    return pd.DataFrame(rows)


def evaluate_all(scores: pd.DataFrame, support: pd.DataFrame, references: pd.DataFrame, threshold_na: float) -> dict[str, pd.DataFrame]:
    thresholds = {"H2-D": ORIGINAL_THRESHOLD, "H2-NA": threshold_na}
    event_rows = []
    metric_rows = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for model in ["H2-D", "H2-NA"]:
        for condition in VALIDATION_CONDITIONS:
            local_scores = scores[(scores["model"] == model) & (scores["condition"] == condition)]
            local_support = support[(support["model"] == model) & (support["condition"] == condition)][["subject", "pid", "supported_hours"]]
            alarms = collapse_alarms(local_scores, thresholds[model])
            event_rows.append(alarms)
            for membership in MEMBERSHIPS:
                eligible, ignored = local_event_inputs(references, membership)
                for tolerance in TOLERANCES:
                    recordings, participants, matches, summary = evaluate_events(eligible, alarms[["subject", "pid", "event_time_sec"]], ignored, local_support, tolerance)
                    config = {"model": model, "condition": condition, "partition": "validation", "membership": membership, "tolerance_sec": tolerance, "threshold": thresholds[model]}
                    metric_rows.append({**config, **summary})
                    for frame, collection in [(recordings, recording_rows), (participants, participant_rows), (matches, match_rows)]:
                        if len(frame):
                            local = frame.copy()
                            for key, value in reversed(list(config.items())):
                                if key not in local.columns:
                                    local.insert(0, key, value)
                            collection.append(local)
    return {"events": pd.concat(event_rows, ignore_index=True), "metrics": pd.DataFrame(metric_rows), "recordings": pd.concat(recording_rows, ignore_index=True), "participants": pd.concat(participant_rows, ignore_index=True), "matches": pd.concat(match_rows, ignore_index=True)}


def aggregate(frame: pd.DataFrame) -> dict:
    return metric_values(int(frame["true_positive"].sum()), int(frame["false_positive"].sum()), int(frame["false_negative"].sum()), float(frame["supported_hours"].sum()))


def bootstrap_pair(left: pd.DataFrame, right: pd.DataFrame, comparison: str, kind: str) -> list[dict]:
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    paired = left[columns].merge(right[columns], on="pid", suffixes=("_left", "_right"), validate="one_to_one")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
        values = {}
        for side in ["left", "right"]:
            values[side] = metric_values(int(sample[f"true_positive_{side}"].sum()), int(sample[f"false_positive_{side}"].sum()), int(sample[f"false_negative_{side}"].sum()), float(sample[f"supported_hours_{side}"].sum()))
        samples.append({"event_f1_difference": values["left"]["f1"] - values["right"]["f1"], "false_alarms_per_hour_difference": values["left"]["false_alarms_per_hour"] - values["right"]["false_alarms_per_hour"]})
    sample_frame = pd.DataFrame(samples)
    points_left = aggregate(left)
    points_right = aggregate(right)
    points = {"event_f1_difference": points_left["f1"] - points_right["f1"], "false_alarms_per_hour_difference": points_left["false_alarms_per_hour"] - points_right["false_alarms_per_hour"]}
    return [{"comparison_kind": kind, "comparison": comparison, "metric": metric, "point_difference": value, "resamples": BOOTSTRAP_RESAMPLES, "seed": BASE_SEED, "lower_95": float(sample_frame[metric].quantile(0.025)), "median": float(sample_frame[metric].quantile(0.5)), "upper_95": float(sample_frame[metric].quantile(0.975))} for metric, value in points.items()]


def bootstrap_all(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[(participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)]
    rows = []
    for condition in VALIDATION_CONDITIONS:
        left = primary[(primary["model"] == "H2-NA") & (primary["condition"] == condition)]
        right = primary[(primary["model"] == "H2-D") & (primary["condition"] == condition)]
        rows.extend(bootstrap_pair(left, right, f"H2-NA_minus_H2-D__{condition}", "between_model"))
    for model in ["H2-D", "H2-NA"]:
        left = primary[(primary["model"] == model) & (primary["condition"] == "H2-HB1-10DB")]
        right = primary[(primary["model"] == model) & (primary["condition"] == "H2-CLEAN")]
        rows.extend(bootstrap_pair(left, right, f"{model}__HB1-10DB_minus_CLEAN", "within_model_degradation"))
    return pd.DataFrame(rows)


def comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index(["model", "condition"])
    rows = []
    for condition in VALIDATION_CONDITIONS:
        original = primary.loc[("H2-D", condition)]
        augmented = primary.loc[("H2-NA", condition)]
        rows.append({"condition": condition, "h2_d_f1": original.f1, "h2_na_f1": augmented.f1, "f1_difference": augmented.f1 - original.f1, "h2_d_false_alarms_per_hour": original.false_alarms_per_hour, "h2_na_false_alarms_per_hour": augmented.false_alarms_per_hour, "false_alarms_per_hour_difference": augmented.false_alarms_per_hour - original.false_alarms_per_hour})
    return pd.DataFrame(rows)


def hypotheses(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index(["model", "condition"])
    od_clean = primary.loc[("H2-D", "H2-CLEAN")]
    na_clean = primary.loc[("H2-NA", "H2-CLEAN")]
    od_hb1 = primary.loc[("H2-D", "H2-HB1-10DB")]
    na_hb1 = primary.loc[("H2-NA", "H2-HB1-10DB")]
    clean_f1 = float(na_clean.f1 - od_clean.f1)
    clean_far = float(na_clean.false_alarms_per_hour - od_clean.false_alarms_per_hour)
    hb1_f1 = float(na_hb1.f1 - od_hb1.f1)
    hb1_far = float(na_hb1.false_alarms_per_hour - od_hb1.false_alarms_per_hour)
    f1_gap_improvement = float((na_hb1.f1 - na_clean.f1) - (od_hb1.f1 - od_clean.f1))
    far_gap_reduction = float((od_hb1.false_alarms_per_hour - od_clean.false_alarms_per_hour) - (na_hb1.false_alarms_per_hour - na_clean.false_alarms_per_hour))
    h86 = clean_f1 > -CLEAN_F1_LOSS_BOUND and clean_far < CLEAN_FAR_INCREASE_BOUND
    h87 = hb1_f1 >= HB1_F1_IMPROVEMENT and hb1_far <= -HB1_FAR_REDUCTION
    h88 = f1_gap_improvement >= HB1_F1_IMPROVEMENT and far_gap_reduction >= HB1_FAR_REDUCTION
    h89 = clean_f1 >= MEANINGFUL_CLEAN_F1_GAIN and clean_far <= 0.0
    robustness = h86 and h87 and h88
    core = robustness and h89
    return pd.DataFrame([
        {"hypothesis": "H8.6_clean_preservation", "value_1": clean_f1, "value_2": clean_far, "supported": h86, "decision": "pass" if h86 else "fail"},
        {"hypothesis": "H8.7_hb1_failure_mitigation", "value_1": hb1_f1, "value_2": hb1_far, "supported": h87, "decision": "pass" if h87 else "fail"},
        {"hypothesis": "H8.8_degradation_gap_reduction", "value_1": f1_gap_improvement, "value_2": far_gap_reduction, "supported": h88, "decision": "pass" if h88 else "fail"},
        {"hypothesis": "H8.9_meaningful_clean_advancement", "value_1": clean_f1, "value_2": clean_far, "supported": h89, "decision": "pass" if h89 else "fail"},
        {"hypothesis": "overall_robustness_method_advance", "value_1": float(robustness), "value_2": np.nan, "supported": robustness, "decision": "advance" if robustness else "stop"},
        {"hypothesis": "overall_core_detector_advance", "value_1": float(core), "value_2": np.nan, "supported": core, "decision": "advance" if core else "do_not_replace"},
    ])


# Section 6: manifests and execution

def validate_manifests(rows: list[dict]) -> None:
    for name, expected in [("train_external_artifact_manifest_v0.1.tsv", 412), ("external_artifact_manifest_v0.1.tsv", 535)]:
        manifest = pd.read_csv(output_dir() / name, sep="\t")
        valid = len(manifest) == expected
        for item in manifest.itertuples(index=False):
            path = data_parent() / item.path_relative_to_data_parent
            valid &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
        record(rows, f"{name}_rehashed", valid, f"{expected} external artifacts")
    validation_manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    no_test = not validation_manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any()
    record(rows, "test_artifacts_absent", no_test, "no test path in manifest")


def main() -> None:
    rows: list[dict] = []
    all_assignments = subject_assignments()
    train_assignments = all_assignments[all_assignments["partition"] == "train"].copy()
    validation_assignments = all_assignments[all_assignments["partition"] == "validation"].copy()
    record(rows, "partition_membership", len(train_assignments) == 82 and train_assignments["pid"].nunique() == 64 and len(validation_assignments) == 20 and validation_assignments["pid"].nunique() == 16 and not set(train_assignments["pid"]) & set(validation_assignments["pid"]), "82/64 train and 20/16 validation; no pid overlap")
    record(rows, "original_model_rehashed", sha256(source_model_path("H2-D")) == ORIGINAL_MODEL_SHA256, ORIGINAL_MODEL_SHA256)
    reconstruct_train_noise(rows, train_assignments)
    final_model, threshold_na = reconstruct_train_fit(rows, train_assignments)

    scores = validation_scores(validation_assignments, final_model)
    saved_scores = pd.read_csv(validation_score_path(), sep="\t")
    record(rows, "validation_probabilities_recomputed", frames_match(scores, saved_scores, ["model", "condition", "subject", "candidate_time_sec"]), f"rows={len(scores)}")
    support = support_table(scores)
    saved_support = pd.read_csv(output_dir() / "validation_support_v0.1.tsv", sep="\t")
    record(rows, "validation_support_recomputed", frames_match(support, saved_support, ["model", "condition", "subject"]), "two models x six conditions")

    outputs = evaluate_all(scores, support, reference_events(validation_assignments), threshold_na)
    output_specs = {
        "validation_predicted_events_v0.1.tsv": (outputs["events"], ["model", "condition", "subject", "event_time_sec"]),
        "validation_event_metrics_v0.1.tsv": (outputs["metrics"], ["model", "condition", "membership", "tolerance_sec"]),
        "validation_event_recordings_v0.1.tsv": (outputs["recordings"], ["model", "condition", "membership", "tolerance_sec", "subject"]),
        "validation_event_participants_v0.1.tsv": (outputs["participants"], ["model", "condition", "membership", "tolerance_sec", "pid"]),
        "validation_event_matches_v0.1.tsv": (outputs["matches"], ["model", "condition", "membership", "tolerance_sec", "subject", "prediction_time_sec"]),
    }
    event_ok = all(frames_match(frame, pd.read_csv(output_dir() / name, sep="\t"), keys) for name, (frame, keys) in output_specs.items())
    record(rows, "validation_events_recomputed", event_ok, "events, metrics, recordings, participants, and matches")

    comparison_table = comparisons(outputs["metrics"])
    bootstrap = bootstrap_all(outputs["participants"])
    decision_table = hypotheses(outputs["metrics"])
    secondary_ok = (
        frames_match(comparison_table, pd.read_csv(output_dir() / "validation_condition_comparisons_v0.1.tsv", sep="\t"), ["condition"])
        and frames_match(bootstrap, pd.read_csv(output_dir() / "paired_participant_bootstrap_v0.1.tsv", sep="\t"), ["comparison", "metric"])
        and frames_match(decision_table, pd.read_csv(output_dir() / "hypothesis_decisions_v0.1.tsv", sep="\t"), ["hypothesis"])
    )
    record(rows, "comparisons_and_decisions_recomputed", secondary_ok, "condition contrasts, 16 bootstrap rows, and six decisions")
    validate_manifests(rows)

    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")
    text = "\n".join([
        "# Block 8 Noise-Augmented Training Output Validation",
        "",
        "**Validation date:** 2026-09-12",
        "**Method:** Independent EDF perturbation reconstruction, participant-grouped refitting, threshold selection, event reconstruction, and artifact rehashing",
        "",
        f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} checks passed.",
        "",
        "The validator does not search a new configuration or access test artifacts.",
        "",
    ])
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", text)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one independent noise-augmentation check failed")


if __name__ == "__main__":
    main()
