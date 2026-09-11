"""Run the frozen Block 8 validation-only raw-signal noise experiment."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import spearmanr

from reviewed_output import verify_or_create_tsv
from run_block7_transfer_validation_v0_1 import context_matrix, normalize_signal, scaler_maps
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


# Section 1: frozen configuration

VERSION = "v0.1"
EXPERIMENT_DIR = "2026-09-11_block8_raw_signal_noise_v0.1"
SOURCE_DERIVED_DIR = "block7_transfer_validation_v0.1"
RESULT_DERIVED_DIR = "block8_raw_signal_noise_v0.1"
PROTOCOL_COMMIT = "4055f2c"
PARTITION = "validation"
MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
THRESHOLD = 0.96
EPOCH_SEC = 30.0
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
BASE_SEED = 20260911
BOOTSTRAP_RESAMPLES = 2000
SNR_TOLERANCE_DB = 0.02
CLEAN_FEATURE_TOLERANCE = 1e-6
PROBABILITY_TOLERANCE = 1e-12
MATERIAL_F1_DROP = 0.03
MATERIAL_FAR_INCREASE = 0.50
ORDER_TOLERANCE = 1e-12

CONDITIONS = {
    "H2-CLEAN": {},
    "H2-BOTH-20DB": {"HB_1": 20.0, "HB_2": 20.0},
    "H2-BOTH-10DB": {"HB_1": 10.0, "HB_2": 10.0},
    "H2-BOTH-0DB": {"HB_1": 0.0, "HB_2": 0.0},
    "H2-HB1-10DB": {"HB_1": 10.0},
    "H2-HB2-10DB": {"HB_2": 10.0},
}
CONDITION_FOLDERS = {
    "H2-BOTH-20DB": "both_20db",
    "H2-BOTH-10DB": "both_10db",
    "H2-BOTH-0DB": "both_0db",
    "H2-HB1-10DB": "hb1_10db",
    "H2-HB2-10DB": "hb2_10db",
}
MODEL_ROLES = {
    "H2-CLEAN": "frozen_clean_wearable",
    "H2-BOTH-20DB": "both_channels_white_noise_20db",
    "H2-BOTH-10DB": "both_channels_white_noise_10db",
    "H2-BOTH-0DB": "both_channels_white_noise_0db",
    "H2-HB1-10DB": "hb1_white_noise_10db",
    "H2-HB2-10DB": "hb2_white_noise_10db",
}


# Section 2: paths and immutable file helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def source_dir() -> Path:
    return data_parent() / "derived" / SOURCE_DERIVED_DIR


def result_dir() -> Path:
    return data_parent() / "derived" / RESULT_DERIVED_DIR


def source_feature_path(subject: str) -> Path:
    return source_dir() / "recording_features" / PARTITION / "hb2" / f"{subject}_features_v0.1.npz"


def generated_feature_path(subject: str, comparator: str) -> Path:
    return result_dir() / "recording_features" / CONDITION_FOLDERS[comparator] / f"{subject}_features_v0.1.npz"


def model_path() -> Path:
    return source_dir() / "models" / "h2_d_model_v0.1.joblib"


def source_score_path() -> Path:
    return source_dir() / "candidate_scores" / "validation_continuous_scores_v0.1.tsv.gz"


def score_path() -> Path:
    return result_dir() / "candidate_scores" / "validation_noise_scores_v0.1.tsv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            raise RuntimeError(f"External score artifact changed: {path}")
        temporary.unlink()
        return
    temporary.replace(path)


# Section 3: frozen validation membership and references

def validation_assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in split[split["partition"] == PARTITION].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": PARTITION})
    result = pd.DataFrame(rows).sort_values("subject")
    if len(result) != 20 or result["pid"].nunique() != 16:
        raise ValueError("Frozen validation assignment must contain 20 recordings and 16 pid groups")
    if set(result["partition"]) != {PARTITION} or result["subject"].duplicated().any():
        raise ValueError("Invalid validation-only assignment")
    return result


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def reference_events(assignments: pd.DataFrame) -> pd.DataFrame:
    membership = pd.read_csv(
        repo_root() / "labels/quality_analysis_membership_v0.1/transition_analysis_membership_v0.1.tsv",
        sep="\t",
    )
    quality = pd.read_csv(
        repo_root() / "labels/signal_quality_flags_v0.3/transition_window_quality_flags_v0.3.tsv",
        sep="\t",
        usecols=["transition_id", "nominal_boundary_sec"],
    )
    result = membership[
        membership["subject"].isin(set(assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
        & membership["partition"].eq(PARTITION)
    ].merge(quality, on="transition_id", validate="one_to_one")
    result["event_time_sec"] = result["nominal_boundary_sec"].astype(float)
    if set(result["partition"]) != {PARTITION}:
        raise ValueError("Reference events escaped validation")
    return result


def local_event_inputs(reference: pd.DataFrame, membership: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = "primary_analysis_eligible" if membership == "primary" else "expanded_quality_analysis_eligible"
    eligible = truth(reference[column])
    columns = ["subject", "pid", "event_time_sec"]
    return reference.loc[eligible, columns], reference.loc[~eligible, columns]


# Section 4: deterministic raw-domain noise propagated through preprocessing

def valid_sample_mask(events: pd.DataFrame, samples: int) -> np.ndarray:
    mask = np.zeros(samples, dtype=bool)
    for item in events.itertuples(index=False):
        start = int(round(float(item.onset) * OUTPUT_SFREQ))
        stop = start + EPOCH_SAMPLES
        if start >= 0 and stop <= samples:
            mask[start:stop] = True
    if not mask.any():
        raise ValueError("No valid samples available for SNR calibration")
    return mask


def noise_seed(subject: str, channel: str) -> int:
    digest = hashlib.sha256(f"{BASE_SEED}|{subject}|{channel}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def load_source_feature(subject: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(source_feature_path(subject), allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["stage"].astype(np.int8),
            values["features"].astype(np.float32),
            values["feature_names"].astype(str).tolist(),
        )


def generate_noise_features(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, hb_scaler, _ = scaler_maps()
    sos = filter_sos()
    generated_rows = []
    calibration_rows = []
    basis_rows = []
    clean_rows = []
    for index, item in enumerate(assignments.itertuples(index=False), start=1):
        print(f"Noise feature generation {index}/{len(assignments)}: {item.subject}", flush=True)
        events = valid_events(item.subject)
        raw = read_uv(item.subject, "headband", HB2)
        clean_filtered = filter_resample(raw, sos)
        mask = valid_sample_mask(events, clean_filtered.shape[1])

        noise_raw = np.empty_like(raw)
        for channel_index, channel in enumerate(HB2):
            rng = np.random.default_rng(noise_seed(item.subject, channel))
            noise_raw[channel_index] = rng.standard_normal(raw.shape[1])
            basis_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "channel": channel,
                    "seed": noise_seed(item.subject, channel),
                    "raw_samples": raw.shape[1],
                    "noise_basis_sha256": array_sha256(noise_raw[channel_index]),
                }
            )
        noise_filtered = filter_resample(noise_raw, sos)

        clean_values = epoch_features(normalize_signal(clean_filtered, HB2, hb_scaler), HB2, events)
        source_values = load_source_feature(item.subject)
        timing_match = np.array_equal(clean_values[0], source_values[0]) and np.array_equal(clean_values[1], source_values[1])
        schema_match = clean_values[3] == source_values[3]
        difference = float(np.max(np.abs(clean_values[2] - source_values[2])))
        clean_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": PARTITION,
                "timing_and_stage_match": timing_match,
                "feature_schema_match": schema_match,
                "maximum_absolute_feature_difference": difference,
                "tolerance": CLEAN_FEATURE_TOLERANCE,
                "reproduction_pass": timing_match and schema_match and difference <= CLEAN_FEATURE_TOLERANCE,
                "source_feature_sha256": sha256(source_feature_path(item.subject)),
            }
        )

        scales: dict[tuple[str, float], float] = {}
        for channel_index, channel in enumerate(HB2):
            clean_rms = float(np.sqrt(np.mean(np.square(clean_filtered[channel_index, mask]))))
            unit_noise_rms = float(np.sqrt(np.mean(np.square(noise_filtered[channel_index, mask]))))
            if clean_rms <= 0 or unit_noise_rms <= 0:
                raise ValueError(f"Invalid SNR calibration RMS: {item.subject}, {channel}")
            for target_snr in [20.0, 10.0, 0.0]:
                scales[(channel, target_snr)] = clean_rms / (10 ** (target_snr / 20.0) * unit_noise_rms)

        for comparator, perturbations in CONDITIONS.items():
            if comparator == "H2-CLEAN":
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
                        "comparator": comparator,
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
            path = generated_feature_path(item.subject, comparator)
            verify_or_create_npz(path, *values)
            centers, _ = context_matrix(values[0], values[2])
            generated_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "partition": PARTITION,
                    "comparator": comparator,
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


# Section 5: frozen scoring and clean reproduction

def verify_model() -> tuple[object, pd.DataFrame]:
    observed = sha256(model_path())
    if observed != MODEL_SHA256:
        raise ValueError("Frozen H2-D model hash changed")
    model = joblib.load(model_path())
    return model, pd.DataFrame(
        [{"model": "H2-D", "expected_sha256": MODEL_SHA256, "observed_sha256": observed, "hash_match": True, "threshold": THRESHOLD}]
    )


def load_condition_feature(subject: str, comparator: str) -> tuple[np.ndarray, np.ndarray]:
    path = source_feature_path(subject) if comparator == "H2-CLEAN" else generated_feature_path(subject, comparator)
    with np.load(path, allow_pickle=False) as values:
        onsets = values["onset"].astype(np.float64)
        features = values["features"].astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite condition feature: {subject}, {comparator}")
    return onsets, features


def score_conditions(assignments: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for comparator in CONDITIONS:
        for item in assignments.itertuples(index=False):
            onsets, features = load_condition_feature(item.subject, comparator)
            centers, matrix = context_matrix(onsets, features)
            probability = model.predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": "H2-D",
                        "partition": PARTITION,
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": centers,
                        "probability": probability,
                    }
                )
            )
            support_rows.append(
                {
                    "comparator": comparator,
                    "partition": PARTITION,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "supported_boundaries": len(centers),
                    "supported_hours": len(centers) * EPOCH_SEC / 3600.0,
                }
            )
    return pd.concat(score_rows, ignore_index=True), pd.DataFrame(support_rows)


def clean_probability_reproduction(scores: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(source_score_path(), sep="\t")
    source = source[source["comparator"] == "H2-D"][["subject", "pid", "candidate_time_sec", "probability"]].sort_values(["subject", "candidate_time_sec"]).reset_index(drop=True)
    clean = scores[scores["comparator"] == "H2-CLEAN"][["subject", "pid", "candidate_time_sec", "probability"]].sort_values(["subject", "candidate_time_sec"]).reset_index(drop=True)
    if not source.drop(columns="probability").equals(clean.drop(columns="probability")):
        raise ValueError("Clean and source validation score rows differ")
    difference = np.abs(source["probability"].to_numpy() - clean["probability"].to_numpy())
    return pd.DataFrame(
        [{"source_comparator": "H2-D", "reproduced_comparator": "H2-CLEAN", "rows": len(clean), "maximum_absolute_probability_difference": float(difference.max()), "tolerance": PROBABILITY_TOLERANCE, "reproduction_pass": bool(difference.max() <= PROBABILITY_TOLERANCE)}]
    )


def score_fidelity(scores: pd.DataFrame) -> pd.DataFrame:
    clean = scores[scores["comparator"] == "H2-CLEAN"].sort_values(["subject", "candidate_time_sec"])
    clean_probability = clean["probability"].to_numpy(dtype=float)
    rows = []
    for comparator in CONDITIONS:
        if comparator == "H2-CLEAN":
            continue
        degraded = scores[scores["comparator"] == comparator].sort_values(["subject", "candidate_time_sec"])
        if not clean[["subject", "candidate_time_sec"]].reset_index(drop=True).equals(degraded[["subject", "candidate_time_sec"]].reset_index(drop=True)):
            raise ValueError(f"Score-fidelity rows differ for {comparator}")
        values = degraded["probability"].to_numpy(dtype=float)
        difference = values - clean_probability
        rows.append(
            {
                "comparator": comparator,
                "rows": len(values),
                "spearman_with_clean": float(spearmanr(clean_probability, values).statistic),
                "mean_absolute_probability_difference": float(np.mean(np.abs(difference))),
                "maximum_absolute_probability_difference": float(np.max(np.abs(difference))),
                "median_probability_shift": float(np.median(difference)),
                "threshold_crossing_fraction": float(np.mean((values >= THRESHOLD) != (clean_probability >= THRESHOLD))),
            }
        )
    return pd.DataFrame(rows)


# Section 6: event evaluation

def collapse_alarms(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= THRESHOLD]
    for (comparator, subject, pid), group in marked.groupby(["comparator", "subject", "pid"], sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        starts = [0]
        starts.extend((np.flatnonzero(np.diff(group["candidate_time_sec"].to_numpy(dtype=float)) > EPOCH_SEC + 1e-6) + 1).tolist())
        stops = starts[1:] + [len(group)]
        for start, stop in zip(starts, stops):
            run = group.iloc[start:stop]
            maximum = float(run["probability"].max())
            best = run[np.isclose(run["probability"], maximum)].sort_values("candidate_time_sec").iloc[0]
            rows.append(
                {"comparator": comparator, "partition": PARTITION, "subject": subject, "pid": int(pid), "event_time_sec": float(best.candidate_time_sec), "probability": float(best.probability), "threshold": THRESHOLD, "run_candidates": len(run)}
            )
    return pd.DataFrame(rows, columns=["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"])


def evaluate_all(scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame) -> dict[str, pd.DataFrame]:
    alarms = collapse_alarms(scores)
    summaries = []
    recordings_all = []
    participants_all = []
    matches_all = []
    for comparator in CONDITIONS:
        local_support = support[support["comparator"] == comparator][["subject", "pid", "supported_hours"]]
        predictions = alarms[alarms["comparator"] == comparator][["subject", "pid", "event_time_sec"]]
        for membership in MEMBERSHIPS:
            eligible, ignored = local_event_inputs(reference, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(eligible, predictions, ignored, local_support, tolerance)
                config = {"comparator": comparator, "model_role": MODEL_ROLES[comparator], "partition": PARTITION, "membership": membership, "tolerance_sec": tolerance, "threshold": THRESHOLD}
                summaries.append({**config, **summary})
                for frame, collection in [(recordings, recordings_all), (participants, participants_all), (matches, matches_all)]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        collection.append(local)
    return {
        "predicted_events": alarms,
        "event_metrics": pd.DataFrame(summaries),
        "event_recordings": pd.concat(recordings_all, ignore_index=True),
        "event_participants": pd.concat(participants_all, ignore_index=True),
        "event_matches": pd.concat(matches_all, ignore_index=True),
    }


def clean_event_reproduction(metrics: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(repo_root() / "experiments/2026-09-06_block7_transfer_validation_v0.1/validation_event_metrics_v0.1.tsv", sep="\t")
    source = source[source["comparator"] == "H2-D"].sort_values(["membership", "tolerance_sec"])
    clean = metrics[metrics["comparator"] == "H2-CLEAN"].sort_values(["membership", "tolerance_sec"])
    rows = []
    for source_row, clean_row in zip(source.itertuples(index=False), clean.itertuples(index=False)):
        differences = [abs(float(getattr(source_row, field)) - float(getattr(clean_row, field))) for field in ["precision", "recall", "f1", "false_alarms_per_hour"]]
        rows.append({"membership": clean_row.membership, "tolerance_sec": clean_row.tolerance_sec, "maximum_absolute_metric_difference": max(differences), "tolerance": PROBABILITY_TOLERANCE, "reproduction_pass": max(differences) <= PROBABILITY_TOLERANCE})
    return pd.DataFrame(rows)


# Section 7: participant uncertainty and hypotheses

def paired_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[(participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)]
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    rows = []
    for comparator in CONDITIONS:
        if comparator == "H2-CLEAN":
            continue
        clean = primary[primary["comparator"] == "H2-CLEAN"][columns]
        degraded = primary[primary["comparator"] == comparator][columns]
        paired = degraded.merge(clean, on="pid", suffixes=("_degraded", "_clean"), validate="one_to_one")
        if len(paired) != 16:
            raise ValueError(f"Expected 16 paired validation participants: {comparator}")
        rng = np.random.default_rng(BASE_SEED)
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
            values = {}
            for side in ["degraded", "clean"]:
                values[side] = metric_values(int(sample[f"true_positive_{side}"].sum()), int(sample[f"false_positive_{side}"].sum()), int(sample[f"false_negative_{side}"].sum()), float(sample[f"supported_hours_{side}"].sum()))
            samples.append({"event_f1_difference": values["degraded"]["f1"] - values["clean"]["f1"], "false_alarms_per_hour_difference": values["degraded"]["false_alarms_per_hour"] - values["clean"]["false_alarms_per_hour"]})
        sample_frame = pd.DataFrame(samples)
        point = {}
        for side in ["degraded", "clean"]:
            point[side] = metric_values(int(paired[f"true_positive_{side}"].sum()), int(paired[f"false_positive_{side}"].sum()), int(paired[f"false_negative_{side}"].sum()), float(paired[f"supported_hours_{side}"].sum()))
        points = {"event_f1_difference": point["degraded"]["f1"] - point["clean"]["f1"], "false_alarms_per_hour_difference": point["degraded"]["false_alarms_per_hour"] - point["clean"]["false_alarms_per_hour"]}
        for metric, value in points.items():
            rows.append({"comparison": f"{comparator}_minus_H2-CLEAN", "metric": metric, "point_difference": value, "resamples": BOOTSTRAP_RESAMPLES, "seed": BASE_SEED, "lower_95": float(sample_frame[metric].quantile(0.025)), "median": float(sample_frame[metric].quantile(0.5)), "upper_95": float(sample_frame[metric].quantile(0.975))})
    return pd.DataFrame(rows)


def hypothesis_decisions(metrics: pd.DataFrame, fidelity: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index("comparator")
    clean = primary.loc["H2-CLEAN"]
    mild = primary.loc["H2-BOTH-20DB"]
    mild_f1_difference = float(mild.f1 - clean.f1)
    mild_far_difference = float(mild.false_alarms_per_hour - clean.false_alarms_per_hour)
    mild_pass = mild_f1_difference > -MATERIAL_F1_DROP and mild_far_difference < MATERIAL_FAR_INCREASE

    fidelity_index = fidelity.set_index("comparator")
    dose_order = ["H2-BOTH-20DB", "H2-BOTH-10DB", "H2-BOTH-0DB"]
    correlations = [float(fidelity_index.loc[name, "spearman_with_clean"]) for name in dose_order]
    mean_differences = [float(fidelity_index.loc[name, "mean_absolute_probability_difference"]) for name in dose_order]
    correlation_monotonic = correlations[0] + ORDER_TOLERANCE >= correlations[1] and correlations[1] + ORDER_TOLERANCE >= correlations[2]
    difference_monotonic = mean_differences[0] <= mean_differences[1] + ORDER_TOLERANCE and mean_differences[1] <= mean_differences[2] + ORDER_TOLERANCE

    hb1 = primary.loc["H2-HB1-10DB"]
    hb2 = primary.loc["H2-HB2-10DB"]
    hb1_loss = float(clean.f1 - hb1.f1)
    hb2_loss = float(clean.f1 - hb2.f1)
    hb1_mad = float(fidelity_index.loc["H2-HB1-10DB", "mean_absolute_probability_difference"])
    hb2_mad = float(fidelity_index.loc["H2-HB2-10DB", "mean_absolute_probability_difference"])
    asymmetry_supported = hb1_loss + ORDER_TOLERANCE >= hb2_loss and hb1_mad + ORDER_TOLERANCE >= hb2_mad

    details = pd.DataFrame(
        [
            {"hypothesis": "H8.3_mild_noise_tolerance", "criterion": "20dB both-channel material bounds", "value_1": mild_f1_difference, "value_2": mild_far_difference, "supported": mild_pass, "decision": "pass" if mild_pass else "fail"},
            {"hypothesis": "H8.4_nested_probability_dose", "criterion": "correlation nonincrease and mean-absolute-difference nondecrease", "value_1": float(correlation_monotonic), "value_2": float(difference_monotonic), "supported": correlation_monotonic and difference_monotonic, "decision": "supported" if correlation_monotonic and difference_monotonic else "not_supported"},
            {"hypothesis": "H8.5_channel_noise_asymmetry", "criterion": "HB1 F1 loss and probability MAD at least HB2", "value_1": hb1_loss - hb2_loss, "value_2": hb1_mad - hb2_mad, "supported": asymmetry_supported, "decision": "supported" if asymmetry_supported else "inconclusive"},
        ]
    )
    dose = pd.DataFrame([{"comparator": name, "target_snr_db": float(name.split("-")[-1].replace("DB", "")), "event_f1": float(primary.loc[name, "f1"]), "false_alarms_per_hour": float(primary.loc[name, "false_alarms_per_hour"]), "spearman_with_clean": float(fidelity_index.loc[name, "spearman_with_clean"]), "mean_absolute_probability_difference": float(fidelity_index.loc[name, "mean_absolute_probability_difference"])} for name in dose_order])
    return details, dose


# Section 8: manifests, controls, and reviewed summary

def external_manifest(assignments: pd.DataFrame, generated: pd.DataFrame) -> pd.DataFrame:
    paths = [("source_clean_feature", source_feature_path(item.subject)) for item in assignments.itertuples(index=False)]
    paths.extend(("generated_noise_feature", data_parent() / item.path_relative_to_data_parent) for item in generated.itertuples(index=False))
    paths.extend([("frozen_h2_model", model_path()), ("source_validation_scores", source_score_path()), ("noise_validation_scores", score_path())])
    return pd.DataFrame([{"artifact_role": role, "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for role, path in paths]).sort_values(["artifact_role", "path_relative_to_data_parent"])


def run_checks(assignments: pd.DataFrame, model_record: pd.DataFrame, generated: pd.DataFrame, calibration: pd.DataFrame, basis: pd.DataFrame, clean_features: pd.DataFrame, scores: pd.DataFrame, support: pd.DataFrame, probability_reproduction: pd.DataFrame, event_reproduction: pd.DataFrame, fidelity: pd.DataFrame, outputs: dict[str, pd.DataFrame], bootstrap: pd.DataFrame, hypotheses: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    context_counts = support.groupby(["subject"])["supported_boundaries"].nunique()
    rows = [
        ("validation_assignment_only", len(assignments) == 20 and assignments["pid"].nunique() == 16 and set(assignments["partition"]) == {PARTITION}, "20 recordings; 16 pid groups"),
        ("frozen_model_hash", bool(model_record.iloc[0].hash_match), MODEL_SHA256),
        ("clean_feature_reproduction", len(clean_features) == 20 and truth(clean_features["reproduction_pass"]).all(), f"maximum={clean_features['maximum_absolute_feature_difference'].max():.12g}"),
        ("deterministic_noise_bases", len(basis) == 40 and basis[["subject", "channel"]].drop_duplicates().shape[0] == 40 and basis["noise_basis_sha256"].str.len().eq(64).all(), "one basis per recording-channel"),
        ("snr_calibration", len(calibration) == 160 and truth(calibration["calibration_pass"]).all(), f"maximum error={calibration['absolute_snr_error_db'].max():.12g} dB"),
        ("generated_feature_artifacts", len(generated) == 100 and truth(generated["all_features_finite"]).all(), "20 recordings x 5 noise conditions"),
        ("six_complete_comparators", set(scores["comparator"]) == set(CONDITIONS) and scores.groupby("comparator")["subject"].nunique().eq(20).all(), "six comparators; 20 recordings each"),
        ("validation_scores_only", set(scores["partition"]) == {PARTITION}, "no non-validation scores"),
        ("clean_probability_reproduction", bool(probability_reproduction.iloc[0].reproduction_pass), f"maximum={probability_reproduction.iloc[0].maximum_absolute_probability_difference:.12g}"),
        ("clean_event_reproduction", len(event_reproduction) == 4 and truth(event_reproduction["reproduction_pass"]).all(), "four event summaries"),
        ("identical_condition_support", len(support) == 120 and context_counts.eq(1).all(), "same context count for all conditions"),
        ("score_fidelity_complete", len(fidelity) == 5 and fidelity["rows"].nunique() == 1, "five degraded-clean comparisons"),
        ("complete_event_outputs", len(outputs["event_metrics"]) == 24, "6 comparators x 2 memberships x 2 tolerances"),
        ("paired_participant_bootstrap", len(bootstrap) == 10 and bootstrap["resamples"].eq(BOOTSTRAP_RESAMPLES).all(), "five comparisons x two metrics"),
        ("hypotheses_evaluated", len(hypotheses) == 3, "H8.3, H8.4, H8.5"),
        ("external_manifest", len(manifest) == 123 and manifest["sha256"].str.len().eq(64).all(), "20 source features, 100 generated features, model, two score files"),
        ("test_artifacts_closed", not manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any(), "no test path in manifest"),
    ]
    return pd.DataFrame([{"check": name, "status": "pass" if passed else "fail", "detail": detail} for name, passed, detail in rows])


def write_readme(result_code_commit: str, metrics: pd.DataFrame, fidelity: pd.DataFrame, bootstrap: pd.DataFrame, hypotheses: pd.DataFrame, dose: pd.DataFrame, checks: pd.DataFrame) -> None:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index("comparator")
    metric_rows = [f"| {name} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |" for name, row in primary.iterrows()]
    fidelity_rows = [f"| {item.comparator} | {item.spearman_with_clean:.4f} | {item.mean_absolute_probability_difference:.4f} | {item.median_probability_shift:+.4f} | {item.threshold_crossing_fraction:.2%} |" for item in fidelity.itertuples(index=False)]
    bootstrap_primary = bootstrap[bootstrap["metric"] == "event_f1_difference"]
    bootstrap_rows = [f"| {item.comparison} | {item.point_difference:+.4f} | {item.lower_95:+.4f} to {item.upper_95:+.4f} |" for item in bootstrap_primary.itertuples(index=False)]
    hypothesis_rows = [f"| {item.hypothesis} | {item.decision} | {bool(item.supported)} |" for item in hypotheses.itertuples(index=False)]
    dose_rows = [f"| {item.target_snr_db:.0f} | {item.event_f1:.4f} | {item.false_alarms_per_hour:.4f} | {item.spearman_with_clean:.4f} | {item.mean_absolute_probability_difference:.4f} |" for item in dose.itertuples(index=False)]
    text = "\n".join(
        [
            "# Block 8 Raw-Signal Noise Robustness v0.1",
            "",
            "**Work date:** 2026-09-11",
            f"**Protocol commit:** `{PROTOCOL_COMMIT}`",
            f"**Result-producing code commit:** `{result_code_commit}`",
            "**Partition accessed:** Validation only",
            "**Model fitting or threshold selection:** None",
            "**Test data accessed:** No",
            "",
            "## Primary Event Result",
            "",
            "| Comparator | Precision | Recall | F1 | False alarms/hour |",
            "|---|---:|---:|---:|---:|",
            *metric_rows,
            "",
            "## Probability Fidelity",
            "",
            "| Comparator | Spearman with clean | Mean absolute difference | Median shift | Threshold crossings |",
            "|---|---:|---:|---:|---:|",
            *fidelity_rows,
            "",
            "## Both-Channel Dose Response",
            "",
            "| SNR dB | Event F1 | False alarms/hour | Spearman with clean | Mean absolute difference |",
            "|---:|---:|---:|---:|---:|",
            *dose_rows,
            "",
            "## Paired Participant F1 Differences",
            "",
            "| Comparison | Point difference | Paired-bootstrap 95% interval |",
            "|---|---:|---:|",
            *bootstrap_rows,
            "",
            "## Frozen Decisions",
            "",
            "| Hypothesis | Decision | Supported |",
            "|---|---|---:|",
            *hypothesis_rows,
            "",
            "## Boundary",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} in-run checks passed. This stationary Gaussian-noise experiment is a controlled validation stress test, not a model of natural wearable artefacts or evidence from a new cohort.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "README.md", text)


# Section 9: execute the frozen experiment

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    args = parser.parse_args()
    if len(args.result_code_commit) < 7:
        raise ValueError("A committed result-producing code hash is required")

    output_dir().mkdir(parents=True, exist_ok=True)
    assignments = validation_assignments()
    model, model_record = verify_model()
    generated, calibration, basis, clean_features = generate_noise_features(assignments)
    if not truth(clean_features["reproduction_pass"]).all() or not truth(calibration["calibration_pass"]).all():
        raise ValueError("Clean feature or SNR calibration control failed before scoring")
    scores, support = score_conditions(assignments, model)
    probability_reproduction = clean_probability_reproduction(scores)
    if not bool(probability_reproduction.iloc[0].reproduction_pass):
        raise ValueError("Clean probability control failed before event evaluation")
    verify_or_create_gzip_tsv(scores, score_path())
    fidelity = score_fidelity(scores)
    outputs = evaluate_all(scores, support, reference_events(assignments))
    event_reproduction = clean_event_reproduction(outputs["event_metrics"])
    if not truth(event_reproduction["reproduction_pass"]).all():
        raise ValueError("Clean event control failed")
    bootstrap = paired_bootstrap(outputs["event_participants"])
    hypotheses, dose = hypothesis_decisions(outputs["event_metrics"], fidelity)
    manifest = external_manifest(assignments, generated)
    checks = run_checks(assignments, model_record, generated, calibration, basis, clean_features, scores, support, probability_reproduction, event_reproduction, fidelity, outputs, bootstrap, hypotheses, manifest)

    reviewed = {
        "frozen_model_verification_v0.1.tsv": model_record,
        "clean_feature_reproduction_v0.1.tsv": clean_features,
        "noise_basis_manifest_v0.1.tsv": basis,
        "noise_calibration_v0.1.tsv": calibration,
        "generated_noise_feature_manifest_v0.1.tsv": generated,
        "clean_probability_reproduction_v0.1.tsv": probability_reproduction,
        "clean_event_reproduction_v0.1.tsv": event_reproduction,
        "validation_support_v0.1.tsv": support,
        "score_fidelity_v0.1.tsv": fidelity,
        "validation_predicted_events_v0.1.tsv": outputs["predicted_events"],
        "validation_event_metrics_v0.1.tsv": outputs["event_metrics"],
        "validation_event_recordings_v0.1.tsv": outputs["event_recordings"],
        "validation_event_participants_v0.1.tsv": outputs["event_participants"],
        "validation_event_matches_v0.1.tsv": outputs["event_matches"],
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decisions_v0.1.tsv": hypotheses,
        "both_channel_dose_response_v0.1.tsv": dose,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in reviewed.items():
        verify_or_create_tsv(frame, output_dir() / name)
    software = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__}
    verify_or_create_text(output_dir() / "software_versions_v0.1.json", json.dumps(software, indent=2, sort_keys=True) + "\n")
    write_readme(args.result_code_commit, outputs["event_metrics"], fidelity, bootstrap, hypotheses, dose, checks)

    primary = outputs["event_metrics"]
    primary = primary[(primary["membership"] == "primary") & (primary["tolerance_sec"] == 15.0)]
    print(primary[["comparator", "precision", "recall", "f1", "false_alarms_per_hour"]].to_string(index=False))
    print(fidelity.to_string(index=False))
    print(hypotheses.to_string(index=False))
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 8 noise experiment check failed")


if __name__ == "__main__":
    main()
