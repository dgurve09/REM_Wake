"""Independently reconstruct and validate Block 8 raw-signal noise outputs."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import spearmanr

from reviewed_output import verify_or_create_tsv
from run_block7_transfer_validation_v0_1 import normalize_signal, scaler_maps
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

EXPERIMENT_DIR = "2026-09-11_block8_raw_signal_noise_v0.1"
SOURCE_DERIVED_DIR = "block7_transfer_validation_v0.1"
RESULT_DERIVED_DIR = "block8_raw_signal_noise_v0.1"
PARTITION = "validation"
MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
THRESHOLD = 0.96
EPOCH_SEC = 30.0
CONTEXT_EPOCHS = 8
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


# Section 2: paths and comparison helpers

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
    folder = CONDITION_FOLDERS[comparator]
    return result_dir() / "recording_features" / folder / f"{subject}_features_v0.1.npz"


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


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


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


# Section 3: validation membership and reference events

def assignments() -> pd.DataFrame:
    split = pd.read_csv(
        repo_root() / "splits/grouped_pid_split_v0.1/pid_split_assignments_v0.1.tsv",
        sep="\t",
        usecols=["pid", "subjects", "partition"],
    )
    rows = []
    for item in split[split["partition"] == PARTITION].itertuples(index=False):
        for subject in str(item.subjects).split(";"):
            rows.append({"subject": subject, "pid": int(item.pid), "partition": PARTITION})
    return pd.DataFrame(rows).sort_values("subject")


def references(local_assignments: pd.DataFrame) -> pd.DataFrame:
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
        membership["subject"].isin(set(local_assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
        & membership["partition"].eq(PARTITION)
    ].merge(quality, on="transition_id", validate="one_to_one")
    result["event_time_sec"] = result["nominal_boundary_sec"].astype(float)
    return result


def event_inputs(reference: pd.DataFrame, membership: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = "primary_analysis_eligible" if membership == "primary" else "expanded_quality_analysis_eligible"
    eligible = truth(reference[column])
    columns = ["subject", "pid", "event_time_sec"]
    return reference.loc[eligible, columns], reference.loc[~eligible, columns]


# Section 4: independent raw-signal and feature reconstruction

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


def load_feature(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["stage"].astype(np.int8),
            values["features"].astype(np.float32),
            values["feature_names"].astype(str).tolist(),
        )


def context_matrix(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contiguous = np.isclose(np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(CONTEXT_EPOCHS - 1, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(counts == CONTEXT_EPOCHS - 1)
    matrix = np.concatenate([features[indices + offset] for offset in range(CONTEXT_EPOCHS)], axis=1)
    return onsets[indices + 4], matrix


def reconstruct_features(
    rows: list[dict], local_assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, hb_scaler, _ = scaler_maps()
    sos = filter_sos()
    generated_rows = []
    calibration_rows = []
    basis_rows = []
    clean_rows = []
    arrays_match = True
    maximum_stored_difference = 0.0

    for index, item in enumerate(local_assignments.itertuples(index=False), start=1):
        print(f"Independent feature reconstruction {index}/{len(local_assignments)}: {item.subject}", flush=True)
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
        source = load_feature(source_feature_path(item.subject))
        timing_match = np.array_equal(clean[0], source[0]) and np.array_equal(clean[1], source[1])
        schema_match = clean[3] == source[3]
        clean_difference = float(np.max(np.abs(clean[2] - source[2])))
        clean_rows.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "partition": PARTITION,
                "timing_and_stage_match": timing_match,
                "feature_schema_match": schema_match,
                "maximum_absolute_feature_difference": clean_difference,
                "tolerance": CLEAN_FEATURE_TOLERANCE,
                "reproduction_pass": timing_match and schema_match and clean_difference <= CLEAN_FEATURE_TOLERANCE,
                "source_feature_sha256": sha256(source_feature_path(item.subject)),
            }
        )

        scales = {}
        for channel_index, channel in enumerate(HB2):
            clean_rms = float(np.sqrt(np.mean(np.square(clean_filtered[channel_index, mask]))))
            noise_rms = float(np.sqrt(np.mean(np.square(noise_filtered[channel_index, mask]))))
            for target_snr in [20.0, 10.0, 0.0]:
                scales[(channel, target_snr)] = clean_rms / (10 ** (target_snr / 20.0) * noise_rms)

        for comparator, perturbations in CONDITIONS.items():
            if comparator == "H2-CLEAN":
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
                        "comparator": comparator,
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
            path = generated_feature_path(item.subject, comparator)
            stored = load_feature(path)
            local_difference = float(np.max(np.abs(expected[2] - stored[2])))
            maximum_stored_difference = max(maximum_stored_difference, local_difference)
            arrays_match &= (
                np.array_equal(expected[0], stored[0])
                and np.array_equal(expected[1], stored[1])
                and expected[3] == stored[3]
                and local_difference <= CLEAN_FEATURE_TOLERANCE
            )
            centers, _ = context_matrix(stored[0], stored[2])
            generated_rows.append(
                {
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "partition": PARTITION,
                    "comparator": comparator,
                    "epochs": len(stored[0]),
                    "context_rows": len(centers),
                    "feature_dimensions": stored[2].shape[1],
                    "all_features_finite": bool(np.isfinite(stored[2]).all()),
                    "path_relative_to_data_parent": path.relative_to(data_parent()).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )

    generated = pd.DataFrame(generated_rows)
    calibration = pd.DataFrame(calibration_rows)
    basis = pd.DataFrame(basis_rows)
    clean = pd.DataFrame(clean_rows)
    comparisons = [
        (generated, "generated_noise_feature_manifest_v0.1.tsv", ["comparator", "subject"]),
        (calibration, "noise_calibration_v0.1.tsv", ["comparator", "subject", "channel"]),
        (basis, "noise_basis_manifest_v0.1.tsv", ["subject", "channel"]),
        (clean, "clean_feature_reproduction_v0.1.tsv", ["subject"]),
    ]
    tables_match = all(
        frames_match(frame, pd.read_csv(output_dir() / name, sep="\t"), sort_by)
        for frame, name, sort_by in comparisons
    )
    record(rows, "raw_noise_features_reconstructed", arrays_match, f"100 arrays; maximum difference={maximum_stored_difference:.12g}")
    record(rows, "noise_control_tables_recomputed", tables_match, "manifests, calibration, bases, and clean reproduction")
    return generated, calibration, basis, clean


# Section 5: probability and event reconstruction

def score_conditions(local_assignments: pd.DataFrame, model) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows = []
    support_rows = []
    for comparator in CONDITIONS:
        for item in local_assignments.itertuples(index=False):
            path = source_feature_path(item.subject) if comparator == "H2-CLEAN" else generated_feature_path(item.subject, comparator)
            onsets, _, features, _ = load_feature(path)
            centers, matrix = context_matrix(onsets, features)
            probabilities = model.predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": "H2-D",
                        "partition": PARTITION,
                        "subject": item.subject,
                        "pid": int(item.pid),
                        "candidate_time_sec": centers,
                        "probability": probabilities,
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


def probability_fidelity(scores: pd.DataFrame) -> pd.DataFrame:
    clean = scores[scores["comparator"] == "H2-CLEAN"].sort_values(["subject", "candidate_time_sec"])
    clean_values = clean["probability"].to_numpy(dtype=float)
    rows = []
    for comparator in CONDITIONS:
        if comparator == "H2-CLEAN":
            continue
        degraded = scores[scores["comparator"] == comparator].sort_values(["subject", "candidate_time_sec"])
        values = degraded["probability"].to_numpy(dtype=float)
        difference = values - clean_values
        rows.append(
            {
                "comparator": comparator,
                "rows": len(values),
                "spearman_with_clean": float(spearmanr(clean_values, values).statistic),
                "mean_absolute_probability_difference": float(np.mean(np.abs(difference))),
                "maximum_absolute_probability_difference": float(np.max(np.abs(difference))),
                "median_probability_shift": float(np.median(difference)),
                "threshold_crossing_fraction": float(np.mean((values >= THRESHOLD) != (clean_values >= THRESHOLD))),
            }
        )
    return pd.DataFrame(rows)


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
                {
                    "comparator": comparator,
                    "partition": PARTITION,
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(best.candidate_time_sec),
                    "probability": float(best.probability),
                    "threshold": THRESHOLD,
                    "run_candidates": len(run),
                }
            )
    columns = ["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"]
    return pd.DataFrame(rows, columns=columns)


def recompute_events(
    scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], bool]:
    alarms = collapse_alarms(scores)
    summaries = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for comparator in CONDITIONS:
        local_support = support[support["comparator"] == comparator][["subject", "pid", "supported_hours"]]
        predictions = alarms[alarms["comparator"] == comparator][["subject", "pid", "event_time_sec"]]
        for membership in MEMBERSHIPS:
            eligible, ignored = event_inputs(reference, membership)
            for tolerance in TOLERANCES:
                recordings, participants, matches, summary = evaluate_events(eligible, predictions, ignored, local_support, tolerance)
                config = {
                    "comparator": comparator,
                    "model_role": MODEL_ROLES[comparator],
                    "partition": PARTITION,
                    "membership": membership,
                    "tolerance_sec": tolerance,
                    "threshold": THRESHOLD,
                }
                summaries.append({**config, **summary})
                for frame, collection in [(recordings, recording_rows), (participants, participant_rows), (matches, match_rows)]:
                    if len(frame):
                        local = frame.copy()
                        for key, value in reversed(list(config.items())):
                            if key not in local.columns:
                                local.insert(0, key, value)
                        collection.append(local)
    outputs = {
        "validation_predicted_events_v0.1.tsv": alarms,
        "validation_event_metrics_v0.1.tsv": pd.DataFrame(summaries),
        "validation_event_recordings_v0.1.tsv": pd.concat(recording_rows, ignore_index=True),
        "validation_event_participants_v0.1.tsv": pd.concat(participant_rows, ignore_index=True),
        "validation_event_matches_v0.1.tsv": pd.concat(match_rows, ignore_index=True),
    }
    keys = {
        "validation_predicted_events_v0.1.tsv": ["comparator", "subject", "event_time_sec"],
        "validation_event_metrics_v0.1.tsv": ["comparator", "membership", "tolerance_sec"],
        "validation_event_recordings_v0.1.tsv": ["comparator", "membership", "tolerance_sec", "subject"],
        "validation_event_participants_v0.1.tsv": ["comparator", "membership", "tolerance_sec", "pid"],
        "validation_event_matches_v0.1.tsv": ["comparator", "membership", "tolerance_sec", "subject", "prediction_time_sec"],
    }
    matched = all(
        frames_match(frame, pd.read_csv(output_dir() / name, sep="\t"), keys[name])
        for name, frame in outputs.items()
    )
    return outputs, matched


# Section 6: participant intervals and frozen decisions

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
        rng = np.random.default_rng(BASE_SEED)
        samples = []
        for _ in range(BOOTSTRAP_RESAMPLES):
            sample = paired.iloc[rng.integers(0, len(paired), size=len(paired))]
            values = {}
            for side in ["degraded", "clean"]:
                values[side] = metric_values(
                    int(sample[f"true_positive_{side}"].sum()),
                    int(sample[f"false_positive_{side}"].sum()),
                    int(sample[f"false_negative_{side}"].sum()),
                    float(sample[f"supported_hours_{side}"].sum()),
                )
            samples.append(
                {
                    "event_f1_difference": values["degraded"]["f1"] - values["clean"]["f1"],
                    "false_alarms_per_hour_difference": values["degraded"]["false_alarms_per_hour"] - values["clean"]["false_alarms_per_hour"],
                }
            )
        sample_frame = pd.DataFrame(samples)
        point = {}
        for side in ["degraded", "clean"]:
            point[side] = metric_values(
                int(paired[f"true_positive_{side}"].sum()),
                int(paired[f"false_positive_{side}"].sum()),
                int(paired[f"false_negative_{side}"].sum()),
                float(paired[f"supported_hours_{side}"].sum()),
            )
        points = {
            "event_f1_difference": point["degraded"]["f1"] - point["clean"]["f1"],
            "false_alarms_per_hour_difference": point["degraded"]["false_alarms_per_hour"] - point["clean"]["false_alarms_per_hour"],
        }
        for metric, value in points.items():
            rows.append(
                {
                    "comparison": f"{comparator}_minus_H2-CLEAN",
                    "metric": metric,
                    "point_difference": value,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BASE_SEED,
                    "lower_95": float(sample_frame[metric].quantile(0.025)),
                    "median": float(sample_frame[metric].quantile(0.5)),
                    "upper_95": float(sample_frame[metric].quantile(0.975)),
                }
            )
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
    dose = pd.DataFrame(
        [
            {
                "comparator": name,
                "target_snr_db": float(name.split("-")[-1].replace("DB", "")),
                "event_f1": float(primary.loc[name, "f1"]),
                "false_alarms_per_hour": float(primary.loc[name, "false_alarms_per_hour"]),
                "spearman_with_clean": float(fidelity_index.loc[name, "spearman_with_clean"]),
                "mean_absolute_probability_difference": float(fidelity_index.loc[name, "mean_absolute_probability_difference"]),
            }
            for name in dose_order
        ]
    )
    return details, dose


# Section 7: external artifact integrity and execution

def validate_external_manifest(rows: list[dict]) -> None:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    valid = len(manifest) == 123
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        valid &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
    no_test = not manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any()
    record(rows, "external_manifest_rehashed", valid, "123 external artifacts")
    record(rows, "test_artifacts_absent", no_test, "no test path in manifest")


def main() -> None:
    rows: list[dict] = []
    local_assignments = assignments()
    membership_ok = len(local_assignments) == 20 and local_assignments["pid"].nunique() == 16 and set(local_assignments["partition"]) == {PARTITION}
    record(rows, "validation_assignment", membership_ok, "20 recordings; 16 pid groups")

    observed_model_hash = sha256(model_path())
    record(rows, "frozen_model_rehashed", observed_model_hash == MODEL_SHA256, observed_model_hash)
    reconstruct_features(rows, local_assignments)

    model = joblib.load(model_path())
    scores, support = score_conditions(local_assignments, model)
    saved_scores = pd.read_csv(score_path(), sep="\t")
    saved_support = pd.read_csv(output_dir() / "validation_support_v0.1.tsv", sep="\t")
    record(rows, "noise_probabilities_recomputed", frames_match(scores, saved_scores, ["comparator", "subject", "candidate_time_sec"]), f"rows={len(scores)}")
    record(rows, "condition_support_recomputed", frames_match(support, saved_support, ["comparator", "subject"]), "six comparators; identical temporal support")

    source = pd.read_csv(source_score_path(), sep="\t")
    source = source[source["comparator"] == "H2-D"].sort_values(["subject", "candidate_time_sec"])
    clean = scores[scores["comparator"] == "H2-CLEAN"].sort_values(["subject", "candidate_time_sec"])
    probability_difference = np.abs(source["probability"].to_numpy() - clean["probability"].to_numpy())
    clean_probability_ok = len(source) == len(clean) and probability_difference.max() <= PROBABILITY_TOLERANCE
    record(rows, "clean_probability_reproduced", clean_probability_ok, f"maximum={probability_difference.max():.12g}")

    fidelity = probability_fidelity(scores)
    saved_fidelity = pd.read_csv(output_dir() / "score_fidelity_v0.1.tsv", sep="\t")
    record(rows, "score_fidelity_recomputed", frames_match(fidelity, saved_fidelity, ["comparator"]), "five degraded-clean comparisons")

    event_outputs, events_match = recompute_events(scores, support, references(local_assignments))
    record(rows, "event_outputs_recomputed", events_match, "alarms, metrics, recordings, participants, and matches")
    metrics = event_outputs["validation_event_metrics_v0.1.tsv"]
    participants = event_outputs["validation_event_participants_v0.1.tsv"]

    bootstrap = paired_bootstrap(participants)
    hypotheses, dose = hypothesis_decisions(metrics, fidelity)
    secondary_ok = (
        frames_match(bootstrap, pd.read_csv(output_dir() / "paired_participant_bootstrap_v0.1.tsv", sep="\t"), ["comparison", "metric"])
        and frames_match(hypotheses, pd.read_csv(output_dir() / "hypothesis_decisions_v0.1.tsv", sep="\t"), ["hypothesis"])
        and frames_match(dose, pd.read_csv(output_dir() / "both_channel_dose_response_v0.1.tsv", sep="\t"), ["target_snr_db"])
    )
    record(rows, "uncertainty_and_decisions_recomputed", secondary_ok, "10 bootstrap rows; three hypotheses; three dose levels")
    validate_external_manifest(rows)

    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")
    text = "\n".join(
        [
            "# Block 8 Raw-Signal Noise Output Validation",
            "",
            "**Validation date:** 2026-09-11",
            "**Scope:** Validation-only raw-signal perturbations, features, probabilities, events, participant intervals, and frozen decisions",
            "**Method:** Independent EDF-to-feature reconstruction, metric reconstruction, and SHA-256 verification",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} checks passed.",
            "",
            "The validator does not fit a model, search a threshold, or access test artifacts.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", text)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 8 raw-signal noise validation check failed")


if __name__ == "__main__":
    main()
