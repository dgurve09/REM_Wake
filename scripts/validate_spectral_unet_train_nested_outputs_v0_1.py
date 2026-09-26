"""Independently validate spectral U-Net reviewed and external outputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from run_spectral_unet_train_nested_v0_1 import (
    BANDS,
    DERIVED_DIR,
    EPOCH_SEC,
    EPOCH_SAMPLES,
    FEATURE_COUNT,
    HB2,
    OUTER_FOLDS,
    OUTPUT_SFREQ,
    PIPELINES,
    SEQUENCE_BINS,
    collapse_events,
    data_parent,
    evaluate_pipelines,
    filter_resample,
    filter_sos,
    normalize,
    output_dir,
    outer_score_path,
    read_uv,
    reference_events,
    sha256,
    train_assignments,
)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def almost_equal(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> bool:
    if len(left) != len(right):
        return False
    left = left.sort_values(columns).reset_index(drop=True)
    right = right.sort_values(columns).reset_index(drop=True)
    if list(left.columns) != list(right.columns):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(left[column], right[column], rtol=1e-10, atol=1e-12, equal_nan=True):
                return False
        elif not left[column].fillna("").astype(str).equals(right[column].fillna("").astype(str)):
            return False
    return True


def recompute_first_cached_epoch(subject: str, onset: float) -> np.ndarray:
    scaler_table = pd.read_csv(
        Path(__file__).resolve().parents[1]
        / "experiments/2026-09-06_block7_feature_generation_validation_v0.1"
        / "train_robust_scalers_v0.1.tsv",
        sep="\t",
    )
    scaler_table = scaler_table[scaler_table["scaler_owner"].eq("HB-2")].set_index("channel")
    scaler = {
        channel: {
            "median_uv": float(scaler_table.loc[channel, "median_uv"]),
            "robust_scale_uv": float(scaler_table.loc[channel, "robust_scale_uv"]),
        }
        for channel in HB2
    }
    signal = normalize(filter_resample(read_uv(subject, "headband", HB2), filter_sos()), HB2, scaler)
    start = int(round(onset * OUTPUT_SFREQ))
    epoch = signal[:, start : start + EPOCH_SAMPLES].reshape(len(HB2), 15, 256)
    frequencies, density = welch(
        epoch, fs=OUTPUT_SFREQ, window="hann", nperseg=256, noverlap=0, axis=-1
    )
    columns = []
    for channel_index in range(len(HB2)):
        for low, high in BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            power = density[channel_index, :, :][:, mask].mean(axis=-1)
            columns.append(np.log10(np.maximum(power, np.finfo(float).eps)))
    return np.stack(columns, axis=-1).astype(np.float32)


def validate() -> pd.DataFrame:
    output = output_dir()
    assignments = train_assignments()
    scores = pd.read_csv(outer_score_path(), sep="\t")
    support = pd.read_csv(output / "outer_support_v0.1.tsv", sep="\t")
    selections = pd.read_csv(output / "inner_threshold_selections_v0.1.tsv", sep="\t")
    stored_events = pd.read_csv(output / "outer_predicted_events_v0.1.tsv", sep="\t")
    stored_metrics = pd.read_csv(output / "outer_event_metrics_v0.1.tsv", sep="\t")
    manifest = pd.read_csv(output / "external_artifact_manifest_v0.1.tsv", sep="\t")
    feature_summary = pd.read_csv(output / "feature_generation_summary_v0.1.tsv", sep="\t")

    unet_events = []
    for outer in range(1, OUTER_FOLDS + 1):
        threshold = float(selections[selections["outer_fold"] == outer].iloc[0].threshold)
        unet_events.append(
            collapse_events(scores[scores["outer_fold"] == outer], threshold, "SPECTRAL-UNET")
        )
    rebuilt_unet = pd.concat(unet_events, ignore_index=True)
    baseline = stored_events[stored_events["pipeline"] == "NESTED-BLSTM-CRF"]
    rebuilt_events = pd.concat([baseline, rebuilt_unet], ignore_index=True, sort=False)
    references = reference_events(assignments)
    rebuilt_metrics = evaluate_pipelines(rebuilt_events, support, references)["metrics"]

    cache_checks = []
    for item in feature_summary.itertuples(index=False):
        path = data_parent() / DERIVED_DIR / "subepoch_features" / f"{item.subject}_spectral_v0.1.npz"
        if not path.exists() or sha256(path) != item.cache_sha256:
            cache_checks.append(False)
            continue
        with np.load(path, allow_pickle=False) as values:
            features = values["features"]
            cache_checks.append(
                features.ndim == 3
                and features.shape[1:] == (15, FEATURE_COUNT)
                and np.isfinite(features).all()
            )

    first_subject = str(feature_summary.sort_values("subject").iloc[0].subject)
    first_path = data_parent() / DERIVED_DIR / "subepoch_features" / f"{first_subject}_spectral_v0.1.npz"
    with np.load(first_path, allow_pickle=False) as values:
        first_onset = float(values["onset"][0])
        first_cached = values["features"][0].astype(np.float32)
    first_recomputed = recompute_first_cached_epoch(first_subject, first_onset)
    direct_recomputation_matches = bool(
        np.allclose(first_cached, first_recomputed, rtol=1e-5, atol=1e-6)
    )

    manifest_checks = []
    for item in manifest.itertuples(index=False):
        path = data_parent() / Path(item.relative_path)
        manifest_checks.append(
            path.exists()
            and path.stat().st_size == int(item.bytes)
            and file_digest(path) == item.sha256
        )

    score_counts = scores.groupby(["subject", "pid", "outer_fold"]).size().reset_index(name="count")
    expected = support.rename(columns={"supported_boundaries": "count"})[
        ["subject", "pid", "outer_fold", "count"]
    ]
    rows = [
        ("train_membership_82_recordings", len(assignments) == 82),
        ("train_membership_64_pid", assignments["pid"].nunique() == 64),
        ("outer_score_rows_75539", len(scores) == 75539),
        ("outer_scores_finite_probability", np.isfinite(scores["probability"]).all() and scores["probability"].between(0, 1).all()),
        ("outer_support_matches_scores", score_counts.sort_values("subject").reset_index(drop=True).equals(expected.sort_values("subject").reset_index(drop=True))),
        ("one_outer_assignment_per_subject", scores.groupby("subject")["outer_fold"].nunique().eq(1).all()),
        ("five_inner_thresholds", len(selections) == 5 and selections["outer_fold"].nunique() == 5),
        ("events_reconstruct", almost_equal(stored_events, rebuilt_events, ["pipeline", "subject", "event_time_sec"])),
        ("metrics_reconstruct", almost_equal(stored_metrics, rebuilt_metrics, ["pipeline", "membership", "tolerance_sec"])),
        ("all_feature_caches_valid", len(cache_checks) == 82 and all(cache_checks)),
        ("first_cached_epoch_recomputes", direct_recomputation_matches),
        ("all_manifest_artifacts_match", len(manifest_checks) == len(manifest) and all(manifest_checks)),
        ("pipeline_pair_complete", set(stored_metrics["pipeline"]) == set(PIPELINES)),
        ("sequence_contract_120_by_10", SEQUENCE_BINS == 120 and FEATURE_COUNT == 10),
        ("support_hours_formula", np.allclose(support["supported_hours"], support["supported_boundaries"] * EPOCH_SEC / 3600.0)),
    ]
    return pd.DataFrame(
        [{"check": name, "status": "pass" if passed else "fail"} for name, passed in rows]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    checks = validate()
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("Validation failed")
    if args.write:
        path = output_dir() / "independent_output_checks_v0.1.tsv"
        value = checks.to_csv(sep="\t", index=False, lineterminator="\n")
        if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != value:
            raise RuntimeError(f"Reviewed validation output changed: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
