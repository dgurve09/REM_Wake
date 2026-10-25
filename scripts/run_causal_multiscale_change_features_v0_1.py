"""Run the frozen causal multiscale change-feature experiment."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.signal import csd, welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import run_causal_bidirectional_transition_v0_1 as causal
import run_full_night_risk_set_formulation_v0_1 as formulation
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import metric_values


# Section 1: frozen configuration

EXPERIMENT_DIR = "2026-10-25_causal_multiscale_change_features_v0.1"
DERIVED_DIR = "causal_multiscale_change_features_v0.1"
PROTOCOL_COMMIT = "7be8ae6"
CANDIDATES = ["BASE-PEAK", "ADV-PEAK"]
SUBWINDOW_STARTS = [-60, -45, -30, -15]
SUBWINDOW_SEC = 15.0
SUBWINDOW_SAMPLES = int(SUBWINDOW_SEC * prior.OUTPUT_SFREQ)
FEATURES_PER_SUBWINDOW = 30
ADVANCED_ONLY_FEATURES = 180
PRIMARY_BUDGET = 0.25
REFRACTORY_SEC = 60.0
C_VALUE = 0.1
MAX_ITER = 1000
BASE_SEED = 20261025
BOOTSTRAP_RESAMPLES = 2000
EPSILON = np.finfo(float).eps


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return root().parent / "REM_W_data"


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root().as_posix()}", "rev-parse", "HEAD"],
        cwd=root(),
        text=True,
    ).strip()


def write_json(path: Path, value: dict) -> None:
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
        raise RuntimeError(f"Reviewed output changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def feature_path(subject: str) -> Path:
    return derived_dir() / "features" / f"{subject}_advanced_features_v0.1.npz"


def model_path(candidate: str, outer: int, phase: str) -> Path:
    safe = candidate.lower().replace("-", "_")
    return derived_dir() / "models" / safe / f"outer_{outer}_{phase}.json.gz"


def score_path(candidate: str) -> Path:
    safe = candidate.lower().replace("-", "_")
    return derived_dir() / "scores" / f"{safe}_outer_oof.tsv.gz"


# Section 2: advanced 15-second features

def subwindow_feature_names() -> list[str]:
    names = []
    for channel in prior.HB2:
        names.extend(f"{channel}_{band}_log10_mean_psd" for band in prior.BANDS)
        names.extend(
            [
                f"{channel}_spectral_entropy",
                f"{channel}_aperiodic_loglog_slope",
                f"{channel}_aperiodic_loglog_intercept",
                f"{channel}_log10_rms",
                f"{channel}_log10_mean_line_length",
                f"{channel}_hjorth_mobility",
                f"{channel}_hjorth_complexity",
            ]
        )
    names.append("HB_1_HB_2_correlation")
    names.extend(f"HB_1_HB_2_{band}_coherence" for band in prior.BANDS)
    if len(names) != FEATURES_PER_SUBWINDOW:
        raise ValueError("Advanced subwindow schema length changed")
    return names


def advanced_feature_names() -> list[str]:
    base = subwindow_feature_names()
    names = [
        f"start_{start:+d}s_{name}"
        for start in SUBWINDOW_STARTS
        for name in base
    ]
    names.extend(f"post_minus_pre_{name}" for name in base)
    names.extend(f"abs_post_minus_pre_{name}" for name in base)
    if len(names) != ADVANCED_ONLY_FEATURES:
        raise ValueError("Advanced feature schema length changed")
    return names


def subwindow_features(blocks: np.ndarray) -> np.ndarray:
    if blocks.ndim != 3 or blocks.shape[1:] != (2, SUBWINDOW_SAMPLES):
        raise ValueError(f"Unexpected 15-second block shape: {blocks.shape}")
    frequencies, density = welch(
        blocks,
        fs=prior.OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    spectral_mask = (frequencies >= 0.5) & (frequencies < 30.0)
    aperiodic_mask = (frequencies >= 2.0) & (frequencies <= 30.0)
    log_frequency = np.log10(frequencies[aperiodic_mask])
    centered_frequency = log_frequency - log_frequency.mean()
    frequency_denominator = float(np.sum(centered_frequency * centered_frequency))
    columns = []

    for channel_index in range(2):
        local_density = np.maximum(density[:, channel_index, :], EPSILON)
        for low, high in prior.BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            columns.append(np.log10(local_density[:, mask].mean(axis=1)))

        spectral_density = local_density[:, spectral_mask]
        total_power = np.maximum(spectral_density.sum(axis=1), EPSILON)
        probability = spectral_density / total_power[:, None]
        entropy = -(probability * np.log(np.maximum(probability, EPSILON))).sum(axis=1)
        entropy /= np.log(probability.shape[1])
        columns.append(entropy)

        log_density = np.log10(local_density[:, aperiodic_mask])
        centered_density = log_density - log_density.mean(axis=1, keepdims=True)
        slope = (centered_density * centered_frequency[None, :]).sum(axis=1)
        slope /= frequency_denominator
        intercept = log_density.mean(axis=1) - slope * log_frequency.mean()
        columns.extend([slope, intercept])

        values = blocks[:, channel_index, :]
        difference_1 = np.diff(values, axis=1)
        difference_2 = np.diff(difference_1, axis=1)
        variance_0 = np.maximum(np.var(values, axis=1), EPSILON)
        variance_1 = np.maximum(np.var(difference_1, axis=1), EPSILON)
        variance_2 = np.maximum(np.var(difference_2, axis=1), EPSILON)
        rms = np.sqrt(np.mean(values * values, axis=1))
        line_length = np.mean(np.abs(difference_1), axis=1)
        mobility = np.sqrt(variance_1 / variance_0)
        complexity = np.sqrt(variance_2 / variance_1) / np.maximum(mobility, EPSILON)
        columns.extend(
            [
                np.log10(np.maximum(rms, EPSILON)),
                np.log10(np.maximum(line_length, EPSILON)),
                mobility,
                complexity,
            ]
        )

    left = blocks[:, 0, :] - blocks[:, 0, :].mean(axis=1, keepdims=True)
    right = blocks[:, 1, :] - blocks[:, 1, :].mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    correlation = np.sum(left * right, axis=1) / np.maximum(denominator, EPSILON)
    columns.append(np.clip(correlation, -1.0, 1.0))

    coherence_frequency, cross_density = csd(
        blocks[:, 0, :],
        blocks[:, 1, :],
        fs=prior.OUTPUT_SFREQ,
        window="hann",
        nperseg=512,
        noverlap=256,
        axis=-1,
    )
    coherence = np.abs(cross_density) ** 2 / np.maximum(
        density[:, 0, :] * density[:, 1, :], EPSILON
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    for low, high in prior.BANDS.values():
        mask = (coherence_frequency >= low) & (coherence_frequency < high)
        columns.append(coherence[:, mask].mean(axis=1))

    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (len(blocks), FEATURES_PER_SUBWINDOW):
        raise ValueError(f"Unexpected advanced feature shape: {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError("Advanced features contain nonfinite values")
    return result


def extract_blocks(signal: np.ndarray, decisions: np.ndarray, start_sec: int) -> np.ndarray:
    starts = np.rint((decisions + start_sec) * prior.OUTPUT_SFREQ).astype(int)
    if starts.min() < 0 or (starts.max() + SUBWINDOW_SAMPLES) > signal.shape[1]:
        raise ValueError("Advanced feature window falls outside signal support")
    sample_index = starts[:, None] + np.arange(SUBWINDOW_SAMPLES)[None, :]
    return signal[:, sample_index].transpose(1, 0, 2)


def feature_bounds(values: np.ndarray) -> bool:
    names = advanced_feature_names()
    raw_names = [name for name in names if not name.startswith(("post_minus", "abs_post"))]
    raw = values[:, : len(raw_names)]
    entropy = [index for index, name in enumerate(raw_names) if name.endswith("spectral_entropy")]
    correlation = [index for index, name in enumerate(raw_names) if name.endswith("correlation")]
    coherence = [index for index, name in enumerate(raw_names) if name.endswith("coherence")]
    mobility = [index for index, name in enumerate(raw_names) if name.endswith("hjorth_mobility")]
    complexity = [index for index, name in enumerate(raw_names) if name.endswith("hjorth_complexity")]
    return bool(
        np.logical_and(raw[:, entropy] >= 0.0, raw[:, entropy] <= 1.0).all()
        and np.logical_and(raw[:, correlation] >= -1.0, raw[:, correlation] <= 1.0).all()
        and np.logical_and(raw[:, coherence] >= 0.0, raw[:, coherence] <= 1.0).all()
        and (raw[:, mobility] >= 0.0).all()
        and (raw[:, complexity] >= 0.0).all()
    )


def load_or_create_advanced(subject: str, decisions: np.ndarray) -> tuple[np.ndarray, dict]:
    path = feature_path(subject)
    names = advanced_feature_names()
    # Record artifact provenance, not whether this invocation read or wrote the cache.
    status = "created"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            cached_decisions = cached["decision_time_sec"].astype(float)
            values = cached["features"].astype(np.float32)
            cached_names = cached["feature_names"].astype(str).tolist()
        if not np.array_equal(cached_decisions, decisions) or cached_names != names:
            raise ValueError(f"Advanced feature cache changed: {subject}")
    else:
        scaler = prior.frozen_scalers()["H2"]
        raw = prior.read_uv(subject, "headband", prior.HB2)
        signal = prior.normalize(
            prior.filter_resample(raw, prior.filter_sos()), prior.HB2, scaler
        )
        parts = [
            subwindow_features(extract_blocks(signal, decisions, start))
            for start in SUBWINDOW_STARTS
        ]
        before, after = parts[-2], parts[-1]
        values = np.column_stack(parts + [after - before, np.abs(after - before)]).astype(
            np.float32
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            decision_time_sec=decisions,
            features=values,
            feature_names=np.asarray(names),
        )
    if values.shape != (len(decisions), ADVANCED_ONLY_FEATURES):
        raise ValueError(f"Advanced feature matrix changed: {subject}, {values.shape}")
    if not np.isfinite(values).all() or not feature_bounds(values):
        raise ValueError(f"Advanced feature validation failed: {subject}")
    return values, {
        "subject": subject,
        "decisions": len(decisions),
        "advanced_features": values.shape[1],
        "all_values_finite": True,
        "feature_bounds_pass": True,
        "cache_status": status,
        "relative_path": path.relative_to(data_parent()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": prior.sha256(path),
    }


def synthetic_controls() -> pd.DataFrame:
    rng = np.random.default_rng(BASE_SEED)
    time = np.arange(SUBWINDOW_SAMPLES) / prior.OUTPUT_SFREQ
    alpha = np.sin(2 * np.pi * 10.0 * time)
    constant = np.zeros_like(time)
    white = rng.normal(size=SUBWINDOW_SAMPLES)
    spectrum = np.fft.rfft(rng.normal(size=SUBWINDOW_SAMPLES))
    frequency = np.fft.rfftfreq(SUBWINDOW_SAMPLES, d=1.0 / prior.OUTPUT_SFREQ)
    pink_scale = np.ones_like(frequency)
    pink_scale[1:] = 1.0 / np.sqrt(frequency[1:])
    pink = np.fft.irfft(spectrum * pink_scale, n=SUBWINDOW_SAMPLES)
    blocks = np.stack(
        [
            np.stack([alpha, alpha]),
            np.stack([constant, constant]),
            np.stack([white, white]),
            np.stack([pink, pink]),
        ]
    )
    values = subwindow_features(blocks)
    names = subwindow_feature_names()
    lookup = {name: index for index, name in enumerate(names)}
    alpha_power = values[0, lookup["HB_1_alpha_log10_mean_psd"]]
    other_power = max(
        values[0, lookup[f"HB_1_{band}_log10_mean_psd"]]
        for band in prior.BANDS
        if band != "alpha"
    )
    white_slope = values[2, lookup["HB_1_aperiodic_loglog_slope"]]
    pink_slope = values[3, lookup["HB_1_aperiodic_loglog_slope"]]
    rows = [
        {
            "check": "alpha_band_localization",
            "observed": float(alpha_power - other_power),
            "expected": ">0",
            "passed": bool(alpha_power > other_power),
        },
        {
            "check": "constant_signal_finite",
            "observed": bool(np.isfinite(values[1]).all()),
            "expected": True,
            "passed": bool(np.isfinite(values[1]).all()),
        },
        {
            "check": "pink_slope_below_white",
            "observed": float(pink_slope - white_slope),
            "expected": "<-0.5",
            "passed": bool(pink_slope < white_slope - 0.5),
        },
    ]
    return pd.DataFrame(rows)


def prepare_recordings(assignments: pd.DataFrame) -> tuple[dict, pd.DataFrame, list[str]]:
    recordings, causal_summary, base_names = causal.prepare_recordings(assignments)
    rows = []
    for number, item in enumerate(assignments.itertuples(index=False), start=1):
        recording = recordings[item.subject]
        advanced, summary = load_or_create_advanced(item.subject, recording["times"])
        recording["BASE-PEAK"] = recording["features"]
        recording["ADV-PEAK"] = np.column_stack([recording["features"], advanced]).astype(
            np.float32
        )
        rows.append({"pid": int(item.pid), **summary})
        print(f"advanced {number:02d}/{len(assignments)} {item.subject}", flush=True)
    names = base_names + advanced_feature_names()
    if len(names) != 260:
        raise ValueError("Combined feature schema changed")
    return recordings, causal_summary.merge(pd.DataFrame(rows), on=["subject", "pid"]), names


# Section 3: target arrays and fixed models

def rows_to_values(rows: pd.DataFrame, recordings: dict, candidate: str) -> np.ndarray:
    result = np.empty((len(rows), recordings[next(iter(recordings))][candidate].shape[1]), dtype=np.float32)
    for subject, indices in rows.groupby("subject", sort=False).groups.items():
        recording = recordings[subject]
        lookup = {
            int(round(float(time) * 2)): index
            for index, time in enumerate(recording["times"])
        }
        local_rows = rows.loc[indices]
        local_index = [lookup[int(round(float(time) * 2))] for time in local_rows.candidate_time_sec]
        result[np.asarray(indices, dtype=int)] = recording[candidate][local_index]
    return result


def candidate_targets(targets: dict, recordings: dict) -> dict:
    negative_rows = targets["negative"]["rows"].reset_index(drop=True)
    positive_rows = targets["CAUSAL-RW"]["rows"].reset_index(drop=True)
    result = {}
    for candidate in CANDIDATES:
        if candidate == "BASE-PEAK":
            negative_values = targets["negative"]["values"]
            positive_values = targets["CAUSAL-RW"]["values"]
        else:
            negative_values = rows_to_values(negative_rows, recordings, candidate)
            positive_values = rows_to_values(positive_rows, recordings, candidate)
        result[candidate] = {
            "rows": pd.concat([negative_rows, positive_rows], ignore_index=True),
            "values": np.concatenate([negative_values, positive_values], axis=0),
        }
        result[candidate]["labels"] = result[candidate]["rows"].label.to_numpy(dtype=int)
    return result


def fit_model(values: np.ndarray, labels: np.ndarray, feature_names: list[str]) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(values)
    model = LogisticRegression(
        C=C_VALUE,
        penalty="l2",
        solver="lbfgs",
        class_weight=None,
        max_iter=MAX_ITER,
        tol=1e-6,
        random_state=BASE_SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(values), labels)
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    payload = {
        "configuration": {
            "C": C_VALUE,
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": None,
            "max_iter": MAX_ITER,
            "tol": 1e-6,
            "random_state": BASE_SEED,
        },
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
    }
    return {"scaler": scaler, "model": model}, payload


def run_fit(
    candidate: str,
    outer: int,
    phase: str,
    fitting_pids: set[int],
    heldout: pd.DataFrame,
    prepared_targets: dict,
    recordings: dict,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    target = prepared_targets[candidate]
    mask = target["rows"].pid.isin(fitting_pids).to_numpy()
    labels = target["labels"][mask]
    names = feature_names[: target["values"].shape[1]]
    fitted, payload = fit_model(target["values"][mask], labels, names)
    frames, support = [], []
    for item in heldout.itertuples(index=False):
        recording = recordings[item.subject]
        probability = fitted["model"].predict_proba(
            fitted["scaler"].transform(recording[candidate])
        )[:, 1]
        frames.append(
            pd.DataFrame(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "phase": phase,
                    "subject": item.subject,
                    "pid": int(item.pid),
                    "candidate_time_sec": recording["times"],
                    "probability": probability,
                }
            )
        )
        support.append(
            {
                "subject": item.subject,
                "pid": int(item.pid),
                "outer_fold": outer,
                "supported_boundaries": len(probability),
                "supported_hours": len(probability) * 15.0 / 3600.0,
            }
        )
    scores = pd.concat(frames, ignore_index=True)
    probability = scores.probability.to_numpy(dtype=float)
    path = model_path(candidate, outer, phase)
    prior.verify_or_create_json_gzip(path, payload)
    summary = {
        "candidate": candidate,
        "outer_fold": outer,
        "phase": phase,
        "input_features": target["values"].shape[1],
        "training_rows": int(mask.sum()),
        "positive_rows": int(labels.sum()),
        "iterations": payload["iterations"],
        "converged": payload["converged"],
        "intercept": float(payload["intercept"][0]),
        "minimum_probability": float(probability.min()),
        "maximum_probability": float(probability.max()),
        "unique_probabilities": int(np.unique(probability).size),
        "exact_endpoint_fraction": float(np.mean((probability == 0.0) | (probability == 1.0))),
        "all_values_finite": bool(np.isfinite(probability).all()),
        "model_relative_path": path.relative_to(data_parent()).as_posix(),
        "model_bytes": path.stat().st_size,
        "model_sha256": prior.sha256(path),
    }
    print(f"fit {candidate} outer={outer} phase={phase}", flush=True)
    return scores, pd.DataFrame(support), summary


# Section 4: peak-first alarm evaluation

def segment_peaks(times: np.ndarray, probability: np.ndarray) -> np.ndarray:
    if len(times) == 0:
        return np.asarray([], dtype=int)
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) > 15.0 + 1e-6) + 1, len(times)]
    selected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = probability[start:stop]
        previous = np.r_[-np.inf, local[:-1]]
        following = np.r_[local[1:], -np.inf]
        peaks = start + np.flatnonzero((local > previous) & (local >= following))
        ranked = sorted(peaks, key=lambda index: (-probability[index], times[index]))
        local_selected = []
        for index in ranked:
            if all(
                abs(times[index] - times[other]) > REFRACTORY_SEC
                for other in local_selected
            ):
                local_selected.append(index)
        selected.extend(local_selected)
    return np.asarray(sorted(selected, key=lambda index: times[index]), dtype=int)


def peak_scores(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for (_, _), group in scores.groupby(["subject", "pid"], sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        selected = segment_peaks(
            group.candidate_time_sec.to_numpy(dtype=float),
            group.probability.to_numpy(dtype=float),
        )
        frames.append(group.iloc[selected])
    return pd.concat(frames, ignore_index=True) if frames else scores.iloc[0:0].copy()


def predictions_from_peaks(peaks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    selected = peaks[peaks.probability.ge(threshold)]
    return selected[["subject", "pid", "candidate_time_sec"]].rename(
        columns={"candidate_time_sec": "event_time_sec"}
    ).reset_index(drop=True)


def threshold_curve(
    peaks: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for threshold in formulation.threshold_grid():
        predictions = predictions_from_peaks(peaks, float(threshold))
        summary, _, _ = causal.evaluate_predictions(
            predictions, support, references, "REM_to_Wake"
        )
        rows.append({"threshold": float(threshold), **summary})
    frame = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    frame["alarm_count_monotonic"] = frame.predicted_events.diff().fillna(0).le(0)
    return frame


def select_threshold(curve: pd.DataFrame) -> dict:
    eligible = curve[curve.false_alarms_per_hour.le(PRIMARY_BUDGET + 1e-12)]
    if eligible.empty:
        raise ValueError("No peak threshold satisfies the alarm budget")
    selected = eligible.sort_values(
        ["f1", "recall", "false_alarms_per_hour", "threshold"],
        ascending=[False, False, True, False],
        kind="stable",
    ).iloc[0].to_dict()
    selected["alarm_budget_per_hour"] = PRIMARY_BUDGET
    selected["selection_rule"] = "max_f1_then_recall_then_min_far_then_threshold_under_budget"
    return selected


def run_nested(
    assignments: pd.DataFrame,
    folds: pd.DataFrame,
    prepared_targets: dict,
    recordings: dict,
    references: pd.DataFrame,
    feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    fold_by_pid = folds.set_index("pid").fold.astype(int).to_dict()
    fold_values = sorted(folds.fold.astype(int).unique())
    fits, curves, selections, outer_scores, outer_support = [], [], [], [], []
    for outer in fold_values:
        inner_scores = {candidate: [] for candidate in CANDIDATES}
        inner_support = {candidate: [] for candidate in CANDIDATES}
        for inner in [value for value in fold_values if value != outer]:
            fitting_pids = {
                int(pid) for pid, fold in fold_by_pid.items() if fold not in {outer, inner}
            }
            heldout = assignments[assignments.pid.map(fold_by_pid).eq(inner)]
            for candidate in CANDIDATES:
                scores, support, summary = run_fit(
                    candidate,
                    outer,
                    f"inner_{inner}",
                    fitting_pids,
                    heldout,
                    prepared_targets,
                    recordings,
                    feature_names,
                )
                inner_scores[candidate].append(scores)
                inner_support[candidate].append(support)
                fits.append(summary)
        for candidate in CANDIDATES:
            scores = pd.concat(inner_scores[candidate], ignore_index=True)
            support = pd.concat(inner_support[candidate], ignore_index=True)
            peaks = peak_scores(scores)
            curve = threshold_curve(peaks, support, references)
            curve.insert(0, "candidate", candidate)
            curve.insert(1, "outer_fold", outer)
            curve["peak_candidates"] = len(peaks)
            curves.append(curve)
            selections.append(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "selection_scope": "REM_to_Wake",
                    **select_threshold(curve),
                }
            )
        fitting_pids = {int(pid) for pid, fold in fold_by_pid.items() if fold != outer}
        heldout = assignments[assignments.pid.map(fold_by_pid).eq(outer)]
        for candidate in CANDIDATES:
            scores, support, summary = run_fit(
                candidate,
                outer,
                "outer_final",
                fitting_pids,
                heldout,
                prepared_targets,
                recordings,
                feature_names,
            )
            outer_scores.append(scores)
            if candidate == CANDIDATES[0]:
                outer_support.append(support)
            fits.append(summary)
    scores = pd.concat(outer_scores, ignore_index=True)
    support = pd.concat(outer_support, ignore_index=True)
    selection_table = pd.DataFrame(selections)
    alarm_frames, peak_rows = [], []
    for candidate in CANDIDATES:
        for outer in fold_values:
            local = scores[scores.candidate.eq(candidate) & scores.outer_fold.eq(outer)]
            peaks = peak_scores(local)
            threshold = float(
                selection_table[
                    selection_table.candidate.eq(candidate)
                    & selection_table.outer_fold.eq(outer)
                ].threshold.iloc[0]
            )
            alarms = predictions_from_peaks(peaks, threshold)
            alarms.insert(0, "candidate", candidate)
            alarms.insert(1, "outer_fold", outer)
            alarms["threshold"] = threshold
            alarm_frames.append(alarms)
            peak_rows.append(
                {
                    "candidate": candidate,
                    "outer_fold": outer,
                    "score_rows": len(local),
                    "peak_candidates": len(peaks),
                    "selected_alarms": len(alarms),
                    "threshold": threshold,
                }
            )
    return {
        "fits": pd.DataFrame(fits),
        "curves": pd.concat(curves, ignore_index=True),
        "selections": selection_table,
        "scores": scores,
        "support": support,
        "alarms": pd.concat(alarm_frames, ignore_index=True),
        "peaks": pd.DataFrame(peak_rows),
    }


# Section 5: outer evaluation and decision

def evaluate_outer(nested: dict, references: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics, participants, matches = [], [], []
    for candidate in CANDIDATES:
        predictions = nested["alarms"][nested["alarms"].candidate.eq(candidate)]
        summary, local_participants, local_matches = causal.evaluate_predictions(
            predictions,
            nested["support"],
            references,
            "REM_to_Wake",
            return_details=True,
        )
        config = {
            "candidate": candidate,
            "scope": "REM_to_Wake",
            "partition": "train_nested_oof",
            "alarm_budget_per_hour": PRIMARY_BUDGET,
            "causal_match_start_sec": 0.0,
            "causal_match_stop_sec": 15.0,
        }
        metrics.append({**config, **summary})
        for key, value in reversed(list(config.items())):
            local_participants.insert(0, key, value)
            if len(local_matches):
                local_matches.insert(0, key, value)
        participants.append(local_participants)
        if len(local_matches):
            matches.append(local_matches)
    return {
        "metrics": pd.DataFrame(metrics),
        "participants": pd.concat(participants, ignore_index=True),
        "matches": pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(),
    }


def participant_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    advanced = participants[participants.candidate.eq("ADV-PEAK")][columns]
    base = participants[participants.candidate.eq("BASE-PEAK")][columns]
    paired = advanced.merge(base, on="pid", suffixes=("_adv", "_base"), validate="one_to_one")
    if len(paired) != 64:
        raise ValueError("Incomplete advanced/base participant comparison")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    for indices in rng.integers(0, len(paired), size=(BOOTSTRAP_RESAMPLES, len(paired))):
        sample = paired.iloc[indices]
        values = {}
        for side in ["adv", "base"]:
            values[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()),
                int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()),
                float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "f1_difference": values["adv"]["f1"] - values["base"]["f1"],
                "recall_difference": values["adv"]["recall"] - values["base"]["recall"],
                "false_alarms_per_hour_difference": (
                    values["adv"]["false_alarms_per_hour"]
                    - values["base"]["false_alarms_per_hour"]
                ),
            }
        )
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {
                "comparison": "ADV-PEAK_minus_BASE-PEAK",
                "metric": metric,
                "resamples": BOOTSTRAP_RESAMPLES,
                "seed": BASE_SEED,
                "lower_95": float(frame[metric].quantile(0.025)),
                "median": float(frame[metric].quantile(0.5)),
                "upper_95": float(frame[metric].quantile(0.975)),
            }
            for metric in frame.columns
        ]
    )


def decision_table(nested: dict, metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics.set_index("candidate")
    f1_difference = float(primary.loc["ADV-PEAK", "f1"] - primary.loc["BASE-PEAK", "f1"])
    far_difference = float(
        primary.loc["ADV-PEAK", "false_alarms_per_hour"]
        - primary.loc["BASE-PEAK", "false_alarms_per_hour"]
    )
    fits = nested["fits"]
    numerical = bool(
        len(fits) == 50
        and fits.converged.all()
        and fits.iterations.lt(MAX_ITER).all()
        and fits.all_values_finite.all()
        and fits.exact_endpoint_fraction.lt(0.01).all()
        and fits.unique_probabilities.gt(100).all()
    )
    monotonic = bool(nested["curves"].alarm_count_monotonic.all())
    point_gate = bool(f1_difference >= 0.03 and far_difference <= 0.05)
    decision = (
        "inconclusive_numerical_stop"
        if not numerical or not monotonic
        else "retain_for_new_confirmation" if point_gate else "stop_v0.1"
    )
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-ADV-CHANGE",
                "comparison": "ADV-PEAK_minus_BASE-PEAK",
                "f1_difference": f1_difference,
                "required_f1_difference": 0.03,
                "false_alarms_per_hour_difference": far_difference,
                "maximum_false_alarm_difference": 0.05,
                "numerical_gate_pass": numerical,
                "alarm_monotonicity_pass": monotonic,
                "point_gate_pass": numerical and monotonic and point_gate,
                "decision": decision,
            }
        ]
    )


def artifact_manifest(feature_summary: pd.DataFrame, fits: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    feature_rows = feature_summary[["relative_path", "bytes", "sha256"]].copy()
    feature_rows.insert(0, "artifact_type", "advanced_feature")
    feature_rows["candidate"] = "ADV-PEAK"
    feature_rows["outer_fold"] = "all"
    feature_rows["phase"] = "feature_cache"
    model_rows = fits[
        ["candidate", "outer_fold", "phase", "model_relative_path", "model_bytes", "model_sha256"]
    ].rename(
        columns={"model_relative_path": "relative_path", "model_bytes": "bytes", "model_sha256": "sha256"}
    )
    model_rows.insert(0, "artifact_type", "model")
    score_rows = []
    for candidate in CANDIDATES:
        path = score_path(candidate)
        prior.verify_or_create_gzip_tsv(path, scores[scores.candidate.eq(candidate)].copy())
        score_rows.append(
            {
                "artifact_type": "outer_score",
                "candidate": candidate,
                "outer_fold": "all",
                "phase": "outer_oof",
                "relative_path": path.relative_to(data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": prior.sha256(path),
            }
        )
    return pd.concat([feature_rows, model_rows, pd.DataFrame(score_rows)], ignore_index=True)


# Section 6: execution

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output_dir().mkdir(parents=True, exist_ok=True)

    assignments = prior.train_assignments()
    folds = prior.frozen_fold_assignments(assignments)
    synthetic = synthetic_controls()
    if not synthetic.passed.all():
        raise ValueError("Synthetic advanced-feature controls failed")
    recordings, feature_summary, combined_names = prepare_recordings(assignments)
    references = causal.load_references(assignments)
    targets, target_summary, event_support = causal.build_targets(assignments, recordings, references)
    supported_ids = set(
        event_support.loc[event_support.causal_context_supported, "transition_id"].astype(int)
    )
    references["causal_primary_eligible"] = (
        prior.truth(references.primary_analysis_eligible)
        & references.transition_id.astype(int).isin(supported_ids)
    )
    schema = pd.DataFrame(
        {
            "feature_index": np.arange(len(combined_names)),
            "feature_name": combined_names,
            "representation": ["base"] * 80 + ["advanced"] * ADVANCED_ONLY_FEATURES,
        }
    )
    audit = pd.DataFrame(
        [
            {"check": "train_recordings", "observed": len(assignments), "expected": 82, "passed": len(assignments) == 82},
            {"check": "train_pid", "observed": assignments.pid.nunique(), "expected": 64, "passed": assignments.pid.nunique() == 64},
            {"check": "grid_decisions", "observed": int(feature_summary.grid_decisions.sum()), "expected": 150967, "passed": int(feature_summary.grid_decisions.sum()) == 150967},
            {"check": "base_features", "observed": 80, "expected": 80, "passed": True},
            {"check": "advanced_only_features", "observed": ADVANCED_ONLY_FEATURES, "expected": 180, "passed": ADVANCED_ONLY_FEATURES == 180},
            {"check": "combined_features", "observed": len(combined_names), "expected": 260, "passed": len(combined_names) == 260},
            {"check": "advanced_values_finite", "observed": bool(feature_summary.all_values_finite.all()), "expected": True, "passed": bool(feature_summary.all_values_finite.all())},
            {"check": "advanced_bounds", "observed": bool(feature_summary.feature_bounds_pass.all()), "expected": True, "passed": bool(feature_summary.feature_bounds_pass.all())},
            {"check": "primary_causal_events", "observed": int(references.causal_primary_eligible.sum()), "expected": 241, "passed": int(references.causal_primary_eligible.sum()) == 241},
            {"check": "rem_to_wake_positive_rows", "observed": len(targets["CAUSAL-RW"]["rows"]), "expected": 180, "passed": len(targets["CAUSAL-RW"]["rows"]) == 180},
            {"check": "future_signal_sec", "observed": float(feature_summary.future_signal_sec.max()), "expected": 0, "passed": feature_summary.future_signal_sec.eq(0).all()},
            {"check": "synthetic_controls", "observed": int(synthetic.passed.sum()), "expected": len(synthetic), "passed": bool(synthetic.passed.all())},
            {"check": "unauthorized_partitions", "observed": ";".join(sorted(assignments.partition.unique())), "expected": "train", "passed": set(assignments.partition) == {"train"}},
        ]
    )
    verify_or_create_tsv(synthetic, output_dir() / "synthetic_feature_controls_v0.1.tsv")
    verify_or_create_tsv(feature_summary, output_dir() / "feature_generation_summary_v0.1.tsv")
    verify_or_create_tsv(schema, output_dir() / "feature_schema_v0.1.tsv")
    verify_or_create_tsv(target_summary, output_dir() / "target_construction_summary_v0.1.tsv")
    verify_or_create_tsv(event_support, output_dir() / "primary_event_causal_support_v0.1.tsv")
    verify_or_create_tsv(audit, output_dir() / "pre_result_support_audit_v0.1.tsv")
    if not audit.passed.all():
        raise ValueError("Pre-result support audit failed")
    if args.audit_only:
        print(f"Pre-result support audit passed {len(audit)}/{len(audit)}; no model was fitted.")
        print(synthetic.to_string(index=False))
        return

    prepared_targets = candidate_targets(targets, recordings)
    nested = run_nested(
        assignments, folds, prepared_targets, recordings, references, combined_names
    )
    evaluated = evaluate_outer(nested, references)
    bootstrap = participant_bootstrap(evaluated["participants"])
    decision = decision_table(nested, evaluated["metrics"])
    manifest = artifact_manifest(feature_summary, nested["fits"], nested["scores"])

    old_scores = pd.read_csv(causal.outer_score_path("CAUSAL-RW"), sep="\t", compression="gzip")
    new_base = nested["scores"][nested["scores"].candidate.eq("BASE-PEAK")]
    reproduction = old_scores.merge(
        new_base,
        on=["outer_fold", "subject", "pid", "candidate_time_sec"],
        suffixes=("_old", "_new"),
        validate="one_to_one",
    )
    reproduction_table = pd.DataFrame(
        [
            {
                "rows": len(reproduction),
                "maximum_absolute_probability_difference": float(
                    np.max(np.abs(reproduction.probability_old - reproduction.probability_new))
                ),
            }
        ]
    )
    checks = pd.DataFrame(
        [
            {"check": "model_fit_rows", "observed": len(nested["fits"]), "expected": 50, "passed": len(nested["fits"]) == 50},
            {"check": "all_models_converged", "observed": bool(nested["fits"].converged.all()), "expected": True, "passed": bool(nested["fits"].converged.all())},
            {"check": "outer_score_rows", "observed": len(nested["scores"]), "expected": 301934, "passed": len(nested["scores"]) == 301934},
            {"check": "outer_pid", "observed": nested["scores"].pid.nunique(), "expected": 64, "passed": nested["scores"].pid.nunique() == 64},
            {"check": "outer_subjects", "observed": nested["scores"].subject.nunique(), "expected": 82, "passed": nested["scores"].subject.nunique() == 82},
            {"check": "inner_alarm_monotonic", "observed": bool(nested["curves"].alarm_count_monotonic.all()), "expected": True, "passed": bool(nested["curves"].alarm_count_monotonic.all())},
            {"check": "baseline_score_rows_reproduced", "observed": len(reproduction), "expected": 150967, "passed": len(reproduction) == 150967},
            {"check": "baseline_score_max_difference", "observed": float(reproduction_table.maximum_absolute_probability_difference.iloc[0]), "expected": "<=1e-12", "passed": float(reproduction_table.maximum_absolute_probability_difference.iloc[0]) <= 1e-12},
            {"check": "bootstrap_rows", "observed": len(bootstrap), "expected": 3, "passed": len(bootstrap) == 3},
            {"check": "validation_or_test_opened", "observed": False, "expected": False, "passed": True},
        ]
    )
    if not checks.passed.all():
        raise ValueError("In-run controls failed")
    outputs = {
        "model_fit_summary_v0.1.tsv": nested["fits"],
        "inner_threshold_curves_v0.1.tsv": nested["curves"],
        "inner_threshold_selections_v0.1.tsv": nested["selections"],
        "outer_peak_summary_v0.1.tsv": nested["peaks"],
        "outer_support_v0.1.tsv": nested["support"],
        "outer_predicted_events_v0.1.tsv": nested["alarms"],
        "outer_event_metrics_v0.1.tsv": evaluated["metrics"],
        "outer_event_participants_v0.1.tsv": evaluated["participants"],
        "outer_event_matches_v0.1.tsv": evaluated["matches"],
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decision_v0.1.tsv": decision,
        "baseline_score_reproduction_v0.1.tsv": reproduction_table,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    write_json(
        output_dir() / "software_versions_v0.1.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "protocol_commit": PROTOCOL_COMMIT,
            "execution_commit": current_commit(),
        },
    )
    print(evaluated["metrics"].to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(decision.to_string(index=False))


if __name__ == "__main__":
    main()
