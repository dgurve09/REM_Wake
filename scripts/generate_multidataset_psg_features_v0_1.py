"""Generate and audit fixed common-channel PSG features without fitting models."""

from __future__ import annotations

import hashlib
import io
import json
import platform
import shutil
import subprocess
import zipfile
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter, csd, sosfilt, welch

from reviewed_output import verify_or_create_tsv


# Section 1: frozen configuration

EXPERIMENT_DIR = "2026-10-25_multidataset_psg_external_generalization_v0.1"
DERIVED_DIR = "multidataset_psg_external_generalization_v0.1"
PROTOCOL_COMMIT = "3880c8d"
CHANNEL_MAP = {
    "BOAS": {"F4": "PSG_F4", "C4": "PSG_C4", "O2": "PSG_O2"},
    "gx36m": {"F4": "F4-A1", "C4": "C4-A1", "O2": "O2-A1"},
    "ISRUC-3": {"F4": "F4-A1", "C4": "C4-A1", "O2": "O2-A1"},
}
CANONICAL_CHANNELS = ["F4", "C4", "O2"]
CHANNEL_PAIRS = [("F4", "C4"), ("F4", "O2"), ("C4", "O2")]
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "sigma": (12.0, 16.0),
    "beta": (16.0, 30.0),
}
SUBWINDOW_STARTS = [-60, -45, -30, -15]
BASE_WINDOW_STARTS = list(range(-240, 0, 30))
BASE_FEATURES = 120
SUBWINDOW_FEATURES = 54
ADVANCED_ONLY_FEATURES = 324
COMBINED_FEATURES = 444
EPSILON = np.finfo(float).eps
BASE_SEED = 20261025


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def project_root() -> Path:
    return repo_root().parents[1]


def data_parent() -> Path:
    return repo_root().parent / "REM_W_data"


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def eligibility_path() -> Path:
    return (
        repo_root()
        / "experiments/2026-10-25_multidataset_psg_pool_audit_v0.1/recording_eligibility_v0.1.tsv"
    )


def current_commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={repo_root().as_posix()}", "rev-parse", "HEAD"],
        cwd=repo_root(),
        text=True,
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    expected = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
        raise RuntimeError(f"Reviewed output changed: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expected, encoding="utf-8")


def feature_path(dataset: str, recording: str) -> Path:
    safe_dataset = dataset.lower().replace("-", "_")
    return derived_dir() / "features" / safe_dataset / f"{recording}_features_v0.1.npz"


# Section 2: feature schemas

def band_names() -> list[str]:
    return [f"{channel}_{band}_log10_mean_psd" for channel in CANONICAL_CHANNELS for band in BANDS]


def base_feature_names() -> list[str]:
    names = [
        f"start_{start:+d}s_{name}"
        for start in BASE_WINDOW_STARTS
        for name in band_names()
    ]
    if len(names) != BASE_FEATURES:
        raise ValueError("Baseline feature schema changed")
    return names


def subwindow_feature_names() -> list[str]:
    names = []
    for channel in CANONICAL_CHANNELS:
        names.extend(f"{channel}_{band}_log10_mean_psd" for band in BANDS)
        names.extend(
            [
                f"{channel}_spectral_entropy",
                f"{channel}_aperiodic_loglog_slope",
                f"{channel}_aperiodic_loglog_intercept",
                f"{channel}_log10_rms",
                f"{channel}_log10_line_length_per_sec",
                f"{channel}_hjorth_mobility_hz",
                f"{channel}_hjorth_complexity",
            ]
        )
    for left, right in CHANNEL_PAIRS:
        names.append(f"{left}_{right}_correlation")
        names.extend(f"{left}_{right}_{band}_coherence" for band in BANDS)
    if len(names) != SUBWINDOW_FEATURES:
        raise ValueError("Subwindow feature schema changed")
    return names


def advanced_feature_names() -> list[str]:
    local = subwindow_feature_names()
    names = [
        f"start_{start:+d}s_{name}"
        for start in SUBWINDOW_STARTS
        for name in local
    ]
    names.extend(f"post_minus_pre_{name}" for name in local)
    names.extend(f"abs_post_minus_pre_{name}" for name in local)
    if len(names) != ADVANCED_ONLY_FEATURES:
        raise ValueError("Advanced feature schema changed")
    return names


# Section 3: native-rate spectral and dynamical features

def welch_density(blocks: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    segment = int(round(4.0 * sampling_frequency))
    return welch(
        blocks,
        fs=sampling_frequency,
        window="hann",
        nperseg=segment,
        noverlap=segment // 2,
        axis=-1,
    )


def bandpower_features(blocks: np.ndarray, sampling_frequency: float) -> np.ndarray:
    frequencies, density = welch_density(blocks, sampling_frequency)
    columns = []
    for channel_index in range(3):
        local = np.maximum(density[:, channel_index, :], EPSILON)
        for low, high in BANDS.values():
            mask = (frequencies >= low) & (frequencies < high)
            columns.append(np.log10(local[:, mask].mean(axis=1)))
    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (len(blocks), 15) or not np.isfinite(result).all():
        raise ValueError("Bandpower feature validation failed")
    return result


def advanced_subwindow_features(blocks: np.ndarray, sampling_frequency: float) -> np.ndarray:
    frequencies, density = welch_density(blocks, sampling_frequency)
    spectral_mask = (frequencies >= 0.5) & (frequencies < 30.0)
    aperiodic_mask = (frequencies >= 2.0) & (frequencies <= 30.0)
    log_frequency = np.log10(frequencies[aperiodic_mask])
    centered_frequency = log_frequency - log_frequency.mean()
    frequency_denominator = float(np.sum(centered_frequency * centered_frequency))
    columns = []

    for channel_index in range(3):
        local_density = np.maximum(density[:, channel_index, :], EPSILON)
        for low, high in BANDS.values():
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
        sample_mobility = np.sqrt(variance_1 / variance_0)
        columns.extend(
            [
                np.log10(np.maximum(np.sqrt(np.mean(values * values, axis=1)), EPSILON)),
                np.log10(np.maximum(np.mean(np.abs(difference_1), axis=1) * sampling_frequency, EPSILON)),
                sample_mobility * sampling_frequency / (2.0 * np.pi),
                np.sqrt(variance_2 / variance_1) / np.maximum(sample_mobility, EPSILON),
            ]
        )

    for left_name, right_name in CHANNEL_PAIRS:
        left_index = CANONICAL_CHANNELS.index(left_name)
        right_index = CANONICAL_CHANNELS.index(right_name)
        left = blocks[:, left_index, :] - blocks[:, left_index, :].mean(axis=1, keepdims=True)
        right = blocks[:, right_index, :] - blocks[:, right_index, :].mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
        columns.append(np.clip(np.sum(left * right, axis=1) / np.maximum(denominator, EPSILON), -1.0, 1.0))
        coherence_frequency, cross_density = csd(
            blocks[:, left_index, :],
            blocks[:, right_index, :],
            fs=sampling_frequency,
            window="hann",
            nperseg=int(round(4.0 * sampling_frequency)),
            noverlap=int(round(2.0 * sampling_frequency)),
            axis=-1,
        )
        coherence = np.abs(cross_density) ** 2 / np.maximum(
            density[:, left_index, :] * density[:, right_index, :], EPSILON
        )
        coherence = np.clip(coherence, 0.0, 1.0)
        for low, high in BANDS.values():
            mask = (coherence_frequency >= low) & (coherence_frequency < high)
            columns.append(coherence[:, mask].mean(axis=1))

    result = np.column_stack(columns).astype(np.float32)
    if result.shape != (len(blocks), SUBWINDOW_FEATURES) or not np.isfinite(result).all():
        raise ValueError("Advanced subwindow feature validation failed")
    return result


def batched_features(blocks: np.ndarray, sampling_frequency: float, advanced: bool) -> np.ndarray:
    function = advanced_subwindow_features if advanced else bandpower_features
    return np.concatenate(
        [function(blocks[start : start + 256], sampling_frequency) for start in range(0, len(blocks), 256)],
        axis=0,
    )


def batched_30s_bandpower(blocks15: np.ndarray, sampling_frequency: float) -> np.ndarray:
    rows = []
    for start in range(0, len(blocks15) - 1, 256):
        stop = min(start + 256, len(blocks15) - 1)
        blocks30 = np.concatenate([blocks15[start:stop], blocks15[start + 1 : stop + 1]], axis=2)
        rows.append(bandpower_features(blocks30, sampling_frequency))
    return np.concatenate(rows, axis=0)


def feature_bounds(values: np.ndarray) -> bool:
    names = advanced_feature_names()[: 4 * SUBWINDOW_FEATURES]
    raw = values[:, : len(names)]
    entropy = [index for index, name in enumerate(names) if name.endswith("spectral_entropy")]
    correlation = [index for index, name in enumerate(names) if name.endswith("correlation")]
    coherence = [index for index, name in enumerate(names) if name.endswith("coherence")]
    nonnegative = [
        index
        for index, name in enumerate(names)
        if name.endswith(("hjorth_mobility_hz", "hjorth_complexity"))
    ]
    return bool(
        np.logical_and(raw[:, entropy] >= 0.0, raw[:, entropy] <= 1.0).all()
        and np.logical_and(raw[:, correlation] >= -1.0, raw[:, correlation] <= 1.0).all()
        and np.logical_and(raw[:, coherence] >= 0.0, raw[:, coherence] <= 1.0).all()
        and (raw[:, nonnegative] >= 0.0).all()
    )


# Section 4: raw-signal access

def absolute_source(relative: str) -> Path:
    return project_root() / relative


def gx_cache_path(recording: str) -> Path:
    return derived_dir() / "gx_edf_cache" / f"{recording}.edf"


def ensure_gx_edf(source: str, recording: str) -> Path:
    archive_relative, member = source.split("::", 1)
    archive = absolute_source(archive_relative)
    target = gx_cache_path(recording)
    with zipfile.ZipFile(archive) as package:
        expected = package.getinfo(member).file_size
        if target.exists() and target.stat().st_size != expected:
            raise RuntimeError(f"Extracted gx36m EDF size changed: {target}")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".edf.partial")
            with package.open(member) as source_file, temporary.open("wb") as destination:
                shutil.copyfileobj(source_file, destination, length=1024 * 1024)
            if temporary.stat().st_size != expected:
                raise RuntimeError(f"Incomplete gx36m extraction: {recording}")
            temporary.replace(target)
    return target


def load_signal(item) -> tuple[np.ndarray, float, Path]:
    path = (
        ensure_gx_edf(item.signal_source, item.recording)
        if item.dataset == "gx36m"
        else absolute_source(item.signal_source)
    )
    edf = edfio.read_edf(path, lazy_load_data=True)
    mapping = CHANNEL_MAP[item.dataset]
    signals = [edf.get_signal(mapping[channel]) for channel in CANONICAL_CHANNELS]
    frequencies = {float(signal.sampling_frequency) for signal in signals}
    if len(frequencies) != 1:
        raise ValueError(f"Mixed EEG sampling rates: {item.recording}")
    sampling_frequency = frequencies.pop()
    if any(signal.physical_dimension.lower() not in {"uv", "µv"} for signal in signals):
        raise ValueError(f"Unexpected EEG unit: {item.recording}")
    values = np.vstack([signal.data for signal in signals]).astype(np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite raw EEG: {item.recording}")
    return values, sampling_frequency, path


# Section 5: per-recording generation

def generate_recording(item) -> dict:
    path = feature_path(item.dataset, item.recording)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            times = cached["decision_time_sec"]
            baseline = cached["baseline"]
            advanced = cached["advanced_only"]
            sampling_frequency = float(cached["sampling_frequency_hz"])
        status = "created"
    else:
        signal, sampling_frequency, raw_path = load_signal(item)
        block_samples = int(round(15.0 * sampling_frequency))
        block_count = int(item.epochs) * 2
        required_samples = block_count * block_samples
        if required_samples > signal.shape[1]:
            raise ValueError(f"Label support exceeds EEG: {item.recording}")
        filtered = sosfilt(
            butter(4, [0.3, 35.0], btype="bandpass", fs=sampling_frequency, output="sos"),
            signal[:, :required_samples],
            axis=1,
        )
        del signal
        blocks15 = filtered.reshape(3, block_count, block_samples).transpose(1, 0, 2)
        features15 = batched_features(blocks15, sampling_frequency, advanced=True)
        features30 = batched_30s_bandpower(blocks15, sampling_frequency)

        decisions = np.arange(16, block_count + 1, dtype=int)
        baseline = np.column_stack(
            [features30[decisions - 16 + 2 * offset] for offset in range(8)]
        ).astype(np.float32)
        parts = [features15[decisions - 4 + offset] for offset in range(4)]
        before, after = features15[decisions - 2], features15[decisions - 1]
        advanced = np.column_stack(parts + [after - before, np.abs(after - before)]).astype(np.float32)
        times = decisions.astype(float) * 15.0
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            decision_time_sec=times,
            baseline=baseline,
            advanced_only=advanced,
            baseline_feature_names=np.asarray(base_feature_names()),
            advanced_feature_names=np.asarray(advanced_feature_names()),
            sampling_frequency_hz=np.asarray(sampling_frequency),
            source_signal_bytes=np.asarray(raw_path.stat().st_size),
        )
        status = "created"

    if baseline.shape != (len(times), BASE_FEATURES):
        raise ValueError(f"Baseline feature shape changed: {item.recording}")
    if advanced.shape != (len(times), ADVANCED_ONLY_FEATURES):
        raise ValueError(f"Advanced feature shape changed: {item.recording}")
    if not np.isfinite(baseline).all() or not np.isfinite(advanced).all():
        raise ValueError(f"Nonfinite feature cache: {item.recording}")
    if not feature_bounds(advanced):
        raise ValueError(f"Advanced feature bounds failed: {item.recording}")
    return {
        "dataset": item.dataset,
        "recording": item.recording,
        "participant_group": item.participant_group,
        "sampling_frequency_hz": sampling_frequency,
        "decisions": len(times),
        "baseline_features": baseline.shape[1],
        "advanced_only_features": advanced.shape[1],
        "combined_features": baseline.shape[1] + advanced.shape[1],
        "earliest_decision_sec": float(times.min()),
        "latest_decision_sec": float(times.max()),
        "future_signal_sec": 0.0,
        "all_values_finite": True,
        "feature_bounds_pass": True,
        "cache_status": status,
        "relative_path": path.relative_to(data_parent()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


# Section 6: synthetic controls and execution

def synthetic_controls() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(BASE_SEED)
    for sampling_frequency in [200.0, 256.0, 500.0]:
        samples = int(15 * sampling_frequency)
        time = np.arange(samples) / sampling_frequency
        alpha = np.sin(2 * np.pi * 10.0 * time)
        white = rng.normal(size=samples)
        block = np.stack([alpha, alpha, alpha])[None, :, :]
        values = advanced_subwindow_features(block, sampling_frequency)[0]
        names = subwindow_feature_names()
        lookup = {name: index for index, name in enumerate(names)}
        alpha_power = values[lookup["F4_alpha_log10_mean_psd"]]
        other_power = max(
            values[lookup[f"F4_{band}_log10_mean_psd"]]
            for band in BANDS
            if band != "alpha"
        )
        rows.append(
            {
                "check": f"alpha_localization_{int(sampling_frequency)}hz",
                "observed": float(alpha_power - other_power),
                "expected": ">0",
                "passed": bool(alpha_power > other_power),
            }
        )
        original = np.vstack([white, white, white])
        modified = original.copy()
        modified[:, samples // 2 :] += 1000.0
        sos = butter(4, [0.3, 35.0], btype="bandpass", fs=sampling_frequency, output="sos")
        left = sosfilt(sos, original, axis=1)
        right = sosfilt(sos, modified, axis=1)
        difference = float(np.max(np.abs(left[:, : samples // 2] - right[:, : samples // 2])))
        rows.append(
            {
                "check": f"causal_filter_{int(sampling_frequency)}hz",
                "observed": difference,
                "expected": "0",
                "passed": difference == 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir().mkdir(parents=True, exist_ok=True)
    eligibility = pd.read_csv(eligibility_path(), sep="\t")
    synthetic = synthetic_controls()
    if not synthetic.passed.all():
        raise ValueError("Synthetic feature controls failed")

    rows = []
    for number, item in enumerate(eligibility.itertuples(index=False), start=1):
        rows.append(generate_recording(item))
        print(f"features {number:03d}/{len(eligibility)} {item.dataset} {item.recording}", flush=True)
    summary = pd.DataFrame(rows)
    schema = pd.DataFrame(
        {
            "feature_index": np.arange(COMBINED_FEATURES),
            "feature_name": base_feature_names() + advanced_feature_names(),
            "representation": ["baseline"] * BASE_FEATURES + ["advanced"] * ADVANCED_ONLY_FEATURES,
        }
    )
    expected_decisions = int((eligibility.epochs.astype(int) * 2 - 15).sum())
    checks = pd.DataFrame(
        [
            {"check": "recordings", "observed": len(summary), "expected": 196, "passed": len(summary) == 196},
            {"check": "participant_groups", "observed": summary.participant_group.nunique(), "expected": 130, "passed": summary.participant_group.nunique() == 130},
            {"check": "datasets", "observed": summary.dataset.nunique(), "expected": 3, "passed": summary.dataset.nunique() == 3},
            {"check": "decision_rows", "observed": int(summary.decisions.sum()), "expected": expected_decisions, "passed": int(summary.decisions.sum()) == expected_decisions},
            {"check": "baseline_features", "observed": BASE_FEATURES, "expected": 120, "passed": BASE_FEATURES == 120},
            {"check": "advanced_only_features", "observed": ADVANCED_ONLY_FEATURES, "expected": 324, "passed": ADVANCED_ONLY_FEATURES == 324},
            {"check": "combined_features", "observed": COMBINED_FEATURES, "expected": 444, "passed": COMBINED_FEATURES == 444},
            {"check": "native_sampling_rates", "observed": ";".join(map(str, sorted(summary.sampling_frequency_hz.unique()))), "expected": "200.0;256.0;500.0", "passed": set(summary.sampling_frequency_hz) == {200.0, 256.0, 500.0}},
            {"check": "all_values_finite", "observed": bool(summary.all_values_finite.all()), "expected": True, "passed": bool(summary.all_values_finite.all())},
            {"check": "feature_bounds", "observed": bool(summary.feature_bounds_pass.all()), "expected": True, "passed": bool(summary.feature_bounds_pass.all())},
            {"check": "future_signal_sec", "observed": float(summary.future_signal_sec.max()), "expected": 0.0, "passed": summary.future_signal_sec.eq(0.0).all()},
            {"check": "synthetic_controls", "observed": int(synthetic.passed.sum()), "expected": len(synthetic), "passed": bool(synthetic.passed.all())},
            {"check": "model_fits_performed", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Multi-dataset feature audit failed")

    verify_or_create_tsv(summary, output_dir() / "feature_generation_summary_v0.1.tsv")
    verify_or_create_tsv(schema, output_dir() / "feature_schema_v0.1.tsv")
    verify_or_create_tsv(synthetic, output_dir() / "synthetic_feature_controls_v0.1.tsv")
    verify_or_create_tsv(checks, output_dir() / "pre_model_feature_audit_v0.1.tsv")
    write_json(
        output_dir() / "software_versions_v0.1.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "edfio": edfio.__version__,
            "protocol_commit": PROTOCOL_COMMIT,
            "execution_commit": current_commit(),
        },
    )
    print(f"Pre-model feature audit passed {len(checks)}/{len(checks)}; no model was fitted.")
    print(checks.to_string(index=False))


if __name__ == "__main__":
    main()
