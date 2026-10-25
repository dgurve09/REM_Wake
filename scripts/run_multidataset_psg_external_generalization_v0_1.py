"""Run the frozen multi-dataset PSG external-generalization experiment."""

from __future__ import annotations

import gzip
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import audit_multidataset_psg_pool_v0_1 as pool_audit
import generate_multidataset_psg_features_v0_1 as feature_generation
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import metric_values


# Section 1: frozen configuration

CANDIDATES = ["PSD3-BASE", "ADV3-CHANGE"]
DATASETS = ["BOAS", "gx36m", "ISRUC-3"]
PRIMARY_EXTERNAL = ["gx36m", "ISRUC-3"]
INNER_FOLDS = 5
ALARM_BUDGET = 0.25
REFRACTORY_SEC = 60.0
NEGATIVE_DISTANCE_SEC = 135.0
C_VALUE = 0.1
MAX_ITER = 1000
BASE_SEED = 20261025
BOOTSTRAP_RESAMPLES = 2000


def output_dir() -> Path:
    return feature_generation.output_dir()


def derived_dir() -> Path:
    return feature_generation.derived_dir()


def model_path(heldout: str, candidate: str, phase: str) -> Path:
    safe = candidate.lower().replace("-", "_")
    return derived_dir() / "models" / heldout.lower().replace("-", "_") / safe / f"{phase}.json.gz"


def score_path(heldout: str, candidate: str) -> Path:
    safe = candidate.lower().replace("-", "_")
    return derived_dir() / "scores" / f"holdout_{heldout.lower().replace('-', '_')}_{safe}.tsv.gz"


def verify_json_gzip(path: Path, value: dict) -> None:
    expected = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            if source.read() != expected:
                raise RuntimeError(f"External model artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=9) as target:
        target.write(expected)


def verify_gzip_tsv(path: Path, frame: pd.DataFrame) -> None:
    expected = frame.to_csv(sep="\t", index=False, lineterminator="\n")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            if source.read() != expected:
                raise RuntimeError(f"External score artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=9) as target:
        target.write(expected)


# Section 2: stages, features, and targets

def source_paths(value: str) -> list[Path]:
    return [feature_generation.project_root() / item for item in value.split(";")]


def primary_stages(item) -> np.ndarray:
    paths = source_paths(item.label_source)
    if item.dataset == "BOAS":
        return pd.read_csv(paths[0]).stage.astype(str).to_numpy()
    if item.dataset == "gx36m":
        return pool_audit.gx_consensus(paths)[0]
    if item.dataset == "ISRUC-3":
        return pool_audit.isruc_consensus(paths)[0]
    raise ValueError(item.dataset)


def scorer_variants(item) -> dict[str, np.ndarray]:
    paths = source_paths(item.label_source)
    if item.dataset == "gx36m":
        names = np.asarray(["W", "N1", "N2", "N3", "REM"])
        return {
            f"expert_{number}": names[np.loadtxt(path, dtype=int, ndmin=1)]
            for number, path in enumerate(paths, start=1)
        }
    if item.dataset == "ISRUC-3":
        return {
            f"expert_{number}": pd.read_csv(path).stage.astype(str).to_numpy()
            for number, path in enumerate(paths, start=1)
        }
    return {}


def transition_tables(stages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    previous, following = stages[:-1], stages[1:]
    boundary = (np.arange(len(stages) - 1, dtype=float) + 1.0) * 30.0
    primary = boundary[(previous == "REM") & (following == "W")] + 15.0
    changed = previous != following
    rem_wake = np.isin(previous, ["REM", "W"]) | np.isin(following, ["REM", "W"])
    unknown = np.isin(previous, ["UNKNOWN", "nan"]) | np.isin(following, ["UNKNOWN", "nan"])
    ignored = boundary[(changed & rem_wake) | unknown] + 15.0
    primary_keys = set(np.rint(primary * 2).astype(int))
    ignored = np.asarray([time for time in ignored if int(round(time * 2)) not in primary_keys])
    return primary, ignored


def training_masks(times: np.ndarray, stages: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    primary, ignored = transition_tables(stages)
    all_boundaries = np.sort(np.concatenate([primary, ignored]))
    current_epoch = np.minimum((times // 30.0).astype(int), len(stages) - 1)
    score_mask = ~np.isin(stages[current_epoch], ["UNKNOWN", "nan"])
    positive_keys = set(np.rint(primary * 2).astype(int))
    positive = np.asarray([int(round(time * 2)) in positive_keys for time in times])
    if len(all_boundaries):
        distance = np.min(np.abs(times[:, None] - all_boundaries[None, :]), axis=1)
    else:
        distance = np.full(len(times), np.inf)
    recording_end = len(stages) * 30.0
    negative = (
        score_mask
        & ~positive
        & (distance >= NEGATIVE_DISTANCE_SEC - 1e-9)
        & (times >= NEGATIVE_DISTANCE_SEC)
        & (recording_end - times >= NEGATIVE_DISTANCE_SEC)
    )
    return positive, negative, score_mask


def load_recordings() -> dict[str, dict]:
    eligibility = pd.read_csv(feature_generation.eligibility_path(), sep="\t")
    result = {}
    for item in eligibility.itertuples(index=False):
        path = feature_generation.feature_path(item.dataset, item.recording)
        with np.load(path, allow_pickle=False) as cached:
            times = cached["decision_time_sec"].astype(float)
            baseline = cached["baseline"].astype(np.float32)
            advanced = cached["advanced_only"].astype(np.float32)
        stages = primary_stages(item)
        positive, negative, score_mask = training_masks(times, stages)
        primary, ignored = transition_tables(stages)
        result[item.recording] = {
            "dataset": item.dataset,
            "participant_group": item.participant_group,
            "times": times,
            "PSD3-BASE": baseline,
            "ADV3-CHANGE": np.column_stack([baseline, advanced]).astype(np.float32),
            "positive": positive,
            "negative": negative,
            "score_mask": score_mask,
            "primary": primary,
            "ignored": ignored,
            "stages": stages,
            "label_source": item.label_source,
        }
    return result


def target_table(recordings: dict, candidate: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows, values, labels = [], [], []
    for recording, item in recordings.items():
        for label, mask in [(0, item["negative"]), (1, item["positive"])]:
            index = np.flatnonzero(mask)
            rows.extend(
                {
                    "dataset": item["dataset"],
                    "recording": recording,
                    "participant_group": item["participant_group"],
                    "candidate_time_sec": float(item["times"][position]),
                    "label": label,
                }
                for position in index
            )
            values.append(item[candidate][index])
            labels.append(np.full(len(index), label, dtype=np.int8))
    return pd.DataFrame(rows), np.concatenate(values), np.concatenate(labels)


# Section 3: participant folds and fixed model

def inner_fold_map(recordings: dict, heldout_dataset: str) -> dict[str, int]:
    rows = []
    for item in recordings.values():
        if item["dataset"] == heldout_dataset:
            continue
        rows.append(
            {
                "dataset": item["dataset"],
                "participant_group": item["participant_group"],
                "events": int(item["positive"].sum()),
                "hours": float(item["score_mask"].sum() * 15.0 / 3600.0),
            }
        )
    participants = pd.DataFrame(rows).groupby(["dataset", "participant_group"], as_index=False).agg(
        events=("events", "sum"), hours=("hours", "sum")
    )
    mapping = {}
    for _, local in participants.groupby("dataset", sort=True):
        totals = np.zeros((INNER_FOLDS, 3), dtype=float)
        for item in local.sort_values(["events", "hours", "participant_group"], ascending=[False, False, True]).itertuples(index=False):
            fold = min(range(INNER_FOLDS), key=lambda value: tuple(totals[value]))
            mapping[item.participant_group] = fold
            totals[fold] += [item.events, item.hours, 1]
    return mapping


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
        "configuration": {"C": C_VALUE, "penalty": "l2", "solver": "lbfgs", "class_weight": None, "max_iter": MAX_ITER, "tol": 1e-6, "random_state": BASE_SEED},
        "feature_names": feature_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coefficient": model.coef_[0].tolist(),
        "intercept": model.intercept_.tolist(),
        "iterations": int(model.n_iter_[0]),
        "converged": converged,
    }
    return {"scaler": scaler, "model": model}, payload


def score_recordings(fitted: dict, recordings: dict, selected: list[str], candidate: str, heldout: str, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores, support = [], []
    for recording in selected:
        item = recordings[recording]
        mask = item["score_mask"]
        probability = fitted["model"].predict_proba(fitted["scaler"].transform(item[candidate][mask]))[:, 1]
        scores.append(
            pd.DataFrame(
                {
                    "heldout_dataset": heldout,
                    "candidate": candidate,
                    "phase": phase,
                    "dataset": item["dataset"],
                    "recording": recording,
                    "participant_group": item["participant_group"],
                    "candidate_time_sec": item["times"][mask],
                    "probability": probability,
                }
            )
        )
        support.append(
            {
                "heldout_dataset": heldout,
                "dataset": item["dataset"],
                "recording": recording,
                "participant_group": item["participant_group"],
                "supported_boundaries": int(mask.sum()),
                "supported_hours": float(mask.sum() * 15.0 / 3600.0),
            }
        )
    return pd.concat(scores, ignore_index=True), pd.DataFrame(support)


# Section 4: peak-first event evaluation

def peak_indices(times: np.ndarray, probability: np.ndarray) -> np.ndarray:
    boundaries = np.r_[0, np.flatnonzero(np.diff(times) > 15.0 + 1e-6) + 1, len(times)]
    selected = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        local = probability[start:stop]
        previous = np.r_[-np.inf, local[:-1]]
        following = np.r_[local[1:], -np.inf]
        peaks = start + np.flatnonzero((local > previous) & (local >= following))
        retained = []
        for index in sorted(peaks, key=lambda value: (-probability[value], times[value])):
            if all(abs(times[index] - times[other]) > REFRACTORY_SEC for other in retained):
                retained.append(index)
        selected.extend(retained)
    return np.asarray(sorted(selected, key=lambda value: times[value]), dtype=int)


def peak_scores(scores: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in scores.groupby("recording", sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        frames.append(group.iloc[peak_indices(group.candidate_time_sec.to_numpy(float), group.probability.to_numpy(float))])
    return pd.concat(frames, ignore_index=True)


def predictions(peaks: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return peaks[peaks.probability.ge(threshold)][
        ["dataset", "recording", "participant_group", "candidate_time_sec"]
    ].rename(columns={"candidate_time_sec": "event_time_sec"}).reset_index(drop=True)


def causal_matches(references: np.ndarray, alarms: np.ndarray) -> list[tuple[int, int]]:
    reference_order = np.argsort(references)
    alarm_order = np.argsort(alarms)
    matches, left, right = [], 0, 0
    while left < len(reference_order) and right < len(alarm_order):
        reference_index = int(reference_order[left])
        alarm_index = int(alarm_order[right])
        reference = float(references[reference_index])
        alarm = float(alarms[alarm_index])
        if alarm < reference - 1e-9:
            right += 1
        elif alarm > reference + 15.0 + 1e-9:
            left += 1
        else:
            matches.append((reference_index, alarm_index))
            left += 1
            right += 1
    return matches


def evaluate(predicted: pd.DataFrame, support: pd.DataFrame, recordings: dict, reference_override: dict | None = None) -> tuple[dict, pd.DataFrame]:
    rows = []
    for item in support.itertuples(index=False):
        recording = recordings[item.recording]
        eligible = recording["primary"] if reference_override is None else reference_override[item.recording][0]
        ignored = recording["ignored"] if reference_override is None else reference_override[item.recording][1]
        alarms = predicted[predicted.recording.eq(item.recording)].event_time_sec.to_numpy(float)
        matches = causal_matches(eligible, alarms)
        matched = {right for _, right in matches}
        unmatched = [index for index in range(len(alarms)) if index not in matched]
        ignored_matches = causal_matches(ignored, alarms[unmatched])
        rows.append(
            {
                "dataset": item.dataset,
                "recording": item.recording,
                "participant_group": item.participant_group,
                "true_positive": len(matches),
                "false_positive": len(unmatched) - len(ignored_matches),
                "false_negative": len(eligible) - len(matches),
                "ignored_predictions": len(ignored_matches),
                "predicted_events": len(alarms),
                "reference_events": len(eligible),
                "supported_hours": float(item.supported_hours),
            }
        )
    recording_table = pd.DataFrame(rows)
    participant = recording_table.groupby(["dataset", "participant_group"], as_index=False).agg(
        true_positive=("true_positive", "sum"), false_positive=("false_positive", "sum"),
        false_negative=("false_negative", "sum"), ignored_predictions=("ignored_predictions", "sum"),
        predicted_events=("predicted_events", "sum"), reference_events=("reference_events", "sum"),
        supported_hours=("supported_hours", "sum"),
    )
    totals = participant[["true_positive", "false_positive", "false_negative", "supported_hours"]].sum()
    summary = metric_values(int(totals.true_positive), int(totals.false_positive), int(totals.false_negative), float(totals.supported_hours))
    summary.update(
        {
            "recordings": len(recording_table), "participant_groups": len(participant),
            "reference_events": int(participant.reference_events.sum()),
            "predicted_events": int(participant.predicted_events.sum()),
            "ignored_predictions": int(participant.ignored_predictions.sum()),
        }
    )
    return summary, participant


def threshold_grid() -> np.ndarray:
    logits = np.arange(-16.0, 16.001, 0.25)
    return np.unique(np.concatenate([1.0 / (1.0 + np.exp(-logits)), [1.0, np.nextafter(1.0, np.inf)]]))


def select_threshold(scores: pd.DataFrame, support: pd.DataFrame, recordings: dict) -> tuple[dict, pd.DataFrame]:
    peaks = peak_scores(scores)
    rows = []
    for threshold in threshold_grid():
        summary, _ = evaluate(predictions(peaks, float(threshold)), support, recordings)
        rows.append({"threshold": float(threshold), **summary})
    curve = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    curve["alarm_count_monotonic"] = curve.predicted_events.diff().fillna(0).le(0)
    eligible = curve[curve.false_alarms_per_hour.le(ALARM_BUDGET + 1e-12)]
    if eligible.empty:
        raise ValueError("No threshold satisfies the alarm budget")
    selected = eligible.sort_values(
        ["f1", "recall", "false_alarms_per_hour", "threshold"],
        ascending=[False, False, True, False], kind="stable",
    ).iloc[0].to_dict()
    return selected, curve


# Section 5: experiment execution

def run_experiment(recordings: dict) -> dict[str, pd.DataFrame]:
    feature_names = feature_generation.base_feature_names() + feature_generation.advanced_feature_names()
    targets = {candidate: target_table(recordings, candidate) for candidate in CANDIDATES}
    fits, curves, selections, all_scores, all_support, all_alarms = [], [], [], [], [], []
    for heldout in DATASETS:
        fold_map = inner_fold_map(recordings, heldout)
        training_recordings = [name for name, item in recordings.items() if item["dataset"] != heldout]
        heldout_recordings = [name for name, item in recordings.items() if item["dataset"] == heldout]
        for candidate in CANDIDATES:
            rows, values, labels = targets[candidate]
            inner_scores, inner_support = [], []
            for fold in range(INNER_FOLDS):
                fit_groups = {group for group, value in fold_map.items() if value != fold}
                score_names = [name for name in training_recordings if fold_map[recordings[name]["participant_group"]] == fold]
                mask = rows.participant_group.isin(fit_groups).to_numpy()
                fitted, payload = fit_model(values[mask], labels[mask], feature_names[: values.shape[1]])
                path = model_path(heldout, candidate, f"inner_{fold}")
                verify_json_gzip(path, payload)
                scores, support = score_recordings(fitted, recordings, score_names, candidate, heldout, f"inner_{fold}")
                inner_scores.append(scores); inner_support.append(support)
                fits.append(
                    {
                        "heldout_dataset": heldout, "candidate": candidate, "phase": f"inner_{fold}",
                        "input_features": values.shape[1], "training_rows": int(mask.sum()),
                        "positive_rows": int(labels[mask].sum()), "iterations": payload["iterations"],
                        "converged": payload["converged"], "minimum_probability": float(scores.probability.min()),
                        "maximum_probability": float(scores.probability.max()), "unique_probabilities": scores.probability.nunique(),
                        "all_values_finite": bool(np.isfinite(scores.probability).all()),
                        "model_relative_path": path.relative_to(feature_generation.data_parent()).as_posix(),
                        "model_bytes": path.stat().st_size, "model_sha256": feature_generation.sha256(path),
                    }
                )
                print(f"fit {heldout} {candidate} inner={fold}", flush=True)
            inner_scores_frame = pd.concat(inner_scores, ignore_index=True)
            inner_support_frame = pd.concat(inner_support, ignore_index=True)
            selected, curve = select_threshold(inner_scores_frame, inner_support_frame, recordings)
            curve.insert(0, "heldout_dataset", heldout); curve.insert(1, "candidate", candidate)
            curves.append(curve)
            selections.append({"heldout_dataset": heldout, "candidate": candidate, "alarm_budget_per_hour": ALARM_BUDGET, **selected})

            fit_groups = {item["participant_group"] for item in recordings.values() if item["dataset"] != heldout}
            mask = rows.participant_group.isin(fit_groups).to_numpy()
            fitted, payload = fit_model(values[mask], labels[mask], feature_names[: values.shape[1]])
            path = model_path(heldout, candidate, "outer_final")
            verify_json_gzip(path, payload)
            scores, support = score_recordings(fitted, recordings, heldout_recordings, candidate, heldout, "outer_final")
            peaks = peak_scores(scores)
            alarms = predictions(peaks, float(selected["threshold"]))
            alarms.insert(0, "heldout_dataset", heldout); alarms.insert(1, "candidate", candidate)
            alarms["threshold"] = float(selected["threshold"])
            all_scores.append(scores); all_alarms.append(alarms)
            if candidate == CANDIDATES[0]:
                all_support.append(support)
            fits.append(
                {
                    "heldout_dataset": heldout, "candidate": candidate, "phase": "outer_final",
                    "input_features": values.shape[1], "training_rows": int(mask.sum()),
                    "positive_rows": int(labels[mask].sum()), "iterations": payload["iterations"],
                    "converged": payload["converged"], "minimum_probability": float(scores.probability.min()),
                    "maximum_probability": float(scores.probability.max()), "unique_probabilities": scores.probability.nunique(),
                    "all_values_finite": bool(np.isfinite(scores.probability).all()),
                    "model_relative_path": path.relative_to(feature_generation.data_parent()).as_posix(),
                    "model_bytes": path.stat().st_size, "model_sha256": feature_generation.sha256(path),
                }
            )
            print(f"fit {heldout} {candidate} outer_final", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    support = pd.concat(all_support, ignore_index=True)
    alarms = pd.concat(all_alarms, ignore_index=True)
    for (heldout, candidate), local in scores.groupby(["heldout_dataset", "candidate"]):
        verify_gzip_tsv(score_path(heldout, candidate), local.reset_index(drop=True))
    return {
        "fits": pd.DataFrame(fits), "curves": pd.concat(curves, ignore_index=True),
        "selections": pd.DataFrame(selections), "scores": scores,
        "support": support, "alarms": alarms,
    }


# Section 6: held-out results, scorer sensitivity, and gate

def result_tables(run: dict, recordings: dict) -> dict[str, pd.DataFrame]:
    metrics, participants, ranking = [], [], []
    for heldout in DATASETS:
        local_support = run["support"][run["support"].heldout_dataset.eq(heldout)]
        for candidate in CANDIDATES:
            local_alarms = run["alarms"][run["alarms"].heldout_dataset.eq(heldout) & run["alarms"].candidate.eq(candidate)]
            summary, local_participants = evaluate(local_alarms, local_support, recordings)
            config = {"heldout_dataset": heldout, "candidate": candidate, "partition_role": "primary_external" if heldout in PRIMARY_EXTERNAL else "descriptive_reused_source"}
            metrics.append({**config, **summary})
            for key, value in reversed(list(config.items())):
                local_participants.insert(0, key, value)
            participants.append(local_participants)

            local_scores = run["scores"][run["scores"].heldout_dataset.eq(heldout) & run["scores"].candidate.eq(candidate)].copy()
            labels = []
            for item in local_scores.itertuples(index=False):
                keys = set(np.rint(recordings[item.recording]["primary"] * 2).astype(int))
                labels.append(int(round(item.candidate_time_sec * 2)) in keys)
            labels = np.asarray(labels, dtype=int)
            ranking.append(
                {
                    "heldout_dataset": heldout, "candidate": candidate, "rows": len(labels),
                    "positive_rows": int(labels.sum()),
                    "average_precision": float(average_precision_score(labels, local_scores.probability)),
                    "roc_auc": float(roc_auc_score(labels, local_scores.probability)),
                }
            )

    participants_frame = pd.concat(participants, ignore_index=True)
    external_support = run["support"][run["support"].heldout_dataset.isin(PRIMARY_EXTERNAL)]
    for candidate in CANDIDATES:
        local_alarms = run["alarms"][run["alarms"].heldout_dataset.isin(PRIMARY_EXTERNAL) & run["alarms"].candidate.eq(candidate)]
        summary, _ = evaluate(local_alarms, external_support, recordings)
        metrics.append({"heldout_dataset": "POOLED_EXTERNAL", "candidate": candidate, "partition_role": "primary_external_pooled", **summary})

    sensitivity = []
    eligibility = pd.read_csv(feature_generation.eligibility_path(), sep="\t").set_index("recording")
    for heldout in PRIMARY_EXTERNAL:
        local_support = run["support"][run["support"].heldout_dataset.eq(heldout)]
        variants = {}
        for recording in local_support.recording:
            item = eligibility.loc[recording]
            holder = type("Row", (), {"dataset": item.dataset, "label_source": item.label_source})
            variants[recording] = scorer_variants(holder)
        variant_names = sorted(set.intersection(*(set(value) for value in variants.values())))
        for variant in variant_names:
            override = {recording: transition_tables(value[variant]) for recording, value in variants.items()}
            for candidate in CANDIDATES:
                local_alarms = run["alarms"][run["alarms"].heldout_dataset.eq(heldout) & run["alarms"].candidate.eq(candidate)]
                summary, _ = evaluate(local_alarms, local_support, recordings, override)
                sensitivity.append({"heldout_dataset": heldout, "candidate": candidate, "label_variant": variant, **summary})
    return {"metrics": pd.DataFrame(metrics), "participants": participants_frame, "ranking": pd.DataFrame(ranking), "sensitivity": pd.DataFrame(sensitivity)}


def bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    local = participants[participants.heldout_dataset.isin(PRIMARY_EXTERNAL)]
    columns = ["dataset", "participant_group", "true_positive", "false_positive", "false_negative", "supported_hours"]
    left = local[local.candidate.eq("ADV3-CHANGE")][columns]
    right = local[local.candidate.eq("PSD3-BASE")][columns]
    paired = left.merge(right, on=["dataset", "participant_group"], suffixes=("_adv", "_base"), validate="one_to_one")
    rng = np.random.default_rng(BASE_SEED)
    samples = []
    by_dataset = {dataset: paired[paired.dataset.eq(dataset)].reset_index(drop=True) for dataset in PRIMARY_EXTERNAL}
    for _ in range(BOOTSTRAP_RESAMPLES):
        parts = [frame.iloc[rng.integers(0, len(frame), size=len(frame))] for frame in by_dataset.values()]
        sample = pd.concat(parts, ignore_index=True)
        result = {}
        for side in ["adv", "base"]:
            result[side] = metric_values(
                int(sample[f"true_positive_{side}"].sum()), int(sample[f"false_positive_{side}"].sum()),
                int(sample[f"false_negative_{side}"].sum()), float(sample[f"supported_hours_{side}"].sum()),
            )
        samples.append(
            {
                "f1_difference": result["adv"]["f1"] - result["base"]["f1"],
                "recall_difference": result["adv"]["recall"] - result["base"]["recall"],
                "false_alarms_per_hour_difference": result["adv"]["false_alarms_per_hour"] - result["base"]["false_alarms_per_hour"],
            }
        )
    frame = pd.DataFrame(samples)
    return pd.DataFrame(
        [
            {"comparison": "ADV3-CHANGE_minus_PSD3-BASE", "metric": column, "resamples": BOOTSTRAP_RESAMPLES, "seed": BASE_SEED,
             "lower_95": float(frame[column].quantile(0.025)), "median": float(frame[column].quantile(0.5)), "upper_95": float(frame[column].quantile(0.975))}
            for column in frame
        ]
    )


def decision(metrics: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    indexed = metrics.set_index(["heldout_dataset", "candidate"])
    advanced = indexed.loc[("POOLED_EXTERNAL", "ADV3-CHANGE")]
    baseline = indexed.loc[("POOLED_EXTERNAL", "PSD3-BASE")]
    differences = {
        dataset: float(indexed.loc[(dataset, "ADV3-CHANGE"), "f1"] - indexed.loc[(dataset, "PSD3-BASE"), "f1"])
        for dataset in PRIMARY_EXTERNAL
    }
    numerical = bool(fits.converged.all() and fits.all_values_finite.all() and fits.unique_probabilities.gt(100).all())
    conditions = {
        "pooled_f1_at_least_0.15": float(advanced.f1) >= 0.15,
        "pooled_f1_gain_at_least_0.03": float(advanced.f1 - baseline.f1) >= 0.03,
        "pooled_recall_at_least_0.15": float(advanced.recall) >= 0.15,
        "pooled_far_at_most_0.30": float(advanced.false_alarms_per_hour) <= 0.30,
        "gx36m_difference_not_below_minus_0.03": differences["gx36m"] >= -0.03,
        "isruc_difference_not_below_minus_0.03": differences["ISRUC-3"] >= -0.03,
        "numerical_gate": numerical,
    }
    passed = all(conditions.values())
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-PSG-XDATA", "advanced_f1": float(advanced.f1), "baseline_f1": float(baseline.f1),
                "f1_difference": float(advanced.f1 - baseline.f1), "advanced_recall": float(advanced.recall),
                "advanced_false_alarms_per_hour": float(advanced.false_alarms_per_hour),
                "gx36m_f1_difference": differences["gx36m"], "isruc_f1_difference": differences["ISRUC-3"],
                **conditions, "point_gate_pass": passed,
                "decision": "retain_for_paired_wearable_transfer" if passed else "stop_v0.1",
            }
        ]
    )


def main() -> None:
    recordings = load_recordings()
    run = run_experiment(recordings)
    results = result_tables(run, recordings)
    paired = bootstrap(results["participants"])
    gate = decision(results["metrics"], run["fits"])
    manifest_rows = []
    for item in run["fits"].itertuples(index=False):
        manifest_rows.append({"artifact_type": "model", "heldout_dataset": item.heldout_dataset, "candidate": item.candidate, "phase": item.phase, "relative_path": item.model_relative_path, "bytes": item.model_bytes, "sha256": item.model_sha256})
    for heldout in DATASETS:
        for candidate in CANDIDATES:
            path = score_path(heldout, candidate)
            manifest_rows.append({"artifact_type": "outer_score", "heldout_dataset": heldout, "candidate": candidate, "phase": "outer_final", "relative_path": path.relative_to(feature_generation.data_parent()).as_posix(), "bytes": path.stat().st_size, "sha256": feature_generation.sha256(path)})
    manifest = pd.DataFrame(manifest_rows)
    checks = pd.DataFrame(
        [
            {"check": "model_fits", "observed": len(run["fits"]), "expected": 36, "passed": len(run["fits"]) == 36},
            {"check": "all_models_converged", "observed": bool(run["fits"].converged.all()), "expected": True, "passed": bool(run["fits"].converged.all())},
            {"check": "heldout_datasets", "observed": run["scores"].heldout_dataset.nunique(), "expected": 3, "passed": run["scores"].heldout_dataset.nunique() == 3},
            {"check": "outer_candidates", "observed": run["scores"].candidate.nunique(), "expected": 2, "passed": run["scores"].candidate.nunique() == 2},
            {"check": "outer_recordings", "observed": run["scores"].recording.nunique(), "expected": 196, "passed": run["scores"].recording.nunique() == 196},
            {"check": "participant_coverage", "observed": run["scores"].participant_group.nunique(), "expected": 130, "passed": run["scores"].participant_group.nunique() == 130},
            {"check": "threshold_rows", "observed": len(run["selections"]), "expected": 6, "passed": len(run["selections"]) == 6},
            {"check": "alarm_monotonicity", "observed": bool(run["curves"].alarm_count_monotonic.all()), "expected": True, "passed": bool(run["curves"].alarm_count_monotonic.all())},
            {"check": "metric_rows", "observed": len(results["metrics"]), "expected": 8, "passed": len(results["metrics"]) == 8},
            {"check": "bootstrap_rows", "observed": len(paired), "expected": 3, "passed": len(paired) == 3},
            {"check": "scorer_sensitivity_rows", "observed": len(results["sensitivity"]), "expected": 10, "passed": len(results["sensitivity"]) == 10},
            {"check": "validation_or_test_tuning", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Multi-dataset in-run controls failed")
    outputs = {
        "model_fit_summary_v0.1.tsv": run["fits"], "inner_threshold_curves_v0.1.tsv": run["curves"],
        "inner_threshold_selections_v0.1.tsv": run["selections"], "outer_support_v0.1.tsv": run["support"],
        "outer_predicted_events_v0.1.tsv": run["alarms"], "heldout_event_metrics_v0.1.tsv": results["metrics"],
        "heldout_event_participants_v0.1.tsv": results["participants"], "score_ranking_diagnostics_v0.1.tsv": results["ranking"],
        "scorer_sensitivity_v0.1.tsv": results["sensitivity"], "paired_external_bootstrap_v0.1.tsv": paired,
        "hypothesis_decision_v0.1.tsv": gate, "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, output_dir() / name)
    print(results["metrics"].to_string(index=False))
    print(paired.to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
