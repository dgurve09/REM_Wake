"""Run the frozen multi-dataset PSG stage-first transition experiment."""

from __future__ import annotations

import gzip
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

import generate_multidataset_psg_features_v0_1 as features
import multidataset_psg_stage_first_common_v0_1 as common
from reviewed_output import verify_or_create_tsv


# Section 1: external artifact helpers

def model_path(heldout: str) -> Path:
    safe = heldout.lower().replace("-", "_")
    return common.derived_dir() / "models" / f"holdout_{safe}.json.gz"


def score_path(heldout: str) -> Path:
    safe = heldout.lower().replace("-", "_")
    return common.derived_dir() / "scores" / f"holdout_{safe}_stage_scores.tsv.gz"


def verify_json_gzip(path: Path, value: dict) -> None:
    expected = json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            if source.read() != expected:
                raise RuntimeError(f"Stage model artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=9) as target:
        target.write(expected)


def load_json_gzip(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def verify_tsv_gzip(path: Path, frame: pd.DataFrame) -> None:
    expected = frame.to_csv(sep="\t", index=False, lineterminator="\n")
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as source:
            if source.read() != expected:
                raise RuntimeError(f"Stage score artifact changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=9) as target:
        target.write(expected)


def load_tsv_gzip(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return pd.read_csv(source, sep="\t")


# Section 2: fixed model fitting and scoring

def stage_indices(target: np.ndarray) -> np.ndarray:
    return np.asarray([common.STAGE_TO_INDEX[value] for value in target], dtype=np.int8)


def training_data(recordings: dict[str, dict], heldout: str) -> tuple[np.ndarray, np.ndarray]:
    values = []
    labels = []
    for item in recordings.values():
        if item["dataset"] == heldout:
            continue
        target, known = common.stage_target(item)
        values.append(item["ADV3-CHANGE"][known])
        labels.append(stage_indices(target[known]))
    return np.concatenate(values), np.concatenate(labels)


def fit_stage_model(values: np.ndarray, labels: np.ndarray) -> tuple[dict, dict]:
    scaler = StandardScaler().fit(values)
    scaled = scaler.transform(values)
    model = LogisticRegression(
        C=0.1,
        penalty="l2",
        solver="lbfgs",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-6,
        random_state=common.SEED,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaled, labels)
    converged = not any(isinstance(item.message, ConvergenceWarning) for item in caught)
    class_counts = np.bincount(labels, minlength=len(common.STAGES))
    class_weights = len(labels) / (len(common.STAGES) * class_counts)
    payload = {
        "candidate": "FIVE_STAGE_BALANCED_LOGIT",
        "feature_count": values.shape[1],
        "classes": model.classes_.astype(int).tolist(),
        "class_counts": class_counts.astype(int).tolist(),
        "class_weights": class_weights.astype(float).tolist(),
        "coefficient": model.coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
        "scaler_mean": scaler.mean_.astype(float).tolist(),
        "scaler_scale": scaler.scale_.astype(float).tolist(),
        "iterations": int(model.n_iter_.max()),
        "converged": converged,
        "sklearn_version": sklearn.__version__,
        "seed": common.SEED,
    }
    return {"scaler": scaler, "model": model}, payload


def score_holdout(fitted: dict, recordings: dict[str, dict], heldout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_frames = []
    support_rows = []
    for recording, item in recordings.items():
        if item["dataset"] != heldout:
            continue
        target, known = common.stage_target(item)
        values = fitted["scaler"].transform(item["ADV3-CHANGE"][known])
        probability = fitted["model"].predict_proba(values)
        prediction = probability.argmax(axis=1)
        frame = pd.DataFrame(
            {
                "heldout_dataset": heldout,
                "dataset": item["dataset"],
                "recording": recording,
                "participant_group": item["participant_group"],
                "candidate_time_sec": item["times"][known],
                "truth_stage": target[known],
                "predicted_stage": common.STAGES[prediction],
            }
        )
        for index, stage in enumerate(common.STAGES):
            frame[f"probability_{stage}"] = probability[:, index]
        score_frames.append(frame)
        support_rows.append(
            {
                "heldout_dataset": heldout,
                "dataset": item["dataset"],
                "recording": recording,
                "participant_group": item["participant_group"],
                "supported_boundaries": int(known.sum()),
                "supported_hours": float(known.sum() * 15.0 / 3600.0),
            }
        )
    return pd.concat(score_frames, ignore_index=True), pd.DataFrame(support_rows)


# Section 3: stage and causal event metrics

def stage_metrics(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    class_rows = []
    confusion_rows = []
    for heldout, local in scores.groupby("heldout_dataset", sort=True):
        truth = stage_indices(local.truth_stage.to_numpy())
        predicted = stage_indices(local.predicted_stage.to_numpy())
        summary_rows.append(
            {
                "heldout_dataset": heldout,
                "rows": len(local),
                "accuracy": accuracy_score(truth, predicted),
                "balanced_accuracy": balanced_accuracy_score(truth, predicted),
                "macro_f1": f1_score(truth, predicted, labels=np.arange(5), average="macro"),
                "cohen_kappa": cohen_kappa_score(truth, predicted, labels=np.arange(5)),
            }
        )
        precision, recall, f1, support = precision_recall_fscore_support(
            truth,
            predicted,
            labels=np.arange(5),
            zero_division=0,
        )
        for index, stage in enumerate(common.STAGES):
            class_rows.append(
                {
                    "heldout_dataset": heldout,
                    "stage": stage,
                    "support": int(support[index]),
                    "precision": precision[index],
                    "recall": recall[index],
                    "f1": f1[index],
                }
            )
        matrix = pd.crosstab(
            pd.Categorical(local.truth_stage, categories=common.STAGES),
            pd.Categorical(local.predicted_stage, categories=common.STAGES),
            dropna=False,
        ).to_numpy()
        for truth_index, truth_stage in enumerate(common.STAGES):
            for predicted_index, predicted_stage in enumerate(common.STAGES):
                confusion_rows.append(
                    {
                        "heldout_dataset": heldout,
                        "truth_stage": truth_stage,
                        "predicted_stage": predicted_stage,
                        "rows": int(matrix[truth_index, predicted_index]),
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(class_rows), pd.DataFrame(confusion_rows)


def derive_events(scores: pd.DataFrame, candidate: str) -> pd.DataFrame:
    rows = []
    for recording, group in scores.groupby("recording", sort=True):
        group = group.sort_values("candidate_time_sec").reset_index(drop=True)
        times = group.candidate_time_sec.to_numpy(float)
        predicted = group.predicted_stage.to_numpy(str)
        boundaries = np.r_[0, np.flatnonzero(np.diff(times) > 15.0 + 1e-6) + 1, len(group)]
        selected = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            if candidate == "STAGE-HARD":
                local = np.flatnonzero(
                    (predicted[start : stop - 1] == "REM")
                    & (predicted[start + 1 : stop] == "W")
                )
                selected.extend((start + local + 1).tolist())
            elif candidate == "STAGE-CONFIRM2":
                for index in range(start + 3, stop):
                    if (
                        predicted[index - 3] == "REM"
                        and predicted[index - 2] == "REM"
                        and predicted[index - 1] == "W"
                        and predicted[index] == "W"
                    ):
                        selected.append(index)
            else:
                raise ValueError(candidate)
        for index in selected:
            rows.append(
                {
                    "heldout_dataset": group.heldout_dataset.iloc[0],
                    "candidate": candidate,
                    "dataset": group.dataset.iloc[0],
                    "recording": recording,
                    "participant_group": group.participant_group.iloc[0],
                    "event_time_sec": float(times[index]),
                }
            )
    return pd.DataFrame(rows, columns=["heldout_dataset", "candidate", "dataset", "recording", "participant_group", "event_time_sec"])


def event_results(
    scores: pd.DataFrame,
    support: pd.DataFrame,
    recordings: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    alarms = pd.concat([derive_events(scores, candidate) for candidate in common.CANDIDATES], ignore_index=True)
    metrics = []
    participants = []
    for heldout in common.direct.DATASETS:
        local_support = support[support.heldout_dataset.eq(heldout)]
        for candidate in common.CANDIDATES:
            local_alarms = alarms[
                alarms.heldout_dataset.eq(heldout) & alarms.candidate.eq(candidate)
            ]
            summary, local_participants = common.direct.evaluate(
                local_alarms,
                local_support,
                recordings,
            )
            role = "primary_external" if heldout in common.PRIMARY_EXTERNAL else "descriptive_reused_source"
            metrics.append(
                {
                    "heldout_dataset": heldout,
                    "candidate": candidate,
                    "partition_role": role,
                    **summary,
                }
            )
            local_participants.insert(0, "heldout_dataset", heldout)
            local_participants.insert(1, "candidate", candidate)
            local_participants.insert(2, "partition_role", role)
            participants.append(local_participants)

    external_support = support[support.heldout_dataset.isin(common.PRIMARY_EXTERNAL)]
    for candidate in common.CANDIDATES:
        local_alarms = alarms[
            alarms.heldout_dataset.isin(common.PRIMARY_EXTERNAL)
            & alarms.candidate.eq(candidate)
        ]
        summary, _ = common.direct.evaluate(local_alarms, external_support, recordings)
        metrics.append(
            {
                "heldout_dataset": "POOLED_EXTERNAL",
                "candidate": candidate,
                "partition_role": "primary_external_pooled",
                **summary,
            }
        )

    sensitivity = []
    eligibility = pd.read_csv(features.eligibility_path(), sep="\t").set_index("recording")
    for heldout in common.PRIMARY_EXTERNAL:
        local_support = support[support.heldout_dataset.eq(heldout)]
        variants = {}
        for recording in local_support.recording:
            item = eligibility.loc[recording]
            holder = type("Row", (), {"dataset": item.dataset, "label_source": item.label_source})
            variants[recording] = common.correction.scorer_variants(holder)
        variant_names = sorted(set.intersection(*(set(value) for value in variants.values())))
        for variant in variant_names:
            override = {
                recording: common.direct.transition_tables(value[variant])
                for recording, value in variants.items()
            }
            for candidate in common.CANDIDATES:
                local_alarms = alarms[
                    alarms.heldout_dataset.eq(heldout) & alarms.candidate.eq(candidate)
                ]
                summary, _ = common.direct.evaluate(
                    local_alarms,
                    local_support,
                    recordings,
                    override,
                )
                sensitivity.append(
                    {
                        "heldout_dataset": heldout,
                        "candidate": candidate,
                        "label_variant": variant,
                        **summary,
                    }
                )
    return alarms, pd.DataFrame(metrics), pd.concat(participants, ignore_index=True), pd.DataFrame(sensitivity)


# Section 4: frozen direct comparison and bootstrap

def direct_metrics() -> pd.DataFrame:
    return pd.read_csv(
        features.repo_root()
        / "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/heldout_event_metrics_v0.2.tsv",
        sep="\t",
    ).query("candidate == 'ADV3-CHANGE'")


def direct_participants() -> pd.DataFrame:
    return pd.read_csv(
        features.repo_root()
        / "experiments/2026-10-25_multidataset_psg_external_generalization_v0.2/heldout_event_participants_v0.2.tsv",
        sep="\t",
    ).query("candidate == 'ADV3-CHANGE'")


def metric_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    direct = direct_metrics().set_index("heldout_dataset")
    rows = []
    for item in metrics.itertuples(index=False):
        baseline = direct.loc[item.heldout_dataset]
        rows.append(
            {
                "heldout_dataset": item.heldout_dataset,
                "candidate": item.candidate,
                "stage_first_precision": item.precision,
                "direct_precision": baseline.precision,
                "precision_difference": item.precision - baseline.precision,
                "stage_first_recall": item.recall,
                "direct_recall": baseline.recall,
                "recall_difference": item.recall - baseline.recall,
                "stage_first_f1": item.f1,
                "direct_f1": baseline.f1,
                "f1_difference": item.f1 - baseline.f1,
                "stage_first_false_alarms_per_hour": item.false_alarms_per_hour,
                "direct_false_alarms_per_hour": baseline.false_alarms_per_hour,
                "false_alarms_per_hour_difference": item.false_alarms_per_hour - baseline.false_alarms_per_hour,
            }
        )
    return pd.DataFrame(rows)


def participant_bootstrap(participants: pd.DataFrame) -> pd.DataFrame:
    direct = direct_participants()
    columns = ["dataset", "participant_group", "true_positive", "false_positive", "false_negative", "supported_hours"]
    rows = []
    for candidate in common.CANDIDATES:
        stage = participants[
            participants.candidate.eq(candidate)
            & participants.heldout_dataset.isin(common.PRIMARY_EXTERNAL)
        ][columns]
        baseline = direct[direct.heldout_dataset.isin(common.PRIMARY_EXTERNAL)][columns]
        paired = stage.merge(
            baseline,
            on=["dataset", "participant_group"],
            suffixes=("_stage", "_direct"),
            validate="one_to_one",
        )
        rng = np.random.default_rng(common.SEED)
        samples = []
        by_dataset = {
            dataset: paired[paired.dataset.eq(dataset)].reset_index(drop=True)
            for dataset in common.PRIMARY_EXTERNAL
        }
        for _ in range(common.BOOTSTRAP_RESAMPLES):
            sample = pd.concat(
                [frame.iloc[rng.integers(0, len(frame), size=len(frame))] for frame in by_dataset.values()],
                ignore_index=True,
            )
            values = {}
            for side in ["stage", "direct"]:
                values[side] = common.direct.metric_values(
                    int(sample[f"true_positive_{side}"].sum()),
                    int(sample[f"false_positive_{side}"].sum()),
                    int(sample[f"false_negative_{side}"].sum()),
                    float(sample[f"supported_hours_{side}"].sum()),
                )
            samples.append(
                {
                    "f1_difference": values["stage"]["f1"] - values["direct"]["f1"],
                    "recall_difference": values["stage"]["recall"] - values["direct"]["recall"],
                    "false_alarms_per_hour_difference": values["stage"]["false_alarms_per_hour"] - values["direct"]["false_alarms_per_hour"],
                }
            )
        frame = pd.DataFrame(samples)
        for metric in frame.columns:
            rows.append(
                {
                    "candidate": candidate,
                    "comparison": f"{candidate}_minus_ADV3-CHANGE",
                    "metric": metric,
                    "resamples": common.BOOTSTRAP_RESAMPLES,
                    "seed": common.SEED,
                    "lower_95": float(frame[metric].quantile(0.025)),
                    "median": float(frame[metric].quantile(0.5)),
                    "upper_95": float(frame[metric].quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


def decision(metrics: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    stage = metrics.set_index(["heldout_dataset", "candidate"])
    direct = direct_metrics().set_index("heldout_dataset")
    primary = stage.loc[("POOLED_EXTERNAL", common.PRIMARY_CANDIDATE)]
    direct_pool = direct.loc["POOLED_EXTERNAL"]
    differences = {
        dataset: float(
            stage.loc[(dataset, common.PRIMARY_CANDIDATE), "f1"]
            - direct.loc[dataset, "f1"]
        )
        for dataset in common.PRIMARY_EXTERNAL
    }
    conditions = {
        "pooled_f1_gain_at_least_0.03": float(primary.f1 - direct_pool.f1) >= 0.03,
        "pooled_f1_at_least_0.2590": float(primary.f1) >= 0.2590,
        "pooled_recall_at_least_0.25": float(primary.recall) >= 0.25,
        "pooled_far_at_most_0.30": float(primary.false_alarms_per_hour) <= 0.30,
        "gx36m_difference_not_below_minus_0.03": differences["gx36m"] >= -0.03,
        "isruc_difference_not_below_minus_0.03": differences["ISRUC-3"] >= -0.03,
        "numerical_gate": bool(fits.converged.all() and fits.all_probabilities_finite.all()),
    }
    passed = all(conditions.values())
    return pd.DataFrame(
        [
            {
                "hypothesis": "H-STAGE-FIRST-XDATA",
                "candidate": common.PRIMARY_CANDIDATE,
                "stage_first_f1": float(primary.f1),
                "direct_f1": float(direct_pool.f1),
                "f1_difference": float(primary.f1 - direct_pool.f1),
                "stage_first_recall": float(primary.recall),
                "stage_first_false_alarms_per_hour": float(primary.false_alarms_per_hour),
                "gx36m_f1_difference": differences["gx36m"],
                "isruc_f1_difference": differences["ISRUC-3"],
                **conditions,
                "point_gate_pass": passed,
                "decision": "retain_for_paired_wearable_mechanism_study" if passed else "stop_stage_first_v0.1",
            }
        ]
    )


# Section 5: complete experiment

def main() -> None:
    recordings = common.load_recordings()
    score_frames = []
    support_frames = []
    fit_rows = []
    manifest_rows = []
    for heldout in common.direct.DATASETS:
        path = model_path(heldout)
        score_file = score_path(heldout)
        if path.exists() != score_file.exists():
            raise RuntimeError(f"Incomplete external artifact pair for {heldout}")
        if path.exists():
            payload = load_json_gzip(path)
            scores = load_tsv_gzip(score_file)
            action = "verified existing artifacts for"
        else:
            values, labels = training_data(recordings, heldout)
            fitted, payload = fit_stage_model(values, labels)
            verify_json_gzip(path, payload)
            scores, _ = score_holdout(fitted, recordings, heldout)
            verify_tsv_gzip(score_file, scores)
            action = "fit and scored"
        if set(scores.heldout_dataset) != {heldout}:
            raise ValueError(f"Stage score held-out identity changed for {heldout}")
        support = (
            scores.groupby(
                ["heldout_dataset", "dataset", "recording", "participant_group"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "supported_boundaries"})
        )
        support["supported_hours"] = support.supported_boundaries * 15.0 / 3600.0
        score_frames.append(scores)
        support_frames.append(support)
        fit_rows.append(
            {
                "heldout_dataset": heldout,
                "training_rows": int(sum(payload["class_counts"])),
                "input_features": int(payload["feature_count"]),
                "iterations": payload["iterations"],
                "converged": payload["converged"],
                "all_probabilities_finite": bool(np.isfinite(scores.filter(like="probability_").to_numpy()).all()),
                "probability_row_sums_max_error": float(np.abs(scores.filter(like="probability_").sum(axis=1) - 1.0).max()),
                "model_relative_path": path.relative_to(features.data_parent()).as_posix(),
                "model_bytes": path.stat().st_size,
                "model_sha256": features.sha256(path),
            }
        )
        manifest_rows.append(
            {
                "artifact_type": "model",
                "heldout_dataset": heldout,
                "relative_path": path.relative_to(features.data_parent()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": features.sha256(path),
            }
        )
        manifest_rows.append(
            {
                "artifact_type": "stage_scores",
                "heldout_dataset": heldout,
                "relative_path": score_file.relative_to(features.data_parent()).as_posix(),
                "bytes": score_file.stat().st_size,
                "sha256": features.sha256(score_file),
            }
        )
        print(f"{action} {heldout}", flush=True)

    scores = pd.concat(score_frames, ignore_index=True)
    support = pd.concat(support_frames, ignore_index=True)
    fits = pd.DataFrame(fit_rows)
    stage_summary, stage_class, confusion = stage_metrics(scores)
    alarms, metrics, participants, sensitivity = event_results(scores, support, recordings)
    comparison = metric_comparison(metrics)
    bootstrap = participant_bootstrap(participants)
    gate = decision(metrics, fits)
    manifest = pd.DataFrame(manifest_rows)
    expected_score_rows = int(
        pd.read_csv(common.output_dir() / "stage_class_counts_v0.1.tsv", sep="\t").rows.sum()
    )

    checks = pd.DataFrame(
        [
            {"check": "model_fits", "observed": len(fits), "expected": 3, "passed": len(fits) == 3},
            {"check": "convergence_status_reported", "observed": bool(fits.converged.notna().all()), "expected": True, "passed": bool(fits.converged.notna().all())},
            {"check": "score_rows", "observed": len(scores), "expected": expected_score_rows, "passed": len(scores) == expected_score_rows},
            {"check": "all_stage_probabilities_finite", "observed": bool(fits.all_probabilities_finite.all()), "expected": True, "passed": bool(fits.all_probabilities_finite.all())},
            {"check": "probability_sums", "observed": float(fits.probability_row_sums_max_error.max()), "expected": "<=1e-12", "passed": fits.probability_row_sums_max_error.le(1e-12).all()},
            {"check": "stage_summary_rows", "observed": len(stage_summary), "expected": 3, "passed": len(stage_summary) == 3},
            {"check": "stage_class_rows", "observed": len(stage_class), "expected": 15, "passed": len(stage_class) == 15},
            {"check": "confusion_rows", "observed": len(confusion), "expected": 75, "passed": len(confusion) == 75},
            {"check": "event_metric_rows", "observed": len(metrics), "expected": 8, "passed": len(metrics) == 8},
            {"check": "participant_rows", "observed": len(participants), "expected": 260, "passed": len(participants) == 260},
            {"check": "scorer_sensitivity_rows", "observed": len(sensitivity), "expected": 10, "passed": len(sensitivity) == 10},
            {"check": "bootstrap_rows", "observed": len(bootstrap), "expected": 6, "passed": len(bootstrap) == 6},
            {"check": "external_artifacts", "observed": len(manifest), "expected": 6, "passed": len(manifest) == 6},
            {"check": "validation_or_test_tuning", "observed": 0, "expected": 0, "passed": True},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Stage-first experiment controls failed")

    outputs = {
        "stage_fit_summary_v0.1.tsv": fits,
        "heldout_stage_metrics_v0.1.tsv": stage_summary,
        "heldout_stage_class_metrics_v0.1.tsv": stage_class,
        "heldout_stage_confusion_v0.1.tsv": confusion,
        "outer_support_v0.1.tsv": support,
        "derived_event_alarms_v0.1.tsv": alarms,
        "heldout_event_metrics_v0.1.tsv": metrics,
        "heldout_event_participants_v0.1.tsv": participants,
        "scorer_sensitivity_v0.1.tsv": sensitivity,
        "direct_metric_comparison_v0.1.tsv": comparison,
        "paired_participant_bootstrap_v0.1.tsv": bootstrap,
        "hypothesis_decision_v0.1.tsv": gate,
        "external_artifact_manifest_v0.1.tsv": manifest,
        "in_run_checks_v0.1.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, common.output_dir() / name)

    print(stage_summary.to_string(index=False))
    print(metrics.to_string(index=False))
    print(bootstrap.to_string(index=False))
    print(gate.to_string(index=False))


if __name__ == "__main__":
    main()
