"""Independently validate the frozen Block 7 descriptive test outputs."""

from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events, metric_values


# Section 1: independent fixed configuration

EXPERIMENT_DIR = "2026-09-06_block7_descriptive_test_v0.1"
DERIVED_DIR = "block7_descriptive_test_v0.1"
TRANSFER_DERIVED_DIR = "block7_transfer_validation_v0.1"
PARTITION = "test"
EPOCH_SEC = 30.0
CONTEXT_EPOCHS = 8
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260906
FIXED_EXCLUSIONS = {"sub-32", "sub-50"}
PSG_OVERLAP_TOLERANCE = 1e-10

MODEL_SHA256 = {
    "P6-D": "243ab382909b30f8288bc24dc1e22b205fcdcea77c2011c2a3c2a452f7969082",
    "P2-D": "a3176329a142c4569a36813b08107b11a0acc4634ef47eb7e983b85fc51f7e51",
    "H2-D": "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066",
}
THRESHOLDS = {"P6-D": 0.99, "P2-D": 0.99, "H2-D": 0.96, "P2-H2-Z": 0.99}
SCORE_SPECS = {
    "P6-D": ("P6-D", "PSG-6"),
    "P2-D": ("P2-D", "PSG-2"),
    "H2-D": ("H2-D", "HB-2"),
    "P2-H2-Z": ("P2-D", "HB-2-PSGscale"),
}
MODEL_ROLES = {
    "P6-D": "direct_six_channel_psg",
    "P2-D": "direct_two_channel_psg",
    "H2-D": "direct_two_channel_wearable",
    "P2-H2-Z": "strict_psg_to_wearable_zero_shot",
}
MODALITY_FOLDERS = {
    "PSG-6": "psg6",
    "PSG-2": "psg2",
    "HB-2": "hb2",
    "HB-2-PSGscale": "hb2_psgscale",
}


# Section 2: paths and comparison helpers

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_parent() -> Path:
    return Path(os.environ.get("REM_W_DATA_ROOT", repo_root().parent / "REM_W_data"))


def output_dir() -> Path:
    return repo_root() / "experiments" / EXPERIMENT_DIR


def derived_dir() -> Path:
    return data_parent() / "derived" / DERIVED_DIR


def feature_path(subject: str, modality: str) -> Path:
    return (
        derived_dir()
        / "recording_features"
        / MODALITY_FOLDERS[modality]
        / f"{subject}_features_v0.1.npz"
    )


def model_path(comparator: str) -> Path:
    name = comparator.lower().replace("-", "_")
    return data_parent() / "derived" / TRANSFER_DERIVED_DIR / "models" / f"{name}_model_v0.1.joblib"


def score_path() -> Path:
    return derived_dir() / "candidate_scores" / "test_continuous_scores_v0.1.tsv.gz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def record(rows: list[dict], name: str, passed: bool, detail: str) -> None:
    rows.append({"check": name, "status": "pass" if passed else "fail", "detail": detail})


def frames_match(left: pd.DataFrame, right: pd.DataFrame, sort_by: list[str]) -> bool:
    try:
        left = left.sort_values(sort_by).reset_index(drop=True)
        right = right.sort_values(sort_by).reset_index(drop=True)
        assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-9,
            atol=1e-10,
        )
        return True
    except AssertionError:
        return False


def verify_or_create_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        actual = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if actual != expected:
            raise RuntimeError(f"Reviewed validation output changed: {path}")
        return
    path.write_text(expected, encoding="utf-8")


# Section 3: independent membership and feature checks

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
    return pd.DataFrame(rows)


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
    rows = membership[
        membership["subject"].isin(set(local_assignments["subject"]))
        & truth(membership["is_primary_label"])
        & membership["transition_type"].eq("REM_to_Wake")
        & membership["partition"].eq(PARTITION)
    ].merge(quality, on="transition_id", validate="one_to_one")
    rows["event_time_sec"] = rows["nominal_boundary_sec"].astype(float)
    return rows


def context_matrix(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contiguous = np.isclose(np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(CONTEXT_EPOCHS - 1, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(counts == CONTEXT_EPOCHS - 1)
    matrix = np.concatenate([features[indices + offset] for offset in range(CONTEXT_EPOCHS)], axis=1)
    return onsets[indices + 4], matrix


def load_feature(subject: str, modality: str) -> dict[str, np.ndarray]:
    with np.load(feature_path(subject, modality), allow_pickle=False) as values:
        return {
            "onset": values["onset"].astype(np.float64),
            "stage": values["stage"].astype(np.int8),
            "features": values["features"].astype(np.float32),
            "feature_names": values["feature_names"].astype(str),
        }


def validate_features(rows: list[dict], local_assignments: pd.DataFrame) -> None:
    saved = pd.read_csv(output_dir() / "generated_test_feature_artifacts_v0.1.tsv", sep="\t")
    record(rows, "feature_manifest_shape", len(saved) == 104 and saved["subject"].nunique() == 26, "104 rows; 26 recordings")
    metadata_ok = True
    parity_ok = True
    context_ok = True
    psg_overlap = 0.0
    for item in local_assignments.itertuples(index=False):
        values = {modality: load_feature(item.subject, modality) for modality in MODALITY_FOLDERS}
        base = values["PSG-6"]
        parity_ok &= all(
            np.array_equal(base["onset"], candidate["onset"])
            and np.array_equal(base["stage"], candidate["stage"])
            for candidate in values.values()
        )
        centers = [context_matrix(candidate["onset"], candidate["features"])[0] for candidate in values.values()]
        context_ok &= all(np.array_equal(centers[0], candidate) for candidate in centers[1:])
        psg_overlap = max(
            psg_overlap,
            float(np.max(np.abs(base["features"][:, : values["PSG-2"]["features"].shape[1]] - values["PSG-2"]["features"]))),
        )
        for modality, candidate in values.items():
            row = saved[(saved["subject"] == item.subject) & (saved["modality"] == modality)].iloc[0]
            path = feature_path(item.subject, modality)
            metadata_ok &= (
                int(row.epochs) == len(candidate["onset"])
                and int(row.base_feature_dimensions) == candidate["features"].shape[1]
                and int(row.bytes) == path.stat().st_size
                and row.sha256 == sha256(path)
                and bool(np.isfinite(candidate["features"]).all())
            )
    stored_parity = pd.read_csv(output_dir() / "test_feature_parity_checks_v0.1.tsv", sep="\t")
    record(rows, "feature_artifacts_reopened", metadata_ok, "shape, size, hash, and finite values")
    record(rows, "feature_epoch_stage_parity", parity_ok and truth(stored_parity["epoch_timing_and_stage_parity"]).all(), "all four modalities")
    record(rows, "feature_context_parity", context_ok and truth(stored_parity["context_center_parity"]).all(), "all four modalities")
    record(rows, "psg_overlap_reproduced", psg_overlap <= PSG_OVERLAP_TOLERANCE and np.isclose(psg_overlap, stored_parity["psg_overlap_max_abs_difference"].max()), f"maximum={psg_overlap:.12g}")


# Section 4: external artifact and probability checks

def validate_external_manifest(rows: list[dict]) -> None:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    valid = len(manifest) == 108
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        valid &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
    record(rows, "external_manifest_rehashed", valid, "108 external artifacts")


def validate_models(rows: list[dict]) -> dict[str, object]:
    saved = pd.read_csv(output_dir() / "frozen_model_verification_v0.1.tsv", sep="\t")
    valid = len(saved) == 3
    models = {}
    for comparator, expected in MODEL_SHA256.items():
        path = model_path(comparator)
        observed = sha256(path)
        stored = saved[saved["comparator"] == comparator].iloc[0]
        valid &= observed == expected == stored.expected_model_sha256 == stored.observed_model_sha256
        models[comparator] = joblib.load(path)
    record(rows, "frozen_models_rehashed", valid, "three exact validation-frozen models")
    return models


def reproduce_scores(
    rows: list[dict], local_assignments: pd.DataFrame, models: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    saved = pd.read_csv(score_path(), sep="\t")
    score_rows = []
    support_rows = []
    for comparator, (model_name, modality) in SCORE_SPECS.items():
        for item in local_assignments.itertuples(index=False):
            values = load_feature(item.subject, modality)
            centers, matrix = context_matrix(values["onset"], values["features"])
            probability = models[model_name].predict_proba(matrix)[:, 1]
            score_rows.append(
                pd.DataFrame(
                    {
                        "comparator": comparator,
                        "model_source": model_name,
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
    recomputed = pd.concat(score_rows, ignore_index=True)
    support = pd.DataFrame(support_rows)
    score_match = frames_match(
        recomputed,
        saved,
        ["comparator", "subject", "candidate_time_sec"],
    )
    saved_support = pd.read_csv(output_dir() / "test_support_v0.1.tsv", sep="\t")
    support_match = frames_match(support, saved_support, ["comparator", "subject"])
    record(rows, "continuous_scores_reproduced", score_match, f"rows={len(recomputed)}")
    record(rows, "test_support_reproduced", support_match, "26 recordings for four comparators")
    return recomputed, support


# Section 5: independent event and bootstrap reconstruction

def collapse_alarms(scores: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = []
    marked = scores[scores["probability"] >= threshold]
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
                    "threshold": float(threshold),
                    "run_candidates": len(run),
                }
            )
    return pd.DataFrame(rows, columns=["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"])


def event_inputs(reference: pd.DataFrame, membership: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    column = "primary_analysis_eligible" if membership == "primary" else "expanded_quality_analysis_eligible"
    eligible = truth(reference[column])
    columns = ["subject", "pid", "event_time_sec"]
    return reference.loc[eligible, columns], reference.loc[~eligible, columns]


def recompute_events(
    rows: list[dict], scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame
) -> pd.DataFrame:
    alarms = pd.concat(
        [collapse_alarms(group, THRESHOLDS[name]) for name, group in scores.groupby("comparator", sort=True)],
        ignore_index=True,
    )
    summaries = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for subset, exclusions in {"all_test": set(), "exclude_sub32_sub50": FIXED_EXCLUSIONS}.items():
        subset_reference = reference[~reference["subject"].isin(exclusions)]
        for comparator in sorted(THRESHOLDS):
            local_support = support[(support["comparator"] == comparator) & ~support["subject"].isin(exclusions)][["subject", "pid", "supported_hours"]]
            predictions = alarms[(alarms["comparator"] == comparator) & ~alarms["subject"].isin(exclusions)][["subject", "pid", "event_time_sec"]]
            for membership in MEMBERSHIPS:
                eligible, ignored = event_inputs(subset_reference, membership)
                for tolerance in TOLERANCES:
                    recordings, participants, matches, summary = evaluate_events(eligible, predictions, ignored, local_support, tolerance)
                    config = {
                        "analysis_subset": subset,
                        "comparator": comparator,
                        "model_role": MODEL_ROLES[comparator],
                        "partition": PARTITION,
                        "membership": membership,
                        "tolerance_sec": tolerance,
                        "threshold": THRESHOLDS[comparator],
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
        "test_predicted_events_v0.1.tsv": (alarms, ["comparator", "subject", "event_time_sec"]),
        "test_event_metrics_v0.1.tsv": (pd.DataFrame(summaries), ["analysis_subset", "comparator", "membership", "tolerance_sec"]),
        "test_event_recordings_v0.1.tsv": (pd.concat(recording_rows, ignore_index=True), ["analysis_subset", "comparator", "membership", "tolerance_sec", "subject"]),
        "test_event_participants_v0.1.tsv": (pd.concat(participant_rows, ignore_index=True), ["analysis_subset", "comparator", "membership", "tolerance_sec", "pid"]),
        "test_event_matches_v0.1.tsv": (pd.concat(match_rows, ignore_index=True), ["analysis_subset", "comparator", "membership", "tolerance_sec", "subject", "prediction_time_sec"]),
    }
    all_match = True
    for name, (frame, sort_by) in outputs.items():
        saved = pd.read_csv(output_dir() / name, sep="\t")
        all_match &= frames_match(frame, saved, sort_by)
    record(rows, "event_outputs_recomputed", all_match, "alarms, metrics, recording, participant, and match rows")
    return outputs["test_event_participants_v0.1.tsv"][0]


def recompute_bootstrap(rows: list[dict], participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[
        (participants["analysis_subset"] == "all_test")
        & (participants["membership"] == "primary")
        & (participants["tolerance_sec"] == 15.0)
    ]
    comparisons = [
        ("P6-D", "P2-D"),
        ("P2-D", "H2-D"),
        ("P2-D", "P2-H2-Z"),
        ("P2-H2-Z", "H2-D"),
    ]
    output = []
    columns = ["pid", "true_positive", "false_positive", "false_negative", "supported_hours"]
    for left_name, right_name in comparisons:
        left = primary[primary["comparator"] == left_name][columns]
        right = primary[primary["comparator"] == right_name][columns]
        paired = left.merge(right, on="pid", suffixes=("_left", "_right"), validate="one_to_one")
        rng = np.random.default_rng(BOOTSTRAP_SEED)
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
        point = {}
        for side in ["left", "right"]:
            point[side] = metric_values(
                int(paired[f"true_positive_{side}"].sum()),
                int(paired[f"false_positive_{side}"].sum()),
                int(paired[f"false_negative_{side}"].sum()),
                float(paired[f"supported_hours_{side}"].sum()),
            )
        points = {
            "event_f1_difference": point["left"]["f1"] - point["right"]["f1"],
            "false_alarms_per_hour_difference": point["left"]["false_alarms_per_hour"] - point["right"]["false_alarms_per_hour"],
        }
        for metric, value in points.items():
            output.append(
                {
                    "comparison": f"{left_name}_minus_{right_name}",
                    "metric": metric,
                    "point_difference": value,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": BOOTSTRAP_SEED,
                    "lower_95": float(sample_frame[metric].quantile(0.025)),
                    "median": float(sample_frame[metric].quantile(0.5)),
                    "upper_95": float(sample_frame[metric].quantile(0.975)),
                }
            )
    result = pd.DataFrame(output)
    saved = pd.read_csv(output_dir() / "test_paired_participant_bootstrap_v0.1.tsv", sep="\t")
    record(rows, "paired_bootstrap_recomputed", frames_match(result, saved, ["comparison", "metric"]), "four paired contrasts; 2,000 resamples")
    return result


def validate_pattern_table(rows: list[dict], paired: pd.DataFrame) -> None:
    saved = pd.read_csv(output_dir() / "validation_pattern_summary_v0.1.tsv", sep="\t")
    metric = pd.read_csv(output_dir() / "test_event_metrics_v0.1.tsv", sep="\t")
    primary = metric[
        (metric["analysis_subset"] == "all_test")
        & (metric["membership"] == "primary")
        & (metric["tolerance_sec"] == 15.0)
    ].set_index("comparator")
    intervals = paired.set_index(["comparison", "metric"])
    specs = [
        ("H7.1_channel_reduction", "P6-D_minus_P2-D", "validation favored P2-D F1 and FAR", "P6-D", "P2-D"),
        ("H7.2_source_to_zero_shot", "P2-D_minus_P2-H2-Z", "validation showed a P2-D F1 advantage; FAR difference inconclusive", "P2-D", "P2-H2-Z"),
        ("H7.2_zero_shot_to_direct_wearable", "P2-H2-Z_minus_H2-D", "validation showed lower zero-shot F1 but fewer false alarms", "P2-H2-Z", "H2-D"),
    ]
    output = []
    for question, comparison, validation_result, left, right in specs:
        f1_interval = intervals.loc[(comparison, "event_f1_difference")]
        far_interval = intervals.loc[(comparison, "false_alarms_per_hour_difference")]
        output.append(
            {
                "question": question,
                "comparison": comparison,
                "validation_result_frozen_before_test": validation_result,
                "test_f1_difference": primary.loc[left, "f1"] - primary.loc[right, "f1"],
                "test_f1_lower_95": f1_interval.lower_95,
                "test_f1_upper_95": f1_interval.upper_95,
                "test_far_difference": primary.loc[left, "false_alarms_per_hour"] - primary.loc[right, "false_alarms_per_hour"],
                "test_far_lower_95": far_interval.lower_95,
                "test_far_upper_95": far_interval.upper_95,
            }
        )
    recomputed = pd.DataFrame(output)
    record(rows, "validation_pattern_table_recomputed", frames_match(recomputed, saved, ["question"]), "three frozen interpretation checks")


# Section 6: execute independent validation

def main() -> None:
    rows: list[dict] = []
    local_assignments = assignments()
    record(rows, "frozen_test_assignments", len(local_assignments) == 26 and local_assignments["pid"].nunique() == 20 and set(local_assignments["partition"]) == {PARTITION}, "26 recordings; 20 pid groups")
    validate_features(rows, local_assignments)
    validate_external_manifest(rows)
    models = validate_models(rows)
    scores, support = reproduce_scores(rows, local_assignments, models)
    participants = recompute_events(rows, scores, support, references(local_assignments))
    paired = recompute_bootstrap(rows, participants)
    validate_pattern_table(rows, paired)

    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")
    text = "\n".join(
        [
            "# Block 7 Descriptive Test Output Validation",
            "",
            "**Validation date:** 2026-09-06",
            "**Scope:** Frozen test features, models, scores, event outputs, and paired contrasts",
            "**Method:** Independent artifact reopening, rehashing, probability reproduction, and metric reconstruction",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} checks passed.",
            "",
            "The validator does not refit a model, select a threshold, or rewrite the primary result files.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", text)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one independent Block 7 test validation check failed")


if __name__ == "__main__":
    main()
