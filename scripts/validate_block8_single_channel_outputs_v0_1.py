"""Independently validate Block 8 single-channel robustness outputs."""

from __future__ import annotations

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

EXPERIMENT_DIR = "2026-09-11_block8_single_channel_robustness_v0.1"
SOURCE_DERIVED_DIR = "block7_transfer_validation_v0.1"
RESULT_DERIVED_DIR = "block8_single_channel_robustness_v0.1"
PARTITION = "validation"
MODEL_SHA256 = "d679d1142abc229b109ca912645b52ed16c4d449a87ee43185da28cafc3e3066"
THRESHOLD = 0.96
EPOCH_SEC = 30.0
CONTEXT_OFFSETS = np.arange(-120.0, 120.0, EPOCH_SEC)
TOLERANCES = [15.0, 45.0]
MEMBERSHIPS = ["primary", "expanded"]
MATERIAL_F1_DROP = 0.03
MATERIAL_FAR_INCREASE = 0.50
COMPARATORS = {
    "H2-INTACT": None,
    "H2-NO-HB1": "HB_1",
    "H2-NO-HB2": "HB_2",
}
MODEL_ROLES = {
    "H2-INTACT": "frozen_two_channel_wearable",
    "H2-NO-HB1": "neutralized_hb1_feature_contribution",
    "H2-NO-HB2": "neutralized_hb2_feature_contribution",
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


def feature_path(subject: str) -> Path:
    return source_dir() / "recording_features" / PARTITION / "hb2" / f"{subject}_features_v0.1.npz"


def model_path() -> Path:
    return source_dir() / "models" / "h2_d_model_v0.1.joblib"


def source_score_path() -> Path:
    return source_dir() / "candidate_scores" / "validation_continuous_scores_v0.1.tsv.gz"


def score_path() -> Path:
    return result_dir() / "candidate_scores" / "validation_ablation_scores_v0.1.tsv.gz"


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
        assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-10)
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


# Section 3: independent membership, references, and feature reconstruction

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


def load_feature(subject: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(feature_path(subject), allow_pickle=False) as values:
        return (
            values["onset"].astype(np.float64),
            values["features"].astype(np.float32),
            values["feature_names"].astype(str).tolist(),
        )


def context_matrix(onsets: np.ndarray, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contiguous = np.isclose(np.diff(onsets), EPOCH_SEC, atol=1e-9, rtol=0.0).astype(np.int8)
    counts = np.convolve(contiguous, np.ones(len(CONTEXT_OFFSETS) - 1, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(counts == len(CONTEXT_OFFSETS) - 1)
    matrix = np.concatenate([features[indices + offset] for offset in range(len(CONTEXT_OFFSETS))], axis=1)
    return onsets[indices + 4], matrix


def expected_mapping(model, names: list[str]) -> pd.DataFrame:
    means = model.named_steps["standardscaler"].mean_
    rows = []
    for context_index, offset in enumerate(CONTEXT_OFFSETS):
        for base_index, name in enumerate(names):
            dimension = context_index * len(names) + base_index
            channel = "HB_1" if name.startswith("HB_1_") else "HB_2"
            rows.append(
                {
                    "dimension_index": dimension,
                    "context_offset_sec": float(offset),
                    "base_feature_index": base_index,
                    "feature_name": name,
                    "channel": channel,
                    "neutral_value": float(means[dimension]),
                    "neutralized_in_comparator": "H2-NO-HB1" if channel == "HB_1" else "H2-NO-HB2",
                }
            )
    return pd.DataFrame(rows)


# Section 4: probability and support reconstruction

def recompute_scores(
    rows: list[dict], local_assignments: pd.DataFrame, model
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_names = load_feature(local_assignments.iloc[0].subject)[2]
    mapping = expected_mapping(model, first_names)
    saved_mapping = pd.read_csv(output_dir() / "feature_ablation_map_v0.1.tsv", sep="\t")
    record(rows, "ablation_map_recomputed", frames_match(mapping, saved_mapping, ["dimension_index"]), "80 dimensions; 40 per channel")
    indices = {
        comparator: mapping.loc[mapping["neutralized_in_comparator"] == comparator, "dimension_index"].to_numpy(dtype=int)
        for comparator in ["H2-NO-HB1", "H2-NO-HB2"]
    }
    means = model.named_steps["standardscaler"].mean_
    score_rows = []
    support_rows = []
    feature_metadata_ok = True
    stored_features = pd.read_csv(output_dir() / "source_validation_feature_manifest_v0.1.tsv", sep="\t")
    for item in local_assignments.itertuples(index=False):
        onsets, features, names = load_feature(item.subject)
        centers, intact = context_matrix(onsets, features)
        stored = stored_features[stored_features["subject"] == item.subject].iloc[0]
        schema_hash = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
        path = feature_path(item.subject)
        feature_metadata_ok &= (
            int(stored.epochs) == len(onsets)
            and int(stored.context_rows) == len(centers)
            and int(stored.feature_dimensions) == intact.shape[1]
            and stored.ordered_schema_sha256 == schema_hash
            and int(stored.bytes) == path.stat().st_size
            and stored.sha256 == sha256(path)
        )
        for comparator in COMPARATORS:
            matrix = intact.copy()
            if comparator != "H2-INTACT":
                selected = indices[comparator]
                matrix[:, selected] = means[selected]
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
    scores = pd.concat(score_rows, ignore_index=True)
    support = pd.DataFrame(support_rows)
    saved_scores = pd.read_csv(score_path(), sep="\t")
    saved_support = pd.read_csv(output_dir() / "validation_support_v0.1.tsv", sep="\t")
    record(rows, "source_features_rehashed", feature_metadata_ok, "20 validation feature arrays")
    record(rows, "ablation_probabilities_recomputed", frames_match(scores, saved_scores, ["comparator", "subject", "candidate_time_sec"]), f"rows={len(scores)}")
    record(rows, "validation_support_recomputed", frames_match(support, saved_support, ["comparator", "subject"]), "identical support for three comparators")
    source = pd.read_csv(source_score_path(), sep="\t")
    source = source[source["comparator"] == "H2-D"].sort_values(["subject", "candidate_time_sec"])
    intact = scores[scores["comparator"] == "H2-INTACT"].sort_values(["subject", "candidate_time_sec"])
    difference = np.abs(source["probability"].to_numpy() - intact["probability"].to_numpy())
    record(rows, "intact_probability_reproduced", len(source) == len(intact) and difference.max() <= 1e-12, f"maximum={difference.max():.12g}")
    return scores, support


# Section 5: event, comparison, and participant-influence reconstruction

def collapse_alarms(scores: pd.DataFrame) -> pd.DataFrame:
    output = []
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
            output.append(
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
    return pd.DataFrame(output, columns=["comparator", "partition", "subject", "pid", "event_time_sec", "probability", "threshold", "run_candidates"])


def recompute_events(
    rows: list[dict], scores: pd.DataFrame, support: pd.DataFrame, reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alarms = collapse_alarms(scores)
    summaries = []
    recording_rows = []
    participant_rows = []
    match_rows = []
    for comparator in COMPARATORS:
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
    metrics = pd.DataFrame(summaries)
    participants = pd.concat(participant_rows, ignore_index=True)
    outputs = {
        "validation_predicted_events_v0.1.tsv": (alarms, ["comparator", "subject", "event_time_sec"]),
        "validation_event_metrics_v0.1.tsv": (metrics, ["comparator", "membership", "tolerance_sec"]),
        "validation_event_recordings_v0.1.tsv": (pd.concat(recording_rows, ignore_index=True), ["comparator", "membership", "tolerance_sec", "subject"]),
        "validation_event_participants_v0.1.tsv": (participants, ["comparator", "membership", "tolerance_sec", "pid"]),
        "validation_event_matches_v0.1.tsv": (pd.concat(match_rows, ignore_index=True), ["comparator", "membership", "tolerance_sec", "subject", "prediction_time_sec"]),
    }
    valid = True
    for name, (frame, sort_by) in outputs.items():
        saved = pd.read_csv(output_dir() / name, sep="\t")
        valid &= frames_match(frame, saved, sort_by)
    record(rows, "event_outputs_recomputed", valid, "alarms, metrics, recordings, participants, matches")
    return metrics, participants


def primary_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    primary = metrics[(metrics["membership"] == "primary") & (metrics["tolerance_sec"] == 15.0)].set_index("comparator")
    intact = primary.loc["H2-INTACT"]
    output = []
    for comparator in ["H2-NO-HB1", "H2-NO-HB2"]:
        ablated = primary.loc[comparator]
        f1_difference = float(ablated.f1 - intact.f1)
        far_difference = float(ablated.false_alarms_per_hour - intact.false_alarms_per_hour)
        output.append(
            {
                "comparison": f"{comparator}_minus_H2-INTACT",
                "ablated_channel": COMPARATORS[comparator],
                "f1_difference": f1_difference,
                "false_alarms_per_hour_difference": far_difference,
                "material_f1_drop_bound": MATERIAL_F1_DROP,
                "material_far_increase_bound": MATERIAL_FAR_INCREASE,
                "materially_adverse": f1_difference <= -MATERIAL_F1_DROP or far_difference >= MATERIAL_FAR_INCREASE,
                "apparent_metric_improvement": f1_difference > 0 or far_difference < 0,
            }
        )
    return pd.DataFrame(output)


def aggregate(frame: pd.DataFrame) -> dict:
    return metric_values(int(frame["true_positive"].sum()), int(frame["false_positive"].sum()), int(frame["false_negative"].sum()), float(frame["supported_hours"].sum()))


def leave_one_out(participants: pd.DataFrame, comparisons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = participants[(participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)]
    full = comparisons.set_index("comparison")
    output = []
    for comparator in ["H2-NO-HB1", "H2-NO-HB2"]:
        comparison = f"{comparator}_minus_H2-INTACT"
        for omitted_pid in sorted(primary["pid"].unique()):
            local = primary[primary["pid"] != omitted_pid]
            intact = aggregate(local[local["comparator"] == "H2-INTACT"])
            ablated = aggregate(local[local["comparator"] == comparator])
            f1_difference = ablated["f1"] - intact["f1"]
            far_difference = ablated["false_alarms_per_hour"] - intact["false_alarms_per_hour"]
            output.append(
                {
                    "comparison": comparison,
                    "ablated_channel": COMPARATORS[comparator],
                    "omitted_pid": int(omitted_pid),
                    "remaining_pid": local["pid"].nunique(),
                    "f1_difference": f1_difference,
                    "false_alarms_per_hour_difference": far_difference,
                    "f1_direction_matches_full": int(np.sign(f1_difference)) == int(np.sign(full.loc[comparison, "f1_difference"])),
                    "far_direction_matches_full": int(np.sign(far_difference)) == int(np.sign(full.loc[comparison, "false_alarms_per_hour_difference"])),
                }
            )
    detail = pd.DataFrame(output)
    summary = []
    for comparison, group in detail.groupby("comparison", sort=True):
        summary.append(
            {
                "comparison": comparison,
                "ablated_channel": group["ablated_channel"].iloc[0],
                "folds": len(group),
                "f1_direction_match_folds": int(group["f1_direction_matches_full"].sum()),
                "far_direction_match_folds": int(group["far_direction_matches_full"].sum()),
                "f1_difference_min": group["f1_difference"].min(),
                "f1_difference_max": group["f1_difference"].max(),
                "far_difference_min": group["false_alarms_per_hour_difference"].min(),
                "far_difference_max": group["false_alarms_per_hour_difference"].max(),
                "participant_stable": bool(group["f1_direction_matches_full"].all() and group["far_direction_matches_full"].all()),
            }
        )
    return detail, pd.DataFrame(summary)


def concentration(participants: pd.DataFrame) -> pd.DataFrame:
    primary = participants[(participants["membership"] == "primary") & (participants["tolerance_sec"] == 15.0)]
    output = []
    for comparator, group in primary.groupby("comparator", sort=True):
        false_positives = int(group["false_positive"].sum())
        top_four = int(group.nlargest(4, "false_positive")["false_positive"].sum())
        output.append(
            {
                "comparator": comparator,
                "pid": group["pid"].nunique(),
                "pid_with_false_positive": int((group["false_positive"] > 0).sum()),
                "total_false_positive": false_positives,
                "top_four_pid_false_positive": top_four,
                "top_four_pid_false_positive_share": top_four / false_positives if false_positives else np.nan,
            }
        )
    return pd.DataFrame(output)


def validate_secondary_tables(rows: list[dict], metrics: pd.DataFrame, participants: pd.DataFrame) -> None:
    comparisons = primary_comparisons(metrics)
    saved_comparisons = pd.read_csv(output_dir() / "primary_ablation_comparisons_v0.1.tsv", sep="\t")
    lopo, lopo_summary = leave_one_out(participants, comparisons)
    saved_lopo = pd.read_csv(output_dir() / "leave_one_pid_out_influence_v0.1.tsv", sep="\t")
    saved_lopo_summary = pd.read_csv(output_dir() / "leave_one_pid_out_summary_v0.1.tsv", sep="\t")
    concentration_table = concentration(participants)
    saved_concentration = pd.read_csv(output_dir() / "participant_false_alarm_concentration_v0.1.tsv", sep="\t")
    valid = (
        frames_match(comparisons, saved_comparisons, ["comparison"])
        and frames_match(lopo, saved_lopo, ["comparison", "omitted_pid"])
        and frames_match(lopo_summary, saved_lopo_summary, ["comparison"])
        and frames_match(concentration_table, saved_concentration, ["comparator"])
    )
    record(rows, "robustness_tables_recomputed", valid, "material effects, 32 leave-one-out folds, and concentration")


# Section 6: external manifest and execution

def validate_external_manifest(rows: list[dict]) -> None:
    manifest = pd.read_csv(output_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t")
    valid = len(manifest) == 23
    for item in manifest.itertuples(index=False):
        path = data_parent() / item.path_relative_to_data_parent
        valid &= path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256
    no_test = not manifest["path_relative_to_data_parent"].str.contains("/test/|test_", case=False, regex=True).any()
    record(rows, "external_manifest_rehashed", valid, "23 external artifacts")
    record(rows, "test_artifacts_absent", no_test, "no test path in manifest")


def main() -> None:
    rows: list[dict] = []
    local_assignments = assignments()
    record(rows, "validation_assignment", len(local_assignments) == 20 and local_assignments["pid"].nunique() == 16 and set(local_assignments["partition"]) == {PARTITION}, "20 recordings; 16 pid groups")
    observed_model_hash = sha256(model_path())
    record(rows, "frozen_model_rehashed", observed_model_hash == MODEL_SHA256, observed_model_hash)
    model = joblib.load(model_path())
    scores, support = recompute_scores(rows, local_assignments, model)
    metrics, participants = recompute_events(rows, scores, support, references(local_assignments))
    validate_secondary_tables(rows, metrics, participants)
    validate_external_manifest(rows)

    checks = pd.DataFrame(rows)
    verify_or_create_tsv(checks, output_dir() / "output_integrity_checks_v0.1.tsv")
    text = "\n".join(
        [
            "# Block 8 Single-Channel Output Validation",
            "",
            "**Validation date:** 2026-09-11",
            "**Scope:** Validation-only feature ablations, probabilities, event outputs, and participant influence",
            "**Method:** Independent feature-map reconstruction, probability reproduction, metric reconstruction, and SHA-256 verification",
            "",
            f"All {int(checks['status'].eq('pass').sum())}/{len(checks)} checks passed.",
            "",
            "The validator does not fit a model, search a threshold, read raw signals, or access test artifacts.",
            "",
        ]
    )
    verify_or_create_text(output_dir() / "OUTPUT_VALIDATION.md", text)
    print(checks.to_string(index=False))
    if not checks["status"].eq("pass").all():
        raise SystemExit("At least one Block 8 independent check failed")


if __name__ == "__main__":
    main()
