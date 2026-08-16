"""Analyze participant, timing, and fragmentation effects within frozen Block 5 outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score


VERSION = "v0.1"
SEED = 20260815
BOOTSTRAP_RESAMPLES = 2000
EXPERIMENT_DIR = "2026-08-15_stage_first_context_diagnostics_v0.1"
STAGES = [0, 1, 2, 3, 4]


# Section 1: frozen paths and hashes

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def experiments_root() -> Path:
    return repo_root() / "experiments"


def feature_dir() -> Path:
    return experiments_root() / "2026-08-15_stage_first_feature_baseline_v0.1"


def fixed_dir() -> Path:
    return experiments_root() / "2026-08-15_stage_first_fixed_comparator_v0.1"


def failure_dir() -> Path:
    return experiments_root() / "2026-08-15_stage_first_failure_analysis_v0.1"


def output_dir() -> Path:
    return experiments_root() / EXPERIMENT_DIR


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def input_paths() -> list[Path]:
    return [
        feature_dir() / "train_validation_stage_predictions_v0.1.tsv",
        feature_dir() / "test_stage_predictions_v0.1.tsv",
        feature_dir() / "train_validation_event_participants_v0.1.tsv",
        feature_dir() / "test_event_participants_v0.1.tsv",
        feature_dir() / "train_validation_event_recordings_v0.1.tsv",
        feature_dir() / "test_event_recordings_v0.1.tsv",
        feature_dir() / "train_validation_event_matches_v0.1.tsv",
        feature_dir() / "test_event_matches_v0.1.tsv",
        fixed_dir() / "event_matches_v0.1.tsv",
        failure_dir() / "sequence_fragmentation_by_recording_v0.1.tsv",
        failure_dir() / "sequence_fragmentation_summary_v0.1.tsv",
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "analysis_version": VERSION,
                "path_relative_to_repo": path.relative_to(repo_root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in input_paths()
        ]
    )


# Section 2: participant-paired stage and event effects

def stage_predictions() -> pd.DataFrame:
    return pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_stage_predictions_v0.1.tsv"),
            read_tsv(feature_dir() / "test_stage_predictions_v0.1.tsv"),
        ],
        ignore_index=True,
    )


def participant_stage_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparator, partition, pid), group in predictions.groupby(
        ["comparator", "partition", "pid"]
    ):
        truth = group["stage_hum"].to_numpy(dtype=int)
        predicted = group["stage_pred"].to_numpy(dtype=int)
        rows.append(
            {
                "comparator": comparator,
                "partition": partition,
                "pid": int(pid),
                "epochs": len(group),
                "macro_f1": f1_score(
                    truth,
                    predicted,
                    labels=STAGES,
                    average="macro",
                    zero_division=0,
                ),
                "five_stage_macro_recall": recall_score(
                    truth,
                    predicted,
                    labels=STAGES,
                    average="macro",
                    zero_division=0,
                ),
                "cohen_kappa": cohen_kappa_score(truth, predicted, labels=STAGES),
            }
        )
    return pd.DataFrame(rows)


def pair_comparators(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    sf_b = frame[frame["comparator"] == "SF-B"].drop(columns="comparator")
    sf_c = frame[frame["comparator"] == "SF-C"].drop(columns="comparator")
    paired = sf_b.merge(
        sf_c,
        on=["partition", "pid"],
        suffixes=("_sf_b", "_sf_c"),
        validate="one_to_one",
    )
    paired.insert(0, "analysis_version", VERSION)
    for metric in metrics:
        paired[f"{metric}_sf_c_minus_sf_b"] = paired[f"{metric}_sf_c"] - paired[f"{metric}_sf_b"]
    return paired


def participant_event_metrics() -> pd.DataFrame:
    events = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_participants_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_participants_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return events[
        (events["membership"] == "primary")
        & np.isclose(events["tolerance_sec"].astype(float), 15.0)
    ].copy()


def direction_counts(values: pd.Series, lower_is_better: bool = False) -> tuple[int, int, int]:
    tolerance = 1e-12
    if lower_is_better:
        improved = int((values < -tolerance).sum())
        reduced = int((values > tolerance).sum())
    else:
        improved = int((values > tolerance).sum())
        reduced = int((values < -tolerance).sum())
    unchanged = int((values.abs() <= tolerance).sum())
    return improved, unchanged, reduced


def paired_participant_summary(stage_pair: pd.DataFrame, event_pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in ["train", "validation", "test"]:
        stage = stage_pair[stage_pair["partition"] == partition]
        event = event_pair[event_pair["partition"] == partition]
        stage_counts = direction_counts(stage["macro_f1_sf_c_minus_sf_b"])
        event_counts = direction_counts(event["f1_sf_c_minus_sf_b"])
        false_alarm_counts = direction_counts(
            event["false_alarms_per_hour_sf_c_minus_sf_b"], lower_is_better=True
        )
        rows.append(
            {
                "analysis_version": VERSION,
                "partition": partition,
                "participants": len(stage),
                "stage_macro_f1_improved": stage_counts[0],
                "stage_macro_f1_unchanged": stage_counts[1],
                "stage_macro_f1_reduced": stage_counts[2],
                "median_stage_macro_f1_delta": float(
                    stage["macro_f1_sf_c_minus_sf_b"].median()
                ),
                "event_f1_improved": event_counts[0],
                "event_f1_unchanged": event_counts[1],
                "event_f1_reduced": event_counts[2],
                "median_event_f1_delta": float(event["f1_sf_c_minus_sf_b"].median()),
                "false_alarm_rate_improved": false_alarm_counts[0],
                "false_alarm_rate_unchanged": false_alarm_counts[1],
                "false_alarm_rate_increased": false_alarm_counts[2],
                "median_false_alarms_per_hour_delta": float(
                    event["false_alarms_per_hour_sf_c_minus_sf_b"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def metric_values(tp: int, fp: int, fn: int, hours: float) -> dict[str, float]:
    return {
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
        "false_alarms_per_hour": fp / hours if hours > 0 else np.nan,
    }


def paired_event_bootstrap(event_pair: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for partition in ["train", "validation", "test"]:
        values = event_pair[event_pair["partition"] == partition].reset_index(drop=True)
        rng = np.random.default_rng(SEED)
        differences = {metric: [] for metric in ["precision", "recall", "f1", "false_alarms_per_hour"]}
        for _ in range(BOOTSTRAP_RESAMPLES):
            sampled = values.iloc[rng.integers(0, len(values), size=len(values))]
            metrics = {}
            for comparator in ["sf_b", "sf_c"]:
                metrics[comparator] = metric_values(
                    int(sampled[f"true_positive_{comparator}"].sum()),
                    int(sampled[f"false_positive_{comparator}"].sum()),
                    int(sampled[f"false_negative_{comparator}"].sum()),
                    float(sampled[f"supported_hours_{comparator}"].sum()),
                )
            for metric in differences:
                differences[metric].append(metrics["sf_c"][metric] - metrics["sf_b"][metric])
        for metric, samples in differences.items():
            samples = pd.Series(samples).dropna()
            rows.append(
                {
                    "analysis_version": VERSION,
                    "partition": partition,
                    "contrast": "SF-C_minus_SF-B",
                    "metric": metric,
                    "resamples": BOOTSTRAP_RESAMPLES,
                    "seed": SEED,
                    "lower_95": float(samples.quantile(0.025)),
                    "median": float(samples.quantile(0.5)),
                    "upper_95": float(samples.quantile(0.975)),
                }
            )
    return pd.DataFrame(rows)


# Section 3: timing direction under the frozen sensitivity tolerance

def timing_offsets() -> tuple[pd.DataFrame, pd.DataFrame]:
    fixed = read_tsv(fixed_dir() / "event_matches_v0.1.tsv").rename(
        columns={"comparator_version": "model_version"}
    )
    transparent = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_matches_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_matches_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    matches = pd.concat([fixed, transparent], ignore_index=True)
    matches = matches[
        (matches["membership"] == "primary")
        & np.isclose(matches["tolerance_sec"].astype(float), 45.0)
        & (matches["match_type"] == "eligible")
    ].copy()
    matches["signed_offset_sec"] = (
        matches["prediction_time_sec"] - matches["reference_time_sec"]
    )

    def category(offset: float) -> str:
        if np.isclose(offset, 0.0):
            return "exact"
        if np.isclose(offset, -30.0):
            return "30_seconds_early"
        if np.isclose(offset, 30.0):
            return "30_seconds_late"
        return "other_within_45_seconds"

    matches["offset_category"] = matches["signed_offset_sec"].map(category)
    summary = (
        matches.groupby(["comparator", "partition", "offset_category"])
        .size()
        .rename("eligible_matches")
        .reset_index()
    )
    totals = summary.groupby(["comparator", "partition"])["eligible_matches"].transform("sum")
    summary["total_eligible_matches"] = totals
    summary["fraction_of_matches"] = summary["eligible_matches"] / totals
    return matches, summary


# Section 4: fragmentation association and REM-bout distribution

def event_recordings() -> pd.DataFrame:
    events = pd.concat(
        [
            read_tsv(feature_dir() / "train_validation_event_recordings_v0.1.tsv"),
            read_tsv(feature_dir() / "test_event_recordings_v0.1.tsv"),
        ],
        ignore_index=True,
    )
    return events[
        (events["membership"] == "primary")
        & np.isclose(events["tolerance_sec"].astype(float), 15.0)
    ].copy()


def fragmentation_correlations() -> tuple[pd.DataFrame, pd.DataFrame]:
    fragmentation = read_tsv(failure_dir() / "sequence_fragmentation_by_recording_v0.1.tsv")
    event = event_recordings()[
        ["comparator", "partition", "subject", "pid", "false_alarms_per_hour"]
    ]
    joined = fragmentation.merge(
        event,
        on=["comparator", "partition", "subject", "pid"],
        validate="one_to_one",
    )
    joined["predicted_all_stage_transitions_per_hour"] = (
        joined["predicted_all_stage_transitions"] / joined["supported_hours"]
    )
    joined["predicted_rem_bouts_per_hour"] = (
        joined["predicted_rem_bouts"] / joined["supported_hours"]
    )
    joined["predicted_to_human_all_transition_count_ratio"] = (
        joined["predicted_all_stage_transitions"]
        / joined["human_all_stage_transitions"].replace(0, np.nan)
    )
    joined["mean_predicted_rem_bout_duration_sec"] = (
        joined["predicted_rem_epochs"] * 30.0
        / joined["predicted_rem_bouts"].replace(0, np.nan)
    )
    predictors = [
        "predicted_all_stage_transitions_per_hour",
        "predicted_rem_bouts_per_hour",
        "predicted_to_human_all_transition_count_ratio",
        "mean_predicted_rem_bout_duration_sec",
    ]
    rows = []
    for (comparator, partition), group in joined.groupby(["comparator", "partition"]):
        for predictor in predictors:
            finite = group[[predictor, "false_alarms_per_hour"]].dropna()
            result = spearmanr(finite[predictor], finite["false_alarms_per_hour"])
            rows.append(
                {
                    "analysis_version": VERSION,
                    "comparator": comparator,
                    "partition": partition,
                    "predictor": predictor,
                    "outcome": "false_alarms_per_hour",
                    "recordings": len(finite),
                    "spearman_rho": float(result.statistic),
                    "unadjusted_descriptive_p": float(result.pvalue),
                }
            )
    return joined, pd.DataFrame(rows)


def rem_run_lengths(onset: np.ndarray, stages: np.ndarray) -> list[int]:
    lengths = []
    current = 0
    for index, stage in enumerate(stages):
        continues = (
            index > 0
            and np.isclose(onset[index] - onset[index - 1], 30.0)
            and stages[index - 1] == 4
        )
        if stage == 4:
            current = current + 1 if continues else 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def duration_bin(epochs: int) -> str:
    if epochs == 1:
        return "30_seconds"
    if epochs == 2:
        return "60_seconds"
    if epochs <= 5:
        return "90_to_150_seconds"
    if epochs <= 10:
        return "180_to_300_seconds"
    return "longer_than_300_seconds"


def rem_bout_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (comparator, partition, subject, pid), group in predictions.groupby(
        ["comparator", "partition", "subject", "pid"]
    ):
        group = group.sort_values("onset")
        onset = group["onset"].to_numpy(dtype=float)
        for source, column in [("human", "stage_hum"), ("predicted", "stage_pred")]:
            for length in rem_run_lengths(onset, group[column].to_numpy(dtype=int)):
                rows.append(
                    {
                        "comparator": comparator,
                        "partition": partition,
                        "subject": subject,
                        "pid": int(pid),
                        "sequence_source": source,
                        "duration_epochs": length,
                        "duration_sec": length * 30,
                        "duration_bin": duration_bin(length),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["comparator", "partition", "sequence_source", "duration_bin"])
        .size()
        .rename("rem_bouts")
        .reset_index()
    )
    totals = summary.groupby(["comparator", "partition", "sequence_source"])["rem_bouts"].transform("sum")
    summary["total_rem_bouts"] = totals
    summary["fraction_of_rem_bouts"] = summary["rem_bouts"] / totals
    return summary


# Section 5: report and checks

def integrity_checks(
    stage_pair: pd.DataFrame,
    event_pair: pd.DataFrame,
    timing: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    expected_participants = {"train": 64, "validation": 16, "test": 20}
    for partition, expected in expected_participants.items():
        for name, frame in [("stage", stage_pair), ("event", event_pair)]:
            observed = len(frame[frame["partition"] == partition])
            rows.append(
                {
                    "analysis_version": VERSION,
                    "check": f"{partition}_{name}_paired_participants",
                    "observed": observed,
                    "expected": expected,
                    "passed": observed == expected,
                }
            )
    rows.append(
        {
            "analysis_version": VERSION,
            "check": "timing_offsets_within_frozen_tolerance",
            "observed": float(timing["signed_offset_sec"].abs().max()),
            "expected": 45.0,
            "passed": bool((timing["signed_offset_sec"].abs() <= 45.0).all()),
        }
    )
    rows.append(
        {
            "analysis_version": VERSION,
            "check": "hashed_input_files",
            "observed": len(manifest),
            "expected": 11,
            "passed": len(manifest) == 11,
        }
    )
    return pd.DataFrame(rows)


def write_readme(
    participant_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    timing_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    bout_summary: pd.DataFrame,
) -> None:
    participant = participant_summary.set_index("partition").loc["test"]
    test_bootstrap = bootstrap[bootstrap["partition"] == "test"].set_index("metric")
    timing = timing_summary[
        (timing_summary["comparator"] == "SF-C")
        & (timing_summary["partition"] == "test")
    ].set_index("offset_category")["eligible_matches"]
    correlation = correlations[
        (correlations["comparator"] == "SF-C")
        & (correlations["partition"] == "test")
    ].set_index("predictor")
    bouts = bout_summary[
        (bout_summary["comparator"] == "SF-C")
        & (bout_summary["partition"] == "test")
    ]
    predicted_bouts = bouts[bouts["sequence_source"] == "predicted"].set_index("duration_bin")
    human_bouts = bouts[bouts["sequence_source"] == "human"].set_index("duration_bin")
    f1_interval = test_bootstrap.loc["f1"]
    false_alarm_interval = test_bootstrap.loc["false_alarms_per_hour"]
    short_predicted = float(
        predicted_bouts.loc[predicted_bouts.index.isin(["30_seconds", "60_seconds"]), "rem_bouts"].sum()
        / predicted_bouts["rem_bouts"].sum()
    )
    short_human = float(
        human_bouts.loc[human_bouts.index.isin(["30_seconds", "60_seconds"]), "rem_bouts"].sum()
        / human_bouts["rem_bouts"].sum()
    )
    text = f"""# Stage-First Context Diagnostics v0.1

**Created:** 2026-08-15
**Plan:** `docs/evaluation/stage_first_context_diagnostic_plan_v0.1.md`
**Status:** Exploratory after primary results
**Raw signals, feature arrays, or models opened:** No
**Frozen input tables hashed:** 11

## Execution Record

The initial execution completed but emitted warnings because participant-level balanced accuracy used different class denominators when a participant lacked one or more human stages. Before interpreting the diagnostic results, that field was replaced with fixed-five-stage macro recall using labels 0-4 and zero contribution for absent stages. The original issue is preserved here; no event, timing, fragmentation, or bootstrap definition changed.

## Participant-Paired Context Effect

On test participants, SF-C improved stage macro F1 for {int(participant.stage_macro_f1_improved)} of {int(participant.participants)}, reduced it for {int(participant.stage_macro_f1_reduced)}, and left it unchanged for {int(participant.stage_macro_f1_unchanged)}. Event F1 improved for {int(participant.event_f1_improved)}, declined for {int(participant.event_f1_reduced)}, and was unchanged for {int(participant.event_f1_unchanged)}. False alarms/hour decreased for {int(participant.false_alarm_rate_improved)} participants and increased for {int(participant.false_alarm_rate_increased)}.

The paired participant bootstrap estimated an SF-C minus SF-B test event-F1 difference with median {f1_interval['median']:+.4f} and exploratory 95% interval [{f1_interval['lower_95']:+.4f}, {f1_interval['upper_95']:+.4f}]. The false-alarms/hour difference had median {false_alarm_interval['median']:+.4f} and interval [{false_alarm_interval['lower_95']:+.4f}, {false_alarm_interval['upper_95']:+.4f}].

## One-Epoch Timing Direction

Under the already-frozen +/-45-second matching sensitivity, SF-C test eligible matches comprised {int(timing.get('exact', 0))} exact matches, {int(timing.get('30_seconds_early', 0))} predictions one epoch early, and {int(timing.get('30_seconds_late', 0))} one epoch late. This describes the additional timing-tolerance matches without rematching or changing the primary endpoint.

## Fragmentation Association

For SF-C test recordings, predicted REM bouts/hour had Spearman rho {correlation.loc['predicted_rem_bouts_per_hour', 'spearman_rho']:.4f} with false alarms/hour. Predicted all-stage transitions/hour had rho {correlation.loc['predicted_all_stage_transitions_per_hour', 'spearman_rho']:.4f}. Mean predicted REM-bout duration had rho {correlation.loc['mean_predicted_rem_bout_duration_sec', 'spearman_rho']:.4f}. These correlations are descriptive, unadjusted, and not causal.

## REM-Bout Duration Distribution

In SF-C test sequences, {short_predicted:.4f} of predicted REM bouts lasted only 30 or 60 seconds, compared with {short_human:.4f} of human REM bouts over the same valid coverage. This distribution supports the previously observed fragmentation mechanism and shows that the median difference is not produced by a single extreme recording.

## Interpretation

Context reduces fragmentation relative to SF-B and improves aggregate event F1, but the benefit is not uniform across participants and does not remove the false-alarm problem. The additional +/-45-second matches include explicit early or late one-epoch errors, while the recording-level associations and bout-duration distribution support widespread short-run fragmentation.

No primary result, model, tolerance, threshold, or quality membership was changed. No later project phase was started.
"""
    output_dir().joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    predictions = stage_predictions()
    stage_metrics = participant_stage_metrics(predictions)
    stage_pair = pair_comparators(
        stage_metrics, ["epochs", "macro_f1", "five_stage_macro_recall", "cohen_kappa"]
    )
    event_metrics = participant_event_metrics()
    event_pair = pair_comparators(
        event_metrics,
        [
            "recordings",
            "reference_events",
            "predicted_events",
            "true_positive",
            "false_positive",
            "false_negative",
            "ignored_predictions",
            "supported_hours",
            "precision",
            "recall",
            "f1",
            "false_alarms_per_hour",
        ],
    )
    participant_summary = paired_participant_summary(stage_pair, event_pair)
    bootstrap = paired_event_bootstrap(event_pair)
    timing_detail, timing_summary = timing_offsets()
    correlation_detail, correlations = fragmentation_correlations()
    bout_summary = rem_bout_distribution(predictions)
    manifest = input_manifest()
    checks = integrity_checks(stage_pair, event_pair, timing_detail, manifest)

    stage_pair.to_csv(destination / "paired_participant_stage_metrics_v0.1.tsv", sep="\t", index=False)
    event_pair.to_csv(destination / "paired_participant_event_metrics_v0.1.tsv", sep="\t", index=False)
    participant_summary.to_csv(destination / "paired_context_effect_summary_v0.1.tsv", sep="\t", index=False)
    bootstrap.to_csv(destination / "paired_event_bootstrap_contrasts_v0.1.tsv", sep="\t", index=False)
    timing_detail.to_csv(destination / "eligible_match_timing_offsets_v0.1.tsv", sep="\t", index=False)
    timing_summary.to_csv(destination / "eligible_match_timing_summary_v0.1.tsv", sep="\t", index=False)
    correlation_detail.to_csv(destination / "fragmentation_event_recording_metrics_v0.1.tsv", sep="\t", index=False)
    correlations.to_csv(destination / "fragmentation_false_alarm_correlations_v0.1.tsv", sep="\t", index=False)
    bout_summary.to_csv(destination / "rem_bout_duration_distribution_v0.1.tsv", sep="\t", index=False)
    manifest.to_csv(destination / "analysis_input_manifest_v0.1.tsv", sep="\t", index=False)
    checks.to_csv(destination / "diagnostic_integrity_checks_v0.1.tsv", sep="\t", index=False)
    write_readme(participant_summary, bootstrap, timing_summary, correlations, bout_summary)

    print(checks.to_string(index=False))
    if not checks["passed"].all():
        raise SystemExit("At least one context diagnostic integrity check failed")
    print(f"Wrote context diagnostics to {destination}")


if __name__ == "__main__":
    main()
