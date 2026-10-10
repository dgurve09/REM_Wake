"""Apply every predeclared alarm budget to held-out outer scores."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import run_full_night_risk_set_formulation_v0_1 as base
import run_paired_enriched_feature_pruning_v0_1 as prior
from reviewed_output import verify_or_create_tsv
from stage_first_event_evaluation_v0_1 import evaluate_events


EXPERIMENT_DIR = "2026-10-08_full_night_risk_set_budget_sensitivity_v0.1"
SOURCE_DIR = "2026-10-07_full_night_risk_set_formulation_v0.1"


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def output_dir() -> Path:
    return root() / "experiments" / EXPERIMENT_DIR


def source_dir() -> Path:
    return root() / "experiments" / SOURCE_DIR


def commit() -> str:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={root().as_posix()}", "rev-parse", "HEAD"],
        cwd=root(), text=True,
    ).strip()


def write_text(path: Path, text: str) -> None:
    expected = text.replace("\r\n", "\n")
    if path.exists():
        if path.read_text(encoding="utf-8").replace("\r\n", "\n") != expected:
            raise RuntimeError(f"Reviewed output changed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(expected, encoding="utf-8")


# Section 1: load and hash-check the fifteen outer score artifacts

def load_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    source_manifest = pd.read_csv(
        source_dir() / "external_artifact_manifest_v0.1.tsv", sep="\t"
    )
    expected = dict(zip(source_manifest.relative_path, source_manifest.sha256))
    frames, artifacts = [], []
    for outer in range(1, 6):
        paths = {
            "SAMP-BAL": base.prior_score_file(outer, "outer_final"),
            "RISK-BAL": base.score_path("RISK-BAL", outer, "outer_final"),
            "RISK-NAT": base.score_path("RISK-NAT", outer, "outer_final"),
        }
        for candidate, path in paths.items():
            relative = path.relative_to(base.data_parent()).as_posix()
            digest = prior.sha256(path)
            if expected.get(relative) != digest:
                raise RuntimeError(f"Source score hash mismatch: {relative}")
            frame = pd.read_csv(path, sep="\t", compression="gzip")
            frame["candidate"] = candidate
            frames.append(frame)
            artifacts.append(
                {"relative_path": relative, "bytes": path.stat().st_size, "sha256": digest}
            )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(artifacts)


# Section 2: apply each fold-specific threshold without refitting

def collapse(scores: pd.DataFrame, selections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in selections.itertuples(index=False):
        local = scores[
            scores.candidate.eq(item.candidate)
            & scores.outer_fold.eq(item.outer_fold)
        ]
        for (subject, pid), group in local.groupby(["subject", "pid"], sort=True):
            times = prior.collapsed_times(
                group.sort_values("candidate_time_sec"), float(item.threshold)
            )
            rows += [
                {
                    "candidate": item.candidate,
                    "outer_fold": int(item.outer_fold),
                    "alarm_budget_per_hour": float(item.alarm_budget_per_hour),
                    "subject": subject,
                    "pid": int(pid),
                    "event_time_sec": float(time),
                    "threshold": float(item.threshold),
                }
                for time in times
            ]
    columns = [
        "candidate", "outer_fold", "alarm_budget_per_hour", "subject",
        "pid", "event_time_sec", "threshold",
    ]
    return pd.DataFrame(rows, columns=columns)


# Section 3: evaluate every predeclared budget on the held-out outer folds

def evaluate(
    events: pd.DataFrame,
    support: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows, participant_frames = [], []
    for budget in base.ALARM_BUDGETS:
        for candidate in base.CANDIDATES:
            predictions = events[
                events.candidate.eq(candidate)
                & np.isclose(events.alarm_budget_per_hour, budget)
            ][["subject", "pid", "event_time_sec"]]
            for membership in base.MEMBERSHIPS:
                eligible, ignored = prior.local_event_inputs(references, membership)
                for tolerance in base.TOLERANCES:
                    _, participants, _, summary = evaluate_events(
                        eligible,
                        predictions,
                        ignored,
                        support[["subject", "pid", "supported_hours"]],
                        tolerance,
                    )
                    config = {
                        "candidate": candidate,
                        "partition": "train_nested_oof",
                        "membership": membership,
                        "tolerance_sec": tolerance,
                        "alarm_budget_per_hour": budget,
                    }
                    metric_rows.append({**config, **summary})
                    local = participants.copy()
                    for key, value in reversed(list(config.items())):
                        local.insert(0, key, value)
                    participant_frames.append(local)
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["alarm_budget_per_hour", "candidate", "membership", "tolerance_sec"]
    )
    participants = pd.concat(participant_frames, ignore_index=True).sort_values(
        ["alarm_budget_per_hour", "candidate", "membership", "tolerance_sec", "pid"]
    )
    return metrics.reset_index(drop=True), participants.reset_index(drop=True)


# Section 4: immutable reviewed outputs

def run(result_code_commit: str) -> None:
    code_commit = commit()
    if not code_commit.startswith(result_code_commit):
        raise ValueError("Result code commit does not match current checkout")

    selections_path = source_dir() / "inner_threshold_selections_v0.1.tsv"
    support_path = source_dir() / "outer_support_v0.1.tsv"
    selections = pd.read_csv(selections_path, sep="\t")
    support = pd.read_csv(support_path, sep="\t")
    scores, artifacts = load_scores()

    assignments = prior.train_assignments()
    references = prior.reference_events(assignments)
    events = collapse(scores, selections)
    metrics, participants = evaluate(events, support, references)

    source_metrics = pd.read_csv(
        source_dir() / "outer_event_metrics_v0.1.tsv", sep="\t"
    )
    reproduced = metrics[np.isclose(metrics.alarm_budget_per_hour, 0.25)]
    reproduced = reproduced[source_metrics.columns].sort_values(
        ["candidate", "membership", "tolerance_sec"]
    ).reset_index(drop=True)
    expected_primary = source_metrics.sort_values(
        ["candidate", "membership", "tolerance_sec"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(
        reproduced, expected_primary, check_dtype=False, rtol=1e-12, atol=1e-12
    )

    artifacts.insert(0, "location_scope", "external_data")
    repository_inputs = pd.DataFrame(
        [
            {
                "location_scope": "repository",
                "relative_path": path.relative_to(root()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": prior.sha256(path),
            }
            for path in [selections_path, support_path]
        ]
    )
    artifacts = pd.concat([artifacts, repository_inputs], ignore_index=True)

    expected_configs = 4 * 3 * 2 * 2
    checks = pd.DataFrame(
        [
            {
                "check": "selection_grid",
                "passed": len(selections) == 60,
                "detail": "3 candidates x 5 folds x 4 budgets",
            },
            {
                "check": "score_artifact_hashes",
                "passed": len(artifacts[artifacts.location_scope.eq("external_data")]) == 15,
                "detail": "15 frozen outer score artifacts verified",
            },
            {
                "check": "score_key_uniqueness",
                "passed": not scores.duplicated(
                    ["candidate", "subject", "pid", "candidate_time_sec"]
                ).any(),
                "detail": "one score per candidate and risk-set boundary",
            },
            {
                "check": "score_coverage",
                "passed": scores.groupby("candidate").size().eq(73656).all(),
                "detail": "73,656 outer scores per candidate",
            },
            {
                "check": "support_scope",
                "passed": len(support) == 82 and support.pid.nunique() == 64,
                "detail": "82 train recordings; 64 participant groups",
            },
            {
                "check": "metric_grid",
                "passed": len(metrics) == expected_configs,
                "detail": "48 predeclared sensitivity configurations",
            },
            {
                "check": "participant_grid",
                "passed": len(participants) == expected_configs * 64,
                "detail": "3,072 participant-configuration rows",
            },
            {
                "check": "primary_reproduction",
                "passed": True,
                "detail": "0.25/h metrics exactly reproduce the validated primary table",
            },
        ]
    )
    if not checks.passed.all():
        raise RuntimeError("A required sensitivity-analysis control failed")

    output = output_dir()
    verify_or_create_tsv(metrics, output / "alarm_budget_event_metrics_v0.1.tsv")
    verify_or_create_tsv(
        participants, output / "alarm_budget_event_participants_v0.1.tsv"
    )
    verify_or_create_tsv(events, output / "alarm_budget_predicted_events_v0.1.tsv")
    verify_or_create_tsv(artifacts, output / "source_artifact_manifest_v0.1.tsv")
    verify_or_create_tsv(checks, output / "in_run_checks_v0.1.tsv")

    source_versions = json.loads(
        (source_dir() / "software_versions_v0.1.json").read_text(encoding="utf-8")
    )
    versions = {
        "git_commit": code_commit,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "source_experiment": SOURCE_DIR,
        "source_result_code_commit": source_versions["git_commit"],
        "analysis_scope": "read-only application of preselected thresholds; no refitting",
    }
    write_text(
        output / "software_versions_v0.1.json",
        json.dumps(versions, indent=2, sort_keys=True) + "\n",
    )

    primary = metrics[
        metrics.membership.eq("primary") & metrics.tolerance_sec.eq(15.0)
    ]
    lines = [
        "# Full-Night Risk-Set Alarm-Budget Sensitivity v0.1",
        "",
        "Train-only participant-grouped nested out-of-fold sensitivity results. "
        "This report applies the four predeclared fold-specific thresholds to frozen "
        "outer scores; it performs no refitting or new threshold selection.",
        "",
        "| Budget (/h) | Candidate | Precision | Recall | F1 | False alarms/hour |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in primary.itertuples(index=False):
        precision = "undefined" if np.isnan(row.precision) else f"{row.precision:.4f}"
        lines.append(
            f"| {row.alarm_budget_per_hour:.2f} | {row.candidate} | {precision} | "
            f"{row.recall:.4f} | {row.f1:.4f} | {row.false_alarms_per_hour:.4f} |"
        )
    lines += [
        "",
        "Metrics use the primary event membership and the +/-15 s tolerance. The alarm "
        "budget constrains inner-fold threshold selection; held-out outer-fold false-alarm "
        "rates can differ from the nominal budget.",
        "",
        "These developmental train-partition results do not establish real-time or clinical "
        "performance. Validation and test partitions remain closed.",
    ]
    write_text(output / "README.md", "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-code-commit", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.result_code_commit)
