"""Validate saved direct-event failure-analysis outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_direct_event_failure_modes_v0_1 import output_dir, sha256
from run_direct_event_baseline_v0_1 import data_parent, repo_root, score_path


# Section 1: helpers

checks = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append(
        {"check": name, "status": "pass" if passed else "fail", "detail": detail}
    )
    if not passed:
        raise AssertionError(f"{name}: {detail}")


# Section 2: validate hashes and accounting

def main() -> None:
    manifest = pd.read_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t")
    failures = []
    for item in manifest.itertuples(index=False):
        root = repo_root() if item.path_base == "repo" else data_parent()
        path = root / item.relative_path
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(item.relative_path)
    record(
        "input_artifact_hashes",
        not failures and len(manifest) == 33,
        f"verified={len(manifest)}, failures={len(failures)}",
    )
    forbidden_roles = manifest["artifact_role"].str.contains("raw|model", case=False, regex=True)
    record(
        "no_raw_or_model_input",
        not bool(forbidden_roles.any()),
        f"forbidden_roles={int(forbidden_roles.sum())}",
    )

    context = pd.read_csv(
        output_dir() / "de_b_test_false_positive_context_v0.1.tsv", sep="\t"
    )
    stage = pd.read_csv(
        output_dir() / "de_b_false_positive_stage_pair_summary_v0.1.tsv", sep="\t"
    )
    distance = pd.read_csv(
        output_dir() / "de_b_false_positive_distance_summary_v0.1.tsv", sep="\t"
    )
    participant = pd.read_csv(
        output_dir() / "de_b_participant_failure_summary_v0.1.tsv", sep="\t"
    )
    timing_details = pd.read_csv(
        output_dir() / "de_b_timing_match_details_v0.1.tsv", sep="\t"
    )
    timing = pd.read_csv(
        output_dir() / "de_b_timing_direction_summary_v0.1.tsv", sep="\t"
    )
    contrast = pd.read_csv(output_dir() / "direct_context_contrast_v0.1.tsv", sep="\t")
    baseline_metrics = pd.read_csv(
        repo_root()
        / "experiments/2026-08-22_direct_event_baseline_v0.1/test_event_metrics_v0.1.tsv",
        sep="\t",
    )
    primary = baseline_metrics[
        (baseline_metrics["membership"] == "primary")
        & (baseline_metrics["tolerance_sec"] == 15.0)
    ].set_index("comparator")

    expected_fp = int(primary.loc["DE-B", "false_positive"])
    record(
        "false_positive_row_count",
        len(context) == expected_fp and not context[["subject", "event_time_sec"]].duplicated().any(),
        f"rows={len(context)}, expected={expected_fp}",
    )
    record(
        "stage_pair_accounting",
        int(stage["false_positive_alarms"].sum()) == expected_fp,
        f"false_positive_sum={int(stage['false_positive_alarms'].sum())}",
    )
    record(
        "distance_bin_accounting",
        int(distance["false_positive_alarms"].sum()) == expected_fp,
        f"false_positive_sum={int(distance['false_positive_alarms'].sum())}",
    )
    record(
        "participant_accounting",
        int(participant["false_positive"].sum()) == expected_fp and len(participant) == 20,
        f"participants={len(participant)}, false_positive_sum={int(participant['false_positive'].sum())}",
    )

    scores = pd.read_csv(score_path("test"), sep="\t", compression="gzip")
    expected_supported = int((scores["comparator"] == "DE-B").sum())
    record(
        "supported_candidate_accounting",
        int(stage["supported_candidates"].sum()) == expected_supported,
        f"categorized={int(stage['supported_candidates'].sum())}, expected={expected_supported}",
    )

    primary_45 = baseline_metrics[
        (baseline_metrics["comparator"] == "DE-B")
        & (baseline_metrics["membership"] == "primary")
        & (baseline_metrics["tolerance_sec"] == 45.0)
    ].iloc[0]
    additional = int(primary_45.true_positive - primary.loc["DE-B", "true_positive"])
    timing_additional = int(
        timing.loc[timing["outside_primary_tolerance"].astype(str).str.lower() == "true", "matches"].sum()
    )
    record(
        "timing_match_accounting",
        len(timing_details) == int(primary_45.true_positive)
        and timing_additional == additional,
        f"matches={len(timing_details)}, additional={timing_additional}",
    )

    a = primary.loc["DE-A"]
    b = primary.loc["DE-B"]
    item = contrast.iloc[0]
    exact = (
        int(item.true_positive_difference) == int(b.true_positive - a.true_positive)
        and int(item.false_positive_difference) == int(b.false_positive - a.false_positive)
        and np.isclose(item.event_f1_difference, b.f1 - a.f1)
        and np.isclose(
            item.false_alarms_per_hour_difference,
            b.false_alarms_per_hour - a.false_alarms_per_hour,
        )
    )
    record("direct_context_contrast", bool(exact), "DE-B minus DE-A recomputed")

    result = pd.DataFrame(checks)
    result.to_csv(output_dir() / "output_integrity_checks_v0.1.tsv", sep="\t", index=False)
    print(result.to_string(index=False))
    print(f"Passed {int((result['status'] == 'pass').sum())}/{len(result)} checks")


if __name__ == "__main__":
    main()
