"""Validate paired participant analysis for endpoint factorization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_direct_endpoint_factorization_participants_v0_1 import (
    RESAMPLES,
    SEED,
    direction_summary,
    input_manifest,
    load_participants,
    output_dir,
    paired_bootstrap,
    repo_root,
    sha256,
)


checks = []


def record(name: str, passed: bool, detail: str) -> None:
    checks.append(
        {"check": name, "status": "pass" if passed else "fail", "detail": detail}
    )
    if not passed:
        raise AssertionError(f"{name}: {detail}")


def main() -> None:
    saved_paired = pd.read_csv(output_dir() / "paired_participant_metrics_v0.1.tsv", sep="\t")
    recomputed_paired = load_participants()
    record(
        "paired_participant_rows",
        len(saved_paired) == 16
        and set(saved_paired["pid"]) == set(recomputed_paired["pid"]),
        f"participants={len(saved_paired)}",
    )
    numeric_columns = saved_paired.select_dtypes(include=[np.number]).columns
    record(
        "paired_metric_recomputation",
        bool(
            np.allclose(
                saved_paired.sort_values("pid")[numeric_columns],
                recomputed_paired.sort_values("pid")[numeric_columns],
                equal_nan=True,
            )
        ),
        f"numeric_columns={len(numeric_columns)}",
    )

    saved_bootstrap = pd.read_csv(
        output_dir() / "paired_bootstrap_summary_v0.1.tsv", sep="\t"
    )
    recomputed_bootstrap = paired_bootstrap(recomputed_paired)
    bootstrap_numeric = [
        "point_difference",
        "resamples",
        "seed",
        "lower_95",
        "median",
        "upper_95",
    ]
    record(
        "paired_bootstrap_recomputation",
        saved_bootstrap[["comparison", "metric"]].equals(
            recomputed_bootstrap[["comparison", "metric"]]
        )
        and bool(
            np.allclose(
                saved_bootstrap[bootstrap_numeric],
                recomputed_bootstrap[bootstrap_numeric],
            )
        ),
        f"resamples={RESAMPLES}, seed={SEED}",
    )

    saved_directions = pd.read_csv(
        output_dir() / "participant_direction_summary_v0.1.tsv", sep="\t"
    )
    recomputed_directions = direction_summary(recomputed_paired)
    record(
        "participant_direction_recomputation",
        saved_directions.equals(recomputed_directions),
        f"metrics={len(saved_directions)}",
    )

    manifest = pd.read_csv(output_dir() / "input_artifact_manifest_v0.1.tsv", sep="\t")
    expected_manifest = input_manifest()
    failures = []
    for item in manifest.itertuples(index=False):
        path = repo_root() / item.relative_path
        if not path.exists() or path.stat().st_size != int(item.bytes) or sha256(path) != item.sha256:
            failures.append(item.relative_path)
    record(
        "input_artifact_hashes",
        not failures and manifest.equals(expected_manifest),
        f"verified={len(manifest)}, failures={len(failures)}",
    )
    record(
        "validation_only_inputs",
        not manifest["relative_path"].str.contains("test", case=False).any(),
        ",".join(manifest["artifact_role"]),
    )

    result = pd.DataFrame(checks)
    result.to_csv(output_dir() / "output_integrity_checks_v0.1.tsv", sep="\t", index=False)
    print(result.to_string(index=False))
    print(f"Passed {int((result['status'] == 'pass').sum())}/{len(result)} checks")


if __name__ == "__main__":
    main()
