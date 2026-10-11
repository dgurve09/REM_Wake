"""Describe post-run numerical stability of interval-aware v0.1 fits."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

import run_interval_aware_localization_v0_1 as experiment
from reviewed_output import verify_or_create_tsv


def audit() -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows = []
    outer_scores = []
    for candidate in experiment.CANDIDATES:
        for outer in range(1, 6):
            phases = [f"inner_{inner}" for inner in range(1, 6) if inner != outer]
            phases.append("outer_final")
            for phase in phases:
                model_path = experiment.model_path(candidate, outer, phase)
                score_path = experiment.score_path(candidate, outer, phase)
                with gzip.open(model_path, "rt", encoding="utf-8") as stream:
                    model = json.load(stream)
                scores = pd.read_csv(score_path, sep="\t", compression="gzip")
                probability = scores.probability.to_numpy(dtype=float)
                coefficient = np.asarray(model["coefficient"], dtype=float)
                fit_rows.append(
                    {
                        "candidate": candidate,
                        "outer_fold": outer,
                        "phase": phase,
                        "score_rows": len(scores),
                        "iterations": int(model["iterations"]),
                        "converged_by_iteration_count": bool(model["converged"]),
                        "maximum_absolute_coefficient": float(np.max(np.abs(coefficient))),
                        "intercept": float(model["intercept"][0]),
                        "probability_minimum": float(probability.min()),
                        "probability_median": float(np.median(probability)),
                        "probability_mean": float(probability.mean()),
                        "probability_maximum": float(probability.max()),
                        "exact_zero_scores": int(np.sum(probability == 0.0)),
                        "exact_one_scores": int(np.sum(probability == 1.0)),
                        "endpoint_score_fraction": float(
                            np.mean((probability == 0.0) | (probability == 1.0))
                        ),
                    }
                )
                if phase == "outer_final":
                    outer_scores.append(scores)

    fits = pd.DataFrame(fit_rows).sort_values(
        ["candidate", "outer_fold", "phase"]
    ).reset_index(drop=True)
    scores = pd.concat(outer_scores, ignore_index=True)
    summary_rows = []
    for candidate, group in scores.groupby("candidate", sort=True):
        probability = group.probability.to_numpy(dtype=float)
        local_fits = fits[fits.candidate.eq(candidate)]
        summary_rows.append(
            {
                "candidate": candidate,
                "outer_score_rows": len(group),
                "probability_minimum": float(probability.min()),
                "probability_median": float(np.median(probability)),
                "probability_mean": float(probability.mean()),
                "probability_maximum": float(probability.max()),
                "exact_zero_scores": int(np.sum(probability == 0.0)),
                "exact_one_scores": int(np.sum(probability == 1.0)),
                "endpoint_score_fraction": float(
                    np.mean((probability == 0.0) | (probability == 1.0))
                ),
                "minimum_intercept": float(local_fits.intercept.min()),
                "maximum_intercept": float(local_fits.intercept.max()),
                "minimum_maximum_absolute_coefficient": float(
                    local_fits.maximum_absolute_coefficient.min()
                ),
                "maximum_maximum_absolute_coefficient": float(
                    local_fits.maximum_absolute_coefficient.max()
                ),
            }
        )
    return fits, pd.DataFrame(summary_rows)


def main() -> None:
    fits, summary = audit()
    output = experiment.output_dir()
    verify_or_create_tsv(fits, output / "post_run_fit_numerical_audit_v0.1.tsv")
    verify_or_create_tsv(summary, output / "post_run_numerical_summary_v0.1.tsv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
