"""Create a deterministic leakage-safe BOAS participant split.

The split uses only participant metadata and pre-model label/quality counts. It
does not read raw EEG, preprocess signals, train a model, or inspect outcomes.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "ds005555"
SNAPSHOT = "1.1.1"
SPLIT_VERSION = "v0.1"
QUALITY_VERSION = "v0.2"
SEED = 20260715
SEARCH_TRIALS = 50_000
PARTITIONS = ("train", "validation", "test")
PARTITION_SIZES = (64, 16, 20)

METRIC_WEIGHTS = {
    "recording_count": 1.0,
    "primary_retained": 4.0,
    "secondary_retained": 2.0,
    "background_retained": 2.0,
    "positive_pid": 4.0,
    "background_only_pid": 2.0,
    "repeated_pid": 1.0,
    "female_pid": 1.0,
    "age_under_30_pid": 1.0,
    "age_30_49_pid": 1.0,
    "age_50_plus_pid": 1.0,
    "age_missing_pid": 1.0,
    "primary_targeted_review": 1.0,
    "background_targeted_review": 1.0,
    "critical_exclusion_pid": 2.0,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_data_root() -> Path:
    return repo_root().parent / "REM_W_data"


def dataset_root() -> Path:
    root = Path(os.environ.get("REM_W_DATA_ROOT", default_data_root()))
    return root / f"boas_{DATASET}_v{SNAPSHOT}"


def output_dir() -> Path:
    return repo_root() / "splits" / "grouped_pid_split_v0.1"


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def participant_summary(data: Path) -> pd.DataFrame:
    participants = read_tsv(data / "participants.tsv")
    if participants["pid"].nunique() != 100:
        raise ValueError("Expected 100 unique pid values")
    sex_counts = participants.groupby("pid")["sex"].nunique(dropna=False)
    if int(sex_counts.max()) != 1:
        raise ValueError("Sex is inconsistent within at least one repeated pid")

    rows = []
    for pid, group in participants.groupby("pid"):
        ages = pd.to_numeric(group["age"], errors="coerce")
        bmis = pd.to_numeric(group["bmi"], errors="coerce")
        age_median = float(ages.median()) if ages.notna().any() else np.nan
        if pd.isna(age_median):
            age_band = "missing"
        elif age_median < 30:
            age_band = "under_30"
        elif age_median < 50:
            age_band = "30_49"
        else:
            age_band = "50_plus"
        rows.append(
            {
                "pid": int(pid),
                "subjects": ";".join(
                    sorted(
                        group["participant_id"],
                        key=lambda value: int(value.replace("sub-", "")),
                    )
                ),
                "recording_count": int(len(group)),
                "sex": str(group["sex"].iloc[0]),
                "age_median": age_median,
                "age_min": float(ages.min()) if ages.notna().any() else np.nan,
                "age_max": float(ages.max()) if ages.notna().any() else np.nan,
                "age_band": age_band,
                "bmi_median": float(bmis.median()) if bmis.notna().any() else np.nan,
                "bmi_min": float(bmis.min()) if bmis.notna().any() else np.nan,
                "bmi_max": float(bmis.max()) if bmis.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("pid")


def quality_summary(root: Path) -> pd.DataFrame:
    quality_root = root / "labels" / "signal_quality_flags_v0.2"
    transitions = read_tsv(quality_root / "transition_window_quality_flags_v0.2.tsv")
    backgrounds = read_tsv(quality_root / "background_window_quality_flags_v0.2.tsv")
    retained_decisions = {"include", "include_mad_sensitivity", "review_targeted"}

    rows = []
    all_pid = sorted(set(transitions["pid"]) | set(backgrounds["pid"]))
    for pid in all_pid:
        transition_rows = transitions[transitions["pid"] == pid]
        background_rows = backgrounds[backgrounds["pid"] == pid]
        primary = transition_rows[
            transition_rows["is_primary_label"].astype(str).str.lower() == "true"
        ]
        secondary = transition_rows[
            transition_rows["is_primary_label"].astype(str).str.lower() != "true"
        ]
        rows.append(
            {
                "pid": int(pid),
                "primary_retained": int(
                    primary["preprocessing_decision"].isin(retained_decisions).sum()
                ),
                "secondary_retained": int(
                    secondary["preprocessing_decision"].isin(retained_decisions).sum()
                ),
                "background_retained": int(
                    background_rows["preprocessing_decision"].isin(retained_decisions).sum()
                ),
                "primary_targeted_review": int(
                    (primary["preprocessing_decision"] == "review_targeted").sum()
                ),
                "background_targeted_review": int(
                    (background_rows["preprocessing_decision"] == "review_targeted").sum()
                ),
                "primary_mad_sensitivity": int(
                    (primary["preprocessing_decision"] == "include_mad_sensitivity").sum()
                ),
                "background_mad_sensitivity": int(
                    (
                        background_rows["preprocessing_decision"]
                        == "include_mad_sensitivity"
                    ).sum()
                ),
                "transition_critical_exclusions": int(
                    (transition_rows["preprocessing_decision"] == "exclude_critical").sum()
                ),
                "background_critical_exclusions": int(
                    (background_rows["preprocessing_decision"] == "exclude_critical").sum()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("pid")


def build_pid_table(root: Path, data: Path) -> pd.DataFrame:
    table = participant_summary(data).merge(
        quality_summary(root), on="pid", how="left", validate="one_to_one"
    )
    quality_columns = [
        column
        for column in table.columns
        if column.endswith("retained")
        or column.endswith("review")
        or column.endswith("sensitivity")
        or column.endswith("exclusions")
    ]
    if table[quality_columns].isna().any().any():
        raise ValueError("At least one pid is missing quality counts")
    table["positive_pid"] = (table["primary_retained"] > 0).astype(int)
    table["background_only_pid"] = (table["primary_retained"] == 0).astype(int)
    table["repeated_pid"] = (table["recording_count"] > 1).astype(int)
    table["female_pid"] = (table["sex"] == "F").astype(int)
    table["male_pid"] = (table["sex"] == "M").astype(int)
    table["age_under_30_pid"] = (table["age_band"] == "under_30").astype(int)
    table["age_30_49_pid"] = (table["age_band"] == "30_49").astype(int)
    table["age_50_plus_pid"] = (table["age_band"] == "50_plus").astype(int)
    table["age_missing_pid"] = (table["age_band"] == "missing").astype(int)
    table["critical_exclusion_pid"] = (
        table["transition_critical_exclusions"]
        + table["background_critical_exclusions"]
        > 0
    ).astype(int)
    return table.sort_values("pid").reset_index(drop=True)


def candidate_valid(sums: dict[str, np.ndarray]) -> bool:
    if np.any(sums["female_pid"] == 0) or np.any(sums["male_pid"] == 0):
        return False
    if np.any(sums["background_only_pid"] == 0):
        return False
    if np.any(sums["repeated_pid"] == 0):
        return False
    if np.any(sums["critical_exclusion_pid"] == 0):
        return False
    minimum_positive = np.array([55, 13, 17])
    return bool(np.all(sums["positive_pid"] >= minimum_positive))


def candidate_score(
    sums: dict[str, np.ndarray], totals: dict[str, float], ratios: np.ndarray
) -> float:
    score = 0.0
    for metric, weight in METRIC_WEIGHTS.items():
        target = totals[metric] * ratios
        relative_error = (sums[metric] - target) / np.maximum(target, 1.0)
        score += weight * float(np.sum(relative_error**2))
    return score


def search_split(table: pd.DataFrame) -> tuple[np.ndarray, float, pd.DataFrame]:
    metrics = list(METRIC_WEIGHTS)
    values = {metric: table[metric].to_numpy(dtype=float) for metric in metrics}
    values["male_pid"] = table["male_pid"].to_numpy(dtype=float)
    totals = {metric: float(values[metric].sum()) for metric in metrics}
    ratios = np.array(PARTITION_SIZES, dtype=float) / sum(PARTITION_SIZES)
    rng = np.random.Generator(np.random.PCG64(SEED))

    best_assignment: np.ndarray | None = None
    best_score = np.inf
    top_candidates: list[dict] = []

    for trial in range(1, SEARCH_TRIALS + 1):
        permutation = rng.permutation(len(table))
        indices = (
            permutation[: PARTITION_SIZES[0]],
            permutation[PARTITION_SIZES[0] : sum(PARTITION_SIZES[:2])],
            permutation[sum(PARTITION_SIZES[:2]) :],
        )
        sums = {
            metric: np.array([values[metric][index].sum() for index in indices])
            for metric in values
        }
        if not candidate_valid(sums):
            continue
        score = candidate_score(sums, totals, ratios)
        if score < best_score:
            assignment = np.empty(len(table), dtype=object)
            for partition, index in zip(PARTITIONS, indices):
                assignment[index] = partition
            best_assignment = assignment
            best_score = score
        top_candidates.append({"trial": trial, "score": score})

    if best_assignment is None:
        raise RuntimeError("No valid split found under the predeclared constraints")
    diagnostics = pd.DataFrame(top_candidates).nsmallest(20, "score")
    diagnostics.insert(0, "split_version", SPLIT_VERSION)
    return best_assignment, best_score, diagnostics


def build_balance_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ratios = dict(zip(PARTITIONS, np.array(PARTITION_SIZES) / sum(PARTITION_SIZES)))
    report_metrics = list(METRIC_WEIGHTS) + ["male_pid"]
    for metric in report_metrics:
        total = float(table[metric].sum())
        for partition in PARTITIONS:
            actual = float(table.loc[table["partition"] == partition, metric].sum())
            target = total * ratios[partition]
            rows.append(
                {
                    "split_version": SPLIT_VERSION,
                    "metric": metric,
                    "partition": partition,
                    "actual": actual,
                    "proportional_target": target,
                    "difference": actual - target,
                    "relative_difference": (actual - target) / max(target, 1.0),
                }
            )
    return pd.DataFrame(rows)


def format_value(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def write_readme(
    destination: Path, table: pd.DataFrame, score: float
) -> None:
    rows = []
    for partition in PARTITIONS:
        group = table[table["partition"] == partition]
        rows.append(
            "| "
            + " | ".join(
                [
                    partition.title(),
                    str(len(group)),
                    str(int(group["recording_count"].sum())),
                    str(int(group["positive_pid"].sum())),
                    str(int(group["primary_retained"].sum())),
                    str(int(group["secondary_retained"].sum())),
                    str(int(group["background_retained"].sum())),
                    str(int(group["repeated_pid"].sum())),
                    str(int(group["critical_exclusion_pid"].sum())),
                    f"{int(group['female_pid'].sum())}/{int(group['male_pid'].sum())}",
                    format_value(float(group["age_median"].median())),
                ]
            )
            + " |"
        )
    table_text = "\n".join(rows)

    text = f"""# Grouped Participant Split v0.1

**Created:** 2026-07-15
**Dataset:** BOAS `ds005555`, snapshot `1.1.1`
**Specification:** `docs/splits/grouped_pid_split_spec_v0.1.md`
**Quality artifact:** `labels/signal_quality_flags_v0.2/`
**Search seed:** `{SEED}`
**Candidate assignments searched:** {SEARCH_TRIALS:,}
**Selected balance score:** {score:.8f}
**Model training performed:** No

## 1. Result

| Partition | `pid` | Recordings | Positive `pid` | Primary retained | Secondary retained | Background retained | Repeated `pid` | Critical-history `pid` | F/M | Median age |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table_text}

No `pid` occurs in more than one partition. All recordings belonging to a repeated `pid` inherit the same assignment.

## 2. Interpretation

The assignment is balanced using pre-model participant, label, and quality counts only. It does not use raw signal values, learned features, predictions, or performance results. The test partition is now locked and must not guide preprocessing, threshold selection, model selection, or error-driven revisions.

Critical-exclusion history remains represented in every partition. Windows marked `exclude_critical` are not counted as retained events or backgrounds, but their rows remain in the quality artifact.

## 3. Outputs

| File | Purpose |
|---|---|
| `pid_split_assignments_v0.1.tsv` | Frozen partition for each `pid` and its recordings |
| `split_balance_summary_v0.1.tsv` | Actual versus proportional target for every balance metric |
| `split_search_diagnostics_v0.1.tsv` | Twenty lowest valid candidate scores |

## 4. Decision

Use this split for the first stage-first and direct-transition comparisons after the label/preprocessing gate passes. Any later change requires a new version and a reason independent of test performance.
"""
    destination.joinpath("README.md").write_text(text, encoding="utf-8")


def main() -> None:
    root = repo_root()
    data = dataset_root()
    destination = output_dir()
    destination.mkdir(parents=True, exist_ok=True)

    table = build_pid_table(root, data)
    assignment, score, diagnostics = search_split(table)
    table["split_version"] = SPLIT_VERSION
    table["quality_version"] = QUALITY_VERSION
    table["search_seed"] = SEED
    table["partition"] = assignment
    table = table.sort_values(["partition", "pid"])

    if table.groupby("pid")["partition"].nunique().max() != 1:
        raise ValueError("Participant leakage detected in split assignment")
    if table["partition"].value_counts().to_dict() != {
        "train": 64,
        "test": 20,
        "validation": 16,
    }:
        raise ValueError("Partition sizes do not match the specification")

    balance = build_balance_summary(table)
    table.to_csv(destination / "pid_split_assignments_v0.1.tsv", sep="\t", index=False)
    balance.to_csv(destination / "split_balance_summary_v0.1.tsv", sep="\t", index=False)
    diagnostics.to_csv(destination / "split_search_diagnostics_v0.1.tsv", sep="\t", index=False)
    write_readme(destination, table, score)

    print(table["partition"].value_counts().to_string())
    print(f"Selected balance score: {score:.8f}")
    print(f"Wrote grouped split artifact to {destination}")


if __name__ == "__main__":
    main()
