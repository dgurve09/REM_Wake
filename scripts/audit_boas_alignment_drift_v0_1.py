"""Quantify overnight drift-like change in saved BOAS pulse-alignment lags."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MIN_USABLE_WINDOWS = 3
PROJECTED_CHANGE_REVIEW_SEC = 2.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def input_path() -> Path:
    return (
        repo_root()
        / "experiments"
        / "2026-07-04_boas_full_signal_alignment"
        / "pulse_alignment_windows.tsv"
    )


def output_dir() -> Path:
    return repo_root() / "experiments" / "2026-08-23_boas_alignment_drift_audit_v0.1"


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite changed reviewed output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def usable_rows(source: pd.DataFrame) -> pd.DataFrame:
    usable = source[
        (source["availability"] == "available")
        & (source["usable_for_alignment"].astype(str).str.lower() == "true")
    ].copy()
    usable["window_center_hour"] = pd.to_numeric(usable["window_center_sec"]) / 3600.0
    usable["best_lag_sec"] = pd.to_numeric(usable["best_lag_sec"])
    return usable


def recording_drift(usable: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject, group in usable.groupby("subject", sort=False):
        group = group.sort_values("window_center_hour")
        if len(group) < MIN_USABLE_WINDOWS:
            continue
        time = group["window_center_hour"].to_numpy(dtype=float)
        lag = group["best_lag_sec"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(time, lag, 1)
        fitted = intercept + slope * time
        residual = float(np.sum((lag - fitted) ** 2))
        total = float(np.sum((lag - lag.mean()) ** 2))
        r_squared = 1.0 - residual / total if total > 0 else np.nan
        observed_span = float(time[-1] - time[0])
        projected_change = float(slope * observed_span)
        rows.append(
            {
                "subject": subject,
                "pid": int(group["pid"].iloc[0]),
                "usable_windows": len(group),
                "first_center_hour": float(time[0]),
                "last_center_hour": float(time[-1]),
                "observed_span_hours": observed_span,
                "slope_sec_per_hour": float(slope),
                "projected_lag_change_sec": projected_change,
                "first_to_last_lag_change_sec": float(lag[-1] - lag[0]),
                "lag_range_sec": float(lag.max() - lag.min()),
                "r_squared": r_squared,
                "projected_change_review": abs(projected_change)
                > PROJECTED_CHANGE_REVIEW_SEC,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "subject", key=lambda value: value.str.replace("sub-", "", regex=False).astype(int)
    )


def summary_table(source: pd.DataFrame, usable: pd.DataFrame, drift: pd.DataFrame) -> pd.DataFrame:
    available = source[source["availability"] == "available"]
    reviews = drift[drift["projected_change_review"]]
    return pd.DataFrame(
        [
            ("source_rows", len(source)),
            ("available_pulse_windows", len(available)),
            ("usable_pulse_windows", len(usable)),
            ("subjects_with_pulse_available", available["subject"].nunique()),
            ("subjects_with_any_usable_window", usable["subject"].nunique()),
            ("subjects_with_at_least_3_usable_windows", len(drift)),
            ("subjects_with_abs_projected_change_over_2_sec", len(reviews)),
            ("median_slope_sec_per_hour", drift["slope_sec_per_hour"].median()),
            ("median_projected_lag_change_sec", drift["projected_lag_change_sec"].median()),
            ("percentile_95_abs_projected_change_sec", drift["projected_lag_change_sec"].abs().quantile(0.95)),
        ],
        columns=["metric", "value"],
    )


def result_readme(summary: pd.DataFrame, drift: pd.DataFrame) -> str:
    values = summary.set_index("metric")["value"]
    reviews = drift[drift["projected_change_review"]].copy()
    review_text = ", ".join(reviews["subject"].tolist()) if len(reviews) else "none"
    return f"""# BOAS Full-Dataset Alignment Drift Audit v0.1

**Work date:** 2026-08-23
**Protocol:** `docs/evaluation/alignment_drift_audit_plan_v0.1.md`
**Input:** Saved July 4 `HB_PULSE` versus `PSG_PULSE` lag estimates
**Model training performed:** No

## Result

| Check | Result |
|---|---:|
| Available pulse windows | {int(values['available_pulse_windows'])} |
| Usable pulse windows | {int(values['usable_pulse_windows'])} |
| Recordings with at least three usable windows | {int(values['subjects_with_at_least_3_usable_windows'])} |
| Median slope | {float(values['median_slope_sec_per_hour']):.4f} sec/hour |
| Median projected lag change | {float(values['median_projected_lag_change_sec']):.4f} sec |
| 95th percentile absolute projected change | {float(values['percentile_95_abs_projected_change_sec']):.4f} sec |
| Recordings above the 2-second review threshold | {int(values['subjects_with_abs_projected_change_over_2_sec'])} |

The review recordings are: {review_text}.

## Interpretation

Drift-like change was not widespread under this proxy: {len(drift) - len(reviews)} of {len(drift)} analyzable recordings stayed within the 2-second projected-change screen. The flagged recordings are retained for review rather than excluded. Pulse waveform differences and cross-correlation instability can produce an apparent slope, so this result does not prove clock drift or exact sample synchronization.

The unchanged primary alignment evidence remains the matching EDF start time, sampling rate, sample count, duration, and extraction indices. This audit narrows the earlier limitation by explicitly quantifying change across the night.
"""


def main() -> None:
    source = pd.read_csv(input_path(), sep="\t")
    usable = usable_rows(source)
    drift = recording_drift(usable)
    summary = summary_table(source, usable, drift)

    if len(source) != 532 or len(usable) != 383 or len(drift) != 82:
        raise RuntimeError("Saved alignment input counts differ from the audited source")

    out_dir = output_dir()
    write_once(
        out_dir / "recording_drift_summary_v0.1.tsv",
        drift.to_csv(sep="\t", index=False, lineterminator="\n"),
    )
    write_once(
        out_dir / "drift_audit_summary_v0.1.tsv",
        summary.to_csv(sep="\t", index=False, lineterminator="\n"),
    )
    write_once(out_dir / "README.md", result_readme(summary, drift))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
