"""Audit the gx36m raw-code correction before regenerating label results."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_1 as previous
import hypnogram_transition_common_v0_2 as corrected
import run_multidataset_hypnogram_transition_uncertainty_v0_1 as analysis
from reviewed_output import verify_or_create_tsv


# Section 1: raw-file inventory

def raw_code_inventory() -> pd.DataFrame:
    rows = []
    staging = corrected.hypno_root() / "data" / "gx36m" / "staging"
    for path in sorted(staging.glob("SLP*_Day*_Expert*.csv")):
        raw = np.loadtxt(path, dtype=int, ndmin=1)
        key = path.stem.rsplit("_Expert", 1)[0]
        rows.append(
            {
                "recording": f"gx36m-{key}",
                "scorer_file": corrected.relative_source(path),
                "epochs": len(raw),
                "code_minus_1": int((raw == -1).sum()),
                "code_0": int((raw == 0).sum()),
                "code_1": int((raw == 1).sum()),
                "code_2": int((raw == 2).sum()),
                "code_3": int((raw == 3).sum()),
                "code_4": int((raw == 4).sum()),
                "other_codes": int((~np.isin(raw, [-1, 0, 1, 2, 3, 4])).sum()),
            }
        )
    return pd.DataFrame(rows)


def recording_changes() -> pd.DataFrame:
    old = {item["recording"]: item for item in previous.gx_records()}
    new = {item["recording"]: item for item in corrected.gx_records()}
    rows = []
    for recording in sorted(new):
        unsafe = old[recording]
        safe = new[recording]
        rows.append(
            {
                "recording": recording,
                "participant_group": safe["participant_group"],
                "epochs": len(safe["stages"]),
                "changed_primary_epochs": int((unsafe["stages"] != safe["stages"]).sum()),
                "unsafe_unknown_epochs": int((unsafe["stages"] == "UNKNOWN").sum()),
                "corrected_unknown_epochs": int((safe["stages"] == "UNKNOWN").sum()),
                "unsafe_events": len(analysis.event_boundaries(unsafe["stages"])),
                "corrected_events": len(analysis.event_boundaries(safe["stages"])),
            }
        )
    return pd.DataFrame(rows)


def result_outputs_present() -> bool:
    result_names = [
        "recording_transition_measures_v0.2.tsv",
        "transition_events_v0.2.tsv",
        "dataset_transition_summary_v0.2.tsv",
        "scorer_uncertainty_summary_v0.2.tsv",
        "repeated_night_summary_v0.2.tsv",
        "hypothesis_decisions_v0.2.tsv",
    ]
    return any((corrected.output_dir() / name).exists() for name in result_names)


# Section 2: reviewed pre-result checkpoint

def main() -> None:
    inventory = raw_code_inventory()
    changes = recording_changes()
    raw_total = int(inventory.epochs.sum())
    minus_one = int(inventory.code_minus_1.sum())
    changed = int(changes.changed_primary_epochs.sum())
    checks = pd.DataFrame(
        [
            {"check": "scorer_files", "observed": len(inventory), "expected": 174, "passed": len(inventory) == 174},
            {"check": "recordings", "observed": len(changes), "expected": 58, "passed": len(changes) == 58},
            {"check": "raw_scorer_epochs", "observed": raw_total, "expected": 184549, "passed": raw_total == 184549},
            {"check": "raw_code_minus_1", "observed": minus_one, "expected": 818, "passed": minus_one == 818},
            {"check": "only_observed_codes_minus_1_to_4", "observed": int(inventory.other_codes.sum()), "expected": 0, "passed": int(inventory.other_codes.sum()) == 0},
            {"check": "affected_scorer_files", "observed": int(inventory.code_minus_1.gt(0).sum()), "expected": int(inventory.code_minus_1.gt(0).sum()), "passed": int(inventory.code_minus_1.gt(0).sum()) > 0},
            {"check": "changed_primary_epochs", "observed": changed, "expected": 81, "passed": changed == 81},
            {"check": "unsafe_unknown_epochs", "observed": int(changes.unsafe_unknown_epochs.sum()), "expected": 1398, "passed": int(changes.unsafe_unknown_epochs.sum()) == 1398},
            {"check": "corrected_unknown_epochs", "observed": int(changes.corrected_unknown_epochs.sum()), "expected": 1479, "passed": int(changes.corrected_unknown_epochs.sum()) == 1479},
            {"check": "unsafe_primary_events", "observed": int(changes.unsafe_events.sum()), "expected": 85, "passed": int(changes.unsafe_events.sum()) == 85},
            {"check": "corrected_primary_events", "observed": int(changes.corrected_events.sum()), "expected": 85, "passed": int(changes.corrected_events.sum()) == 85},
            {"check": "corrected_results_present", "observed": result_outputs_present(), "expected": False, "passed": not result_outputs_present()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("gx36m label-code correction audit failed")
    outputs = {
        "gx_raw_code_inventory_v0.2.tsv": inventory,
        "gx_recording_label_correction_v0.2.tsv": changes,
        "pre_analysis_correction_audit_v0.2.tsv": checks,
    }
    for name, frame in outputs.items():
        verify_or_create_tsv(frame, corrected.output_dir() / name)
    print(f"gx36m correction audit passed {len(checks)}/{len(checks)} checks.")
    print(f"Raw -1 labels: {minus_one}; changed primary epochs: {changed}.")


if __name__ == "__main__":
    main()
