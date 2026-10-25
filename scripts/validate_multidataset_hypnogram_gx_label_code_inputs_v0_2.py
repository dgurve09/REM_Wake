"""Independently validate the gx36m label-code correction input checkpoint."""

from __future__ import annotations

import numpy as np
import pandas as pd

import hypnogram_transition_common_v0_2 as corrected
from reviewed_output import verify_or_create_tsv


def main() -> None:
    inventory = pd.read_csv(corrected.output_dir() / "gx_raw_code_inventory_v0.2.tsv", sep="\t")
    changes = pd.read_csv(corrected.output_dir() / "gx_recording_label_correction_v0.2.tsv", sep="\t")
    audit = pd.read_csv(corrected.output_dir() / "pre_analysis_correction_audit_v0.2.tsv", sep="\t")

    paths = sorted((corrected.hypno_root() / "data" / "gx36m" / "staging").glob("SLP*_Day*_Expert*.csv"))
    raw = np.concatenate([np.loadtxt(path, dtype=int, ndmin=1) for path in paths])
    records = corrected.gx_records()
    unknown = sum(int((item["stages"] == "UNKNOWN").sum()) for item in records)
    safe_scorers = all(
        np.isin(value, np.append(corrected.KNOWN_STAGES, "UNKNOWN")).all()
        for item in records
        for value in item["scorers"].values()
    )
    checks = pd.DataFrame(
        [
            {"check": "raw_file_count_reconstructed", "observed": len(paths), "expected": 174, "passed": len(paths) == 174},
            {"check": "raw_epoch_count_reconstructed", "observed": len(raw), "expected": 184549, "passed": len(raw) == 184549},
            {"check": "raw_vocabulary_reconstructed", "observed": ",".join(map(str, np.unique(raw))), "expected": "-1,0,1,2,3,4", "passed": np.array_equal(np.unique(raw), [-1, 0, 1, 2, 3, 4])},
            {"check": "minus_one_count_reconstructed", "observed": int((raw == -1).sum()), "expected": 818, "passed": int((raw == -1).sum()) == 818},
            {"check": "saved_inventory_total", "observed": int(inventory.code_minus_1.sum()), "expected": 818, "passed": int(inventory.code_minus_1.sum()) == 818},
            {"check": "changed_primary_epochs", "observed": int(changes.changed_primary_epochs.sum()), "expected": 81, "passed": int(changes.changed_primary_epochs.sum()) == 81},
            {"check": "corrected_unknown_epochs", "observed": unknown, "expected": 1479, "passed": unknown == 1479},
            {"check": "safe_scorer_vocabulary", "observed": safe_scorers, "expected": True, "passed": safe_scorers},
            {"check": "primary_audit_passed", "observed": bool(audit.passed.all()), "expected": True, "passed": bool(audit.passed.all())},
            {"check": "corrected_result_absent", "observed": (corrected.output_dir() / "transition_events_v0.2.tsv").exists(), "expected": False, "passed": not (corrected.output_dir() / "transition_events_v0.2.tsv").exists()},
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Independent gx36m input validation failed")
    verify_or_create_tsv(
        checks,
        corrected.output_dir() / "independent_input_validation_v0.2.tsv",
    )
    print(f"Independent gx36m input validation passed {len(checks)}/{len(checks)} checks.")


if __name__ == "__main__":
    main()
