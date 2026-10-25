"""Validate reviewed common-channel PSG feature caches."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

import generate_multidataset_psg_features_v0_1 as features
from reviewed_output import verify_or_create_tsv


# Section 1: helpers

def load(name: str) -> pd.DataFrame:
    return pd.read_csv(features.output_dir() / name, sep="\t")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# Section 2: independent cache checks

def main() -> None:
    summary = load("feature_generation_summary_v0.1.tsv")
    schema = load("feature_schema_v0.1.tsv")
    synthetic = load("synthetic_feature_controls_v0.1.tsv")
    audit = load("pre_model_feature_audit_v0.1.tsv")
    eligibility = pd.read_csv(features.eligibility_path(), sep="\t")
    eligibility = eligibility.set_index(["dataset", "recording"])

    cache_pass = []
    hash_pass = []
    time_pass = []
    for item in summary.itertuples(index=False):
        path = features.data_parent() / item.relative_path
        hash_pass.append(path.exists() and path.stat().st_size == int(item.bytes) and sha256(path) == item.sha256)
        with np.load(path, allow_pickle=False) as cached:
            times = cached["decision_time_sec"].astype(float)
            baseline = cached["baseline"]
            advanced = cached["advanced_only"]
            base_names = cached["baseline_feature_names"].astype(str).tolist()
            advanced_names = cached["advanced_feature_names"].astype(str).tolist()
        expected_epochs = int(eligibility.loc[(item.dataset, item.recording), "epochs"])
        expected_times = np.arange(16, 2 * expected_epochs + 1, dtype=float) * 15.0
        cache_pass.append(
            baseline.shape == (len(times), features.BASE_FEATURES)
            and advanced.shape == (len(times), features.ADVANCED_ONLY_FEATURES)
            and np.isfinite(baseline).all()
            and np.isfinite(advanced).all()
            and base_names == features.base_feature_names()
            and advanced_names == features.advanced_feature_names()
        )
        time_pass.append(np.array_equal(times, expected_times))

    checks = pd.DataFrame(
        [
            {"check": "cache_count", "observed": len(summary), "expected": 196, "passed": len(summary) == 196},
            {"check": "cache_schema_and_values", "observed": int(sum(cache_pass)), "expected": len(cache_pass), "passed": all(cache_pass)},
            {"check": "cache_time_grids", "observed": int(sum(time_pass)), "expected": len(time_pass), "passed": all(time_pass)},
            {"check": "cache_hashes", "observed": int(sum(hash_pass)), "expected": len(hash_pass), "passed": all(hash_pass)},
            {"check": "decision_rows", "observed": int(summary.decisions.sum()), "expected": 378054, "passed": int(summary.decisions.sum()) == 378054},
            {"check": "feature_schema", "observed": len(schema), "expected": 444, "passed": len(schema) == 444 and schema.feature_index.nunique() == 444},
            {"check": "baseline_schema", "observed": int(schema.representation.eq("baseline").sum()), "expected": 120, "passed": int(schema.representation.eq("baseline").sum()) == 120},
            {"check": "advanced_schema", "observed": int(schema.representation.eq("advanced").sum()), "expected": 324, "passed": int(schema.representation.eq("advanced").sum()) == 324},
            {"check": "synthetic_controls", "observed": int(synthetic.passed.astype(bool).sum()), "expected": len(synthetic), "passed": synthetic.passed.astype(bool).all()},
            {"check": "primary_audit", "observed": int(audit.passed.astype(bool).sum()), "expected": len(audit), "passed": audit.passed.astype(bool).all()},
            {
                "check": "model_outputs_present",
                "observed": 0,
                "expected": 0,
                "passed": (
                    not any(features.output_dir().glob("model_fit*"))
                    and not (features.derived_dir() / "models").exists()
                ),
            },
        ]
    )
    if not checks.passed.all():
        print(checks.to_string(index=False))
        raise ValueError("Multi-dataset feature validation failed")
    verify_or_create_tsv(checks, features.output_dir() / "independent_feature_validation_v0.1.tsv")
    print(f"Independent feature validation passed {len(checks)}/{len(checks)} checks.")
    print(f"Verified {len(hash_pass)} feature-cache hashes.")


if __name__ == "__main__":
    main()
