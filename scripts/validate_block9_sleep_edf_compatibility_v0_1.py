"""Independently validate Block 9 Sleep-EDF compatibility outputs."""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import edfio
import numpy as np
import pandas as pd


EXPERIMENT = "2026-10-04_block9_sleep_edf_compatibility_v0.1"
EXPECTED_PILOTS = {
    "sleep-cassette/SC4001E0-PSG.edf": "sleep-cassette/SC4001EC-Hypnogram.edf",
    "sleep-telemetry/ST7011J0-PSG.edf": "sleep-telemetry/ST7011JP-Hypnogram.edf",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return repo_root().parent / "REM_W_data" / "sleep_edfx_v1.0.0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> pd.DataFrame:
    output = repo_root() / "experiments" / EXPERIMENT
    inventory = pd.read_csv(output / "official_file_inventory_v0.1.tsv", sep="\t")
    pairing = pd.read_csv(output / "manifest_pairing_v0.1.tsv", sep="\t")
    grouping = pd.read_csv(output / "participant_grouping_summary_v0.1.tsv", sep="\t")
    channels = pd.read_csv(output / "pilot_channel_timing_v0.1.tsv", sep="\t")
    annotation_summary = pd.read_csv(output / "pilot_annotation_summary_v0.1.tsv", sep="\t")
    transitions = pd.read_csv(output / "pilot_rem_to_wake_events_v0.1.tsv", sep="\t")
    matrix = pd.read_csv(output / "compatibility_matrix_v0.1.tsv", sep="\t")
    decision = pd.read_csv(output / "compatibility_decision_v0.1.tsv", sep="\t")

    checks = []
    hash_pass = True
    for row in inventory.itertuples(index=False):
        hash_pass &= sha256(data_root() / row.relative_path) == row.expected_sha256 == row.observed_sha256
    checks.append(("official_hashes", hash_pass, f"{len(inventory)} files"))

    annex_paths = set()
    for line in (data_root() / "SHA256SUMS.txt").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+(.+)", line)
        if match:
            annex_paths.add(match.group(2))
    pair_pass = (len(pairing) == 197 and pairing.psg_path.nunique() == 197 and
                 pairing.hypnogram_path.nunique() == 197 and
                 set(pairing.psg_path).issubset(annex_paths) and
                 set(pairing.hypnogram_path).issubset(annex_paths))
    checks.append(("manifest_pairs", pair_pass, "197 unique PSG/hypnogram pairs"))

    total = grouping[grouping.study == "all"].iloc[0]
    checks.append(("participant_grouping", int(total.recordings) == 197 and int(total.participants) == 100,
                   f"{int(total.recordings)} recordings; {int(total.participants)} participants"))

    pilot_pass = True
    annotation_labels = set()
    transition_count = 0
    for psg_relative, hyp_relative in EXPECTED_PILOTS.items():
        psg = edfio.read_edf(data_root() / psg_relative, lazy_load_data=True)
        hyp = edfio.read_edf(data_root() / hyp_relative, lazy_load_data=True)
        observed = set(psg.labels)
        pilot_pass &= {"EEG Fpz-Cz", "EEG Pz-Oz"}.issubset(observed)
        pilot_pass &= all(float(psg.get_signal(label).sampling_frequency) == 100.0
                          for label in ["EEG Fpz-Cz", "EEG Pz-Oz"])
        annotations = sorted(hyp.annotations, key=lambda item: item.onset)
        annotation_labels.update(item.text for item in annotations)
        transition_count += sum(left.text == "Sleep stage R" and right.text == "Sleep stage W"
                                for left, right in zip(annotations[:-1], annotations[1:]))
    checks.append(("pilot_edf_headers", pilot_pass, "both fixed EEG derivations at 100 Hz"))
    checks.append(("pilot_label_reconstruction", {"Sleep stage W", "Sleep stage R"}.issubset(annotation_labels)
                   and transition_count == len(transitions), f"{transition_count} R-to-W transitions"))

    channel_pass = (set(channels[channels.channel.str.startswith("EEG ")].channel) ==
                    {"EEG Fpz-Cz", "EEG Pz-Oz"})
    checks.append(("stored_channel_claim", channel_pass, "Fpz-Cz and Pz-Oz"))
    annotation_pass = (annotation_summary.all_duration_multiple_30.astype(bool).all() and
                       annotation_summary.all_nonoverlap.astype(bool).all() and
                       not annotation_summary.all_within_psg_support.astype(bool).all())
    checks.append(("stored_annotation_failure", annotation_pass,
                   "30-second nonoverlap passes; complete PSG support fails"))

    matrix_index = matrix.set_index("criterion")
    no_go_pass = (matrix_index.loc["annotation_timing", "status"] == "fail" and
                  matrix_index.loc["boas_feature_semantics", "status"] == "fail" and
                  bool(matrix_index.loc["boas_feature_semantics", "blocks_direct_generalization"]) and
                  decision.iloc[0].outcome == "complete_no_go" and
                  not bool(decision.iloc[0].direct_external_model_evaluation_authorized) and
                  not bool(decision.iloc[0].full_dataset_download_authorized))
    checks.append(("decision_reconstruction", no_go_pass,
                   "annotation support and channel semantics block evaluation"))

    result = pd.DataFrame([{"check": name, "status": "pass" if passed else "fail", "detail": detail}
                           for name, passed, detail in checks])
    if not result.status.eq("pass").all():
        raise ValueError("Validation failed:\n" + result.to_string(index=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate()
    print(result.to_string(index=False))
    if args.report:
        value = result.to_csv(sep="\t", index=False, lineterminator="\n")
        if args.report.exists() and args.report.read_text(encoding="utf-8") != value:
            raise RuntimeError("Validation report changed")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    main()
