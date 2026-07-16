"""Run minimal preprocessing v0.2 with coverage-aware quality flags."""

import validate_minimal_preprocessing_v0_1 as pipeline


pipeline.PREPROCESSING_VERSION = "v0.2"
pipeline.QUALITY_VERSION = "v0.3"
pipeline.EXPERIMENT_DIR_NAME = "2026-07-15_minimal_preprocessing_v0.2"
pipeline.SPECIFICATION_PATH = (
    "docs/preprocessing/minimal_wearable_eeg_preprocessing_spec_v0.2.md"
)
pipeline.EXPECTED_TRAIN_TRANSITIONS = 302
pipeline.EXPECTED_TRAIN_BACKGROUNDS = 2761


if __name__ == "__main__":
    pipeline.main()
