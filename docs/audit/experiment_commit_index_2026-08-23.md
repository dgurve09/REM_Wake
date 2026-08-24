# Experiment First-Commit Index

**Prepared:** 2026-08-23
**Repository base reviewed:** `d936ea4`

This index records the first Git commit containing each historical experiment folder. It improves artifact discovery but must not be misread as the code commit used to execute a run. For several past experiments, protocol, code, and result files first appear together, so Git cannot independently establish their within-session order.

| Experiment | First repository commit | Commit date |
|---|---|---|
| `2026-06-24_boas_sub53_pilot` | `4f9bada` | 2026-06-24 |
| `2026-06-25_to_2026-06-28_boas_e0_metadata_readiness` | `88182c0` | 2026-06-28 |
| `2026-06-25_to_2026-06-28_boas_sub53_transition_quality` | `88182c0` | 2026-06-28 |
| `2026-06-29_to_2026-07-05_boas_e0_transition_inventory` | `4b0014c` | 2026-06-30 |
| `2026-07-04_boas_full_edf_acquisition` | `bf1bcef` | 2026-07-04 |
| `2026-07-04_boas_full_signal_alignment` | `bf1bcef` | 2026-07-04 |
| `2026-07-04_boas_sub53_signal_alignment` | `bf1bcef` | 2026-07-04 |
| `2026-07-11_boas_headband_window_quality` | `657f4c7` | 2026-07-11 |
| `2026-07-15_minimal_preprocessing_v0.1` | `664bc0f` | 2026-07-15 |
| `2026-07-15_minimal_preprocessing_v0.2` | `664bc0f` | 2026-07-15 |
| `2026-07-18_block4_artifact_integrity_v0.1` | `7d24fc7` | 2026-07-19 |
| `2026-08-15_event_matching_validation_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-15_stage_first_comparison_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-15_stage_first_context_diagnostics_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-15_stage_first_failure_analysis_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-15_stage_first_feature_baseline_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-15_stage_first_fixed_comparator_v0.1` | `176fdd8` | 2026-08-16 |
| `2026-08-22_direct_endpoint_factorization_participant_analysis_v0.1` | `ab76720` | 2026-08-22 |
| `2026-08-22_direct_endpoint_factorization_v0.1` | `ab76720` | 2026-08-22 |
| `2026-08-22_direct_endpoint_threshold_robustness_v0.1` | `ab76720` | 2026-08-22 |
| `2026-08-22_direct_event_baseline_v0.1` | `ab76720` | 2026-08-22 |
| `2026-08-22_direct_event_failure_analysis_v0.1` | `ab76720` | 2026-08-22 |
| `2026-08-22_direct_endpoint_contribution_analysis_v0.1` | `d936ea4` | 2026-08-22 |

The three 2026-08-23 audit experiment folders are local and uncommitted at the time of this index. Their first-commit field must be added after the eventual verified commit rather than predicted in advance.

The empty local folder `2026-07-18_post_gate_stage_first_label_readiness_v0.1` has no files and no Git history, so it is not an experiment artifact and is intentionally omitted.
