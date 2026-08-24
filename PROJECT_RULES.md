# Project Working Rules

## Research Integrity

- Cite only verifiable publications and official data sources.
- Do not fabricate results, work dates, hours, citations, or experimental history.
- Distinguish established knowledge, working hypotheses, observations, and conclusions.
- Record negative, failed, and inconclusive experiments rather than hiding them.
- Keep clinical interpretations within the limits of the available cohort and labels.

## Implementation

- Use simple, section-by-section notebooks for exploratory work.
- Prefer clear, minimal code over unnecessary abstractions or infrastructure.
- Check the existing environment before installing a package.
- Add a dependency only when the standard library and installed packages are insufficient.
- Move repeated, stable logic into small reusable modules only when justified.

## Organization and Reproducibility

- Keep the repository root limited to essential project-level files.
- Store notebooks, reusable code, documentation, manifests, experiments, and reviewed results in clearly named directories.
- Assign each experiment a dated run identifier and retain its configuration, dataset version, code commit, metrics, outcome, and notes.
- Never overwrite an experiment result. Archive superseded reviewed outputs and preserve failed runs.
- Keep raw datasets, binary feature arrays, full-night score artifacts, model weights, and large temporary artifacts outside Git.
- Compact reviewed tabular predictions or labeled-row scores may be retained when they are necessary for exact metric recomputation. State this explicitly and do not describe them as external-only artifacts.
- Record dataset sources, versions, paths, file counts, and checksums or official manifests.
- Treat reviewed experiment folders as immutable. A validator should compare stored outputs without rewriting them; any changed result requires a new versioned experiment folder.

## Experimental Records

- Before adding a new model or method, record the known baseline, why it may be insufficient, and the hypothesis being tested.
- For each experiment, retain the configuration, input data version, participant split, metrics, result, failure mode, interpretation, and next decision.
- Separate routine setup, data acquisition, cleaning, and format conversion from experiments that resolve a stated uncertainty.
- Treat model tuning as support work unless it is testing a documented technical hypothesis.
- Prefer simple baselines before complex architectures or adaptation methods.
- Follow `docs/planning/ml_experiment_record_requirements.md` when creating any model or quantitative baseline experiment.
- Commit a result-producing protocol before running the experiment whenever Git chronology is intended to evidence prespecification. Do not rely on a later combined protocol/result commit to establish execution order.

## Version Control and Research Records

- Commit and push meaningful work multiple times per week when practical, with at least one verified push during each active calendar week.
- Before pushing, verify the working tree, branch, upstream, staged diff, and remote commit.
- Exclude credentials, direct or non-public participant information, private working material, machine-specific paths, and temporary artifacts. Public pseudonymous dataset identifiers and metadata may be retained only when required for reproducible grouping or audit and permitted by the dataset license.
- Use concise, professional language in code, documentation, commit messages, and reports.
- For each meaningful update, record what changed, why it changed, the uncertainty or hypothesis addressed, the work performed, the result, limitations, and the next decision.
- Maintain dated weekly technical records using actual contemporaneous information.
