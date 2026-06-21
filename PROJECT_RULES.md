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
- Keep raw datasets, generated arrays, model weights, and large temporary artifacts outside Git.
- Record dataset sources, versions, paths, file counts, and checksums or official manifests.

## Version Control and Research Records

- Commit and push meaningful work multiple times per week when practical, with at least one verified push during each active calendar week.
- Before pushing, verify the working tree, branch, upstream, staged diff, and remote commit.
- Exclude credentials, participant information, private working material, machine-specific paths, and temporary artifacts.
- Use concise, professional language in code, documentation, commit messages, and reports.
- For each meaningful update, record what changed, why it changed, the uncertainty or hypothesis addressed, the work performed, the result, limitations, and the next decision.
- Maintain dated weekly technical and SR&ED records using actual contemporaneous information.

