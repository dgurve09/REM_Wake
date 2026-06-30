# ML Experiment Record Requirements

**Prepared:** 2026-06-30
**Project phase:** Block 3 documentation guardrail
**Applies from:** first model or quantitative baseline experiment
**Status:** Active working rule

## 1. Purpose

This project should not treat model training as evidence by itself. Each quantitative experiment must show what was known before the run, what uncertainty was being tested, what was done, what happened, and what decision followed.

These requirements apply to stage-first baselines, direct transition baselines, transfer tests, robustness tests, adaptation experiments, temporal-localization analyses, and any later prototype thresholding work.

## 2. Minimum Record for Each Experiment

Each experiment folder should include a `README.md` or plan/result note with:

- run identifier and date;
- code commit used for the run;
- dataset snapshot, local input path description, and generated input version;
- label specification version and any deviation from it;
- participant grouping and train/validation/test split rule;
- known-method baseline or comparator;
- hypothesis or uncertainty being tested;
- method summary and reason for using it;
- configuration, random seed, window size, channels, preprocessing, and metrics;
- result tables or artifact links;
- failure mode or limitation;
- interpretation and next decision.

## 3. Baseline-First Rule

Before adding a complex model, adaptation method, or architecture change, record the simplest relevant comparator.

Examples for this project:

- stage-first sleep staging followed by transition derivation before direct transition claims;
- simple feature-based direct transition baseline before a CNN;
- zero-shot or no-adaptation PSG-to-headband transfer before fine-tuning;
- missing-channel and noisy-channel stress tests before adding robustness methods.

If a known method fails, keep the actual metric, split, configuration, and observed failure mode. Do not summarize it later as only "did not work."

## 4. Routine Work Versus Uncertainty Tests

Keep routine preparation separate from experiments that test a stated uncertainty.

Routine preparation includes:

- downloading metadata or EDF files;
- converting file formats;
- cleaning paths and manifests;
- checking package availability;
- fixing simple script errors;
- producing plots or tables from already-decided analyses.

Uncertainty tests include:

- testing whether a stage-first baseline misses REM-to-Wake boundaries under 30-second label uncertainty;
- testing whether direct boundary detection adds event-level value over stage-derived transitions;
- testing whether PSG-trained methods transfer to wearable channels;
- testing whether label uncertainty changes the model-ranking decision;
- testing whether adaptation is needed because a documented baseline fails.

## 5. Negative and Inconclusive Results

Failed, negative, and inconclusive experiments should remain in the experiment folder with the same structure as successful experiments.

Record:

- what was expected;
- what failed or remained unclear;
- whether the failure came from data quality, label definition, model assumption, split design, implementation error, or metric choice;
- whether the next decision is retry, narrow, redesign, or stop.

## 6. Notebook Style

Exploratory notebooks should remain simple and sectioned:

1. purpose and hypothesis;
2. environment and input check;
3. data loading;
4. label or split check;
5. method;
6. metrics;
7. interpretation;
8. next decision.

When a notebook produces a reviewed result, export the key table or figure into the dated experiment folder and summarize the decision in Markdown.
