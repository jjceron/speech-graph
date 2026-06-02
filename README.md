# NLP Speech-Graph Analysis

This project analyzes annotated speech transcripts using NLP and graph-based metrics. Its main goal is to transform participant speech into directed word graphs and evaluate whether discourse structure is associated with behavioral, cognitive, academic, and clinical metadata.

## Purpose

The pipeline extracts linguistic and graph features from transcript files, merges them with participant-level metadata, and generates analysis-ready outputs for exploratory research. The project is focused on identifying patterns in speech organization, discourse markers, and activity-specific performance.

## What the Project Does

- Parses annotated transcript files from `data/processed/Transcripciones`.
- Selects target speaker data, by default `spk_1`.
- Splits transcripts by activity when required.
- Tokenizes speech and builds directed word-adjacency graphs.
- Computes graph metrics such as:
  - number of nodes and edges
  - largest connected component
  - largest strongly connected component
  - graph density
  - average shortest path
  - diameter
  - clustering
  - repeated edges
  - short cycles such as L1, L2, and L3
- Supports sliding-window analysis, mainly using 30-word windows.
- Optionally compares observed graph metrics against randomized word-order baselines.
- Extracts discourse labels from transcript annotations such as `[[...]]`.
- Merges graph metrics with participant metadata from `df_dataset.xlsx`.
- Produces correlation tables, group profiles, quality-control files, Markdown reports, and visual summaries.

## Main Research Questions

This project is designed to explore whether speech-graph and discourse-label features are associated with:

- age and school year
- cognitive, motor, planning, and total scores
- Barratt impulsivity measures
- narrative coherence and language task performance
- participant groups such as `Tipo`, educational level, gender, or school
- differences between activity types, especially target activities `2`, `6`, and `7`

The expected analysis is exploratory: the goal is to detect robust linguistic and graph-based signals that may help characterize discourse organization, task performance, and group-level differences.

## Project Structure

```text
src/
  analysis/          Statistical summaries and activity-focused reports
  config/            Default paths and analysis settings
  emotions/          Optional emotion lexicon support
  features/          Discourse-label feature extraction
  graphs/            Word-graph metric computation
  io/                Path and code normalization utilities
  pipeline/          End-to-end extraction and analysis workflows
  preprocessing/     Transcript parsing, annotation cleanup, and tokenization
  visualization/     Figures, plots, and visual reports

data/                Local input data, not versioned
docs/                Local documentation, not versioned
outputs/             Generated results