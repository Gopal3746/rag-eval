# rag-eval

A compact evaluation harness for retrieval-augmented generation systems. It ships with a 24-question golden set, retrieval metrics, deterministic answer-quality scoring, and a CI regression gate that fails when quality drops below configured thresholds.

## What it evaluates

- Retrieval precision@k and recall@k against known relevant documents.
- Mean reciprocal rank (MRR) for ranking quality.
- Deterministic faithfulness scoring based on support from retrieved context.
- Deterministic relevance scoring against known-correct reference answers.
- Configurable regression thresholds suitable for CI.

The included retriever and extractive answerer are intentionally small fixtures. The main focus is the evaluation workflow: golden examples, metrics, reports, tests, and regression gating.

## Architecture

```text
Golden Q&A + document corpus
           |
           v
      BM25 retrieval
           |
           v
    extractive answer
           |
           v
 retrieval metrics + deterministic answer scoring
           |
           v
        JSON report
           |
           v
     regression gate
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make verify
```

`make verify` runs Ruff, pytest, and the complete evaluation gate.

## Run the evaluator

```bash
rag-eval run --config config/ci.yaml
```

Write to another report path:

```bash
rag-eval run --config config/ci.yaml --output reports/experiment.json
```

Run without enforcing thresholds:

```bash
rag-eval run --config config/ci.yaml --no-gate
```

## Baseline

The checked-in fixture currently evaluates 24 golden examples at `k=3`.

| Metric | Baseline |
|---|---:|
| precision@3 | 0.3333 |
| recall@3 | 1.0000 |
| MRR | 1.0000 |
| faithfulness | 1.0000 |
| relevance | 0.9000 |

Precision@3 is `1/3` because each fixture question has one known relevant document while three documents are retrieved.

## Regression gating

Thresholds live in `config/ci.yaml`. The CLI exits non-zero when any required metric drops below its configured threshold, which makes the evaluator usable as a CI quality gate.

The GitHub Actions workflow runs:

1. dependency installation,
2. Ruff linting,
3. pytest,
4. the golden-set evaluation,
5. regression threshold enforcement,
6. report upload.

## Project layout

```text
rag-eval/
├── config/ci.yaml
├── data/
│   ├── corpus.jsonl
│   └── golden.jsonl
├── reports/
│   └── baseline.json
├── src/rag_eval/
│   ├── cli.py
│   ├── config.py
│   ├── evaluator.py
│   ├── gate.py
│   ├── generator.py
│   ├── io.py
│   ├── judge.py
│   ├── metrics.py
│   ├── models.py
│   └── retrieval.py
├── tests/
├── Makefile
├── README.md
└── VALIDATION.md
```

## Scope

This repository is deliberately small. It demonstrates how to define a golden dataset, measure retrieval quality, score generated answers reproducibly, and enforce quality thresholds in CI without requiring external services or API credentials.
