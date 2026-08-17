# Validation

Validated in the build environment on August 16, 2026.

## Automated checks

```text
pytest: 6 passed
Golden examples: 24
Regression gate: PASS
```

Deterministic baseline metrics:

| Metric | Baseline |
|---|---:|
| precision@3 | 0.3333 |
| recall@3 | 1.0000 |
| MRR | 1.0000 |
| faithfulness | 1.0000 |
| relevance | 0.9000 |

Faithfulness and relevance are scored by a deterministic lexical judge so the evaluation is reproducible, secret-free, and safe to run in CI.

## Reproduce locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make verify
```
