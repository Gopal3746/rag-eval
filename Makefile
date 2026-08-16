.PHONY: install test lint eval verify clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

eval:
	rag-eval run --config config/ci.yaml

verify: lint test eval

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov reports/*.json
