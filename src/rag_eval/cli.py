from __future__ import annotations

import argparse
import json
import sys

from .config import load_config
from .evaluator import evaluate
from .gate import RegressionGateError, enforce_thresholds
from .io import load_documents, load_golden, write_json
from .judge import HeuristicJudge


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-eval", description="Evaluate a small RAG system")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run the golden-set evaluation")
    run.add_argument("--config", default="config/ci.yaml")
    run.add_argument("--output", default="reports/latest.json")
    run.add_argument("--no-gate", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command != "run":
        raise AssertionError("unreachable")

    cfg = load_config(args.config)
    documents = load_documents(cfg["corpus"])
    examples = load_golden(cfg["golden"])
    report = evaluate(documents, examples, judge=HeuristicJudge(), k=int(cfg["k"]))
    write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2))

    if not args.no_gate:
        try:
            enforce_thresholds(report["summary"], cfg["thresholds"])
        except RegressionGateError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        print("Regression gate: PASS")


if __name__ == "__main__":
    main()
