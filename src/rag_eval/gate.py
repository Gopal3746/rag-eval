from __future__ import annotations


class RegressionGateError(RuntimeError):
    pass


def check_thresholds(summary: dict, thresholds: dict[str, float]) -> list[str]:
    failures: list[str] = []
    for metric, minimum in thresholds.items():
        actual = float(summary.get(metric, 0.0))
        if actual < float(minimum):
            failures.append(f"{metric}: {actual:.4f} < required {float(minimum):.4f}")
    return failures


def enforce_thresholds(summary: dict, thresholds: dict[str, float]) -> None:
    failures = check_thresholds(summary, thresholds)
    if failures:
        raise RegressionGateError("Regression gate failed:\n- " + "\n- ".join(failures))
