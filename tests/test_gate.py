import pytest

from rag_eval.gate import RegressionGateError, enforce_thresholds


def test_gate_passes_at_threshold() -> None:
    enforce_thresholds({"recall_at_k": 0.95}, {"recall_at_k": 0.95})


def test_gate_fails_below_threshold() -> None:
    with pytest.raises(RegressionGateError):
        enforce_thresholds({"recall_at_k": 0.94}, {"recall_at_k": 0.95})
