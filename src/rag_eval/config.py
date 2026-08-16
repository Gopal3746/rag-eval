from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    required = {"corpus", "golden", "k", "thresholds"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing config keys: {sorted(missing)}")
    return cfg
