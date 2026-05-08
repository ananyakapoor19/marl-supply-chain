"""Configuration loading and merging utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    """Load YAML config file and apply optional overrides dict."""
    with open(path, "r") as f:
        cfg: dict = yaml.safe_load(f) or {}
    if overrides:
        cfg = merge_configs(cfg, overrides)
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
