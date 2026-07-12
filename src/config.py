"""Config loading for BIOPINN.

All hyperparameters and physical constants live in configs/*.yaml, never
hard-coded in modules. load_config() reads configs/default_config.yaml and
then deep-merges an optional experiment override file on top, so experiment
configs only need to list the fields they change.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default_config.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `base`, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(experiment: str | Path | None = None) -> dict[str, Any]:
    """Load default_config.yaml, optionally overridden by an experiment file.

    Args:
        experiment: Name of an experiment config (e.g. "experiment_1"),
            a path to a YAML file, or None to load only the defaults.

    Returns:
        A merged config dict.
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if experiment is None:
        return config

    experiment_path = Path(experiment)
    if not experiment_path.exists():
        candidate = CONFIGS_DIR / f"{experiment}.yaml"
        if not candidate.exists():
            candidate = CONFIGS_DIR / experiment
        experiment_path = candidate

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {experiment}")

    with open(experiment_path) as f:
        override = yaml.safe_load(f) or {}

    return _deep_merge(config, override)


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a paths.<key> entry from config to an absolute Path under REPO_ROOT."""
    relative = config["paths"][key]
    return REPO_ROOT / relative
