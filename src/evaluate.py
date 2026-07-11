"""Six-metric evaluation against publication thresholds. [Phase 8 — not yet
implemented]

Will provide: RMSE, MAE, R^2, L2 relative error, mean PDE residual, and
penetration RMSE against the held-out test set, computed globally and
decomposed by tumor zone / NP-size range / time range, plus pass/fail
against configs' evaluation.thresholds and the H1-H6 hypothesis summary.
Local module: loads artifacts/biopinn_model.pt + processed test data, never
retrains.
"""

from __future__ import annotations

import numpy as np


def compute_metrics(C_pred: np.ndarray, C_true: np.ndarray, config: dict) -> dict:
    """RMSE, MAE, R^2, L2 relative error over a set of predictions."""
    raise NotImplementedError("Phase 8")


def compute_pde_residual_stats(model, data: dict, microenv: dict, config: dict) -> dict:
    raise NotImplementedError("Phase 8")


def compute_penetration_rmse(model, data: dict, config: dict) -> float:
    raise NotImplementedError("Phase 8")


def decompose_by(metric_fn, data: dict, key: str, bins: list, config: dict) -> dict:
    """Decompose a metric by zone / NP-size range / time range."""
    raise NotImplementedError("Phase 8")


def evaluate_hypotheses(metrics: dict, config: dict) -> dict:
    """Pass/fail summary for H1-H6 against configs' evaluation.hypotheses targets."""
    raise NotImplementedError("Phase 8")


def full_evaluation_report(model, data: dict, microenv: dict, config: dict) -> dict:
    """Top-level entry point used by scripts/run_evaluation.py."""
    raise NotImplementedError("Phase 8")
