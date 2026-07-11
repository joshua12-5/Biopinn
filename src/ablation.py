"""Physics-informed vs. unconstrained ablation study (H5). [Phase 9 — not yet
implemented]

Will provide: training of an identical-architecture baseline network with
w_phys=0 (data + BC + IC only, via configs/experiment_2.yaml), comparison of
RMSE/R^2 and PDE-residual statistics (mean, max, physical-consistency %)
against full BIOPINN, and a Wilcoxon signed-rank test on the paired
PDE-residual-magnitude distributions. Local module.
"""

from __future__ import annotations

import numpy as np


def train_baseline(config: dict, data: dict, microenv: dict) -> dict:
    """Train the w_phys=0 baseline (reuses src/train.py with experiment_2.yaml)."""
    raise NotImplementedError("Phase 9")


def compare_residuals(biopinn_model, baseline_model, data: dict, microenv: dict, config: dict) -> dict:
    """Mean/max PDE residual + physical-consistency % for both models."""
    raise NotImplementedError("Phase 9")


def wilcoxon_test(residuals_biopinn: np.ndarray, residuals_baseline: np.ndarray) -> dict:
    """Wilcoxon signed-rank test on paired PDE-residual magnitudes. Returns statistic + p-value."""
    raise NotImplementedError("Phase 9")


def ablation_report(biopinn_model, baseline_model, data: dict, microenv: dict, config: dict) -> dict:
    """Top-level entry point used by scripts/run_ablation.py."""
    raise NotImplementedError("Phase 9")
