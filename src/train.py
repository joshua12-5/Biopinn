"""Two-phase Adam -> L-BFGS training engine. [Phase 5 — not yet implemented]

Will provide: Phase-1 Adam training (StepLR schedule, gradient clipping),
Phase-2 L-BFGS fine-tuning (strong Wolfe line search) seeded from the Adam
state, per-component loss logging, best-checkpoint saving, the
convergence/patience criterion, and the NaN-recovery safeguards (lr
reduction + w_phys ramp). Runs on Colab; used unchanged by src/ablation.py
locally to train the w_phys=0 baseline on CPU for the ablation study.
"""

from __future__ import annotations

from src.model import BIOPINN


def train_adam_phase(model: BIOPINN, data: dict, microenv: dict, config: dict) -> dict:
    """Run Phase-1 Adam optimization. Returns loss history."""
    raise NotImplementedError("Phase 5")


def train_lbfgs_phase(model: BIOPINN, data: dict, microenv: dict, config: dict) -> dict:
    """Run Phase-2 L-BFGS fine-tuning from the Adam-trained state. Returns loss history."""
    raise NotImplementedError("Phase 5")


def train(config: dict, data: dict, microenv: dict) -> dict:
    """Full two-phase training run. Saves best checkpoint + normalization stats.

    Returns a dict with the trained model and combined loss history.
    """
    raise NotImplementedError("Phase 5")
