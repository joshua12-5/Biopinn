"""BIOPINN network architecture. [Phase 4 — not yet implemented]

Will provide: the 5x64 tanh fully-connected network with Xavier init and the
hard-IC output transform C_NN = sigmoid(f_theta(r,t)) * t_norm, plus an
optional random Fourier feature input encoding. Shared module: constructed
identically on Colab (for training) and locally (for loading a checkpoint
into src/biology.py, src/evaluate.py, and app/server.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BIOPINN(nn.Module):
    """Physics-informed network mapping (r_norm, t_norm) -> C_norm."""

    def __init__(self, config: dict):
        super().__init__()
        raise NotImplementedError("Phase 4")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Phase 4")


def load_checkpoint(checkpoint_path: str, config: dict) -> BIOPINN:
    """Load a trained BIOPINN model from a Colab-produced .pt checkpoint."""
    raise NotImplementedError("Phase 4")
