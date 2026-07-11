"""Composite PINN loss: L_data + L_phys + L_bc + L_neu + L_ic. [Phase 4 — not
yet implemented]

Will provide the five loss components (data fidelity, PDE residual via
torch.autograd.grad for dC/dt, dC/dr, d2C/dr2, Dirichlet surface, Neumann
symmetry at r=0, and initial condition), combined with the configured
weights (w_data, w_phys, w_bc, w_neu, w_ic). Used by src/train.py on Colab
and by src/evaluate.py locally (to compute the PDE-residual metric on the
held-out test set) and src/ablation.py (w_phys=0 baseline).
"""

from __future__ import annotations

import torch

from src.model import BIOPINN


def pde_residual(model: BIOPINN, r_t: torch.Tensor, microenv: dict) -> torch.Tensor:
    """Compute the PDE residual dC/dt - D_eff*(d2C/dr2 + 2/r*dC/dr) + k_d*C at collocation points."""
    raise NotImplementedError("Phase 4")


def data_loss(model: BIOPINN, r_t: torch.Tensor, C_label: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Phase 4")


def physics_loss(model: BIOPINN, r_t_colloc: torch.Tensor, microenv: dict) -> torch.Tensor:
    raise NotImplementedError("Phase 4")


def dirichlet_bc_loss(model: BIOPINN, r_t_surface: torch.Tensor, C0_norm: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Phase 4")


def neumann_bc_loss(model: BIOPINN, r_t_center: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Phase 4")


def ic_loss(model: BIOPINN, r_t_ic: torch.Tensor) -> torch.Tensor:
    raise NotImplementedError("Phase 4")


def composite_loss(model: BIOPINN, batch: dict, microenv: dict, config: dict) -> dict:
    """Return a dict of individual losses plus 'total', weighted per config['loss']."""
    raise NotImplementedError("Phase 4")
