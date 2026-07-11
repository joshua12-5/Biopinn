"""Forward-Euler FDM solver for the augmented Fickian reaction-diffusion PDE.
[Phase 2 — not yet implemented]

Solves dC/dt = D_eff(r) * [d2C/dr2 + (2/r) dC/dr] - k_d(r) * C on a 1D radial
spherical grid, with Dirichlet BC at r=R, Neumann symmetry at r=0 (ghost
point), and a CFL guard that auto-reduces dt when D_eff_max*dt/dr^2 exceeds
the configured limit. Used on Colab to generate the labeled synthetic dataset
consumed by src/data_pipeline.py, and locally as the ground-truth reference
in src/evaluate.py and src/optimize.py's speedup study.
"""

from __future__ import annotations

import numpy as np


def check_cfl(D_eff_max: float, dt: float, dr: float, config: dict) -> tuple[bool, float]:
    """Return (is_stable, safe_dt). safe_dt equals dt if stable, else the reduced value."""
    raise NotImplementedError("Phase 2")


def solve_fdm(
    R_um: float,
    d_NP_nm: float,
    C0_uM: float,
    k_d_per_hr: float,
    t_max_hr: float,
    config: dict,
) -> dict:
    """Run the forward-Euler solver for one parameter combination.

    Returns a dict with radial grid `r`, time grid `t`, and the concentration
    field `C` of shape [N_t, N_r].
    """
    raise NotImplementedError("Phase 2")
