"""Three-zone tumor microenvironment model. [Phase 1 — not yet implemented]

Will provide: radial grid construction, Stokes-Einstein free diffusivity,
steady-state oxygen gradient, zone assignment (proliferating rim / quiescent
zone / necrotic core) from the oxygen field, and the spatially varying
effective diffusion coefficient D_eff(r) and decay-rate multiplier k_d(r)
used by the FDM solver (src/fdm_solver.py) and the PINN physics loss
(src/losses.py).

Shared module: imported unchanged by both the Colab data-generation notebook
and the local analysis/dashboard code.
"""

from __future__ import annotations

import numpy as np


def stokes_einstein_diffusivity(d_NP_m: float, T: float, eta: float, k_B: float) -> float:
    """D_free = k_B * T / (3 * pi * eta * d_NP). Returns m^2/s."""
    raise NotImplementedError("Phase 1")


def radial_grid(R_um: float, N_r: int, r_min_um: float) -> np.ndarray:
    """Uniform radial grid from r_min_um to R_um with N_r points."""
    raise NotImplementedError("Phase 1")


def oxygen_gradient(r: np.ndarray, config: dict) -> np.ndarray:
    """Steady-state oxygen diffusion-consumption profile over the radial grid."""
    raise NotImplementedError("Phase 1")


def assign_zones(r: np.ndarray, oxygen: np.ndarray, config: dict) -> np.ndarray:
    """Return per-grid-point zone labels derived from the oxygen gradient."""
    raise NotImplementedError("Phase 1")


def effective_diffusivity(r: np.ndarray, d_NP_nm: float, config: dict) -> np.ndarray:
    """D_eff(r) = f_zone(r) * D_free, per grid point."""
    raise NotImplementedError("Phase 1")


def decay_rate_field(r: np.ndarray, k_d_base: float, config: dict) -> np.ndarray:
    """k_d(r) = k_d_base * zone_multiplier(r), per grid point."""
    raise NotImplementedError("Phase 1")
