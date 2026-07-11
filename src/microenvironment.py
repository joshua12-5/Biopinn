"""Three-zone tumor microenvironment model.

Provides: radial grid construction, Stokes-Einstein free diffusivity, the
steady-state oxygen gradient in a spherical tumor, zone assignment
(proliferating rim / quiescent zone / necrotic core) derived from that
oxygen field, and the resulting spatially varying effective diffusion
coefficient D_eff(r) and decay-rate field k_d(r) used by the FDM solver
(src/fdm_solver.py) and the PINN physics loss (src/losses.py).

Shared module: imported unchanged by both the Colab data-generation notebook
and the local analysis/dashboard code.
"""

from __future__ import annotations

import numpy as np

ZONE_ORDER = ("proliferating_rim", "quiescent_zone", "necrotic_core")


def stokes_einstein_diffusivity(d_NP_nm: float, T: float, eta: float, k_B: float) -> float:
    """Free diffusivity of a spherical nanoparticle via Stokes-Einstein.

    D_free = k_B * T / (3 * pi * eta * d_NP), equivalent to k_B*T/(6*pi*eta*r_NP).

    Args:
        d_NP_nm: nanoparticle diameter in nanometers.
        T: absolute temperature in Kelvin.
        eta: dynamic viscosity of the medium in Pa*s.
        k_B: Boltzmann constant in J/K.

    Returns:
        D_free in m^2/s.
    """
    d_NP_m = d_NP_nm * 1e-9
    return k_B * T / (3 * np.pi * eta * d_NP_m)


def radial_grid(R_um: float, N_r: int, r_min_um: float) -> np.ndarray:
    """Uniform radial grid of N_r points from r_min_um to R_um (inclusive)."""
    return np.linspace(r_min_um, R_um, N_r)


def oxygen_gradient(r: np.ndarray, config: dict, R: np.ndarray | float | None = None) -> np.ndarray:
    """Steady-state oxygen diffusion-consumption profile over the radial grid.

    Solves D_O2 * lap(O2) = k_O2 * O2 in spherical coordinates with a fixed
    surface concentration, using the closed-form solution
        O2(r) = O2_surface * R * sinh(r*lambda) / (r * sinh(R*lambda))
    where lambda = sqrt(k_O2 / D_O2). The r=0 singularity is resolved via
    the L'Hopital limit O2(0) = O2_surface * R * lambda / sinh(R*lambda).

    Args:
        r: radial positions (um). May be a single simulation's radial grid
            or an arbitrary batch of points from many simulations.
        config: full config dict (reads microenvironment.oxygen and constants).
        R: tumor radius (um) for each point in `r`. Defaults to r[-1], i.e.
            "r is one simulation's grid and its own last point is R" (the
            src/fdm_solver.py usage). Pass an array the same shape as `r`
            (or a scalar) to evaluate points drawn from different tumor
            radii, e.g. a mixed-simulation PINN collocation batch.

    Returns:
        Oxygen concentration (% saturation) at each point in `r`.
    """
    ox = config["microenvironment"]["oxygen"]
    D_O2 = ox["D_oxygen_um2_per_hr"]
    k_O2 = ox["consumption_rate_per_hr"]
    O2_surface = ox["surface_o2_percent"]

    r = np.asarray(r, dtype=float)
    R = np.asarray(r[-1] if R is None else R, dtype=float)
    lam = np.sqrt(k_O2 / D_O2)

    def R_at(mask: np.ndarray) -> np.ndarray:
        return R if R.ndim == 0 else R[mask]

    O2 = np.empty_like(r, dtype=float)
    nonzero = r > 0
    R_nz = R_at(nonzero)
    r_nz = r[nonzero]
    O2[nonzero] = O2_surface * R_nz * np.sinh(r_nz * lam) / (r_nz * np.sinh(R_nz * lam))
    if np.any(~nonzero):
        R_z = R_at(~nonzero)
        O2[~nonzero] = O2_surface * R_z * lam / np.sinh(R_z * lam)
    return O2


def assign_zones(r: np.ndarray, oxygen: np.ndarray, config: dict) -> np.ndarray:
    """Assign each grid point to a zone based on local oxygen concentration.

    proliferating_rim: O2 > hypoxia threshold (normoxic, active division)
    quiescent_zone:     anoxia threshold < O2 <= hypoxia threshold (hypoxic, dormant)
    necrotic_core:      O2 <= anoxia threshold (dead/dying)
    """
    ox = config["microenvironment"]["oxygen"]
    hypoxia = ox["hypoxia_threshold_percent"]
    anoxia = ox["necrotic_threshold_percent"]

    zones = np.empty(len(r), dtype=object)
    zones[oxygen > hypoxia] = "proliferating_rim"
    zones[(oxygen > anoxia) & (oxygen <= hypoxia)] = "quiescent_zone"
    zones[oxygen <= anoxia] = "necrotic_core"
    return zones


def effective_diffusivity(
    r: np.ndarray, d_NP_nm: np.ndarray | float, config: dict, R: np.ndarray | float | None = None
) -> np.ndarray:
    """D_eff(r) = f_zone(r) * D_free, per point, in um^2/hr.

    D_free is computed via Stokes-Einstein (m^2/s) and converted to um^2/hr
    to match the radial grid (um) and time grid (hr) used by the FDM solver.
    `d_NP_nm` and `R` may be scalars (one simulation) or arrays the same
    shape as `r` (a batch of points drawn from different simulations, e.g.
    a PINN collocation batch -- see oxygen_gradient's `R` argument).
    """
    const = config["constants"]
    D_free_m2_s = stokes_einstein_diffusivity(
        np.asarray(d_NP_nm, dtype=float), const["T"], const["eta"], const["k_B"]
    )
    D_free_um2_hr = D_free_m2_s * 1e12 * 3600.0

    oxygen = oxygen_gradient(r, config, R=R)
    zones = assign_zones(r, oxygen, config)
    f_zone = config["microenvironment"]["f_zone"]

    f_vals = np.array([f_zone[z] for z in zones])
    return D_free_um2_hr * f_vals


def decay_rate_field(
    r: np.ndarray, k_d_base: np.ndarray | float, config: dict, R: np.ndarray | float | None = None
) -> np.ndarray:
    """k_d(r) = k_d_base * zone_multiplier(r), per point (1/hr).

    `k_d_base` and `R` may be scalars or arrays the same shape as `r` (see
    effective_diffusivity's docstring for the batched-collocation use case).
    """
    oxygen = oxygen_gradient(r, config, R=R)
    zones = assign_zones(r, oxygen, config)
    multiplier = config["microenvironment"]["k_d_multiplier"]

    f_vals = np.array([multiplier[z] for z in zones])
    return np.asarray(k_d_base, dtype=float) * f_vals
