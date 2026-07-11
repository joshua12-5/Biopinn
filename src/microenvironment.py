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


def oxygen_gradient(r: np.ndarray, config: dict) -> np.ndarray:
    """Steady-state oxygen diffusion-consumption profile over the radial grid.

    Solves D_O2 * lap(O2) = k_O2 * O2 in spherical coordinates with a fixed
    surface concentration, using the closed-form solution
        O2(r) = O2_surface * R * sinh(r*lambda) / (r * sinh(R*lambda))
    where lambda = sqrt(k_O2 / D_O2). The r=0 singularity is resolved via
    the L'Hopital limit O2(0) = O2_surface * R * lambda / sinh(R*lambda).

    Args:
        r: radial grid (um), r[-1] is treated as the tumor radius R.
        config: full config dict (reads microenvironment.oxygen and constants).

    Returns:
        Oxygen concentration (% saturation) at each grid point.
    """
    ox = config["microenvironment"]["oxygen"]
    D_O2 = ox["D_oxygen_um2_per_hr"]
    k_O2 = ox["consumption_rate_per_hr"]
    O2_surface = ox["surface_o2_percent"]

    R = r[-1]
    lam = np.sqrt(k_O2 / D_O2)

    O2 = np.empty_like(r, dtype=float)
    nonzero = r > 0
    O2[nonzero] = O2_surface * R * np.sinh(r[nonzero] * lam) / (r[nonzero] * np.sinh(R * lam))
    O2[~nonzero] = O2_surface * R * lam / np.sinh(R * lam)
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


def effective_diffusivity(r: np.ndarray, d_NP_nm: float, config: dict) -> np.ndarray:
    """D_eff(r) = f_zone(r) * D_free, per grid point, in um^2/hr.

    D_free is computed via Stokes-Einstein (m^2/s) and converted to um^2/hr
    to match the radial grid (um) and time grid (hr) used by the FDM solver.
    """
    const = config["constants"]
    D_free_m2_s = stokes_einstein_diffusivity(d_NP_nm, const["T"], const["eta"], const["k_B"])
    D_free_um2_hr = D_free_m2_s * 1e12 * 3600.0

    oxygen = oxygen_gradient(r, config)
    zones = assign_zones(r, oxygen, config)
    f_zone = config["microenvironment"]["f_zone"]

    D_eff = np.array([D_free_um2_hr * f_zone[z] for z in zones])
    return D_eff


def decay_rate_field(r: np.ndarray, k_d_base: float, config: dict) -> np.ndarray:
    """k_d(r) = k_d_base * zone_multiplier(r), per grid point (1/hr)."""
    oxygen = oxygen_gradient(r, config)
    zones = assign_zones(r, oxygen, config)
    multiplier = config["microenvironment"]["k_d_multiplier"]

    k_d = np.array([k_d_base * multiplier[z] for z in zones])
    return k_d
