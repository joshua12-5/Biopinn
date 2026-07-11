"""Forward-Euler FDM solver for the augmented Fickian reaction-diffusion PDE.

Solves dC/dt = D_eff(r) * [d2C/dr2 + (2/r) dC/dr] - k_d(r) * C on a 1D radial
spherical grid, with Dirichlet BC at r=R, Neumann symmetry at r=0 (ghost
point), and a CFL guard that keeps the internal time step stable. Used on
Colab to generate the labeled synthetic dataset consumed by
src/data_pipeline.py, and locally as the ground-truth reference in
src/evaluate.py and src/optimize.py's speedup study.

The output time grid (N_t_initial points, per configs/default_config.yaml)
is kept fixed regardless of how fine the CFL-stable step must be: realistic
D_eff values (thousands of um^2/hr) routinely require an internal step many
times smaller than t_max_hr/(N_t_initial-1), especially for small tumors
with fast-diffusing small nanoparticles. Storing every internal step would
blow up memory (a 100um tumor + 10nm nanoparticle can require tens of
millions of steps), so the solver sub-steps between each *output* time point
with a CFL-safe internal dt and only records the state at the requested
output times.
"""

from __future__ import annotations

import numpy as np

from src.microenvironment import decay_rate_field, effective_diffusivity, radial_grid


def check_cfl(D_eff_max: float, dt: float, dr: float, config: dict) -> tuple[bool, float]:
    """Check the diffusion CFL condition D_eff_max*dt/dr^2 < cfl_limit.

    Returns (is_stable, safe_dt): safe_dt equals dt if stable, otherwise the
    reduced time step dt_safe = cfl_safety_factor * dr^2 / D_eff_max.
    """
    fdm_cfg = config["fdm"]
    cfl = D_eff_max * dt / dr**2
    if cfl < fdm_cfg["cfl_limit"]:
        return True, dt
    dt_safe = fdm_cfg["cfl_safety_factor"] * dr**2 / D_eff_max
    return False, dt_safe


def solve_fdm(
    R_um: float,
    d_NP_nm: float,
    C0_uM: float,
    k_d_per_hr: float,
    t_max_hr: float,
    config: dict,
    k_el_per_hr: float = 0.0,
    D_eff_override: float | None = None,
) -> dict:
    """Run the forward-Euler solver for one parameter combination.

    Args:
        R_um: tumor radius (um).
        d_NP_nm: nanoparticle diameter (nm), sets D_eff(r) via Stokes-Einstein
            (ignored if D_eff_override is given).
        C0_uM: surface drug concentration (uM) at t=0.
        k_d_per_hr: base drug decay rate (1/hr), scaled per zone (always
            zone-varying, even when D_eff_override is set).
        t_max_hr: simulation duration (hr).
        config: full config dict.
        k_el_per_hr: optional plasma elimination rate (1/hr). When > 0 the
            Dirichlet boundary follows C(R,t) = C0 * exp(-k_el*t) to fold in
            first-order PK decay of the surface concentration; 0 keeps the
            boundary fixed at C0.
        D_eff_override: if given, use this single spatially-constant
            diffusivity (um^2/hr) everywhere instead of the zone-resolved
            D_eff(r) -- used by src/optimize.py's H3 homogeneous-vs-
            heterogeneous comparison (a homogeneous medium at the arithmetic
            mean of the three zone D_eff values, with k_d(r) left
            zone-varying, per the spec's literal "homogeneous diffusion
            coefficient" comparison).

    Returns:
        Dict with radial grid `r` (um), output time grid `t` (hr, length
        N_t_initial), concentration field `C` of shape [N_t, N_r] (uM), the
        `D_eff` (um^2/hr) and `k_d` (1/hr) fields actually used, and the
        `n_substeps` used internally per output interval to satisfy CFL.
    """
    fdm_cfg = config["fdm"]
    N_r = fdm_cfg["N_r"]
    N_t = fdm_cfg["N_t_initial"]
    r_min_um = fdm_cfg["r_min_um"]

    r = radial_grid(R_um, N_r, r_min_um)
    dr = r[1] - r[0]

    if D_eff_override is None:
        D_eff = effective_diffusivity(r, d_NP_nm, config)
    else:
        D_eff = np.full(N_r, D_eff_override)
    k_d = decay_rate_field(r, k_d_per_hr, config)
    D_eff_max = D_eff.max()

    t = np.linspace(0.0, t_max_hr, N_t)
    dt_output = t[1] - t[0]

    is_stable, dt_internal = check_cfl(D_eff_max, dt_output, dr, config)
    n_substeps = 1 if is_stable else int(np.ceil(dt_output / dt_internal))
    dt_internal = dt_output / n_substeps

    def boundary_value(t_now: float) -> float:
        if k_el_per_hr > 0:
            return C0_uM * np.exp(-k_el_per_hr * t_now)
        return C0_uM

    r_mid = r[1:-1]
    D_mid = D_eff[1:-1]
    k_mid = k_d[1:-1]

    C = np.zeros((N_t, N_r))
    C[0, -1] = boundary_value(0.0)

    state = C[0, :].copy()
    for out_idx in range(1, N_t):
        t_start = t[out_idx - 1]
        for step in range(n_substeps):
            C_left = state[:-2]
            C_mid = state[1:-1]
            C_right = state[2:]

            d2C_dr2 = (C_right - 2.0 * C_mid + C_left) / dr**2
            dC_dr = (C_right - C_left) / (2.0 * dr)
            diffusion = D_mid * (d2C_dr2 + (1.0 / r_mid) * dC_dr)

            new_state = state.copy()
            new_state[1:-1] = C_mid + dt_internal * (diffusion - k_mid * C_mid)
            new_state[1:-1] = np.maximum(new_state[1:-1], 0.0)
            new_state[0] = new_state[1]  # Neumann symmetry at r=0 (ghost-point method)
            new_state[-1] = boundary_value(t_start + (step + 1) * dt_internal)  # Dirichlet at r=R

            state = new_state

        C[out_idx, :] = state

    return {
        "r": r,
        "t": t,
        "C": C,
        "D_eff": D_eff,
        "k_d": k_d,
        "n_substeps": n_substeps,
        "dt_internal": dt_internal,
    }
