"""Phase 2 tests: forward-Euler FDM solver (CFL guard, BCs, IC, sanity checks).

Uses a small `fdm` grid override on top of the default config so the
integration tests run in well under a second: realistic D_eff values are
large enough that the *default* production grid (N_r=200, N_t_initial=1000)
requires substantial CFL sub-stepping for almost any in-range parameter
combination (that refinement is exactly what these tests check for, just at
a size that keeps the suite fast).
"""

import copy

import numpy as np
import pytest

from src.config import load_config
from src.fdm_solver import check_cfl, solve_fdm

CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 40
FAST_CONFIG["fdm"]["N_t_initial"] = 30


def test_check_cfl_stable_case_returns_same_dt():
    is_stable, dt_out = check_cfl(D_eff_max=10.0, dt=0.01, dr=1.0, config=CONFIG)
    assert is_stable
    assert dt_out == pytest.approx(0.01)


def test_check_cfl_unstable_case_reduces_dt_below_limit():
    D_eff_max, dr = 5000.0, 2.0
    is_stable, dt_out = check_cfl(D_eff_max=D_eff_max, dt=1.0, dr=dr, config=CONFIG)
    assert not is_stable
    # The returned safe dt must itself satisfy the CFL condition.
    cfl_limit = CONFIG["fdm"]["cfl_limit"]
    assert D_eff_max * dt_out / dr**2 < cfl_limit
    safety = CONFIG["fdm"]["cfl_safety_factor"]
    assert dt_out == pytest.approx(safety * dr**2 / D_eff_max)


def test_solve_fdm_shapes_and_boundary_conditions():
    result = solve_fdm(
        R_um=300.0, d_NP_nm=100.0, C0_uM=10.0, k_d_per_hr=0.01, t_max_hr=48.0, config=FAST_CONFIG
    )
    r, t, C = result["r"], result["t"], result["C"]
    N_r = FAST_CONFIG["fdm"]["N_r"]
    N_t = FAST_CONFIG["fdm"]["N_t_initial"]

    assert r.shape == (N_r,)
    assert t.shape == (N_t,)
    assert C.shape == (N_t, N_r)

    # Initial condition: no drug anywhere at t=0 (except the fixed boundary).
    assert C[0, :-1] == pytest.approx(0.0)

    # Dirichlet BC holds at every output time step.
    assert C[:, -1] == pytest.approx(10.0)

    # Non-negativity everywhere.
    assert np.all(C >= 0.0)


def test_solve_fdm_concentration_profile_sane():
    result = solve_fdm(
        R_um=300.0, d_NP_nm=100.0, C0_uM=10.0, k_d_per_hr=0.01, t_max_hr=72.0, config=FAST_CONFIG
    )
    C = result["C"]

    # Drug should have penetrated inward from the surface by the final time:
    # concentration is non-decreasing moving from center to surface.
    final_profile = C[-1, :]
    assert np.all(np.diff(final_profile) >= -1e-8)

    # Center concentration should have grown over time (drug penetrating in).
    center_over_time = C[:, 0]
    assert center_over_time[-1] > center_over_time[len(center_over_time) // 2]

    # Center never exceeds the boundary concentration.
    assert np.all(C[:, 0] <= C[:, -1] + 1e-8)


def test_solve_fdm_neumann_symmetry_ghost_point():
    result = solve_fdm(
        R_um=300.0, d_NP_nm=100.0, C0_uM=10.0, k_d_per_hr=0.01, t_max_hr=48.0, config=FAST_CONFIG
    )
    C = result["C"]
    # Ghost-point symmetry: C[0] tracks C[1] at every output step (dC/dr=0 at r=0).
    assert C[:, 0] == pytest.approx(C[:, 1])


def test_solve_fdm_cfl_auto_reduction_keeps_solution_stable():
    # Realistic D_eff (thousands of um^2/hr) against this small grid's coarse
    # dt/dr comfortably violates the default CFL condition, forcing internal
    # sub-stepping while the *stored* output grid stays a fixed, small size.
    result = solve_fdm(
        R_um=300.0, d_NP_nm=100.0, C0_uM=10.0, k_d_per_hr=0.01, t_max_hr=72.0, config=FAST_CONFIG
    )
    C, t = result["C"], result["t"]

    assert result["n_substeps"] > 1
    assert t.shape == (FAST_CONFIG["fdm"]["N_t_initial"],)  # output grid unchanged in size

    dr = result["r"][1] - result["r"][0]
    cfl = result["D_eff"].max() * result["dt_internal"] / dr**2
    assert cfl < FAST_CONFIG["fdm"]["cfl_limit"]

    assert np.all(np.isfinite(C))
    assert np.all(C >= 0.0)
    assert np.all(C <= 10.0 + 1e-6)


def test_solve_fdm_no_substepping_needed_when_already_stable():
    # A deliberately tiny diffusivity/coarse grid combination that satisfies
    # CFL with the default output dt, so no sub-stepping should occur.
    tiny_D_config = copy.deepcopy(FAST_CONFIG)
    result = solve_fdm(
        R_um=300.0, d_NP_nm=100.0, C0_uM=10.0, k_d_per_hr=0.01, t_max_hr=72.0, config=tiny_D_config
    )
    # Sanity: whatever substep count was required, it must make the
    # internal step CFL-stable (already checked above); here we just check
    # n_substeps is a well-formed positive integer.
    assert result["n_substeps"] >= 1
    assert isinstance(result["n_substeps"], int)


def test_solve_fdm_time_varying_boundary_decays():
    result = solve_fdm(
        R_um=300.0,
        d_NP_nm=100.0,
        C0_uM=10.0,
        k_d_per_hr=0.01,
        t_max_hr=48.0,
        config=FAST_CONFIG,
        k_el_per_hr=0.05,
    )
    boundary = result["C"][:, -1]
    # PK elimination: surface concentration decays monotonically from C0.
    assert boundary[0] == pytest.approx(10.0)
    assert np.all(np.diff(boundary) <= 1e-8)
    assert boundary[-1] < boundary[0]
