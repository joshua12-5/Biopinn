"""Phase 1 tests: microenvironment model (grid, Stokes-Einstein, oxygen,
zones, spatial D_eff / k_d)."""

import numpy as np
import pytest

from src.config import load_config
from src.microenvironment import (
    assign_zones,
    decay_rate_field,
    effective_diffusivity,
    oxygen_gradient,
    radial_grid,
    stokes_einstein_diffusivity,
)

CONFIG = load_config()


def test_stokes_einstein_decreases_with_size():
    const = CONFIG["constants"]
    D_small = stokes_einstein_diffusivity(10.0, const["T"], const["eta"], const["k_B"])
    D_large = stokes_einstein_diffusivity(200.0, const["T"], const["eta"], const["k_B"])
    assert D_small > D_large > 0


def test_stokes_einstein_matches_formula():
    const = CONFIG["constants"]
    d_NP_nm = 100.0
    D = stokes_einstein_diffusivity(d_NP_nm, const["T"], const["eta"], const["k_B"])
    expected = const["k_B"] * const["T"] / (3 * np.pi * const["eta"] * (d_NP_nm * 1e-9))
    assert D == pytest.approx(expected)


def test_stokes_einstein_endpoint_values_match_known_physical_magnitude():
    """Pins the two endpoint values against the physically correct magnitude
    for this config's constants (k_B=1.380649e-23 J/K, T=310 K,
    eta=1.2e-3 Pa*s): D_free = k_B*T/(3*pi*eta*d_NP) gives ~3.8e-11 m^2/s at
    d_NP=10nm and ~1.9e-12 m^2/s at d_NP=200nm. A prior version of the
    project's manuscript printed values ~5x lower than this (~7.3e-12 and
    ~3.7e-13) -- that was a manuscript-prose error, not a bug in this
    formula, but pinned here so a future regression in the formula or
    constants would be caught."""
    const = CONFIG["constants"]
    D_10nm = stokes_einstein_diffusivity(10.0, const["T"], const["eta"], const["k_B"])
    D_200nm = stokes_einstein_diffusivity(200.0, const["T"], const["eta"], const["k_B"])
    assert D_10nm == pytest.approx(3.8e-11, rel=0.02)
    assert D_200nm == pytest.approx(1.9e-12, rel=0.02)


def test_radial_grid_shape_and_bounds():
    r = radial_grid(R_um=400.0, N_r=200, r_min_um=0.001)
    assert r.shape == (200,)
    assert r[0] == pytest.approx(0.001)
    assert r[-1] == pytest.approx(400.0)
    assert np.all(np.diff(r) > 0)


def test_oxygen_gradient_decreases_toward_center():
    r = radial_grid(R_um=400.0, N_r=200, r_min_um=0.001)
    O2 = oxygen_gradient(r, CONFIG)
    assert O2.shape == r.shape
    assert np.all(O2 >= 0)
    # Surface oxygen should approach the configured surface value.
    surface_o2 = CONFIG["microenvironment"]["oxygen"]["surface_o2_percent"]
    assert O2[-1] == pytest.approx(surface_o2, rel=1e-6)
    # Oxygen falls monotonically from surface to center.
    assert np.all(np.diff(O2) >= 0)
    assert O2[0] < O2[-1]


def test_oxygen_gradient_no_nan_at_center():
    r = radial_grid(R_um=100.0, N_r=50, r_min_um=0.0)
    O2 = oxygen_gradient(r, CONFIG)
    assert np.all(np.isfinite(O2))


def test_larger_tumor_is_more_hypoxic_at_center():
    r_small = radial_grid(R_um=100.0, N_r=200, r_min_um=0.001)
    r_large = radial_grid(R_um=500.0, N_r=200, r_min_um=0.001)
    O2_small = oxygen_gradient(r_small, CONFIG)
    O2_large = oxygen_gradient(r_large, CONFIG)
    assert O2_large[0] < O2_small[0]


def test_assign_zones_labels_valid_and_ordered_outward():
    r = radial_grid(R_um=500.0, N_r=300, r_min_um=0.001)
    O2 = oxygen_gradient(r, CONFIG)
    zones = assign_zones(r, O2, CONFIG)
    assert set(np.unique(zones)) <= {"proliferating_rim", "quiescent_zone", "necrotic_core"}
    # Surface must be proliferating rim (normoxic boundary condition).
    assert zones[-1] == "proliferating_rim"
    # A large tumor should have a necrotic core.
    assert zones[0] == "necrotic_core"


def test_small_tumor_may_have_no_necrotic_core():
    r = radial_grid(R_um=100.0, N_r=200, r_min_um=0.001)
    O2 = oxygen_gradient(r, CONFIG)
    zones = assign_zones(r, O2, CONFIG)
    # Small enough tumors should stay fully oxygenated/proliferating.
    assert zones[-1] == "proliferating_rim"


def test_effective_diffusivity_zone_ordering():
    r = radial_grid(R_um=500.0, N_r=300, r_min_um=0.001)
    D_eff = effective_diffusivity(r, d_NP_nm=100.0, config=CONFIG)
    assert np.all(D_eff > 0)
    # Necrotic core (f_zone=0.5) has higher D_eff than the dense proliferating
    # rim (f_zone=0.2), despite both sharing the same D_free.
    assert D_eff[0] > D_eff[-1]


def test_decay_rate_field_zone_multipliers():
    r = radial_grid(R_um=500.0, N_r=300, r_min_um=0.001)
    k_d = decay_rate_field(r, k_d_base=0.02, config=CONFIG)
    assert np.all(k_d > 0)
    # Rim binds/decays fastest (1.5x), necrotic core slowest (0.3x).
    assert k_d[-1] == pytest.approx(0.02 * 1.5)
    assert k_d[0] == pytest.approx(0.02 * 0.3)


def test_batched_pointwise_evaluation_matches_per_sim_grid():
    # A mixed-simulation batch (different R and d_NP per point, via the R=
    # and array d_NP_nm arguments) must reproduce exactly what a per-sim
    # radial_grid + effective_diffusivity call would give for each point --
    # this is what src/losses.py relies on for a collocation batch drawn
    # from many simulations at once.
    R_a, d_NP_a, k_d_base_a = 200.0, 50.0, 0.02
    R_b, d_NP_b, k_d_base_b = 450.0, 150.0, 0.03

    r_a = np.array([1.0, 100.0, 199.0])
    r_b = np.array([1.0, 225.0, 449.0])

    r_batch = np.concatenate([r_a, r_b])
    R_batch = np.concatenate([np.full(3, R_a), np.full(3, R_b)])
    d_NP_batch = np.concatenate([np.full(3, d_NP_a), np.full(3, d_NP_b)])
    k_d_base_batch = np.concatenate([np.full(3, k_d_base_a), np.full(3, k_d_base_b)])

    D_eff_batch = effective_diffusivity(r_batch, d_NP_batch, CONFIG, R=R_batch)
    k_d_batch = decay_rate_field(r_batch, k_d_base_batch, CONFIG, R=R_batch)

    D_eff_a_expected = effective_diffusivity(r_a, d_NP_a, CONFIG, R=R_a)
    D_eff_b_expected = effective_diffusivity(r_b, d_NP_b, CONFIG, R=R_b)
    k_d_a_expected = decay_rate_field(r_a, k_d_base_a, CONFIG, R=R_a)
    k_d_b_expected = decay_rate_field(r_b, k_d_base_b, CONFIG, R=R_b)

    np.testing.assert_allclose(D_eff_batch[:3], D_eff_a_expected)
    np.testing.assert_allclose(D_eff_batch[3:], D_eff_b_expected)
    np.testing.assert_allclose(k_d_batch[:3], k_d_a_expected)
    np.testing.assert_allclose(k_d_batch[3:], k_d_b_expected)


def test_oxygen_gradient_default_R_unchanged_for_single_grid():
    # Backward compatibility: omitting R still uses r[-1], exactly the
    # original single-simulation-grid behavior relied on by src/fdm_solver.py.
    r = radial_grid(R_um=300.0, N_r=100, r_min_um=0.001)
    O2_default = oxygen_gradient(r, CONFIG)
    O2_explicit = oxygen_gradient(r, CONFIG, R=r[-1])
    np.testing.assert_array_equal(O2_default, O2_explicit)
