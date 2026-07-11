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
