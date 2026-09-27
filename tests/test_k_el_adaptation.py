"""Tests for the Nano-Tumor Database clearance adaptation (docs/NTD_ADAPTATION.md).

k_el_per_hr as a 6th sampled parameter (time-varying Dirichlet boundary
C(R,t) = C0*exp(-k_el*t)), backward compatibility with the legacy
5-parameter / 7-input setup, and the configurable absolute thresholds.
"""

import copy

import numpy as np
import pytest
import torch

from src.biology import predict_concentration_field
from src.config import load_config
from src.data_pipeline import (
    BASE_PARAM_ORDER,
    PARAM_ORDER,
    _solve_one,
    build_dataset,
    check_input_dim,
    default_k_el,
    latin_hypercube_sample,
    param_order,
)
from src.evaluate import concentration_threshold_uM, resolve_test_simulations
from src.model import BIOPINN
from src.optimize import exposure_profile, subtherapeutic_threshold_uM

CONFIG = load_config()
LEGACY = load_config("experiment_legacy_5param")


def _fast(config):
    cfg = copy.deepcopy(config)
    cfg["fdm"]["N_r"] = 25
    cfg["fdm"]["N_t_initial"] = 15
    cfg["dataset"]["split"] = {"train": 4, "val": 2, "test": 2}
    cfg["dataset"]["points_per_sim"] = {"data": 40, "collocation": 60, "bc_surface": 10, "bc_center": 10, "ic": 10}
    return cfg


def test_default_config_samples_k_el_over_ntd_range():
    assert CONFIG["dataset"]["sample_k_el"] is True
    assert CONFIG["dataset"]["parameter_ranges"]["k_el_per_hr"] == [0.0136, 0.381]
    assert CONFIG["model"]["input_dim"] == 8
    assert param_order(CONFIG) == PARAM_ORDER
    assert default_k_el(CONFIG) == pytest.approx(0.0684)
    sample = latin_hypercube_sample(30, CONFIG, seed=0)
    assert sample.shape == (30, 6)
    assert np.all((sample[:, 5] >= 0.0136) & (sample[:, 5] <= 0.381))


def test_legacy_config_restores_5_params_and_constant_boundary():
    assert param_order(LEGACY) == BASE_PARAM_ORDER
    assert default_k_el(LEGACY) == 0.0
    check_input_dim(LEGACY)
    ds = build_dataset(_fast(LEGACY), seed=3, save=False)
    assert "k_el_per_hr" not in ds["stats"]
    assert ds["splits"]["train"]["data_X"].shape[1] == 7
    assert ds["splits"]["train"]["bc_surface_y"] == pytest.approx(1.0)


def test_check_input_dim_rejects_mismatch():
    cfg = copy.deepcopy(CONFIG)
    cfg["model"]["input_dim"] = 7
    with pytest.raises(ValueError, match="input_dim"):
        check_input_dim(cfg)


def test_solve_one_applies_decaying_surface_boundary():
    cfg = _fast(CONFIG)
    k_el = 0.1
    sim = _solve_one((0, np.array([300.0, 100.0, 5.0, 0.01, 48.0, k_el]), cfg))
    assert sim["k_el_per_hr"] == k_el
    np.testing.assert_allclose(sim["C"][1:, -1], 5.0 * np.exp(-k_el * sim["t"][1:]), rtol=1e-9)
    # A 5-value row (legacy) means k_el = 0: constant surface concentration.
    legacy = _solve_one((0, np.array([300.0, 100.0, 5.0, 0.01, 48.0]), cfg))
    assert legacy["k_el_per_hr"] == 0.0
    assert np.all(legacy["C"][:, -1] == 5.0)


def test_resolve_test_simulations_treats_missing_k_el_as_zero():
    cfg = _fast(CONFIG)
    params = [{"sim_id": 0, "R_um": 300.0, "d_NP_nm": 100.0, "C0_uM": 5.0, "k_d_per_hr": 0.01, "t_max_hr": 48.0}]
    (sim,) = resolve_test_simulations(cfg, sim_params=params)
    assert sim["k_el_per_hr"] == 0.0
    assert np.all(sim["C"][:, -1] == 5.0)


def test_predict_uses_checkpoint_stats_layout_for_legacy_models():
    """A 7-input model + 5-key normalization stats still works under the new code."""
    torch.manual_seed(0)
    model = BIOPINN(LEGACY)
    stats5 = {p: {"min": 0.0, "max": 1.0} for p in BASE_PARAM_ORDER}
    r, t = np.linspace(0.001, 300.0, 9), np.linspace(0.0, 48.0, 5)
    params = {"R_um": 300.0, "d_NP_nm": 100.0, "C0_uM": 5.0, "k_d_per_hr": 0.01, "t_max_hr": 48.0}
    C = predict_concentration_field(model, r, t, params, stats5)
    assert C.shape == (5, 9) and np.all(np.isfinite(C))

    # 8-input model: k_el may be omitted (defaults to 0.0) or given explicitly.
    model8 = BIOPINN(CONFIG)
    stats6 = {**stats5, "k_el_per_hr": {"min": 0.0136, "max": 0.381}}
    C_a = predict_concentration_field(model8, r, t, params, stats6)
    C_b = predict_concentration_field(model8, r, t, {**params, "k_el_per_hr": 0.2}, stats6)
    assert C_a.shape == C_b.shape == (5, 9)
    assert not np.allclose(C_a, C_b)


def test_concentration_threshold_modes():
    assert concentration_threshold_uM(CONFIG, C0_uM=10.0) == pytest.approx(1.0)  # absolute default
    assert concentration_threshold_uM(LEGACY, C0_uM=10.0) == pytest.approx(1.0)  # 0.1 * 10
    assert concentration_threshold_uM(LEGACY, C0_uM=4.0) == pytest.approx(0.4)
    assert concentration_threshold_uM(LEGACY, C0_uM=4.0, relative_fraction=0.25) == pytest.approx(1.0)
    bad = copy.deepcopy(CONFIG)
    bad["evaluation"]["penetration_threshold"]["mode"] = "nope"
    with pytest.raises(ValueError):
        concentration_threshold_uM(bad, 1.0)


def test_exposure_profile_peak_vs_final():
    C = np.array([[0.0, 0.0, 1.0], [0.5, 2.0, 0.5], [0.1, 0.2, 0.05]])  # rise then decay
    np.testing.assert_allclose(exposure_profile(C, CONFIG), [0.5, 2.0, 1.0])  # peak (default)
    np.testing.assert_allclose(exposure_profile(C, LEGACY), [0.1, 0.2, 0.05])  # final (legacy)
    assert subtherapeutic_threshold_uM(CONFIG) == pytest.approx(0.1)
