"""Phase 7 tests: Hill equation / viability / cytotoxicity / penetration
depth, plus the loaded-checkpoint prediction path and the H4 hypothesis
check (rim viability < 20%, core viability > 60% at t=72hr)."""

import copy

import numpy as np
import pytest

from src.biology import (
    compute_biological_response,
    cytotoxicity_map,
    evaluate_h4_hypothesis,
    hill_death_rate,
    logistic_growth,
    penetration_depth,
    predict_concentration_field,
    survival_fraction,
    viability_map,
)
from src.config import load_config
from src.data_pipeline import build_dataset
from src.fdm_solver import solve_fdm
from src.microenvironment import radial_grid
from src.model import BIOPINN
from src.train import train

CONFIG = load_config()


def test_hill_death_rate_matches_formula_and_bounds():
    C = np.array([0.0, 0.5, 1.0, 2.0, 100.0])
    delta = hill_death_rate(C, CONFIG)

    bio = CONFIG["biology"]
    delta_max = bio["delta_max_per_day"] / 24.0
    IC50, n = bio["IC50_uM"], bio["n_hill"]
    expected = delta_max * C**n / (IC50**n + C**n)

    np.testing.assert_allclose(delta, expected)
    assert delta[0] == 0.0  # no drug, no kill
    assert delta[-1] == pytest.approx(delta_max, rel=1e-2)  # saturates at delta_max
    assert np.all(np.diff(delta) >= 0)  # monotonically increasing in C


def test_hill_death_rate_clips_negative_concentration():
    delta = hill_death_rate(np.array([-5.0]), CONFIG)
    assert delta[0] == 0.0


def test_survival_fraction_no_drug_stays_at_one():
    t = np.linspace(0, 72, 50)
    C_rt = np.zeros((50, 10))
    f_alive = survival_fraction(C_rt, t, CONFIG)
    np.testing.assert_allclose(f_alive, 1.0)


def test_survival_fraction_starts_at_one_and_decreases_with_constant_drug():
    t = np.linspace(0, 72, 50)
    C_rt = np.full((50, 10), 5.0)  # constant, well above IC50
    f_alive = survival_fraction(C_rt, t, CONFIG)

    assert f_alive[0, 0] == pytest.approx(1.0)
    assert np.all(f_alive >= 0.0) and np.all(f_alive <= 1.0)
    # Monotonically non-increasing over time at each radius.
    assert np.all(np.diff(f_alive, axis=0) <= 1e-12)
    assert f_alive[-1, 0] < f_alive[0, 0]


def test_viability_and_cytotoxicity_are_consistent_with_survival():
    t = np.linspace(0, 72, 30)
    rng = np.random.default_rng(0)
    C_rt = rng.uniform(0, 10, size=(30, 15))

    f_alive = survival_fraction(C_rt, t, CONFIG)
    viability = viability_map(C_rt, t, CONFIG)
    cytotox = cytotoxicity_map(C_rt, t, CONFIG)

    np.testing.assert_allclose(viability, f_alive * 100.0)
    np.testing.assert_allclose(cytotox, 1.0 - f_alive)
    np.testing.assert_allclose(viability / 100.0 + cytotox, 1.0)


def test_penetration_depth_basic_cases():
    r = np.linspace(0, 400, 5)  # [0, 100, 200, 300, 400]
    config = CONFIG

    # Nothing exceeds threshold -> zero penetration.
    cyt_none = np.zeros((1, 5))
    assert penetration_depth(cyt_none, r, config)[0] == 0.0

    # Everything exceeds threshold -> full radius penetration (innermost point qualifies).
    cyt_all = np.full((1, 5), 0.9)
    assert penetration_depth(cyt_all, r, config)[0] == pytest.approx(400.0)

    # Only the outer half (indices 3, 4 -> r=300, 400) exceeds threshold:
    # depth = R - r[innermost qualifying] = 400 - 300 = 100.
    cyt_partial = np.array([[0.1, 0.2, 0.3, 0.6, 0.9]])
    assert penetration_depth(cyt_partial, r, config)[0] == pytest.approx(100.0)


def test_logistic_growth_matches_formula_and_zero_at_boundaries():
    k_p, N_max = 0.05, 1.0
    N = np.array([0.0, 0.5, 1.0])
    growth = logistic_growth(N, k_p, N_max)
    np.testing.assert_allclose(growth, k_p * N * (1 - N / N_max))
    assert growth[0] == 0.0  # no cells, no growth
    assert growth[-1] == pytest.approx(0.0, abs=1e-12)  # at carrying capacity, growth stalls
    assert growth[1] > 0.0


def test_predict_concentration_field_shape_and_hard_ic():
    model = BIOPINN(CONFIG)
    r = radial_grid(R_um=300.0, N_r=20, r_min_um=0.001)
    t = np.linspace(0.0, 72.0, 10)
    sim_params = {"R_um": 300.0, "d_NP_nm": 100.0, "C0_uM": 10.0, "k_d_per_hr": 0.01, "t_max_hr": 72.0}
    norm_stats = {
        "R_um": {"min": 100.0, "max": 500.0},
        "d_NP_nm": {"min": 10.0, "max": 200.0},
        "C0_uM": {"min": 0.1, "max": 20.0},
        "k_d_per_hr": {"min": 0.005, "max": 0.05},
        "t_max_hr": {"min": 24.0, "max": 72.0},
    }

    C_rt = predict_concentration_field(model, r, t, sim_params, norm_stats)
    assert C_rt.shape == (10, 20)
    assert np.all(np.isfinite(C_rt))
    assert np.all(C_rt >= 0.0)
    # Hard-IC transform: C(r, t=0) = 0 exactly, for every untrained model too.
    np.testing.assert_allclose(C_rt[0, :], 0.0, atol=1e-6)


def test_compute_biological_response_end_to_end_shapes():
    model = BIOPINN(CONFIG)
    r = radial_grid(R_um=400.0, N_r=25, r_min_um=0.001)
    t = np.linspace(0.0, 72.0, 15)
    sim_params = {"R_um": 400.0, "d_NP_nm": 100.0, "C0_uM": 10.0, "k_d_per_hr": 0.01, "t_max_hr": 72.0}
    norm_stats = {
        "R_um": {"min": 100.0, "max": 500.0},
        "d_NP_nm": {"min": 10.0, "max": 200.0},
        "C0_uM": {"min": 0.1, "max": 20.0},
        "k_d_per_hr": {"min": 0.005, "max": 0.05},
        "t_max_hr": {"min": 24.0, "max": 72.0},
    }

    result = compute_biological_response(model, r, t, sim_params, norm_stats, CONFIG)
    for key in ("C", "viability", "cytotoxicity"):
        assert result[key].shape == (15, 25)
    assert result["penetration_depth"].shape == (15,)
    assert np.all(np.isfinite(result["viability"]))
    assert np.all((result["viability"] >= 0) & (result["viability"] <= 100))


def test_evaluate_h4_hypothesis_detects_pass_and_fail():
    # Large tumor so a genuine necrotic core forms (see tests/test_microenvironment.py).
    r = radial_grid(R_um=500.0, N_r=200, r_min_um=0.001)
    t = np.array([0.0, 36.0, 72.0])

    from src.microenvironment import assign_zones, oxygen_gradient

    oxygen = oxygen_gradient(r, CONFIG, R=500.0)
    zones = assign_zones(r, oxygen, CONFIG)

    # Craft a viability field that is exactly low in the rim zone and exactly
    # high everywhere else -- a clean, unambiguous H4-passing case regardless
    # of exactly where the zone boundaries fall.
    viability_row = np.where(zones == "proliferating_rim", 5.0, 90.0)
    viability_pass = np.tile(viability_row, (len(t), 1))
    result_pass = evaluate_h4_hypothesis(viability_pass, r, t, CONFIG, R_um=500.0)
    assert result_pass["t_hr"] == pytest.approx(72.0)
    assert result_pass["rim_pass"] is True
    assert result_pass["core_pass"] is True
    assert result_pass["overall_pass"] is True
    assert result_pass["rim_viability_pct"] < CONFIG["biology"]["h4_rim_viability_max"]
    assert result_pass["core_viability_pct"] > CONFIG["biology"]["h4_core_viability_min"]

    # Flip it: uniformly high viability everywhere -> rim criterion fails.
    viability_fail = np.full((len(t), len(r)), 90.0)
    result_fail = evaluate_h4_hypothesis(viability_fail, r, t, CONFIG, R_um=500.0)
    assert result_fail["rim_pass"] is False
    assert result_fail["overall_pass"] is False


def test_h4_direction_holds_for_a_well_chosen_fdm_ground_truth_case():
    # A large, slowly-diffusing-nanoparticle, low-dose case: the rim (exposed
    # to the boundary concentration from t=0) should be more heavily killed
    # by 72hr than the deep core, which limited diffusion hasn't fully
    # reached. This validates the biology math's *direction* against real
    # physics (FDM), independent of any trained PINN's accuracy.
    #
    # Note: a parameter sweep across the full C0/k_d range did not find a
    # combination clearing *both* H4 numeric thresholds (rim<20%, core>60%)
    # simultaneously at exactly t=72hr for this model's zone-diffusivity
    # factors (necrotic core has the *highest* D_eff of the three zones per
    # the guide's own spec, so once the rim barrier is crossed the interior
    # tends to re-equilibrate quickly, coupling rim and core viability more
    # tightly than H4's idealized two-tier picture assumes). That's an
    # empirical finding for Phase 8's full-test-set evaluation to report
    # pass/fail on, not something to force here -- this test only checks the
    # directional relationship the biology math should always produce.
    config = CONFIG
    fdm_result = solve_fdm(
        R_um=500.0, d_NP_nm=200.0, C0_uM=0.65, k_d_per_hr=0.05, t_max_hr=72.0, config=config
    )
    r, t, C_rt = fdm_result["r"], fdm_result["t"], fdm_result["C"]
    viability = viability_map(C_rt, t, config)

    h4 = evaluate_h4_hypothesis(viability, r, t, config, R_um=500.0)
    assert h4["rim_viability_pct"] < h4["core_viability_pct"]


def test_biology_pipeline_from_a_trained_checkpoint(tmp_path):
    # End-to-end per Phase 7's own verification bar: produce maps "from a
    # loaded checkpoint" (not raw FDM/synthetic C(r,t)).
    fast_config = copy.deepcopy(CONFIG)
    fast_config["fdm"]["N_r"] = 25
    fast_config["fdm"]["N_t_initial"] = 15
    fast_config["dataset"]["split"] = {"train": 4, "val": 2, "test": 2}
    fast_config["dataset"]["points_per_sim"] = {
        "data": 40,
        "collocation": 60,
        "bc_surface": 10,
        "bc_center": 10,
        "ic": 10,
    }
    fast_config["training"]["adam"]["iters"] = 20
    fast_config["training"]["lbfgs"]["iters"] = 5

    dataset = build_dataset(fast_config, seed=13, save=False)
    train_result = train(fast_config, dataset, save=False)
    model = train_result["model"]
    norm_stats = dataset["stats"]

    r = radial_grid(R_um=400.0, N_r=30, r_min_um=0.001)
    t = np.linspace(0.0, 72.0, 12)
    sim_params = {"R_um": 400.0, "d_NP_nm": 100.0, "C0_uM": 10.0, "k_d_per_hr": 0.01, "t_max_hr": 72.0}

    result = compute_biological_response(model, r, t, sim_params, norm_stats, fast_config)
    assert result["C"].shape == (12, 30)
    assert np.all(np.isfinite(result["viability"]))
    assert np.all(np.isfinite(result["cytotoxicity"]))
    assert np.all(np.isfinite(result["penetration_depth"]))

    from src.microenvironment import assign_zones, oxygen_gradient

    oxygen = oxygen_gradient(r, fast_config, R=400.0)
    zones = assign_zones(r, oxygen, fast_config)
    assert set(np.unique(zones)) <= {"proliferating_rim", "quiescent_zone", "necrotic_core"}
