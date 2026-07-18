"""Phase 3 tests: LHS sampling, point sampling, normalization, and the
end-to-end small-dataset pipeline.

Uses a small `fdm`/`dataset` override on top of the default config (few
simulations, coarse grid, few points per sim) so the suite runs in well
under a second, per the same reasoning as tests/test_fdm_solver.py.
"""

import copy
import json

import numpy as np
import pytest

from src.config import load_config
from src.data_pipeline import (
    PARAM_ORDER,
    build_dataset,
    build_split_tensors,
    compute_normalization_stats,
    generate_dataset,
    latin_hypercube_sample,
    sample_training_points,
)

CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 25
FAST_CONFIG["fdm"]["N_t_initial"] = 15
FAST_CONFIG["dataset"]["split"] = {"train": 4, "val": 2, "test": 2}
FAST_CONFIG["dataset"]["points_per_sim"] = {
    "data": 40,
    "collocation": 60,
    "bc_surface": 10,
    "bc_center": 10,
    "ic": 10,
}


def test_latin_hypercube_sample_shape_and_bounds():
    sample = latin_hypercube_sample(50, FAST_CONFIG, seed=0)
    assert sample.shape == (50, 5)
    ranges = FAST_CONFIG["dataset"]["parameter_ranges"]
    for i, p in enumerate(PARAM_ORDER):
        lo, hi = ranges[p]
        assert np.all(sample[:, i] >= lo)
        assert np.all(sample[:, i] <= hi)


def test_latin_hypercube_sample_reproducible_with_seed():
    a = latin_hypercube_sample(20, FAST_CONFIG, seed=42)
    b = latin_hypercube_sample(20, FAST_CONFIG, seed=42)
    assert np.array_equal(a, b)


def test_generate_dataset_returns_solved_sims():
    sims = generate_dataset(3, FAST_CONFIG, seed=1)
    assert len(sims) == 3
    N_r = FAST_CONFIG["fdm"]["N_r"]
    N_t = FAST_CONFIG["fdm"]["N_t_initial"]
    for sim in sims:
        assert sim["r"].shape == (N_r,)
        assert sim["t"].shape == (N_t,)
        assert sim["C"].shape == (N_t, N_r)
        for p in PARAM_ORDER:
            assert sim[p] > 0


def test_generate_dataset_parallel_matches_sequential():
    sims_seq = generate_dataset(6, FAST_CONFIG, seed=8, n_jobs=1)
    sims_par = generate_dataset(6, FAST_CONFIG, seed=8, n_jobs=2)
    assert len(sims_par) == len(sims_seq)
    for a, b in zip(sims_seq, sims_par):
        assert a["sim_id"] == b["sim_id"]
        assert a["R_um"] == pytest.approx(b["R_um"])
        assert a["d_NP_nm"] == pytest.approx(b["d_NP_nm"])
        np.testing.assert_allclose(a["C"], b["C"])


def test_sample_training_points_counts_and_bounds():
    sims = generate_dataset(1, FAST_CONFIG, seed=2)
    sim = sims[0]
    rng = np.random.default_rng(0)
    pts = sample_training_points(sim, FAST_CONFIG, rng=rng)

    pcfg = FAST_CONFIG["dataset"]["points_per_sim"]
    assert len(pts["data"]["r"]) == pcfg["data"]
    assert len(pts["collocation"]["r"]) == pcfg["collocation"]
    assert len(pts["bc_surface"]["r"]) == pcfg["bc_surface"]
    assert len(pts["bc_center"]["r"]) == pcfg["bc_center"]
    assert len(pts["ic"]["r"]) == pcfg["ic"]

    # bc_surface sits exactly on the tumor surface with the surface concentration.
    assert np.all(pts["bc_surface"]["r"] == sim["R_um"])
    assert np.all(pts["bc_surface"]["C"] == sim["C0_uM"])

    # bc_center sits exactly at the innermost grid point (Neumann location).
    assert np.all(pts["bc_center"]["r"] == sim["r"][0])

    # ic points are all at t=0 with C=0.
    assert np.all(pts["ic"]["t"] == 0.0)
    assert np.all(pts["ic"]["C"] == 0.0)

    # collocation points stay within the physical domain.
    assert np.all(pts["collocation"]["r"] >= sim["r"][0])
    assert np.all(pts["collocation"]["r"] <= sim["R_um"])
    assert np.all(pts["collocation"]["t"] <= sim["t_max_hr"])


def test_compute_normalization_stats_matches_actual_range():
    sims = generate_dataset(6, FAST_CONFIG, seed=3)
    stats = compute_normalization_stats(sims, FAST_CONFIG)
    for p in PARAM_ORDER:
        values = np.array([s[p] for s in sims])
        assert stats[p]["min"] == pytest.approx(values.min())
        assert stats[p]["max"] == pytest.approx(values.max())


def test_build_split_tensors_columns_and_normalization_ranges():
    sims = generate_dataset(3, FAST_CONFIG, seed=4)
    stats = compute_normalization_stats(sims, FAST_CONFIG)
    rng = np.random.default_rng(0)
    tensors = build_split_tensors(sims, stats, FAST_CONFIG, rng=rng)

    n_expected = 3 * FAST_CONFIG["dataset"]["points_per_sim"]["data"]
    assert tensors["data_X"].shape == (n_expected, 7)
    assert tensors["data_y"].shape == (n_expected, 1)

    # r_norm and t_norm columns must lie in [0, 1] for every category.
    for cat in ("data", "collocation", "bc_surface", "bc_center", "ic"):
        X = tensors[f"{cat}_X"]
        assert np.all(X[:, 0] >= 0.0) and np.all(X[:, 0] <= 1.0 + 1e-6)
        assert np.all(X[:, 1] >= 0.0) and np.all(X[:, 1] <= 1.0 + 1e-6)
        # Parameter-conditioning columns are also normalized to [0, 1].
        assert np.all(X[:, 2:] >= -1e-6) and np.all(X[:, 2:] <= 1.0 + 1e-6)

    # Hard-IC sanity: bc_surface targets are always C_norm=1, ic targets are always 0.
    assert tensors["bc_surface_y"] == pytest.approx(1.0)
    assert tensors["ic_y"] == pytest.approx(0.0)

    # bc_surface/bc_center sit exactly at r_norm=1 / r_norm=0 respectively.
    assert tensors["bc_surface_X"][:, 0] == pytest.approx(1.0)
    assert np.all(tensors["bc_center_X"][:, 0] < 1e-3)

    # ic points sit exactly at t_norm=0.
    assert tensors["ic_X"][:, 1] == pytest.approx(0.0)


def test_build_dataset_end_to_end_small(tmp_path):
    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])
    # resolve_path joins REPO_ROOT with paths.processed; point REPO_ROOT at a
    # throwaway tmp_path for the duration of this test so nothing touches
    # the real repo's data/ directory.
    import src.config as cfg_module

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["processed"] = "processed"

        result = build_dataset(config, seed=7, save=True)

        assert set(result["sims"].keys()) == {"train", "val", "test"}
        split_cfg = config["dataset"]["split"]
        assert len(result["sims"]["train"]) == split_cfg["train"]
        assert len(result["sims"]["val"]) == split_cfg["val"]
        assert len(result["sims"]["test"]) == split_cfg["test"]

        processed_dir = tmp_path / "processed"
        for split_name in ("train", "val", "test"):
            assert (processed_dir / f"{split_name}.npz").exists()
        stats_path = processed_dir / "normalization_stats.json"
        assert stats_path.exists()

        with open(stats_path, encoding="utf-8") as f:
            stats_on_disk = json.load(f)
        assert set(stats_on_disk.keys()) == set(PARAM_ORDER)

        loaded = np.load(processed_dir / "train.npz")
        assert "data_X" in loaded
        assert loaded["data_X"].shape[1] == 7

        sim_params_path = processed_dir / "sim_params.json"
        assert sim_params_path.exists()
        with open(sim_params_path, encoding="utf-8") as f:
            sim_params = json.load(f)
        assert set(sim_params.keys()) == {"train", "val", "test"}
        assert len(sim_params["test"]) == split_cfg["test"]
        for entry in sim_params["test"]:
            assert set(entry.keys()) == {"sim_id"} | set(PARAM_ORDER)
        test_sim_ids = {e["sim_id"] for e in sim_params["test"]}
        assert test_sim_ids == {sim["sim_id"] for sim in result["sims"]["test"]}
    finally:
        cfg_module.REPO_ROOT = original_root
