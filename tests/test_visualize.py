"""Phase 11 tests: figure + animation generation (src/visualize.py).

Each plotting function is checked two ways: with save_path it must write a
valid, non-empty file; without save_path it must return a matplotlib Figure
(or, for the animation, a FuncAnimation) for interactive/further use.
"""

from __future__ import annotations

import matplotlib.figure
import matplotlib.animation
import numpy as np
import pytest

from src.config import load_config
from src.visualize import (
    animate_concentration,
    plot_ablation_comparison,
    plot_concentration_heatmap,
    plot_cytotoxicity_map,
    plot_effectiveness_surface,
    plot_homogeneous_comparison,
    plot_pde_residual_histogram,
    plot_penetration_depth,
    plot_pinn_vs_fdm_overlay,
    plot_viability_map,
)

CONFIG = load_config()

N_R, N_T = 15, 10
R = np.linspace(50.0, 400.0, N_R)
T = np.linspace(0.0, 72.0, N_T)


def _synthetic_field(peak: float = 10.0) -> np.ndarray:
    rr, tt = np.meshgrid(R, T)
    return peak * (tt / T[-1]) * (1.0 - rr / (R[-1] * 1.2))


def _assert_saved_file(path) -> None:
    assert path.exists()
    assert path.stat().st_size > 0


def test_plot_concentration_heatmap(tmp_path):
    C_rt = _synthetic_field()
    fig = plot_concentration_heatmap(C_rt, R, T, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "conc.png"
    result = plot_concentration_heatmap(C_rt, R, T, CONFIG, save_path=str(save_path))
    assert result == str(save_path)
    _assert_saved_file(save_path)


def test_plot_penetration_depth(tmp_path):
    depth_t = np.linspace(0.0, 150.0, N_T)
    fig = plot_penetration_depth(depth_t, T, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "depth.png"
    plot_penetration_depth(depth_t, T, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_viability_map(tmp_path):
    V_rt = np.clip(100.0 - _synthetic_field(peak=120.0), 0.0, 100.0)
    fig = plot_viability_map(V_rt, R, T, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "viability.png"
    plot_viability_map(V_rt, R, T, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_cytotoxicity_map(tmp_path):
    Cyt_rt = np.clip(_synthetic_field(peak=1.0), 0.0, 1.0)
    fig = plot_cytotoxicity_map(Cyt_rt, R, T, CONFIG, R_um=R[-1])
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "cytotox.png"
    plot_cytotoxicity_map(Cyt_rt, R, T, CONFIG, R_um=R[-1], save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_effectiveness_surface(tmp_path):
    d_NP_grid = np.linspace(10.0, 200.0, 5)
    C0_grid = np.linspace(0.1, 20.0, 4)
    eta_grid = np.outer(np.linspace(0.1, 1.0, 5), np.linspace(0.2, 0.9, 4))

    fig = plot_effectiveness_surface(eta_grid, d_NP_grid, C0_grid, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "surface.png"
    plot_effectiveness_surface(eta_grid, d_NP_grid, C0_grid, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_homogeneous_comparison(tmp_path):
    hetero = {"r": R, "t": T, "C": _synthetic_field(peak=8.0)}
    homo = {"r": R, "t": T, "C": _synthetic_field(peak=5.0)}

    fig = plot_homogeneous_comparison(hetero, homo, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "h3.png"
    plot_homogeneous_comparison(hetero, homo, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_ablation_comparison_with_and_without_residual_arrays(tmp_path):
    rng = np.random.default_rng(0)
    result_with_residuals = {
        "biopinn": {"mean_abs_residual": 1e-4, "max_abs_residual": 5e-3},
        "baseline": {"mean_abs_residual": 5e-2, "max_abs_residual": 1.0},
        "residuals_biopinn": np.abs(rng.normal(1e-4, 1e-5, size=200)),
        "residuals_baseline": np.abs(rng.normal(5e-2, 1e-2, size=200)),
    }
    fig = plot_ablation_comparison(result_with_residuals, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "ablation_full.png"
    plot_ablation_comparison(result_with_residuals, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)

    result_without_residuals = {
        "residual_comparison": {
            "biopinn": {"mean_abs_residual": 1e-4, "max_abs_residual": 5e-3},
            "baseline": {"mean_abs_residual": 5e-2, "max_abs_residual": 1.0},
        }
    }
    save_path_2 = tmp_path / "ablation_bars_only.png"
    plot_ablation_comparison(result_without_residuals, CONFIG, save_path=str(save_path_2))
    _assert_saved_file(save_path_2)


def test_plot_pde_residual_histogram(tmp_path):
    residuals = np.abs(np.random.default_rng(1).normal(1e-3, 5e-4, size=500))
    fig = plot_pde_residual_histogram(residuals, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "residual_hist.png"
    plot_pde_residual_histogram(residuals, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_plot_pinn_vs_fdm_overlay(tmp_path):
    sim = {
        "r": R,
        "t": T,
        "C": _synthetic_field(peak=10.0),
        "sim_id": 7,
        "R_um": R[-1],
        "d_NP_nm": 100.0,
        "C0_uM": 10.0,
        "t_max_hr": T[-1],
    }
    C_pred = sim["C"] * 0.95

    fig = plot_pinn_vs_fdm_overlay(sim, C_pred, CONFIG)
    assert isinstance(fig, matplotlib.figure.Figure)

    save_path = tmp_path / "overlay.png"
    plot_pinn_vs_fdm_overlay(sim, C_pred, CONFIG, save_path=str(save_path))
    _assert_saved_file(save_path)


def test_animate_concentration_returns_funcanimation_without_save_path():
    C_rt = _synthetic_field()
    anim = animate_concentration(C_rt, R, T, CONFIG, n_frames=5)
    assert isinstance(anim, matplotlib.animation.FuncAnimation)


def test_animate_concentration_saves_gif(tmp_path):
    C_rt = _synthetic_field()
    save_path = tmp_path / "concentration.gif"
    result = animate_concentration(C_rt, R, T, CONFIG, save_path=str(save_path), fps=5, n_frames=5)
    assert result == str(save_path)
    _assert_saved_file(save_path)
