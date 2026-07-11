"""Figure + animation generation. [Phase 11 — not yet implemented]

Will provide: 2D concentration heatmaps, penetration-depth-vs-time plots,
viability/cytotoxicity maps, the eta(d_NP,C0) treatment-effectiveness
surface, heterogeneous-vs-homogeneous profile comparison, the ablation
residual comparison figure, PDE-residual histograms, PINN-vs-FDM overlays,
and time-series animations. Shared plotting primitives used by both
scripts/make_figures.py (static PNG export) and app/server.py (interactive
Plotly figures in the dashboard).
"""

from __future__ import annotations


def plot_concentration_heatmap(C_rt, r, t, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def plot_penetration_depth(depth_t, t, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def plot_viability_map(V_rt, r, t, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def plot_effectiveness_surface(eta_grid, d_NP_grid, C0_grid, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def plot_homogeneous_comparison(hetero_result: dict, homo_result: dict, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def plot_ablation_comparison(ablation_result: dict, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")


def animate_concentration(C_rt, r, t, config: dict, save_path: str | None = None):
    raise NotImplementedError("Phase 11")
