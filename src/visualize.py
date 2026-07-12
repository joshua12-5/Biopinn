"""Figure + animation generation.

2D concentration heatmaps, penetration-depth-vs-time plots,
viability/cytotoxicity maps (with a three-zone overlay), the
eta(d_NP,C0) treatment-effectiveness surface, the heterogeneous-vs-
homogeneous profile comparison (H3), the ablation residual comparison (H5),
PDE-residual histograms, PINN-vs-FDM overlays, and time-series concentration
animations. Shared plotting primitives used by both scripts/make_figures.py
(static PNG export) and scripts/run_evaluation.py / scripts/run_ablation.py,
so there is exactly one implementation of each figure.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe default for scripts/servers; callers
# that need an interactive backend can select one before importing this module.

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _save_or_return(fig: plt.Figure, save_path: str | None):
    """Save + close the figure if a path is given, else return it (e.g. for
    interactive use or further customization by the caller)."""
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return save_path
    return fig


def plot_concentration_heatmap(C_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict, save_path: str | None = None):
    """2D concentration heatmap (r vs t) + radial profile snapshots."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].pcolormesh(r, t, C_rt, cmap="hot_r", shading="auto")
    plt.colorbar(im, ax=axes[0], label="Concentration (uM)")
    axes[0].set_xlabel("Radial distance (um)")
    axes[0].set_ylabel("Time (hr)")
    axes[0].set_title("Spatiotemporal concentration field")

    snapshot_hours = [h for h in (6, 24, 48, 72) if h <= t[-1]] or [t[-1]]
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(snapshot_hours)))
    for hours, color in zip(snapshot_hours, colors):
        idx = int(np.argmin(np.abs(t - hours)))
        axes[1].plot(r, C_rt[idx, :], color=color, linewidth=2, label=f"t={t[idx]:.0f}hr")
    axes[1].set_xlabel("Radial distance (um)")
    axes[1].set_ylabel("Concentration (uM)")
    axes[1].set_title("Radial concentration profiles")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Drug concentration C(r,t)")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_penetration_depth(depth_t: np.ndarray, t: np.ndarray, config: dict, save_path: str | None = None):
    """Penetration depth vs. time."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t, depth_t, "b-", linewidth=2.5)
    ax.fill_between(t, 0, depth_t, alpha=0.15, color="blue")
    ax.set_xlabel("Time (hr)")
    ax.set_ylabel("Penetration depth (um)")
    ax.set_title("Drug penetration depth vs. time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_viability_map(V_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict, save_path: str | None = None):
    """Viability heatmap + final-time viable/dead profile."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].pcolormesh(r, t, V_rt, cmap="RdYlGn", vmin=0, vmax=100, shading="auto")
    plt.colorbar(im, ax=axes[0], label="Viability (%)")
    axes[0].set_xlabel("Radial distance (um)")
    axes[0].set_ylabel("Time (hr)")
    axes[0].set_title("Cell viability V(r,t)")

    final_viab = V_rt[-1, :]
    axes[1].fill_between(r, 0, final_viab, where=final_viab >= 50, color="green", alpha=0.6, label="Viable (>=50%)")
    axes[1].fill_between(r, 0, final_viab, where=final_viab < 50, color="red", alpha=0.6, label="Dead (<50%)")
    axes[1].set_xlabel("Radial distance (um)")
    axes[1].set_ylabel("Viability (%)")
    axes[1].set_title(f"Final viability profile (t={t[-1]:.0f}hr)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(0, 105)

    fig.suptitle("Tumor cell viability")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_cytotoxicity_map(
    Cyt_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict, R_um: float | None = None, save_path: str | None = None
):
    """Cytotoxicity heatmap + zone-resolved final-time profile (three-zone overlay)."""
    from src.microenvironment import assign_zones, oxygen_gradient

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im = axes[0].pcolormesh(r, t, Cyt_rt * 100, cmap="Reds", vmin=0, vmax=100, shading="auto")
    plt.colorbar(im, ax=axes[0], label="Cytotoxicity (%)")
    axes[0].set_xlabel("Radial distance (um)")
    axes[0].set_ylabel("Time (hr)")
    axes[0].set_title("Cytotoxicity Cyt(r,t)")

    oxygen = oxygen_gradient(r, config, R=R_um)
    zones = assign_zones(r, oxygen, config)
    zone_colors = {"proliferating_rim": "tab:red", "quiescent_zone": "tab:orange", "necrotic_core": "tab:blue"}
    final_cytotox = Cyt_rt[-1, :] * 100
    for zone_name, color in zone_colors.items():
        mask = zones == zone_name
        if mask.any():
            axes[1].scatter(r[mask], final_cytotox[mask], s=8, color=color, label=zone_name.replace("_", " "))
    threshold = config["biology"]["penetration_cytotoxicity_threshold"] * 100
    axes[1].axhline(threshold, color="gray", linestyle="--", label="penetration threshold")
    axes[1].set_xlabel("Radial distance (um)")
    axes[1].set_ylabel("Cytotoxicity (%)")
    axes[1].set_title(f"Zone-resolved cytotoxicity (t={t[-1]:.0f}hr)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.suptitle("Cytotoxicity + three-zone overlay")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_effectiveness_surface(
    eta_grid: np.ndarray, d_NP_grid: np.ndarray, C0_grid: np.ndarray, config: dict, save_path: str | None = None
):
    """Treatment-effectiveness surface eta(d_NP, C0), with the optimum marked."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(C0_grid, d_NP_grid, eta_grid, cmap="viridis", shading="auto")
    plt.colorbar(im, ax=ax, label="eta (volume-averaged kill fraction)")

    best_i, best_j = np.unravel_index(np.argmax(eta_grid), eta_grid.shape)
    ax.scatter([C0_grid[best_j]], [d_NP_grid[best_i]], color="red", marker="*", s=200, label="optimum")
    ax.set_xlabel("C0 (uM)")
    ax.set_ylabel("d_NP (nm)")
    ax.set_title("Treatment-effectiveness surface eta(d_NP, C0)")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_homogeneous_comparison(hetero_result: dict, homo_result: dict, config: dict, save_path: str | None = None):
    """Heterogeneous-vs-homogeneous diffusion comparison (H3): final radial
    profile overlay + a concentration-difference heatmap over time.

    hetero_result/homo_result: dicts with 'r', 't', 'C' (e.g. from
    src.fdm_solver.solve_fdm; both must share the same r/t grid, as they do
    when produced by the same solve_fdm call with only D_eff_override changed).
    """
    r, t = hetero_result["r"], hetero_result["t"]
    C_h, C_g = hetero_result["C"], homo_result["C"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(r, C_h[-1, :], "b-", linewidth=2, label="heterogeneous (3-zone)")
    axes[0].plot(r, C_g[-1, :], "r--", linewidth=2, label="homogeneous (mean D_eff)")
    threshold = 0.1 * config["biology"]["IC50_uM"]
    axes[0].axhline(threshold, color="gray", linestyle=":", label="sub-therapeutic threshold")
    axes[0].set_xlabel("Radial distance (um)")
    axes[0].set_ylabel("Concentration (uM)")
    axes[0].set_title(f"Final concentration profile (t={t[-1]:.0f}hr)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    diff = C_h - C_g
    vmax = np.abs(diff).max() or 1.0
    im = axes[1].pcolormesh(r, t, diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    plt.colorbar(im, ax=axes[1], label="heterogeneous - homogeneous (uM)")
    axes[1].set_xlabel("Radial distance (um)")
    axes[1].set_ylabel("Time (hr)")
    axes[1].set_title("Concentration difference")

    fig.suptitle("Heterogeneous vs. homogeneous diffusion (H3)")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_ablation_comparison(ablation_result: dict, config: dict, save_path: str | None = None):
    """Ablation residual comparison (H5): mean-residual bar chart, and a
    distribution histogram if the raw per-point residual arrays are present
    (src.ablation.compare_residuals's residuals_biopinn/residuals_baseline;
    src.ablation.ablation_report strips those before JSON export, so pass
    compare_residuals's own return value here for the full figure)."""
    rc = ablation_result.get("residual_comparison", ablation_result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["BIOPINN", "baseline (w_phys=0)"]
    means = [rc["biopinn"]["mean_abs_residual"], rc["baseline"]["mean_abs_residual"]]
    improvement = means[1] / means[0] if means[0] else float("inf")
    axes[0].bar(labels, means, color=["tab:blue", "tab:red"])
    axes[0].set_ylabel("Mean |PDE residual|")
    axes[0].set_yscale("log")
    axes[0].set_title(f"PDE residual comparison ({improvement:.1f}x improvement)")

    if "residuals_biopinn" in rc and "residuals_baseline" in rc:
        max_val = max(rc["residuals_biopinn"].max(), rc["residuals_baseline"].max())
        bins = np.linspace(0, max_val if max_val > 0 else 1.0, 50)
        axes[1].hist(rc["residuals_biopinn"], bins=bins, alpha=0.6, label="BIOPINN", color="tab:blue")
        axes[1].hist(rc["residuals_baseline"], bins=bins, alpha=0.6, label="baseline", color="tab:red")
        axes[1].set_xlabel("|PDE residual|")
        axes[1].set_ylabel("count")
        axes[1].set_title("Residual distributions")
        axes[1].legend(fontsize=8)
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "residual arrays not provided", ha="center", va="center")

    fig.suptitle("Ablation study: physics-informed vs. unconstrained")
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_pde_residual_histogram(residuals: np.ndarray, config: dict, save_path: str | None = None):
    """Distribution of |PDE residual| over a batch of collocation points."""
    threshold = config["evaluation"]["thresholds"]["mean_pde_residual"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(residuals, bins=60, color="steelblue", alpha=0.85)
    ax.axvline(residuals.mean(), color="black", linestyle="-", label=f"mean = {residuals.mean():.2e}")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.0e}")
    ax.set_xlabel("|PDE residual|")
    ax.set_ylabel("count")
    ax.set_title("PDE-residual distribution")
    ax.legend()
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def plot_pinn_vs_fdm_overlay(sim: dict, C_pred: np.ndarray, config: dict, save_path: str | None = None):
    """Radial concentration snapshots comparing a PINN prediction against
    its FDM reference, for one simulation (sim: dict with 'r','t','C',
    'sim_id','R_um','d_NP_nm','C0_uM')."""
    r, t, C_true = sim["r"], sim["t"], sim["C"]
    t_max_hr = sim.get("t_max_hr", t[-1])

    snapshot_hours = [h for h in (6, 24, 48, 72) if h <= t_max_hr] or [t_max_hr]
    fig, axes = plt.subplots(1, len(snapshot_hours), figsize=(4.5 * len(snapshot_hours), 4.5), squeeze=False)
    for ax, hours in zip(axes[0], snapshot_hours):
        idx = int(np.argmin(np.abs(t - hours)))
        ax.plot(r, C_true[idx, :], "k-", linewidth=2, label="FDM reference")
        ax.plot(r, C_pred[idx, :], "r--", linewidth=2, label="PINN prediction")
        ax.set_xlabel("radius (um)")
        ax.set_ylabel("concentration (uM)")
        ax.set_title(f"t={t[idx]:.0f}hr")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(
        f"PINN vs. FDM -- sim {sim.get('sim_id', '?')} "
        f"(R={sim['R_um']:.0f}um, d_NP={sim['d_NP_nm']:.0f}nm, C0={sim['C0_uM']:.1f}uM)"
    )
    fig.tight_layout()
    return _save_or_return(fig, save_path)


def animate_concentration(
    C_rt: np.ndarray, r: np.ndarray, t: np.ndarray, config: dict, save_path: str | None = None, fps: int = 10, n_frames: int = 100
):
    """Animated GIF of the radial concentration profile evolving over time."""
    fig, ax = plt.subplots(figsize=(7, 5))
    (line,) = ax.plot(r, C_rt[0, :], "b-", linewidth=2)
    ax.set_xlim(r[0], r[-1])
    ax.set_ylim(0, C_rt.max() * 1.05 if C_rt.max() > 0 else 1.0)
    ax.set_xlabel("Radial distance (um)")
    ax.set_ylabel("Concentration (uM)")
    title = ax.set_title(f"t={t[0]:.1f}hr")
    ax.grid(alpha=0.3)

    n_frames = min(n_frames, len(t))
    frame_indices = np.linspace(0, len(t) - 1, n_frames).astype(int)

    def update(frame_num: int):
        idx = frame_indices[frame_num]
        line.set_ydata(C_rt[idx, :])
        title.set_text(f"t={t[idx]:.1f}hr")
        return line, title

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=1000 // fps, blit=False)

    if save_path:
        anim.save(save_path, writer="pillow", fps=fps)
        plt.close(fig)
        return save_path
    return anim
