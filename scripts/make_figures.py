#!/usr/bin/env python3
"""CLI: generate the full set of publication figures from a trained checkpoint.

Loads artifacts/biopinn_model.pt (never retrains) and produces, for one
representative test-set simulation: a concentration heatmap, penetration
depth vs. time, viability + cytotoxicity maps (three-zone overlay), and an
animated GIF of the concentration profile. Also produces the surrogate
treatment-effectiveness surface eta(d_NP, C0) at a representative tumor
radius, and the heterogeneous-vs-homogeneous diffusion comparison (H3) at
the parameter regime where the effect is genuinely visible (see
tests/test_optimize.py::test_h3_holds_for_a_well_chosen_case -- the guide's
own example parameters oversaturate the tumor and show no contrast). If a
w_phys=0 ablation baseline checkpoint is present (artifacts/*_baseline.pt,
written by scripts/run_ablation.py), also produces the ablation residual
comparison figure (H5).

All figures are saved to paths.results/figures/.

Usage:
    python scripts/make_figures.py [--experiment NAME]

Requires paths.model_checkpoint and a processed dataset (paths.processed) to
already exist -- run notebooks/biopinn_train.ipynb first if they don't.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ablation import compare_residuals
from src.biology import compute_biological_response
from src.config import load_config, resolve_path
from src.evaluate import load_test_sim_params, resolve_test_simulations
from src.fdm_solver import solve_fdm
from src.microenvironment import stokes_einstein_diffusivity
from src.model import load_checkpoint
from src.optimize import grid_search_radius
from src.visualize import (
    animate_concentration,
    plot_ablation_comparison,
    plot_concentration_heatmap,
    plot_cytotoxicity_map,
    plot_effectiveness_surface,
    plot_homogeneous_comparison,
    plot_penetration_depth,
    plot_viability_map,
)

import numpy as np
import torch

# A low-dose, slow-diffusing, large-tumor regime found via parameter sweep
# where the heterogeneous-vs-homogeneous diffusion contrast is genuinely
# visible (the guide's suggested "typical" demo parameters, e.g. C0=10uM,
# saturate the whole tumor for both models within 72hr).
H3_DEMO_PARAMS = {"d_NP_nm": 200.0, "R_um": 500.0, "C0_uM": 0.105, "k_d_per_hr": 0.05, "t_max_hr": 72.0}


def _load_normalization_stats(config: dict) -> dict:
    stats_path = resolve_path(config, "normalization_stats")
    with open(stats_path) as f:
        return json.load(f)


def make_representative_sim_figures(model, config: dict, norm_stats: dict, output_dir: Path) -> dict:
    """Concentration heatmap, penetration depth, viability, cytotoxicity,
    and animation for the first test-set simulation, driven by the PINN
    surrogate's own predictions."""
    sim_params = load_test_sim_params(config)[0]
    sims = resolve_test_simulations(config, sim_params=[sim_params])
    sim = sims[0]
    r, t = sim["r"], sim["t"]

    response = compute_biological_response(model, r, t, sim_params, norm_stats, config)

    plot_concentration_heatmap(response["C"], r, t, config, save_path=str(output_dir / "concentration_heatmap.png"))
    plot_penetration_depth(response["penetration_depth"], t, config, save_path=str(output_dir / "penetration_depth.png"))
    plot_viability_map(response["viability"], r, t, config, save_path=str(output_dir / "viability_map.png"))
    plot_cytotoxicity_map(
        response["cytotoxicity"], r, t, config, R_um=sim_params["R_um"], save_path=str(output_dir / "cytotoxicity_map.png")
    )
    animate_concentration(response["C"], r, t, config, save_path=str(output_dir / "concentration_animation.gif"))

    print(f"  sim {sim_params['sim_id']} (R={sim_params['R_um']:.0f}um, d_NP={sim_params['d_NP_nm']:.0f}nm, "
          f"C0={sim_params['C0_uM']:.2f}uM): concentration_heatmap.png, penetration_depth.png, "
          "viability_map.png, cytotoxicity_map.png, concentration_animation.gif")
    return sim_params


def make_effectiveness_surface_figure(model, config: dict, norm_stats: dict, output_dir: Path) -> None:
    radii = config["optimization"]["radii_um"]
    R_um = radii[len(radii) // 2]
    result = grid_search_radius(model, R_um, config, norm_stats)
    plot_effectiveness_surface(
        result["eta_grid"], result["d_NP_grid_nm"], result["C0_grid_uM"], config,
        save_path=str(output_dir / "effectiveness_surface.png"),
    )
    print(f"  R={R_um:.0f}um: effectiveness_surface.png (optimum d_NP*={result['d_NP_star_nm']:.1f}nm, "
          f"C0*={result['C0_star_uM']:.2f}uM, eta*={result['max_eta']:.4f})")


def make_h3_figure(config: dict, output_dir: Path) -> None:
    p = H3_DEMO_PARAMS
    hetero_result = solve_fdm(p["R_um"], p["d_NP_nm"], p["C0_uM"], p["k_d_per_hr"], p["t_max_hr"], config)

    const = config["constants"]
    D_free_m2_s = stokes_einstein_diffusivity(p["d_NP_nm"], const["T"], const["eta"], const["k_B"])
    D_free_um2_hr = D_free_m2_s * 1e12 * 3600.0
    D_eff_homogeneous = D_free_um2_hr * float(np.mean(list(config["microenvironment"]["f_zone"].values())))
    homog_result = solve_fdm(
        p["R_um"], p["d_NP_nm"], p["C0_uM"], p["k_d_per_hr"], p["t_max_hr"], config, D_eff_override=D_eff_homogeneous
    )

    plot_homogeneous_comparison(hetero_result, homog_result, config, save_path=str(output_dir / "h3_homogeneous_comparison.png"))
    print(f"  d_NP={p['d_NP_nm']}nm, R={p['R_um']}um, C0={p['C0_uM']}uM: h3_homogeneous_comparison.png")


def make_ablation_figure(config: dict, norm_stats: dict, output_dir: Path) -> bool:
    checkpoint_path = resolve_path(config, "model_checkpoint")
    baseline_path = checkpoint_path.with_name(checkpoint_path.stem + "_baseline" + checkpoint_path.suffix)
    if not baseline_path.exists():
        print(f"  no baseline checkpoint at {baseline_path} -- run scripts/run_ablation.py first to enable this figure.")
        return False

    biopinn_model = load_checkpoint(str(checkpoint_path), config)
    baseline_model = load_checkpoint(str(baseline_path), config)

    processed_dir = resolve_path(config, "processed")
    with np.load(processed_dir / "test.npz") as test_npz:
        collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)

    residual_comparison = compare_residuals(biopinn_model, baseline_model, collocation_X, config, norm_stats)
    plot_ablation_comparison(residual_comparison, config, save_path=str(output_dir / "ablation_comparison.png"))
    print("  ablation_comparison.png")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    args = parser.parse_args()

    config = load_config(args.experiment)

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)
    norm_stats = _load_normalization_stats(config)

    output_dir = resolve_path(config, "results") / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating representative-simulation figures...")
    make_representative_sim_figures(model, config, norm_stats, output_dir)

    print("\nGenerating treatment-effectiveness surface...")
    make_effectiveness_surface_figure(model, config, norm_stats, output_dir)

    print("\nGenerating heterogeneous-vs-homogeneous diffusion comparison (H3)...")
    make_h3_figure(config, output_dir)

    print("\nGenerating ablation residual comparison (H5)...")
    make_ablation_figure(config, norm_stats, output_dir)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
