#!/usr/bin/env python3
"""CLI: generate the full manuscript Results & Discussion asset pack.

Loads the trained checkpoint (+ optionally a w_phys=0 ablation baseline) and
the held-out test set, computes every figure (Fig 4.1-4.10) and table
(Table 4.1-4.9) via src/results.py, and writes them to results/paper/:
  - fig_4_1_*.png / .pdf ... fig_4_10_*.png / .pdf   (300 DPI)
  - table_4_1.csv ... table_4_9.csv
  - BIOPINN_results_tables.docx                       (all 9 tables, captioned)
  - results_manifest.json                             (source/config/key-values per asset)

Never retrains -- consumes artifacts/ exactly as the other scripts/*.py do.
Table 4.8 (ablation) is skipped with a clear note if no baseline checkpoint
is present (run scripts/run_ablation.py first to produce one).

Usage:
    python scripts/generate_results.py [--experiment NAME] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from docx import Document
from docx.shared import Pt

from src import results as R
from src.biology import evaluate_h4_hypothesis, predict_concentration_field, viability_map
from src.config import load_config, resolve_path
from src.evaluate import evaluate_h2_hypothesis, resolve_test_simulations
from src.microenvironment import radial_grid
from src.model import load_checkpoint
from src.optimize import optimize_all_radii, speedup_study


def _sanitize_for_json(obj):
    """Recursively converts numpy/pandas scalars to native Python types (same
    conversion _default performs for json.dump) and replaces NaN/Infinity
    with None -- json.dump's default allow_nan=True would otherwise emit a
    literal NaN/Infinity token, which is not valid JSON and breaks strict
    parsers (e.g. a div-by-zero guard in table_4_6's percentage-difference
    column, or an empty resistance zone elsewhere, both legitimate results
    at some parameter combinations)."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if hasattr(obj, "tolist"):
        return _sanitize_for_json(obj.tolist())
    if hasattr(obj, "item"):
        return _sanitize_for_json(obj.item())
    return obj


def _add_table_to_doc(doc: Document, number: str, caption: str, df, footnote: str | None = None) -> None:
    doc.add_heading(f"Table {number} — {caption}", level=2)
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = "Light Grid Accent 1"
    header_cells = table.rows[0].cells
    for i, col in enumerate(df.columns):
        header_cells[i].text = str(col)
        for run in header_cells[i].paragraphs[0].runs:
            run.font.bold = True
    for _, row in df.iterrows():
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = f"{value:.4g}" if isinstance(value, float) else str(value)
    if footnote:
        note = doc.add_paragraph(footnote)
        note.runs[0].font.size = Pt(9)
        note.runs[0].font.italic = True
    doc.add_paragraph()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for re-solving the test set's FDM reference")
    args = parser.parse_args()

    config = load_config(args.experiment)
    paper_cfg = config["paper"]

    checkpoint_path = resolve_path(config, "model_checkpoint")
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.")
        print("Run notebooks/biopinn_train.ipynb on Colab first, then drop its artifacts here.")
        sys.exit(1)

    processed_dir = resolve_path(config, "processed")
    if not (processed_dir / "test.npz").exists():
        print(f"No processed test set found at {processed_dir}.")
        print("Run notebooks/biopinn_train.ipynb (or src.data_pipeline.build_dataset) first.")
        sys.exit(1)

    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path), config)
    with open(resolve_path(config, "normalization_stats"), encoding="utf-8") as f:
        norm_stats = json.load(f)

    baseline_checkpoint_path = checkpoint_path.with_name(checkpoint_path.stem + "_baseline" + checkpoint_path.suffix)
    baseline_model = load_checkpoint(str(baseline_checkpoint_path), config) if baseline_checkpoint_path.exists() else None
    if baseline_model is None:
        print(f"No ablation baseline checkpoint at {baseline_checkpoint_path} -- Table 4.8 will be skipped.")
        print("Run scripts/run_ablation.py first to enable it.")

    print(f"Re-solving FDM reference for the test set (n_jobs={args.n_jobs})...")
    sims = resolve_test_simulations(config, n_jobs=args.n_jobs)
    print(f"  {len(sims)} test simulations resolved.")

    with np.load(processed_dir / "test.npz") as test_npz:
        collocation_X = torch.as_tensor(test_npz["collocation_X"], dtype=torch.float32)

    output_dir = resolve_path(config, "results") / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"config_used": args.experiment or "default_config.yaml", "baseline_params": R.baseline_params(config), "figures": {}, "tables": {}}

    # ----------------------------------------------------------------- #
    # Figures
    # ----------------------------------------------------------------- #
    print("\nComputing figures...")

    print("  Fig 4.1 -- spatiotemporal concentration heatmap")
    fig, meta = R.fig_4_1_concentration_heatmap(model, config, norm_stats)
    manifest["figures"]["4.1"] = {"files": R._save_figure(fig, output_dir, "fig_4_1_concentration_heatmap"), **meta}

    print("  Fig 4.2 -- radial concentration profiles")
    fig, meta = R.fig_4_2_radial_profiles(model, config, norm_stats)
    manifest["figures"]["4.2"] = {"files": R._save_figure(fig, output_dir, "fig_4_2_radial_concentration_profiles"), **meta}

    print("  solving the nanoparticle-diameter FDM sweep (shared by Table 4.1/4.2/4.4 and Fig 4.3)...")
    diameter_sweep = R.solve_diameter_sweep(config)

    print("  Fig 4.3 -- PINN vs. FDM comparison at t=24hr")
    fig, meta = R.fig_4_3_pinn_vs_fdm_t24(model, config, norm_stats, diameter_sweep=diameter_sweep)
    manifest["figures"]["4.3"] = {"files": R._save_figure(fig, output_dir, "fig_4_3_pinn_vs_fdm_profile_comparison"), **meta}

    print("  Fig 4.4 -- predicted vs. reference scatter (full test set)")
    fig, meta = R.fig_4_4_scatter_pred_vs_ref(model, config, norm_stats, sims)
    manifest["figures"]["4.4"] = {"files": R._save_figure(fig, output_dir, "fig_4_4_predicted_vs_reference_scatter"), **meta}

    print("  Fig 4.5 -- training loss convergence")
    history = R.load_training_history(config)
    fig, meta = R.fig_4_5_training_loss(history, config)
    manifest["figures"]["4.5"] = {"files": R._save_figure(fig, output_dir, "fig_4_5_training_loss_convergence"), **meta}

    print("  Fig 4.6 -- penetration depth vs. time")
    fig, meta = R.fig_4_6_penetration_vs_time(model, config, norm_stats)
    manifest["figures"]["4.6"] = {"files": R._save_figure(fig, output_dir, "fig_4_6_penetration_depth_vs_time"), **meta}

    print("  Fig 4.7 -- spatial viability at t=72hr")
    fig, meta = R.fig_4_7_viability_t72(model, config, norm_stats)
    manifest["figures"]["4.7"] = {"files": R._save_figure(fig, output_dir, "fig_4_7_spatial_viability"), **meta}

    print("  Fig 4.8 -- cytotoxicity evolution")
    fig, meta = R.fig_4_8_cytotoxicity_evolution(model, config, norm_stats)
    manifest["figures"]["4.8"] = {"files": R._save_figure(fig, output_dir, "fig_4_8_cytotoxicity_evolution"), **meta}

    print("  solving the heterogeneous/homogeneous D_eff comparison (shared by Table 4.6 and Fig 4.9)...")
    hetero_homog = R.solve_hetero_and_homogeneous(config)

    print("  Fig 4.9 -- heterogeneous vs. homogeneous D_eff")
    fig, meta = R.fig_4_9_hetero_vs_homog(config, solved=hetero_homog)
    manifest["figures"]["4.9"] = {"files": R._save_figure(fig, output_dir, "fig_4_9_heterogeneous_vs_homogeneous"), **meta}

    print("  running the grid search at the baseline radius (shared by Table 4.7 and Fig 4.10)...")
    radius_results = optimize_all_radii(model, config, norm_stats)
    baseline_R = R.baseline_params(config)["R_um"]

    print("  Fig 4.10 -- treatment effectiveness surface")
    fig, meta = R.fig_4_10_effectiveness_surface(model, config, norm_stats, grid_result=radius_results[baseline_R])
    manifest["figures"]["4.10"] = {"files": R._save_figure(fig, output_dir, "fig_4_10_effectiveness_surface"), **meta}

    # ----------------------------------------------------------------- #
    # Tables
    # ----------------------------------------------------------------- #
    print("\nComputing tables...")
    tables: dict[str, tuple] = {}

    print("  Table 4.1 -- FDM simulation summary")
    tables["4.1"] = R.table_4_1_fdm_summary(config, diameter_sweep=diameter_sweep)

    print("  Table 4.2 -- max penetration depth vs. diameter")
    tables["4.2"] = R.table_4_2_penetration_vs_diameter(config, diameter_sweep=diameter_sweep)

    print("  Table 4.3 -- PINN test-set metrics")
    tables["4.3"] = R.table_4_3_pinn_metrics(model, config, norm_stats, sims, collocation_X)

    print("  Table 4.4 -- penetration depth analysis")
    tables["4.4"] = R.table_4_4_penetration_analysis(model, config, norm_stats, diameter_sweep=diameter_sweep)

    print("  Table 4.5 -- viability summary")
    tables["4.5"] = R.table_4_5_viability_summary(model, config, norm_stats)

    print("  Table 4.6 -- heterogeneous vs. homogeneous comparison")
    tables["4.6"] = R.table_4_6_hetero_vs_homog(config, solved=hetero_homog)

    print("  Table 4.7 -- optimization results")
    tables["4.7"] = R.table_4_7_optimization_results(model, config, norm_stats, radius_results=radius_results)

    if baseline_model is not None:
        print("  Table 4.8 -- ablation study")
        tables["4.8"] = R.table_4_8_ablation(model, baseline_model, config, norm_stats, sims, collocation_X)
    else:
        tables["4.8"] = None

    for number, result in tables.items():
        if result is None:
            continue
        df, meta = result
        df.to_csv(output_dir / f"table_4_{number.split('.')[1]}.csv", index=False, encoding="utf-8")
        manifest["tables"][number] = {"csv": str(output_dir / f"table_4_{number.split('.')[1]}.csv"), **meta}

    # ----------------------------------------------------------------- #
    # Hypothesis summary (Table 4.9) -- assembled from the tables/figures above
    # ----------------------------------------------------------------- #
    print("  Table 4.9 -- hypothesis evaluation summary")
    h1_meta = tables["4.3"][1]
    h2 = evaluate_h2_hypothesis(model, norm_stats, config)

    baseline_p = R.baseline_params(config)
    r_h4 = radial_grid(baseline_p["R_um"], 200, config["fdm"]["r_min_um"])
    t_h4 = np.linspace(0.0, baseline_p["t_max_hr"], 60)
    C_h4 = predict_concentration_field(model, r_h4, t_h4, baseline_p, norm_stats)
    V_h4 = viability_map(C_h4, t_h4, config)
    h4 = evaluate_h4_hypothesis(V_h4, r_h4, t_h4, config, R_um=baseline_p["R_um"])

    h3_meta = tables["4.6"][1]

    if tables["4.8"] is not None:
        h5_pass = tables["4.8"][1]["H5_pass"]
    else:
        h5_pass = False

    print(f"  running the PINN-vs-FDM speedup study ({paper_cfg['hypotheses']['H6_speedup_n_combinations']} combinations)...")
    speedup = speedup_study(model, config, norm_stats, n_combinations=paper_cfg["hypotheses"]["H6_speedup_n_combinations"])
    table_4_7_times = [row["Computation Time (sec)"] for row in tables["4.7"][0].to_dict("records")]
    h6_pass = bool(
        speedup["speedup_ratio_mean"] >= 10 ** config["evaluation"]["hypotheses"]["H6_speedup_orders_of_magnitude"]
        and float(np.mean(table_4_7_times)) <= paper_cfg["hypotheses"]["H6_max_pinn_time_s"]
    )
    manifest["H6_speedup_study"] = speedup

    hypotheses = {
        "H1": {
            "name": "PINN Accuracy", "expected": "R² > 0.990, RMSE < 0.05 μM (target)",
            "evidence": "Table 4.3", "pass": bool(h1_meta["H1_pass_on_pooled_metrics"]),
        },
        "H2": {
            "name": "NP Size vs. Depth", "expected": "72hr penetration-depth difference (10nm vs. 200nm) > 100 μm",
            "evidence": "Table 4.4", "pass": bool(h2["pass"]),
        },
        "H3": {
            "name": "Spatial Heterogeneity", "expected": f"Steeper gradient, sub-therapeutic zone >= {h3_meta['H3_threshold_pct']:.0f}% larger",
            "evidence": "Table 4.6", "pass": bool(h3_meta["H3_pass"]),
        },
        "H4": {
            "name": "Viability Pattern", "expected": "Rim < 20%, Core > 60% at t=72hr",
            "evidence": "Table 4.5", "pass": bool(h4["overall_pass"]),
        },
        "H5": {
            "name": "Physics vs. Data-Driven", "expected": ">= 10x lower PDE residual, Wilcoxon significant",
            "evidence": "Table 4.8", "pass": bool(h5_pass),
        },
        "H6": {
            "name": "Surrogate Speedup", "expected": ">= 2 orders of magnitude speedup, <= 5s per configuration",
            "evidence": "Table 4.7", "pass": h6_pass,
        },
    }
    table_4_9 = R.table_4_9_hypothesis_summary(hypotheses)
    table_4_9.to_csv(output_dir / "table_4_9.csv", index=False, encoding="utf-8")
    manifest["tables"]["4.9"] = {"csv": str(output_dir / "table_4_9.csv"), "hypotheses": hypotheses}

    # ----------------------------------------------------------------- #
    # Compiled Word document
    # ----------------------------------------------------------------- #
    print("\nCompiling BIOPINN_results_tables.docx...")
    doc = Document()
    doc.add_heading("BIOPINN — Results & Discussion: Tables", level=1)
    doc.add_paragraph(
        f"Every value in this document was computed from the trained checkpoint, the held-out "
        f"{len(sims)}-simulation test set, and the analysis routines in src/ -- see "
        f"results_manifest.json in this folder for the exact source and configuration behind "
        f"every number."
    )

    order = [
        ("4.1", f"FDM Simulation Summary Statistics (Baseline: R={baseline_p['R_um']:.0f} μm, d_NP={baseline_p['d_NP_nm']:.0f} nm)"),
        ("4.2", "Maximum Penetration Depth at t=72hr Across Nanoparticle Diameters"),
        ("4.3", f"PINN Test Set Evaluation Metrics (Mean ± SD, N={len(sims)} test simulations)"),
        ("4.4", f"Penetration Depth Analysis Across Nanoparticle Sizes (R={baseline_p['R_um']:.0f}μm, C0={baseline_p['C0_uM']:.0f}μM, t={baseline_p['t_max_hr']:.0f}hr)"),
        ("4.5", f"Viability Analysis Summary at t={baseline_p['t_max_hr']:.0f}hr (R={baseline_p['R_um']:.0f}μm, C0={baseline_p['C0_uM']:.0f}μM)"),
        ("4.6", f"Heterogeneous vs. Homogeneous D_eff Model Comparison (d_NP={baseline_p['d_NP_nm']:.0f}nm, R={baseline_p['R_um']:.0f}μm, t={baseline_p['t_max_hr']:.0f}hr)"),
        ("4.7", "Optimization Results: Optimal Nanoparticle Parameters for Different Tumor Sizes"),
        ("4.8", "Ablation Study: PINN vs. Unconstrained NN Baseline"),
        ("4.9", "Hypothesis Evaluation Summary"),
    ]
    for number, caption in order:
        if number == "4.8" and tables["4.8"] is None:
            doc.add_heading(f"Table {number} — {caption}", level=2)
            doc.add_paragraph("Unavailable: no ablation baseline checkpoint found. Run scripts/run_ablation.py first.")
            continue
        df = table_4_9 if number == "4.9" else tables[number][0]
        footnote = None
        if number == "4.8":
            footnote = f"Wilcoxon signed-rank test (baseline residuals > BIOPINN residuals): p = {tables['4.8'][1]['wilcoxon_p_value']:.3e}"
        _add_table_to_doc(doc, number, caption, df, footnote=footnote)

    docx_path = output_dir / "BIOPINN_results_tables.docx"
    doc.save(docx_path)
    print(f"  saved {docx_path}")

    # ----------------------------------------------------------------- #
    # Manifest
    # ----------------------------------------------------------------- #
    manifest_path = output_dir / "results_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(manifest), f, indent=2)
    print(f"\nSaved 10 figures (PNG+PDF), 9 tables (CSV), the compiled docx, and results_manifest.json to {output_dir}")

    print("\n--- Hypothesis summary ---")
    for key, h in hypotheses.items():
        print(f"  {key} ({h['name']}): {'SUPPORTED' if h['pass'] else 'NOT SUPPORTED'} -- {h['evidence']}")


if __name__ == "__main__":
    main()
