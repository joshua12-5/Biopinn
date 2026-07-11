# BIOPINN

A Physics-Informed Neural Network (PINN) platform that predicts nanoparticle
drug transport, penetration depth, and tumor-cell viability inside a 3-zone
tumor spheroid, then uses the trained network as a fast surrogate to
optimize nanoparticle size and dose.

> **Status:** Phase 0 (scaffold) complete. See `## Build phases` below —
> this README will be filled in as each phase lands.

## What this is

BIOPINN solves an augmented Fickian reaction-diffusion PDE for drug
concentration `C(r, t)` across a spherical tumor with three biological
zones (proliferating rim, quiescent zone, necrotic core), trains a
physics-informed neural network as a fast surrogate for that PDE, and uses
the surrogate to explore how nanoparticle size and dose affect drug
penetration and tumor-cell kill.

## Two-part execution split

Heavy compute (synthetic dataset generation + PINN training) runs on
**Google Colab** (free T4 GPU). Everything else — biology maps, evaluation,
ablation, optimization, visualization, and the results dashboard — runs
**locally on CPU**, consuming the trained artifacts. The `src/` package is
the single source of truth imported unchanged by both sides.

```
notebooks/biopinn_train.ipynb   [Colab, GPU]   data generation + training
        │  saves biopinn_model.pt + normalization_stats.json to Drive
        ▼
artifacts/                      [Local, CPU]   drop the downloaded files here
        │
        ▼
scripts/run_evaluation.py, run_ablation.py, run_optimization.py,
make_figures.py, run_dashboard.py
```

## Repository layout

```
BIOPINN/
├── configs/           YAML hyperparameters (default + experiment overrides)
├── notebooks/          Colab training notebook
├── data/                raw FDM sims, processed train/val/test tensors, viability maps
├── artifacts/            handoff: trained checkpoint + normalization stats
├── src/                   shared package: microenvironment, FDM solver, data
│                          pipeline, PINN model + losses, training engine,
│                          biology, evaluation, ablation, optimization, visualization
├── scripts/                local CLI entry points (consume artifacts/, never retrain)
├── app/                     FastAPI + Plotly results dashboard
└── tests/                    unit tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Verify the scaffold:

```bash
pytest tests/ -v
```

## Round trip (filled in as later phases land)

1. **Run the notebook on Colab** (`notebooks/biopinn_train.ipynb`) to
   generate the synthetic FDM dataset and train BIOPINN. *(Phase 6)*
2. **Download artifacts from Drive** into `artifacts/`:
   `biopinn_model.pt`, `normalization_stats.json`, plus the processed
   dataset into `data/processed/`.
3. **Run local analysis:**
   - `python scripts/run_evaluation.py` — six-metric report + H1/H2/H4 pass/fail. *(Phase 8)*
   - `python scripts/run_ablation.py` — physics-informed vs. unconstrained baseline, H5. *(Phase 9)*
   - `python scripts/run_optimization.py` — optimal `(d_NP*, C0*)` per tumor radius, H3/H6. *(Phase 10)*
   - `python scripts/make_figures.py` — all publication figures. *(Phase 11)*
   - `python scripts/run_dashboard.py` — interactive results dashboard. *(Phase 12)*

## Build phases

This system is being built incrementally, one phase at a time, per the
BIOPINN build protocol. Each phase adds modules on top of the shared `src/`
package without touching earlier phases.

| Phase | Delivers | Status |
|---|---|---|
| 0 | Scaffold: repo structure, configs, requirements | done |
| 1 | Microenvironment model (`src/microenvironment.py`) | pending |
| 2 | FDM solver (`src/fdm_solver.py`) | pending |
| 3 | Data pipeline (`src/data_pipeline.py`) | pending |
| 4 | PINN core + losses (`src/model.py`, `src/losses.py`) | pending |
| 5 | Training engine (`src/train.py`) | pending |
| 6 | Colab notebook (`notebooks/biopinn_train.ipynb`) | pending |
| 7 | Biology module (`src/biology.py`) | pending |
| 8 | Evaluation (`src/evaluate.py`, `scripts/run_evaluation.py`) | pending |
| 9 | Ablation study (`src/ablation.py`, `scripts/run_ablation.py`) | pending |
| 10 | Optimization + efficiency (`src/optimize.py`, `scripts/run_optimization.py`) | pending |
| 11 | Visualization (`src/visualize.py`, `scripts/make_figures.py`) | pending |
| 12 | Results dashboard (`app/`, `scripts/run_dashboard.py`) | pending |
| 13 | Polish: full docs, remaining tests, cleanup | pending |

## Scientific model summary

- **Governing PDE:** `dC/dt = D_eff(r)*[d2C/dr2 + (2/r)dC/dr] - k_d*C`,
  Dirichlet at the tumor surface, Neumann symmetry at the center.
- **Three-zone microenvironment:** proliferating rim / quiescent zone /
  necrotic core, each with a distinct diffusion correction factor and
  binding-rate multiplier derived from a steady-state oxygen gradient.
- **Biological response:** Hill-equation death rate → survival fraction →
  viability and cytotoxicity maps → penetration depth.
- **PINN:** 5×64 tanh network, hard initial-condition output transform,
  composite loss (data + physics + Dirichlet + Neumann + IC), two-phase
  Adam → L-BFGS training.
- **Optimization:** the trained PINN as a differentiable surrogate for a
  grid search over nanoparticle diameter and dose, at four tumor radii.

Full details, hyperparameters, and acceptance thresholds live in
`configs/default_config.yaml` and are documented per-module as each phase
lands.

## License

TBD.
