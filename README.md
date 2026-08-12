# BIOPINN

A Physics-Informed Neural Network (PINN) platform that predicts nanoparticle
drug transport, penetration depth, and tumor-cell viability inside a 3-zone
tumor spheroid, then uses the trained network as a fast surrogate to
optimize nanoparticle size and dose.

> **Status:** all 14 build phases complete (0–13 per the original build
> protocol, plus Phase 14: manuscript Results & Discussion asset generation).
> See [LOGBOOK.md](LOGBOOK.md) for a plain-language, dated diary of how the
> project was built, including what was tested and what went wrong along the
> way.

## What this is

BIOPINN solves an augmented Fickian reaction-diffusion PDE for drug
concentration `C(r, t)` across a spherical tumor with three biological
zones (proliferating rim, quiescent zone, necrotic core), trains a
physics-informed neural network as a fast surrogate for that PDE, and uses
the surrogate to explore how nanoparticle size and dose affect drug
penetration and tumor-cell kill — all wired up to a local results dashboard
for interactive exploration.

## Scientific model summary

- **Governing PDE:** `dC/dt = D_eff(r)*[d2C/dr2 + (2/r)dC/dr] - k_d*C`,
  Dirichlet at the tumor surface (`C(R,t) = C0`), Neumann symmetry at the
  center (`dC/dr = 0`). The `2/r` term is the correct spherically-symmetric
  Laplacian; `src/fdm_solver.py` and `src/losses.py` are kept consistent
  with each other on this.
- **Three-zone microenvironment:** proliferating rim / quiescent zone /
  necrotic core, each with a distinct diffusion correction factor
  (`f_zone`, applied to Stokes–Einstein `D_free`) and drug-binding-rate
  multiplier, derived from a closed-form steady-state oxygen
  diffusion–consumption gradient.
- **Biological response:** Hill-equation drug-induced death rate →
  time-integrated survival fraction (hazard form, always in `(0,1]`) →
  viability `V(r,t)` and cytotoxicity `Cyt(r,t)` maps → penetration depth.
- **PINN:** 5×96 tanh MLP, Xavier init, **parametric 7-dim input**
  `(r_norm, t_norm, R_norm, d_NP_norm, C0_norm, k_d_norm, t_max_norm)` so
  *one* trained model generalizes across the full 5D physical parameter
  space (extends the base "(r_norm, t_norm)" skeleton — required so the
  optimization surrogate can query arbitrary `(d_NP, C0, R)` combinations
  without retraining), hard initial-condition output transform
  (`C_NN = sigmoid(f_theta(x)) * t_norm`), composite loss (data + physics +
  Dirichlet + Neumann + IC), two-phase Adam → L-BFGS training.
- **Optimization:** the trained PINN as a fast differentiable-free
  surrogate objective for a grid search over nanoparticle diameter and
  dose, run separately at four tumor radii (200/300/400/500 μm).

Full hyperparameters, physical constants, and acceptance thresholds live in
`configs/default_config.yaml`; every `src/` module reads from it rather than
hard-coding numbers.

## Two-part execution split

Heavy compute (synthetic dataset generation + PINN training) is meant to run
on a GPU (2,000 FDM simulations + a 20k-iteration Adam phase + an
L-BFGS phase — impractical on a CPU-only laptop). Everything else — biology
maps, evaluation, ablation, optimization, visualization, and the results
dashboard — runs **locally on CPU**, consuming the trained artifacts. The
`src/` package is the single source of truth, imported unchanged
everywhere, so there is no duplicated physics or model logic.

`notebooks/biopinn_train.ipynb` auto-detects its environment and runs
either way, with no manual edits:

- **Google Colab** (free T4 GPU): clones the repo, installs dependencies,
  mounts Google Drive, and redirects every output path there so it survives
  the runtime being recycled.
- **A local Jupyter kernel** (e.g. with your own NVIDIA GPU): skips the
  clone/Drive steps and writes directly into this checkout's own
  `artifacts/` and `data/` folders. If PyTorch reports no GPU because it's
  the CPU-only build, the notebook prints the exact `pip install` command
  to switch to a CUDA build.

If you know you're only ever running locally, `notebooks/biopinn_train_local.ipynb`
is the same notebook with all the Colab-detection branching removed —
simpler to read, identical behavior to the local path above.

**Windows + data generation:** both notebooks' "generate the dataset" cell always
runs single-threaded on Windows, even with multiple CPU cores available. This is
deliberate — Windows' multiprocessing re-imports the Jupyter kernel launcher as
`__main__` in each worker process, which fails to bootstrap and crashes with
`BrokenProcessPool` if you try to parallelize `ProcessPoolExecutor` work directly
from a notebook cell. For real multi-core speed on Windows (or anywhere), run
generation as a standalone script instead, which parallelizes safely because it has
a real `if __name__ == "__main__":` guard:

```
python scripts/generate_dataset.py --experiment experiment_1 --n-jobs 8
```

Then open the training notebook, set `DATA_ALREADY_GENERATED = True` next to
`QUICK_TEST`, and run the rest of the notebook — it loads this script's output via
`load_processed_dataset()` instead of regenerating it. Omit `--experiment` to
generate the full 2,000-sim production dataset; `--n-jobs` defaults to all CPU
cores.

```
notebooks/biopinn_train.ipynb   [Colab or local GPU]   data generation + training
        │  saves biopinn_model.pt + normalization_stats.json
        ▼
artifacts/                      [Local, CPU]   (already there if trained locally;
        │                                       drop the downloaded files here if trained on Colab)
        ▼
scripts/run_evaluation.py, run_ablation.py, run_optimization.py,
make_figures.py, run_dashboard.py
```

Local scripts **never retrain** — they load `artifacts/biopinn_model.pt` and
the processed dataset and fail with a clear message if those aren't present
yet.

## Repository layout

```
BIOPINN/
├── configs/            YAML hyperparameters (default_config.yaml + experiment overrides)
├── notebooks/          Colab training notebook (data generation + training)
├── data/                raw FDM sims / processed train-val-test tensors / viability maps
├── artifacts/            handoff: trained checkpoint + normalization stats (dropped in from Drive)
├── src/                   shared package -- single source of truth for both Colab and local
├── scripts/                local CLI entry points (consume artifacts/, never retrain)
├── app/                     FastAPI + Plotly results dashboard
├── results/                 evaluation / ablation / optimization / paper reports + figures (generated)
└── tests/                    unit tests (pytest)
```

## Module reference

### `src/` — shared package

| Module | What it does | Used by |
|---|---|---|
| `config.py` | Loads `configs/default_config.yaml`, deep-merges an optional experiment override on top, resolves `paths.*` to absolute repo-root paths. | everything |
| `microenvironment.py` | Radial grid, Stokes–Einstein `D_free`, steady-state oxygen gradient, three-zone assignment, `D_eff(r)` / `k_d(r)` fields (batched or pointwise). | FDM solver, losses, biology, evaluate, optimize, dashboard |
| `fdm_solver.py` | Forward-Euler solver for the governing PDE, with a CFL guard that auto-reduces the internal time step (decoupled from the stored output grid, so memory stays bounded even for the worst-case small-R/small-d_NP corner) and Dirichlet/Neumann boundary handling. | Colab data generation, `optimize.py`'s H3 comparison |
| `data_pipeline.py` | Latin Hypercube Sampling over the 5D parameter space, parallel FDM dataset generation, data/collocation/BC/IC point sampling, normalization, train/val/test tensor + `sim_params.json` export. | Colab notebook |
| `model.py` | The `BIOPINN` network (5×96 tanh MLP, optional Random Fourier Features, hard-IC output transform) + checkpoint load/save. | training, everything downstream |
| `losses.py` | The five composite loss terms (data, physics/PDE-residual via autograd, Dirichlet BC, Neumann BC, IC), plus `composite_loss_chunked`/`physics_loss_chunked` -- gradient-accumulated, memory-bounded equivalents (identical resulting gradient, bounded peak GPU memory) used when `training.max_points_per_chunk` is set. | training, evaluation (PDE-residual stats), ablation |
| `train.py` | Two-phase Adam → L-BFGS training engine: gradient clipping, StepLR, `w_phys` warmup ramp, NaN-recovery safeguard, checkpointing. Training is full-batch (every iteration sees the whole train split at once); `configs/default_config.yaml`'s `training.max_points_per_chunk` (default 1,000,000) caps peak memory by computing/backward()-ing each loss term in point-count-bounded chunks instead of one shot -- needed since the PDE-residual term's second-order autograd over millions of collocation points can exceed GPU memory otherwise (`CUDA out of memory`). Unset it to restore the original single-shot behavior on a GPU with enough VRAM to not need it. **Resumable**: `training.checkpoint_every` (default 100) periodically saves model weights + Adam's momentum buffers + LR schedule position (not just the model) to `<model_checkpoint>_resume.pt` during the Adam phase; if that file exists the next time `train()` runs, it resumes from the exact epoch it left off at -- bit-identical to an uninterrupted run, not just "close enough" -- instead of restarting from epoch 0. The file is deleted once training fully completes. Set to 0 to disable. Both phases track the best validation-scored state seen and restore it at the end, rather than keeping whatever epoch/closure happened to run last -- L-BFGS's line search in particular can wander through many closures with no guarantee the final one generalizes best, so it can only improve on the Adam phase's result, never quietly regress it. The score is `loss.w_data * val_data + loss.w_phys * val_phys` (the same weights the training objective itself uses), not an unweighted sum -- since `phys` can swing over orders of magnitude while `data` moves by a few percent, an unweighted sum could let a big physics-residual win mask a real regression in data fit. | Colab notebook, `ablation.py` (baseline training) |
| `biology.py` | Hill-equation death rate → survival fraction → viability/cytotoxicity maps → penetration depth; queries a loaded checkpoint for `C(r,t)` over an arbitrary parameter combination. | evaluation, optimization, dashboard |
| `evaluate.py` | Six-metric evaluation (RMSE, MAE, R², L2 relative error, mean PDE residual, penetration RMSE) against the held-out test set, decomposed by zone/NP-size/time; H1/H2/H4 hypothesis checks. | `run_evaluation.py`, ablation, dashboard |
| `ablation.py` | Trains an identical-architecture `w_phys=0` baseline, compares PDE-residual statistics against the primary model, Wilcoxon signed-rank test (H5). | `run_ablation.py`, dashboard |
| `optimize.py` | Grid search over `(d_NP, C0)` at 4 tumor radii maximizing volume-averaged kill fraction η; homogeneous-vs-heterogeneous diffusion comparison (H3); PINN-vs-FDM speedup study (H6). | `run_optimization.py`, `make_figures.py`, dashboard |
| `visualize.py` | Every figure: concentration/viability/cytotoxicity heatmaps, penetration-depth curves, the η(d_NP,C0) surface, the H3 and ablation comparison figures, PDE-residual histogram, PINN-vs-FDM overlay, and a GIF animation. One implementation per figure, shared by `run_evaluation.py` and `make_figures.py`. | `run_evaluation.py`, `make_figures.py` |
| `results.py` | Computes the manuscript's Results & Discussion chapter: Fig 4.1–4.10 and Table 4.1–4.9, at the manuscript's fixed baseline (R=400μm, d_NP=100nm, C0=10μM, k_d=0.01/hr, t=72hr) and its three standard sweeps. Calls into the modules above for every number; never reimplements physics. | `generate_results.py` |

### `scripts/` — local CLI entry points (consume `artifacts/`, never retrain)

| Script | What it prints/saves |
|---|---|
| `run_evaluation.py` | Six-metric report (global + decomposed) + H1/H2/H4 pass/fail → `results/evaluation/` (metrics.json/csv, PDE-residual histogram, PINN-vs-FDM overlay). |
| `run_ablation.py` | Trains the `w_phys=0` baseline locally, PDE-residual comparison + Wilcoxon p-value (H5) → `results/ablation/ablation_report.json`. |
| `run_optimization.py` | Per-radius `(d_NP*, C0*)` + max η, the homogeneous-vs-heterogeneous comparison (H3), and the PINN-vs-FDM speedup study (H6) → `results/optimization/optimization_report.json`. |
| `make_figures.py` | All publication figures for one representative test simulation + the effectiveness surface + H3 + (if a baseline checkpoint exists) the ablation figure → `results/figures/`. |
| `run_dashboard.py` | Launches the interactive results dashboard (see below). |
| `generate_results.py` | The full manuscript Results & Discussion asset pack, in APA (7th ed.) style: 10 figures (PNG+PDF, 300 DPI, no caption baked into the image) and 9 tables (CSV + one compiled `BIOPINN_results_tables_<mode>.docx` with APA three-line borders, bold "Table N" + italicized title, "Note." footnotes) → `results/paper/`, plus `FIGURE_CAPTIONS_<mode>.md` (bold "Figure N" + italicized title, since captions live outside the image files) and `results_manifest_<mode>.json` recording the source/config/key values behind every number. `--numbering {chapter,sequential}` picks `<mode>`: `chapter` (default, e.g. Table 4.1 — unchanged from before) or `sequential` (plain Table 1, Figure 1 — run once per mode to get both, filenames never collide). Skips Table 4.8/8 (ablation) with a clear note if no baseline checkpoint is present. |

### `app/` — results dashboard

FastAPI backend (`app/server.py`) loads the checkpoint once and serves it
live: `POST /api/predict` queries the PINN surrogate for the current
parameter settings (near-instant), while `/api/optimization`,
`/api/evaluation`, and `/api/ablation` are computed once in background
threads and cached (they re-solve FDM references / run the 4-radius grid
search), with the frontend polling until each is ready. The frontend
(`app/templates/index.html`, `app/static/`) is a single-page vanilla
HTML/CSS/JS app with a locally vendored Plotly bundle (no CDN dependency at
runtime): a persistent parameter sidebar plus a responsive card grid
covering the concentration field, penetration depth, viability/cytotoxicity
with a three-zone overlay, treatment effectiveness + resistance-risk map,
the per-radius optimization table (compared against your current
settings), the six-metric quality panel with a PDE-residual histogram and
PINN-vs-FDM overlay, and the ablation comparison — plus PNG/JSON/CSV
export. Never retrains.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                  # Windows (cmd.exe):    .venv\Scripts\activate.bat
pip install -e .
pip install -r requirements.txt
```

Verify the scaffold:

```bash
pytest tests/ -v
```

Tested on Linux, macOS, and Windows 10/11 (PowerShell and cmd.exe). Every
path in `src/`/`scripts/`/`app/` is built with `pathlib`, so nothing
hard-codes a POSIX-only path separator. Two Windows-specific notes:

- **PowerShell's script execution policy** may block `Activate.ps1` the
  first time, with an error like "running scripts is disabled on this
  system." Fix once per machine (as an administrator, or per-user):
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **`pip install torch`** installs the CPU-only build by default on every
  platform, Windows included. If this machine has an NVIDIA GPU, install a
  CUDA build instead — `notebooks/biopinn_train.ipynb`'s GPU-detection cell
  prints the exact command for your setup, or see
  [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

## Full round trip

1. **Run the training notebook** (`notebooks/biopinn_train.ipynb`), either:
   - **On Colab** (no local GPU needed): open it in Colab, select a T4 GPU
     runtime, and run all cells top to bottom. It installs dependencies,
     mounts your Google Drive, clones/installs `src/`, and saves everything
     there.
   - **Locally** (if this machine has an NVIDIA GPU): open it in a local
     Jupyter kernel from inside `notebooks/` and run all cells — it
     auto-detects it isn't on Colab, skips the clone/Drive steps, and
     writes straight into this checkout's `artifacts/`/`data/` folders.
     Make sure PyTorch can see the GPU first (`torch.cuda.is_available()`
     in cell 5 — if it's `False` because you have the CPU-only build, the
     notebook prints the exact command to reinstall the CUDA build).

   Either way, it generates the 2,000-simulation LHS-sampled FDM dataset
   (CFL-enforced), runs two-phase Adam → L-BFGS training with logged loss
   curves, and saves `biopinn_model.pt`, `normalization_stats.json`,
   `training_history.json`, and the processed train/val/test dataset (+
   `sim_params.json`). A `QUICK_TEST` toggle near the top switches to a
   tiny dev-scale run for smoke-testing the notebook itself.
2. **If you trained on Colab**, download the artifacts from Drive into this
   repo (skip this step if you trained locally — they're already there):
   - `biopinn_model.pt`, `normalization_stats.json`, `training_history.json` → `artifacts/`
   - the processed dataset → `data/processed/`
3. **Run local analysis** (any order; all consume `artifacts/` + `data/processed/`, none retrain):
   ```bash
   python scripts/run_evaluation.py          # six metrics + H1/H2/H4
   python scripts/run_ablation.py            # trains the w_phys=0 baseline locally, H5
   python scripts/run_optimization.py        # per-radius (d_NP*, C0*), H3/H6
   python scripts/make_figures.py            # all publication figures -> results/figures/
   python scripts/run_dashboard.py           # interactive dashboard at http://127.0.0.1:8000
   python scripts/generate_results.py        # manuscript Fig 4.1-4.10 + Table 4.1-4.9 -> results/paper/
   ```
   Pass `--experiment experiment_1` to any of them to point at the small
   dev-scale config instead of `configs/default_config.yaml` (useful for a
   fast smoke test with a dev-trained checkpoint). `generate_results.py`
   additionally needs `artifacts/training_history.json` (saved directly by
   the notebook's training-artifacts cell) for the loss-curve figure, and a
   `<model_checkpoint>_baseline.pt` (from `run_ablation.py`) to include the
   ablation table -- it runs and produces the other 9 tables + 10 figures
   without either.

## Configuration system

`configs/default_config.yaml` is the single source of truth for physical
constants, the microenvironment model, the FDM solver, the dataset spec,
the PINN architecture, loss weights, the training schedule, the
optimization grid, and evaluation thresholds. Every `src/` module reads
from it via `src/config.py::load_config` rather than hard-coding numbers.
`configs/experiment_1.yaml` (fast local/dev smoke test — ~20 tiny
simulations, coarse FDM grid, short training) and `configs/experiment_2.yaml`
(ablation baseline: `w_phys=0`) are deep-merged on top of the default when
passed as `--experiment NAME`.

## Hypotheses & acceptance targets

| # | Hypothesis | Target | Checked by |
|---|---|---|---|
| H1 | Test-set concentration prediction accuracy | RMSE < 0.05 μM, R² > 0.990 | `run_evaluation.py` |
| H2 | Smaller nanoparticles penetrate deeper | 10nm vs. 200nm penetration-depth difference > 100 μm @ R=400μm, t=72hr | `run_evaluation.py` |
| H3 | Three-zone heterogeneity matters | steeper concentration gradient + larger sub-therapeutic zone than a homogeneous-diffusion model | `run_optimization.py` |
| H4 | Zone-dependent viability | rim viability < 20%, core viability > 60% @ t=72hr | `run_evaluation.py` |
| H5 | Physics constraint improves PDE consistency | PDE residual ≥ 10× lower than an identical unconstrained (`w_phys=0`) baseline, Wilcoxon-significant | `run_ablation.py` |
| H6 | Surrogate speedup | ≥ 2 orders of magnitude faster than re-solving FDM | `run_optimization.py` |

All six metrics (RMSE, MAE, R², L2 relative error, mean PDE residual,
penetration RMSE) are also reported globally and decomposed by tumor zone,
nanoparticle-size range, and time range in `run_evaluation.py`'s output.
Every threshold is defined once, in `configs/default_config.yaml`'s
`evaluation` section — nowhere else.

*(Note: several hypotheses, including H3–H5, hold cleanly in specific
parameter regimes found via sweeps — e.g. H3 at a low, just-above-threshold
dose — rather than at every point in the 5D parameter space; at saturating
doses the whole tumor equilibrates within the simulation window and there's
no contrast left to detect. This is a property of the underlying physical
model at those parameters, not an implementation bug; see the docstrings in
`tests/test_optimize.py` and `tests/test_biology.py` for the specific
regimes and the reasoning. `generate_results.py`'s Table 4.6/Fig 4.9 and
Hypothesis H3 compute this honestly at the manuscript's own literal
baseline (C0=10μM) rather than substituting a different, contrast-showing
regime — so a real trained model may legitimately report H3 as "not
supported" in `results/paper/`, alongside a manifest note explaining why;
that is the correct, non-cherry-picked output of replacing a manuscript's
illustrative `[SAMPLE]` figures with a real computation, not a bug in the
pipeline.)*

## Testing

```bash
pytest tests/ -v
```

Each `src/` module (and the dashboard's pure helpers) has a corresponding
`tests/test_*.py` using small synthetic configs/data, so the suite runs in
well under a minute on CPU and never requires a trained checkpoint or GPU.

## License

TBD.
