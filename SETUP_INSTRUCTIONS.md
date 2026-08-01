# BIOPINN — Step-by-Step Setup Guide

This guide walks you from the unzipped `biopinn.zip` file to a running
results dashboard, in order. It assumes no prior familiarity with the
project. Follow the sections in order the first time through.

**The short version:** train once on Google Colab (free GPU) → download a
few small files → run everything else on your own laptop, forever, without
retraining.

---

## 0. What you're setting up

BIOPINN predicts how a drug carried by a nanoparticle spreads through a
tumor, how deep it penetrates, and how many tumor cells it kills — using a
neural network trained to obey the underlying physics (a Physics-Informed
Neural Network, or PINN). Once trained, that network is fast enough to
power an interactive dashboard where you drag sliders (tumor size,
nanoparticle size, dose...) and see the predicted outcome update live.

Training happens **once**, on Google Colab's free GPU (your laptop doesn't
need one). Everything else — evaluation, figures, the dashboard — runs
locally on an ordinary CPU.

---

## 1. Prerequisites

Install these before you start:

| Requirement | Why | Get it |
|---|---|---|
| Python 3.11 (or newer 3.11.x) | Runs everything locally | [python.org/downloads](https://www.python.org/downloads/) |
| A Google account | Runs the free-GPU training notebook on Colab | you probably already have one |
| ~2 GB free disk space | Python packages (mainly PyTorch) + generated data | — |

You do **not** need a GPU on your own machine, and you do **not** need to
install anything Colab-side manually — the notebook installs its own
dependencies.

Check your Python version:

```bash
python3 --version
# should print Python 3.11.x (3.10/3.12 will likely also work, but 3.11 is what this was built and tested against)
```

---

## 2. Unzip the project

Unzip `biopinn.zip` wherever you keep code, then move into it:

```bash
unzip biopinn.zip -d biopinn
cd biopinn
```

You should see this layout:

```
biopinn/
├── README.md            <- full reference docs (module-by-module, config system, etc.)
├── configs/              hyperparameters
├── notebooks/            training notebooks (Colab or local GPU)
├── src/                  shared code (physics, model, training, evaluation...)
├── scripts/               command-line tools you'll run locally
├── app/                    the results dashboard
├── tests/                  automated tests
├── data/, artifacts/, results/   empty folders that fill up as you go
```

---

## 3. Set up the local Python environment

From inside the `biopinn/` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows (PowerShell): .venv\Scripts\Activate.ps1
                                  # Windows (cmd.exe):    .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

This installs PyTorch, NumPy, SciPy, Matplotlib, FastAPI, and everything
else needed. It can take a few minutes the first time (PyTorch is a large
download).

> **Windows PowerShell only:** if `Activate.ps1` fails with something like
> "running scripts is disabled on this system," PowerShell's execution
> policy is blocking it — a one-time fix (as an administrator, or
> per-user): `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then
> try activating again.

**Verify it worked:**

```bash
pytest tests/ -v
```

You should see all 137 tests pass, finishing in well under a minute. If
this fails, stop here and fix your environment before continuing —
everything downstream depends on this working.

> Every time you come back to work on this project in a new terminal, run
> `source .venv/bin/activate` (from inside `biopinn/`) again before running
> anything else.

---

## 4. Train the model (one-time; minutes for a sanity run, hours for full scale)

This step needs a GPU to be practical at full scale. You have two options:
**Google Colab** (free GPU, no hardware of your own needed — Option A/B
below), or **your own NVIDIA GPU locally** if you have one (skip straight
to "Training locally instead of Colab" further down).

You have two ways to get the code onto Colab — pick one.

### Option A — you have a GitHub account (recommended, simplest)

1. Create a new empty repository on GitHub (public or private).
2. Push the unzipped folder to it:
   ```bash
   cd biopinn
   git init
   git add .
   git commit -m "Initial import"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. Open `notebooks/biopinn_train.ipynb` on your computer, find the setup
   cell near the top that sets `GITHUB_REPO_URL = "https://github.com/..."`,
   and change that value to point at **your** repo instead.
4. Upload the edited notebook to [Google Colab](https://colab.research.google.com)
   (File → Upload notebook), or upload it to Google Drive and open it with
   Colab from there.

### Option B — no GitHub, upload the code directly to Drive

1. Zip just the `biopinn/` folder again (or reuse `biopinn.zip`) and upload
   it to your Google Drive, e.g. into `My Drive/biopinn.zip`.
2. Open `notebooks/biopinn_train.ipynb` in Colab (Upload notebook, or open
   it from Drive).
3. In the notebook's setup cell (the one that would normally `git clone`
   using `GITHUB_REPO_URL`), replace the clone step with:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip -q "/content/drive/MyDrive/biopinn.zip" -d /content/
   %cd /content/biopinn
   !pip install -e .
   ```
   (adjust the path to wherever you uploaded the zip). It's fine that the
   notebook mounts Drive again later in its own "Mount Google Drive" cell —
   Colab just reuses the existing mount. The rest of the notebook is
   unchanged — it imports the shared `src/` package exactly the same way
   either option gets it onto the machine.

### Run the notebook

1. In Colab: **Runtime → Change runtime type → T4 GPU** (or any available
   GPU), then **Save**.
2. **Runtime → Run all.**
3. The notebook will, in order: install dependencies, mount your Google
   Drive (you'll be asked to authorize this — click through the prompts),
   generate the synthetic training dataset, and train the network in two
   phases. Progress/loss prints as it goes.
4. Find the `QUICK_TEST` toggle a few cells down (in the "Mount Google
   Drive & load config" section). It **defaults to `True`**, which runs a
   small few-minute sanity version of the whole pipeline — good for
   confirming everything works before committing to a long run. Set it to
   `False` and re-run for the real, full-scale run (2,000 simulated
   tumors; budget roughly 1–3.5 hours total on a T4, mostly data
   generation). Free Colab sessions cap out around 12 hours with no
   built-in resume, so if a run might exceed that, either use Colab Pro,
   generate the dataset locally first (see "Training locally instead of
   Colab" below) and upload it, or reduce `n_simulations` in
   `configs/default_config.yaml`.
5. When it finishes, it has saved everything to
   `My Drive/BIOPINN_outputs/` (the `DRIVE_OUTPUT_DIR` variable in the same
   cell as `QUICK_TEST`, change it there if you want a different location):
   - `BIOPINN_outputs/artifacts/biopinn_model.pt` (the trained network)
   - `BIOPINN_outputs/artifacts/normalization_stats.json` (small file the network needs alongside it)
   - `BIOPINN_outputs/artifacts/training_history.json` (loss curves, only needed for the manuscript figure pack in step 6)
   - `BIOPINN_outputs/data/processed/*.npz` (the train/validation/test data split)

You can close the browser tab once training is running — Colab keeps
executing — but check back periodically, since free Colab sessions can
time out after a few hours of inactivity.

### Training locally instead of Colab

If you have your own NVIDIA GPU, you don't need Colab at all —
`notebooks/biopinn_train_local.ipynb` is a simplified version of the same
notebook with no Colab/Drive branching, writing straight into this
checkout's own `artifacts/` and `data/` folders. Open it with
`jupyter lab` (or in VS Code / PyCharm) instead of Colab, then run all
cells the same way. It has the same `QUICK_TEST` toggle.

If PyTorch reports no GPU on your machine, it's almost always the
CPU-only build — reinstall with the CUDA build matching your driver:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(replace `cu121` with whatever CUDA tag matches your GPU driver, per
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) —
`nvidia-smi`'s "CUDA Version" is a ceiling, not an exact match
requirement, so a slightly older `cuXXX` tag than that number is normal
and fine).

**Generating the dataset on Windows:** both training notebooks' data
generation cell always runs single-threaded on Windows, even with
multiple CPU cores, because Windows can't safely parallelize
`ProcessPoolExecutor` work from inside a Jupyter cell (see
Troubleshooting, step 10, for why). For real multi-core speed on Windows,
generate the dataset as a standalone script first instead of running that
cell:

```bash
python scripts/generate_dataset.py --experiment experiment_1 --n-jobs 8
```

(drop `--experiment` for the full 2,000-sim production dataset; `--n-jobs`
defaults to all CPU cores). Then open the training notebook, set
`DATA_ALREADY_GENERATED = True` next to `QUICK_TEST`, and run the rest of
the notebook — it loads the script's output instead of regenerating it.

---

## 5. Bring the trained model back to your laptop

**If you trained locally** (the "Training locally instead of Colab"
section above), skip this step — the notebook already wrote everything
straight into this checkout's own `artifacts/` and `data/processed/`.

**If you trained on Colab**, download from `My Drive/BIOPINN_outputs/`
into your local `biopinn/` folder:

| From Drive (`BIOPINN_outputs/...`) | To (local path) |
|---|---|
| `artifacts/biopinn_model.pt` | `biopinn/artifacts/biopinn_model.pt` |
| `artifacts/normalization_stats.json` | `biopinn/artifacts/normalization_stats.json` |
| `artifacts/training_history.json` | `biopinn/artifacts/training_history.json` (optional — only needed for the manuscript figure pack, step 6) |
| `data/processed/` (train.npz, val.npz, test.npz, sim_params.json) | `biopinn/data/processed/` |

After this, `artifacts/` and `data/processed/` should each contain real
files (not just the placeholder `.gitkeep`).

From here on, **nothing retrains locally** — every script and the
dashboard just load these files and run instantly on CPU.

---

## 6. Run the local analysis scripts

Make sure your virtual environment is active (`source .venv/bin/activate`),
then from inside `biopinn/`, run any of these — order doesn't matter:

```bash
# Six-metric accuracy report against the held-out test set
python scripts/run_evaluation.py

# Trains a quick comparison baseline locally and reports the physics-vs-no-physics improvement
python scripts/run_ablation.py

# Finds the best nanoparticle size + dose at 4 different tumor sizes
python scripts/run_optimization.py

# Generates every figure (heatmaps, penetration plots, comparison charts...) into results/figures/
python scripts/make_figures.py
```

Each one prints a report to your terminal and saves results (JSON/CSV/PNG)
under `results/`. `run_ablation.py` takes the longest since it trains one
extra small model locally (still CPU-only, a few minutes).

If you're preparing a research paper or report, see the next section —
there's a dedicated script that generates a full, captioned figure and
table pack in one command.

---

## 7. Generate figures & tables for a research paper

`scripts/generate_results.py` produces a complete, publication-ready
Results & Discussion asset pack in one command, formatted per **APA (7th
edition)** conventions — every figure and table a manuscript needs,
computed from your actual trained model and data, no manual chart-making
or reformatting required.

```bash
python scripts/generate_results.py
```

This takes a few minutes on an ordinary CPU (it re-solves several reference
simulations and runs a small optimization search along the way). Everything
is written to `results/paper/`.

**Numbering scheme:** add `--numbering sequential` for plain APA
journal-article numbering (Table 1, Figure 1...) instead of the default
`--numbering chapter` (Table 4.1, Figure 4.1... — a dissertation-chapter
convention, also valid APA style). Filenames never collide between the two
modes, so you can run the command twice (once per value) and get both sets
side by side in `results/paper/`.

**10 figures**, each saved as both a PNG and a PDF at 300 DPI (use the PDF
for LaTeX/Word, the PNG for quick previews or slides). Per APA figure
convention, **no caption is drawn inside the image itself** — only axis
labels, legend, and data — the caption text lives in a separate
`FIGURE_CAPTIONS_<mode>.md` file (bold "Figure N" + italicized title, matching
APA style), for you to place under each figure when you drop it into your
paper:

| File (chapter mode; drop the `4_` for sequential mode) | What it shows |
|---|---|
| `fig_4_1_concentration_heatmap` | Drug concentration across space and time, one heatmap |
| `fig_4_2_radial_concentration_profiles` | Concentration vs. radial distance, overlaid at six time points |
| `fig_4_3_pinn_vs_fdm_profile_comparison` | Model prediction vs. the ground-truth reference simulation at 24hr, with an error panel underneath |
| `fig_4_4_predicted_vs_reference_scatter` | Accuracy scatter plot across the entire held-out test set, R² annotated |
| `fig_4_5_training_loss_convergence` | How each training loss term dropped over time (needs `training_history.json`, step 5) |
| `fig_4_6_penetration_depth_vs_time` | Penetration depth over time, five nanoparticle sizes compared |
| `fig_4_7_spatial_viability` | Cell survival by radial position, with the tumor's three zones shaded and labeled |
| `fig_4_8_cytotoxicity_evolution` | Cell death progressing across five time points, side by side |
| `fig_4_9_heterogeneous_vs_homogeneous` | Whether the three-zone tumor model actually changes the predicted drug spread vs. treating the tumor as uniform |
| `fig_4_10_effectiveness_surface` | Best nanoparticle size + dose combination, visualized as a heatmap with the optimum marked |

**9 tables**, each saved as its own CSV and also compiled together into one
Word document, formatted as APA's "three-line table" (no vertical rules, no
shading — just a rule above the header, below the header, and below the
last row), with a bold "Table N" + italicized title above each table and
any footnote as an italicized "Note." label + regular text below it:

- `table_4_1.csv` through `table_4_9.csv` — one file per table
- `BIOPINN_results_tables_<mode>.docx` — all nine tables in a single
  document, ready to copy straight into a paper

**`results_manifest_<mode>.json`** — records exactly what parameters,
config, and source data went into every figure and table, so every number
in the pack is traceable back to how it was computed.

**Requirements:** the trained checkpoint (step 5) is always required.
`artifacts/training_history.json` (also step 5) is needed specifically for
the training-loss figure (Fig 4.5) — if it's missing, the script stops
immediately with a clear message rather than silently producing an
incomplete pack (see the Troubleshooting section below). Table 4.8 (the
physics-informed vs. plain-network comparison) is the one piece that's
optional: it only appears if you've already run `python
scripts/run_ablation.py` once, since that's what trains the comparison
network it needs. Every other figure and table is produced either way.

**Using the output in a paper:**
- LaTeX: `\includegraphics{fig_4_1_concentration_heatmap.pdf}`, with the
  matching caption copied from `FIGURE_CAPTIONS_<mode>.md`
- Word/Google Docs: drag in the `.png` files (add the caption from
  `FIGURE_CAPTIONS_<mode>.md` underneath), or copy tables straight out of
  `BIOPINN_results_tables_<mode>.docx`
- Spreadsheet: import any individual `table_4_*.csv`

Point it at a different config the same way as the other scripts, e.g. the
small dev-scale config for a fast end-to-end check:

```bash
python scripts/generate_results.py --experiment experiment_1
python scripts/generate_results.py --experiment experiment_1 --numbering sequential
```

**One result worth understanding before writing it up:** Table 4.6 and
Figure 4.9 test whether the tumor's three-zone structure (proliferating
rim / quiescent zone / necrotic core) meaningfully changes the predicted
drug spread compared to treating the tumor as uniform. At a high,
fully-saturating drug dose, that difference can shrink to nearly nothing —
that's a real, physically-expected property of the underlying model (a
saturated tumor has little contrast left to detect), not a script error.
If your output there looks smaller than you expected, check
`results_manifest_<mode>.json` for the exact parameters used before drawing
conclusions in a "Discussion" section.

---

## 8. Launch the interactive dashboard

```bash
python scripts/run_dashboard.py
```

Then open **http://127.0.0.1:8000** in your browser.

You'll see parameter sliders on the left (tumor radius, nanoparticle
diameter, dose, decay rate, simulation duration) — moving any of them
re-queries the trained model and updates every chart on the right almost
instantly. The evaluation, optimization, and ablation panels take a little
longer to appear the first time (they're computed once in the background
and cached) — that's expected, not a bug; refresh isn't needed, they'll
fill in on their own.

To stop the dashboard, go back to the terminal and press `Ctrl+C`.

Useful flags:

```bash
python scripts/run_dashboard.py --port 8080          # use a different port
python scripts/run_dashboard.py --experiment experiment_1   # point at a different config
```

---

## 9. Everyday quick-reference

Once steps 1–5 are done, this is all you need on a normal day:

```bash
cd biopinn
source .venv/bin/activate
python scripts/run_dashboard.py
```

Full command cheat sheet:

| Command | What it does |
|---|---|
| `pytest tests/ -v` | Run the automated test suite |
| `python scripts/run_evaluation.py` | Six-metric accuracy report |
| `python scripts/run_ablation.py` | Physics-informed vs. plain-neural-network comparison |
| `python scripts/run_optimization.py` | Best nanoparticle size + dose per tumor size |
| `python scripts/make_figures.py` | Save all publication figures |
| `python scripts/generate_results.py` | Save the full manuscript figure + table pack to `results/paper/` |
| `python scripts/run_dashboard.py` | Launch the interactive dashboard |

---

## 10. Troubleshooting

**Data generation crashes with `BrokenProcessPool` in a notebook on Windows**
Windows can't safely parallelize `ProcessPoolExecutor` work directly from a
Jupyter cell — each worker process tries to re-import the Jupyter kernel
launcher as `__main__` to bootstrap itself and fails, since there's no real
script file there to protect it with an `if __name__ == "__main__":` guard.
Both training notebooks already work around this by defaulting to
`n_jobs=1` (single-threaded) on Windows automatically, so the generation
cell itself won't crash — it'll just be slower than on Linux/Mac/Colab.

For real multi-core speed on Windows, don't generate from the notebook at
all: run it as a standalone script instead, which parallelizes safely
because it's a real script with a proper `__main__` guard:

```bash
python scripts/generate_dataset.py --experiment experiment_1 --n-jobs 8
```

Then set `DATA_ALREADY_GENERATED = True` next to `QUICK_TEST` in the
notebook and run the rest of it — it loads this script's output instead of
regenerating it. See "Training locally instead of Colab" in step 4 for
more detail.

**"No checkpoint found at artifacts/biopinn_model.pt"**
You skipped or haven't finished step 5. The scripts (and dashboard) only
read trained artifacts, they never train anything themselves (except
`run_ablation.py`'s small local comparison model). Go back to step 4/5.

**`generate_results.py` fails with "No training history found"**
Download `artifacts/training_history.json` from Drive too (step 5) — it's a
small file the notebook now saves alongside the checkpoint, needed only for
the training-loss figure in the manuscript pack.

**`pip install` fails or hangs on PyTorch**
Slow/unstable internet is the usual cause — just retry `pip install -r
requirements.txt`. If you're on an ARM Mac or an unusual platform and it
still fails, install PyTorch first by itself following the exact command
for your system from [pytorch.org/get-started](https://pytorch.org/get-started/locally/),
then re-run `pip install -r requirements.txt`.

**Colab: it seems to hang**
Restart the runtime (Runtime → Restart runtime) and run again — this is
usually a one-off hiccup. Also double check Runtime → Change runtime type is
actually set to a GPU.

**Training crashes with `CUDA out of memory`**
This is a real, reproducible memory limit, not a hiccup — training is
full-batch (every iteration processes the entire train split at once, no
mini-batching), and the physics loss's second-order autograd over millions
of collocation points can genuinely exceed GPU memory at full 10,000-sim
production scale (seen in practice on a 22GB GPU).
`configs/default_config.yaml`'s `training.max_points_per_chunk` (default
`1000000`) already guards against this: it computes and backpropagates each
loss term in point-count-bounded chunks instead of one giant forward pass —
the resulting gradient (and therefore what the model learns) is
mathematically identical to the unchunked version, only peak memory
differs. If you still hit `CUDA out of memory` with the default in place,
lower it (e.g. `500000` or `250000`) and re-run; if you have VRAM to spare
and want marginally less looping overhead, raise it, or delete the key
entirely to restore the original single-shot behavior.

**Training looks stuck -- no output for a very long time**
`train_adam_phase`/`train_lbfgs_phase` now print every single epoch/closure
by default (`log_every=1`), specifically so this doesn't happen -- if you're
not seeing a new line every few seconds, something is genuinely off rather
than just a sparse logging interval. Check `nvidia-smi` (or Colab's resource
panel, or Task Manager if training locally) for GPU/CPU utilization -- if
it's non-zero and stable, it's still actively working, just slower than
expected (e.g. a weaker local GPU); if it's near zero, that's a real hang
worth investigating rather than waiting out. If you want *less* console
noise instead (25,000 lines is a lot to scroll through), raise `log_every`
back up when calling `train_adam_phase`/`train_lbfgs_phase` directly, or
edit their default in `src/train.py`. `configs/default_config.yaml`'s
`dataset.n_simulations: 2000` (scaled down from an earlier 10,000-sim
default specifically because full-batch training at that scale could take
~97 hours worst-case, past any single Colab session) should already keep a
full run within a single session; if it's still running long, the biggest
remaining lever is `training.adam.iters`/`training.lbfgs.iters`.

**Dashboard opens but panels never fill in / show a "not ready" message forever**
The Evaluation/Optimization/Ablation panels do real computation the first
time (re-solving physics simulations for the held-out test set) and can
take a minute or two, especially on an older CPU — leave the tab open. If
it truly never finishes, check the terminal running
`scripts/run_dashboard.py` for an error message.

**"Address already in use" when launching the dashboard**
Something else is using port 8000. Either stop that process or run
`python scripts/run_dashboard.py --port 8080` and open
`http://127.0.0.1:8080` instead.

**Tests fail right after a fresh install**
Make sure the virtual environment is active (`source .venv/bin/activate`)
and that both `pip install -e .` and `pip install -r requirements.txt`
completed without errors — scroll up in the install output for the actual
failure.

---

## 11. Where to go next

`README.md` (in the unzipped folder) has the full technical reference: a
module-by-module breakdown of every file, the physics being modeled, the
configuration system, and the specific accuracy/hypothesis targets the
model is checked against. This guide gets you running; the README explains
what's actually happening under the hood.
