# Nano-Tumor Database clearance adaptation

This branch lets BIOPINN take a drug surface concentration that decays with
plasma clearance, with the clearance rate taken from real in-vivo data. It
also adds a pipeline for that data and fixes metrics that stop making sense
once the surface concentration decays.

## TL;DR

| What | Before | Now (default) |
|---|---|---|
| Surface boundary | `C(R,t) = C0` | `C(R,t) = C0·exp(-k_el·t)` |
| Sampled parameters | 5 (R, d_NP, C0, k_d, t_max) | 6 (+ `k_el_per_hr` in [0.0136, 0.381] hr⁻¹) |
| Model input | 7 columns | 8 columns (`model.input_dim: 8`) |
| Paper baseline | — | `k_el_per_hr: 0.0684` (NTD median, t½ ≈ 10 h) |
| Penetration threshold | `C > 0.1·C0` | `C > 1 µM` (absolute, = IC50); relative mode selectable |
| Under-dose zone | final-time `C < 0.1 µM` | per-radius **peak** `C < 0.1 µM` (identical for a constant boundary) |

`--experiment experiment_legacy_5param` switches all of the above back to the
old behavior. Use it for checkpoints trained before this change.

## Data source

**Nano-Tumor Database, 2023 version**: Chen et al. (2023), *ACS Nano*
17(20):19810–19831, doi:10.1021/acsnano.3c04037. Data mirror:
<https://github.com/UFPBPK/Nano-Tumor-Database-Version2023>.

The input we used is a tidy long-format table with 12,590 rows (534 in-vivo
mouse datasets × tissue × time point). Each row is the percent of injected
dose (`%ID`) in Tumor, Heart, Liver, Spleen, Lung, Kidney, Blood, Plasma or
Serum. Each row also carries nanoparticle metadata (size as TEM and/or
hydrodynamic diameter, shape, zeta potential, PDI, core material, targeting),
the tumour model, the tumour volume and the dose. The raw file is **not
committed**: it is third-party data (~6 MB) and the repo git-ignores data. Put
it at `data/external/ntd/raw/` and run:

```bash
python scripts/adapt_ntd.py --input data/external/ntd/raw/<file>.csv --fdm-demo   # or .xlsx (needs openpyxl)
```

The small derived tables are committed in `data/external/ntd/` (see its
README).

### What the pipeline does

1. **Parse fields.**
   - Size: the hydrodynamic diameter is preferred, falling back to TEM,
     because Stokes–Einstein needs the hydrodynamic size. The database's own
     "numeric size" column prefers TEM, and switching changes which 10–200 nm
     bin 26 datasets fall into.
   - Tumour volume becomes an equivalent-sphere radius. Dose (mg/kg) and body
     weight are parsed too.
   - QC: 42 rows with more than 100 %ID (impossible for a whole organ) are
     dropped.
2. **Define scope.** IV route, spherical particles, 10–200 nm. That leaves 357
   of 534 datasets.
3. **Fit plasma clearance.** For each dataset, take the blood, plasma or serum
   series with the most time points and fit `ln C = a − k_el·t` over 0–72 h,
   which is BIOPINN's `t_max` window. A fit counts as good if k_el > 0,
   R² ≥ 0.8 and there are at least 3 points; **152** in-scope datasets pass.
   The results:
   - k_el: 5th percentile 0.0136, median 0.0684, 95th percentile 0.381 hr⁻¹
   - median half-life ≈ 10 h

   The p5–p95 interval is the new LHS range, and the median is the paper
   baseline.
4. **Tumour uptake.** Cmax, Tmax, AUC and %ID at 24/48/72 h per dataset, plus
   medians per 10–50 / 50–100 / 100–150 / 150–200 nm bin. These are only for
   a qualitative, rank/direction check against BIOPINN's volume-averaged
   tumour concentration.

## Limitations: read before citing

- **No intratumoral spatial data.** `%ID` is a whole-organ amount. It has no
  radial coordinate and is not a local µM concentration. Converting it would
  need drug molecular weight and loading, which the database doesn't carry.
  So **none of it is used as PINN training labels**; `L_data` is still the
  synthetic FDM field.
- **Different system.** The mouse tumours have an equivalent radius of about
  2.3–10 mm (median ≈ 3 mm), are vascularised, and take up the carrier. BIOPINN
  models a 100–500 µm avascular spheroid with surface-only diffusion of the
  drug. The database only informs how the **surface/plasma concentration
  decays in time**.
- **Mono-exponential fit.** Early distribution phases are folded into one
  effective rate over 0–72 h. A two-phase fit would be more faithful.
- **Size doesn't predict clearance here.** Spearman ρ(d_NP, k_el) = −0.01
  (p = 0.87). So k_el is sampled independently of d_NP rather than regressed
  on it.
- **Only a weak external check.** ρ(d_NP, tumour %ID at 24 h) = −0.10
  (p = 0.085, n = 299). The direction agrees with "smaller particles do
  better" but the effect is weak.
- **Uniform LHS over k_el.** Sampling is uniform over a ~28× range, so the
  training set is weighted toward fast clearance (sample median ≈ 0.2 hr⁻¹ vs
  the NTD median 0.068 hr⁻¹). Log-uniform sampling would match the data
  better; this isn't implemented yet.

## Code changes

- `src/data_pipeline.py`
  - `PARAM_ORDER` now includes `k_el_per_hr`.
  - `param_order(config | norm_stats)` picks 5 or 6 columns. The inference
    paths use the checkpoint's `normalization_stats.json` keys, so old
    checkpoints keep working.
  - `_solve_one` passes `k_el_per_hr` to `solve_fdm`, which already supported
    it; a 5-value row means k_el = 0.
  - Surface BC targets are `C0·exp(-k_el·t)`, which is exactly the FDM
    boundary.
  - `check_input_dim` fails fast if `model.input_dim` doesn't match the
    columns.
- `src/losses.py`: denormalisation follows the stats layout. The PDE residual
  itself doesn't depend on k_el.
- `src/biology.py`, `src/evaluate.py`, `src/optimize.py`, `src/results.py`,
  `scripts/make_figures.py`, `app/server.py`:
  - Every `sim_params` dict and every `solve_fdm` call carries `k_el_per_hr`.
    It defaults to `paper.baseline.k_el_per_hr`, or 0 when k_el is disabled;
    `sim_params.get("k_el_per_hr", 0.0)` still works.
  - `resolve_test_simulations` treats a missing key as 0, which is how the
    old datasets were generated.
  - The dashboard shows a k_el slider only for k_el-conditioned models.
- Thresholds:
  - `evaluation.penetration_threshold` has `mode: absolute | relative`,
    `absolute_uM: 1.0` and `relative_fraction: 0.1`.
  - The under-dose threshold was already absolute (0.1 × IC50 = 0.1 µM). It is
    now `biology.subtherapeutic_threshold_uM`, and
    `biology.subtherapeutic_basis: peak | final` chooses what it is compared
    against. With a decaying boundary the final-time field goes to ~0
    everywhere, so "final" would call the whole tumour under-dosed.
- Notebooks: the validation-plot cell builds its parameter row from
  `param_order(norm_stats=...)`.
- Configs:
  - `default_config.yaml` holds the new range, baseline, input_dim and
    thresholds.
  - `experiment_legacy_5param.yaml` restores the old behavior.
  - `experiment_ntd.yaml` is a dev-scale run with the data-derived values
    pinned and its own paths.

## Verification on this branch

- `pytest tests/`: **168 passed**. That is 160 existing tests, several updated
  for the 8-column layout and the decaying BC target, plus 8 new ones in
  `tests/test_k_el_adaptation.py`. The new tests cover the legacy layout
  round-trip, 5-key stats with a 7-input model, the decaying boundary in the
  FDM output, threshold modes, and peak vs final.
- Dev-scale end-to-end run with `experiment_ntd` (40 sims, N_r = 50):
  - **Data:** `generate_dataset.py` gives `data_X` of shape (11200, 8).
  - **Training:** 300 Adam + 25 L-BFGS iterations on CPU in about 50 s.
    Losses (unweighted): data 0.120 → 0.040, physics 388 → 0.047,
    boundary 0.153 → 0.067, Neumann 7e-4 → 7e-7. Final val_data 0.030,
    val_phys 0.033.
  - **Downstream scripts:** `run_evaluation.py`, `run_optimization.py`,
    `make_figures.py` and `generate_results.py` all ran to completion. The
    dev model is far too small to meet any accuracy target; this run only
    checks plumbing.
- FDM demo (R = 400 µm, d_NP = 100 nm, C0 = 10 µM, k_d = 0.01 hr⁻¹):

  | k_el | Volume-averaged C at 24 h | at 72 h |
  |---|---|---|
  | 0 | 9.71 µM | 9.72 µM |
  | 0.0684 hr⁻¹ (NTD median) | 2.30 µM | 0.087 µM |

  So clearance changes exposure, viability and the optimisation results
  substantially. **Every paper number has to be regenerated from a retrained
  model.**

### Consequences to expect when you retrain

- **H2** is evaluated at t = 72 h only. With k_el at the baseline, the surface
  concentration at 72 h (≈ 0.07 µM) is already below the 1 µM threshold, so
  both penetration depths are 0 and H2 can't pass. Consider measuring
  penetration as the maximum over time, or at the time of peak exposure.
- The H3 demo regime (C0 = 0.105 µM, `make_figures.py` / `test_optimize.py`)
  was tuned for a constant boundary, so it is pinned to k_el = 0.
- The ablation baseline (`experiment_2`) and any existing checkpoint must be
  retrained, or loaded with `experiment_legacy_5param`.

## Known issue, not fixed: the hard-IC transform caps the surface value

`model.py` uses `C_NN = sigmoid(f)·t_norm`. Because sigmoid ≤ 1, the model
**can't output more than t_norm anywhere**, including at r = R. There the
Dirichlet target is 1 (or `exp(-k_el·t)`) from t = 0 onward.

- With a constant boundary and t ~ U(0,1), the boundary loss can't go below
  E[(1 − t)²] = **1/3**. That matches the stuck ~0.4 `bc_loss` plateau
  described in `configs/default_config.yaml` under `loss.w_bc`. Raising
  `w_bc` can't fix it.
- With the NTD k_el values the floor is about 0.06. The dev runs sit right on
  it: val boundary MSE was 0.067–0.069 against a floor of 0.064.

**Candidate fix, tried and not committed.** Replace the gate with
`g(t) = 1 − exp(−t_norm/τ)`, τ ≈ 0.01. This keeps C(r,0) = 0 exactly but
caps only the first few % of the window, which is the unavoidable IC/BC corner
discontinuity. It drops the theoretical floor from 0.064 to 0.0024.

A/B test on the dev dataset (2000 Adam + 50 L-BFGS iterations, same seed):

| | Linear gate | Exponential gate (τ = 0.01) |
|---|---|---|
| Train boundary loss | 0.064 | 0.050 (below the old floor) |
| **Val boundary MSE** | 0.0690 | 0.0683 (≈ 1% better) |
| val_data | 0.0295 | 0.0353 (worse) |
| val_phys | 0.032 | 0.041 (worse); train physics loss 0.034 → 0.27 |

The steep gate makes ∂C/∂t stiff near t = 0. Since the short run showed no
clear benefit on validation, the change was **not applied**. Before adopting
it, try it on a full-scale run, possibly with τ ≈ 0.02–0.05 or with the
boundary enforced by construction instead. The patch is small: an
`ic_time_gate()` helper in `src/model.py` plus a `model.hard_ic_time_scale`
config key.

## Paper vs. code discrepancies found

These refer to the DSTF 2026 manuscript "BioPINN: A Physics-Informed Neural
Network Surrogate…" compared with this repo at commit `3e4f0e3`.

| Item | Manuscript | Code |
|---|---|---|
| Network | 5 × 64, inputs (r_norm, t_norm) | 5 × 96, 7 inputs (now 8) |
| Output transform | σ(f)·t_norm·C0 | σ(f)·t_norm (normalised output) |
| Loss weights | w_data = 1, w_BC = w_IC = 10 | w_data = 6, w_bc = 20, w_neu = 10, w_ic = 10 |
| Points per simulation | N_data = 3000, N_col = 5000–10000 | 600 data, 1500 collocation |
| O₂ consumption k_O2 | 0.1 hr⁻¹ | 1.0 hr⁻¹ |

- **Zone boundaries at R = 400 µm.**
  - Paper: quiescent from 0.85R, necrotic from 0.50R.
  - Code: 293 µm (0.73R) and 164 µm (0.41R).
  - With the paper's k_O2 = 0.1 there would be no hypoxic zones at all (the
    minimum O₂ is 2.7%).
- **Tissue diffusivity.** Table 4 gives D_eff,tissue = 35 µm²/hr for 100 nm.
  The code gives 2725 (rim) to 6812 (core) µm²/hr.
- **Penetration and zone concentrations.** Tables 4–6 give a 91 µm
  penetration depth for 100 nm and a core concentration of 1.6 µM. With the
  code's physics the R = 400 µm tumour reaches 9.7/10 µM volume-averaged by
  24 h, so these values can't be reproduced. The README notes that the
  manuscript's figures were illustrative `[SAMPLE]` values.
- **Abstract vs. Table 10.** The abstract says the "largest diameter
  particles" work best; Table 10 and the conclusion say the smallest (10 nm).
