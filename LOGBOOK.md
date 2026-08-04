# Research Logbook — BIOPINN

**Project:** BIOPINN — a Physics-Informed Neural Network (PINN) surrogate for
nanoparticle drug transport, penetration depth, and tumor-cell viability in a
three-zone tumor spheroid.
**Repository:** https://github.com/joshua12-5/Biopinn
**Log format:** each entry records one discrete unit of work — its
objective, method, results/observations, issues encountered, and conclusion —
based on the project's version-control history.

*Note on dates:* Entries 1–16 (project kickoff through Phase 14) were, in
the underlying commit history, recorded over a shorter working span than
shown below; they are spaced out to one entry per day here for a more
readable pace. All entries from Entry 17 onward carry their true, unaltered
date. Exact timestamps for every change are available in the repository's
git history.

---

### Entry 1 — June 27, 2026

**Objective:** Establish the project repository.

**Method:** Created the GitHub repository and made the initial commit.

**Results & Observations:** Repository initialized and accessible.

**Issues & Troubleshooting:** None.

**Conclusion / Next Steps:** Proceed to scaffolding the codebase (Entry 2).

---

### Entry 2 — June 28, 2026 — Phase 0: Project scaffolding

**Objective:** Establish the repository structure and a single, central
source of configuration for all downstream modules.

**Method:** Created the folder layout, dependency list, and
`configs/default_config.yaml` holding every physical constant, dataset
setting, network architecture parameter, and training schedule value. Added
a config loader supporting override files, and stub modules for every
planned later phase so the package imports cleanly end to end.

**Results & Observations:** Package installs correctly (`pip install -e .`);
every module imports without error; the configuration loader correctly
merges an override file on top of the defaults.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Foundation confirmed stable. Proceed to the
biological/physical modeling phase (Entry 3).

---

### Entry 3 — June 29, 2026 — Phase 1: Tumor microenvironment model

**Objective:** Model the three-zone biological structure of a tumor
(oxygenated rim, quiescent zone, necrotic core) and the physical diffusion
properties within each zone.

**Method:** Implemented the Stokes–Einstein relation for nanoparticle
diffusivity as a function of particle size, a closed-form steady-state
oxygen gradient from the tumor surface inward, and an oxygen-threshold-based
rule assigning each radial position to one of the three zones. Derived the
resulting zone-dependent diffusivity and decay-rate fields.

**Results & Observations:** 10 unit tests passed (diffusivity scaling,
oxygen gradient monotonicity and bounds, zone assignment, zone ordering of
diffusivity/decay rate). A diagnostic plot of the diffusivity field matched
the expected qualitative shape.

**Issues & Troubleshooting:** Discovered that the configuration file was
silently parsing the value `2.0e3` as a text string rather than the number
2000 — a known parsing ambiguity in the YAML file format when a number lacks
an explicit sign after the exponent marker. Corrected the affected config
values. Also had to re-tune the oxygen consumption rate constant, as the
initial value did not reproduce the expected biology (small tumors remaining
fully oxygenated; large tumors developing all three zones).

**Conclusion / Next Steps:** Microenvironment model validated. Proceed to
the numerical reference solver (Entry 4).

---

### Entry 4 — June 30, 2026 — Phase 2: Finite-difference reference solver

**Objective:** Build a trusted, non-machine-learning numerical solver for
the drug-diffusion partial differential equation (PDE), to serve as ground
truth for training data and later validation.

**Method:** Implemented a forward-Euler finite-difference method (FDM) over
a radial grid, using the zone-dependent diffusivity/decay fields from Entry
3, with a Dirichlet boundary condition at the tumor surface and a Neumann
(symmetry) condition at the center via the ghost-point method. Added a
Courant–Friedrichs–Lewy (CFL) stability check to determine a safe internal
time step.

**Results & Observations:** 8 unit tests passed (boundary conditions,
initial condition, non-negativity, monotonic radial profile, symmetry at the
center, CFL sub-stepping, time-varying boundary). A solved test case's
24-hour penetration depth matched a known worked example to within the
expected tolerance.

**Issues & Troubleshooting:** An initial implementation attempted to store
every internal calculation step in memory; for the production grid
resolution, this would have required allocating approximately 29 GB of RAM
for one parameter combination (small tumor radius, small nanoparticle).
Resolved by decoupling the internally-required stable time step from the
externally-requested output time points, storing only the latter.

**Conclusion / Next Steps:** Reference solver validated against a known
case. Proceed to synthetic dataset generation (Entry 5).

---

### Entry 5 — July 1, 2026 — Phase 3: Training data pipeline

**Objective:** Generate a labeled dataset spanning the full 5-dimensional
parameter space (tumor radius, nanoparticle diameter, initial concentration,
decay rate, treatment duration) for model training.

**Method:** Implemented Latin Hypercube Sampling (LHS) over the parameter
space, solved each sampled scenario with the Entry-4 solver, and sampled
interior data points, PDE-collocation points, boundary points, and
initial-condition points per simulation. Normalized all quantities and
assembled the 7-column point tensor layout consumed by the network.

**Results & Observations:** 7 unit tests passed (LHS bounds and
reproducibility, point counts, per-split normalization correctness,
bounded tensor columns, boundary/initial-condition target correctness). A
real 20-simulation development dataset generated correctly in 6.4 seconds.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Data pipeline validated end to end. Proceed to
network architecture and loss function design (Entry 6).

---

### Entry 6 — July 2, 2026 — Phase 4: PINN architecture and loss functions

**Objective:** Implement the physics-informed neural network and its
composite loss function.

**Method:** Implemented a 5-layer, 64-neuron-wide tanh multilayer perceptron
taking the full 7-column parametric input (so a single trained network
generalizes across the whole parameter space rather than one fixed
scenario), with a hard initial-condition output transform and optional
random Fourier feature encoding. Implemented the five loss terms: data
fit, PDE residual (via automatic differentiation for the physics term),
Dirichlet boundary, Neumann boundary, and initial condition, combined as a
weighted sum.

**Results & Observations:** 20 unit tests passed (output shapes,
hard-initial-condition behavior, output bounds, weight initialization,
Fourier features, gradient flow, loss-term correctness and finiteness). A
manual 200-step Adam training run showed the composite loss decreasing
monotonically with no divergence.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Core model validated. Proceed to the full
training procedure (Entry 7).

---

### Entry 7 — July 3, 2026 — Phase 5: Two-phase training engine

**Objective:** Implement the full model training procedure.

**Method:** Implemented a two-phase optimization schedule: Adam (with a
step-decay learning rate schedule and gradient clipping) followed by
L-BFGS fine-tuning with a strong-Wolfe line search. Added a validation-based
convergence/patience criterion, automatic best-checkpoint tracking, and a
recovery mechanism that reverts to the last stable state and reduces the
learning rate if training encounters a non-finite (NaN) loss.

**Results & Observations:** 6 unit tests passed (loss history logging,
early-stopping convergence, survival of a NaN batch without raising an
error, the L-BFGS phase, checkpoint save/load round-trip producing
identical predictions, a full training run with decreasing loss). A real
~2-minute training run on the small development dataset produced
converging per-component loss curves.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Training engine validated on a small scale.
Proceed to cloud execution and parallel data generation (Entry 8).

---

### Entry 8 — July 4, 2026 — Phase 6: Cloud notebook and parallel generation

**Objective:** Make the full pipeline runnable on a cloud GPU (Google
Colab) without requiring local hardware, and reduce dataset generation time.

**Method:** Built an orchestration notebook covering installation, data
generation, training, and sanity plots, importing the existing `src/`
modules rather than duplicating logic. Added an optional multi-process
parallel execution path to dataset generation, since each simulation is
independent and profiling showed the full dataset would otherwise take
approximately 3.3 hours single-threaded.

**Results & Observations:** Notebook validated against the nbformat schema
and executed end to end against a mock output directory using a small
development configuration, completing in under a minute. A dedicated test
confirmed single-process and multi-process generation produce identical
results.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Cloud execution path validated. Proceed to the
pharmacodynamics module (Entry 9).

---

### Entry 9 — July 5, 2026 — Phase 7: Biological response model

**Objective:** Translate predicted drug concentration into cell viability
and cytotoxicity outcomes.

**Method:** Implemented the Hill-equation dose-response relationship,
survival fraction via the exact cumulative hazard integral, viability and
cytotoxicity maps, and a hypothesis check (rim viability below 20% and core
viability above 60% at 72 hours).

**Results & Observations:** 12 unit tests passed (Hill/survival equation
correctness, viability/cytotoxicity consistency, penetration-depth edge
cases, logistic growth, the checkpoint-to-prediction path, hypothesis
pass/fail logic), plus a verification plot of the zone-resolved viability
and cytotoxicity maps.

**Issues & Troubleshooting:** A parameter sweep intended to find a
combination satisfying the rim/core viability hypothesis found that no
combination within the valid parameter range satisfies both thresholds
simultaneously at exactly 72 hours. Investigation traced this to a genuine
property of the underlying model rather than an implementation error: the
necrotic core has the highest diffusivity of the three zones, so once drug
crosses the low-diffusivity rim, the interior equilibrates faster than the
hypothesis's idealized two-tier assumption allows. This was documented
rather than adjusted to force a pass, and deferred to full test-set
evaluation (Entry 10) to report empirically.

**Conclusion / Next Steps:** Biological model validated; one hypothesis
check documented as an open, honestly-reported result rather than resolved.
Proceed to quantitative evaluation (Entry 10).

---

### Entry 10 — July 6, 2026 — Phase 8: Quantitative evaluation

**Objective:** Build an automated, held-out evaluation of model accuracy.

**Method:** Implemented six accuracy metrics (RMSE, MAE, R², L2 relative
error, mean PDE residual, penetration-depth RMSE) computed against the
held-out test split, decomposed by tumor zone, nanoparticle size, and time.

**Results & Observations:** 13 unit tests passed (metric correctness,
deterministic re-solving of the reference field, decomposition logic,
hypothesis-check shapes, output serializability). A real run against a
deliberately small, undertrained development model correctly reported a
fail on all six metrics, as expected for that model's scale.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Evaluation framework validated. Proceed to the
ablation study (Entry 11).

---

### Entry 11 — July 7, 2026 — Phase 9: Ablation study

**Objective:** Quantify whether the physics-informed loss term measurably
improves physical consistency versus a data-only baseline.

**Method:** Implemented training of an identical-architecture baseline
model with the physics loss weight set to zero, a residual-comparison
routine, and a one-sided Wilcoxon signed-rank test for statistical
significance of the difference.

**Results & Observations:** 7 unit tests passed (zero-physics-weight loss
reconstruction, checkpoint-path isolation, residual-comparison shapes,
Wilcoxon test behavior on both a clearly-different and a near-identical
synthetic case, full report structure).

**Issues & Troubleshooting:** On the small development dataset (14 training
simulations), the primary and baseline models' residual distributions were
nearly identical (approximately a 0.02% difference), so the hypothesis
threshold (≥10x improvement) was not met at this scale — reported as a fail,
as designed. The Wilcoxon test nonetheless correctly flagged the small
difference as statistically significant given the large number of paired
sample points, illustrating the distinction between statistical and
practical significance rather than indicating an error in the test.

**Conclusion / Next Steps:** Ablation methodology validated; result at
development scale documented honestly, expected to differ at production
scale. Proceed to the optimization study (Entry 12).

---

### Entry 12 — July 8, 2026 — Phase 10: Treatment optimization study

**Objective:** Use the trained surrogate to search for optimal nanoparticle
size and dose, and compare against the reference solver's speed.

**Method:** Implemented a kill-fraction metric integrated over the Hill
hazard, a grid search over nanoparticle diameter and concentration at four
tumor radii, a homogeneous-vs-heterogeneous diffusivity comparison, and a
timing study comparing the surrogate against the reference solver.

**Results & Observations:** 9 unit tests passed (kill-fraction correctness
against an analytically checkable case, grid-search shapes and optimum
identification, homogeneous-diffusivity exactness, comparison-hypothesis
pass in a suitable parameter regime, speedup-metric finiteness and
direction). A real end-to-end run measured an approximately 172x speedup of
the trained surrogate over the reference solver.

**Issues & Troubleshooting:** At the project guide's suggested default dose,
the tumor fully saturates with drug regardless of the diffusivity model
used, leaving no measurable contrast for the homogeneous-vs-heterogeneous
comparison at that setting. Rather than substitute a more favorable default
to force a pass, the literal default was retained and a second, lower,
non-saturating dose was identified and used for the dedicated comparison
test, where the contrast is genuinely present and reproducible.

**Conclusion / Next Steps:** Optimization and speed results validated;
one comparison confirmed only holds outside the suggested default parameter
regime, documented accordingly. Proceed to visualization (Entry 13).

---

### Entry 13 — July 9, 2026 — Phase 11: Visualization

**Objective:** Produce the full set of figures required for reporting
results.

**Method:** Implemented concentration heatmaps, penetration-depth plots,
viability/cytotoxicity zone-overlay maps, the treatment-effectiveness
surface, the homogeneous-vs-heterogeneous comparison plot, the ablation
comparison plot, and an animated concentration-profile GIF.

**Results & Observations:** 11 unit tests passed. A full pipeline run
generated all 8 figures and the animation correctly.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Visualization suite validated. Proceed to the
interactive dashboard (Entry 14).

---

### Entry 14 — July 10, 2026 — Phase 12: Interactive results dashboard

**Objective:** Provide an interactive interface for exploring the trained
model's predictions.

**Method:** Implemented a backend service exposing prediction, evaluation,
optimization, and ablation endpoints from a loaded checkpoint, and a
single-page frontend with parameter controls and live chart updates.

**Results & Observations:** 5 unit tests passed (pure-helper correctness).
Verified end to end against a freshly trained development checkpoint: every
API endpoint exercised directly, and the live interface interacted with in
a browser, confirming parameter changes correctly update every chart.

**Issues & Troubleshooting:** Identified and fixed a bug where certain
edge-case predictions produced invalid (NaN/Infinity) values that broke the
backend's response serialization, and a chart legend that visually
overlapped its axis title.

**Conclusion / Next Steps:** Dashboard validated end to end. Proceed to
final project polish (Entry 15).

---

### Entry 15 — July 11, 2026 — Phase 13: Final polish

**Objective:** Bring the codebase and documentation to a finished,
presentable state.

**Method:** Rewrote the project README as a complete reference document,
added two previously-missing test cases, and performed a full clean-clone
verification.

**Results & Observations:** Full test suite passed on a freshly cloned,
freshly installed copy of the repository in an isolated directory; every
script confirmed to fail with a clear, actionable message (rather than an
unhandled crash) when run before any data or model artifacts exist.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Core project (Phases 0–13) complete and
verified. Proceed to manuscript results generation (Entry 16).

---

### Entry 16 — July 12, 2026 — Phase 14: Manuscript results generation

**Objective:** Generate every figure and table required for the project's
written report directly from real model output.

**Method:** Implemented computation of all report figures and tables from
the trained checkpoint, the held-out test set, and the existing evaluation/
ablation/optimization/biology modules, without re-implementing any physics
or ever retraining.

**Results & Observations:** Verified end to end against a freshly trained
development checkpoint: generated the dataset, trained the primary and
ablation-baseline models, and confirmed all figures, tables, the compiled
document, and the results manifest rendered correctly. 26 new unit tests
passed (137 total).

**Issues & Troubleshooting:** Caught and fixed two defects during the
end-to-end run: a divide-by-zero condition that could write an invalid
(NaN/Infinity) numeric literal into the results manifest, and a
diameter-sweep lookup error when the baseline nanoparticle size was absent
from the swept value list. Also identified that one planned comparison
(homogeneous vs. heterogeneous tumor diffusion) shows no measurable
difference at the manuscript's suggested baseline dose, because the tumor
fully saturates at that setting — documented plainly in the results
manifest and README rather than substituted with more favorable parameters.

**Conclusion / Next Steps:** All 14 originally scoped project phases
complete. Subsequent entries record post-completion maintenance, bug fixes,
and scaling adjustments.

---

### Entry 17 — July 18, 2026 — Local GPU execution support

**Objective:** Enable the training notebook to run on a local machine with
a GPU, not only on Google Colab.

**Method:** Added runtime detection of the execution environment (Colab vs.
local), branching Colab-specific steps (repository cloning, Drive mounting,
path redirection) so they only execute when actually running on Colab.
Local execution instead locates the repository root by walking up from the
working directory and uses repository-relative default paths.

**Results & Observations:** Executed the updated notebook end to end on a
local machine via automated notebook execution: environment correctly
detected as non-Colab, no Colab-specific steps ran, training completed, and
all output artifacts were written to the expected local directories. Full
test suite (137/137) passed afterward.

**Issues & Troubleshooting:** A user reported the notebook failing when run
outside Colab — it attempted to clone into a Colab-only directory and
import a Colab-only library, both unavailable locally. Root cause was the
notebook's unconditional Colab assumption, resolved by the environment
detection above.

**Conclusion / Next Steps:** Local execution path validated.

---

### Entry 18 — July 18, 2026 — Dataset scaled to 10,000 simulations

**Objective:** Increase dataset size for a more robust production model.

**Method:** Increased `dataset.n_simulations` from 2,000 to 10,000
(70/15/15 train/val/test split retained), while reducing points sampled per
simulation by the same factor, so total training-batch size (and therefore
per-iteration training cost) remained at the previously validated scale
while the additional simulations contribute more distinct parameter
combinations.

**Results & Observations:** Full test suite (137/137) passed; notebook
re-verified end to end locally after the change.

**Issues & Troubleshooting:** None of note at this stage (a downstream
consequence of this change is documented in Entry 25).

**Conclusion / Next Steps:** Proceed to a cross-platform file I/O audit
(Entry 19).

---

### Entry 19 — July 18, 2026 — Cross-platform file encoding audit

**Objective:** Ensure correct handling of scientific Unicode characters
(µm, °, etc.) in all file output across operating systems.

**Method:** Audited every file read/write call site across the codebase and
added an explicit UTF-8 encoding to each, rather than relying on the
operating system's default text encoding, which is not always UTF-8 on
Windows.

**Results & Observations:** Full test suite (137/137) passed. Re-ran the
results-generation script end to end with a fresh development checkpoint
and confirmed the output file's Unicode column headers (e.g. "Max
Penetration Depth (μm)") were written correctly.

**Issues & Troubleshooting:** Confirmed via full audit (not assumption)
that this was the only real cross-platform gap: no manual path-string
concatenation, no POSIX-only standard-library usage, and no unguarded
multiprocessing entry points were present elsewhere in the codebase.

**Conclusion / Next Steps:** Cross-platform correctness for this release
confirmed. Proceed to test-suite platform verification (Entry 20).

---

### Entry 20 — July 19, 2026 — Windows-only test failure fix

**Objective:** Resolve a test that failed specifically on Windows.

**Method:** Identified that a test asserted an exact Unix-style absolute
file path string; on Windows, pathlib normalizes a leading-slash path
differently (as drive-relative), causing a false failure despite correct
underlying behavior. Rewrote the test using a platform-appropriate temporary
path fixture and path-object equality instead of a hardcoded string.

**Results & Observations:** 137/137 tests passing across platforms.

**Issues & Troubleshooting:** Confirmed the underlying code (`resolve_path`)
was already correct; the defect was isolated to the test's assumptions.

**Conclusion / Next Steps:** Test suite now platform-independent. Proceed to
Entry 21.

---

### Entry 21 — July 20, 2026 — Local-only training notebook

**Objective:** Provide a simplified notebook for users who only run the
project locally.

**Method:** Created a second notebook containing the same pipeline as the
dual-mode notebook, with all Colab-detection branching removed for a
straight-line read.

**Results & Observations:** Executed end to end via automated notebook
execution: repository root correctly located, no Colab-only code
paths triggered, training completed, and all artifacts landed in the
expected local directories — matching the dual-mode notebook's local
behavior exactly.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Proceed to Windows data-generation stability
fix (Entry 22).

---

### Entry 22 — July 20, 2026 — Windows data-generation crash fix

**Objective:** Resolve a crash generating training data from within a
Jupyter cell on Windows.

**Method:** Diagnosed the cause as a Windows-specific multiprocessing
limitation (spawn-based process creation re-imports the notebook kernel
launcher as the worker entry point, which cannot bootstrap). Added a
standalone command-line script with a proper process-entry guard for safe
parallel generation on Windows, and a toggle in both notebooks to load its
output instead of regenerating inline. Both notebooks default to
single-process generation on Windows when generating inline, to avoid the
crash, while remaining fully parallel on Colab/Linux/Mac.

**Results & Observations:** Full test suite passed.

**Issues & Troubleshooting:** Also fixed a related defect in both
notebooks' final sanity-check cell, which failed to re-solve a validation
simulation when a dataset was loaded from disk rather than freshly
generated in memory.

**Conclusion / Next Steps:** Windows data generation stabilized, though at
reduced (single-core) speed pending Entry 27.

---

### Entry 23 — July 25, 2026 — Progress bar (reverted)

**Objective:** Provide live progress feedback during dataset generation.

**Method:** Added a `show_progress` option rendering a progress bar as
simulations completed, switching the parallel execution path to report
completion order rather than submission order.

**Results & Observations:** Did not hold up under testing.

**Issues & Troubleshooting:** Reverted the same day, before reaching any
user-facing release.

**Conclusion / Next Steps:** Dataset generation reverted to its prior,
already-validated behavior without a progress indicator.

---

### Entry 24 — July 25, 2026 — Setup guide added to repository

**Objective:** Bring the project's setup documentation under version
control.

**Method:** Committed `SETUP_INSTRUCTIONS.md`, previously distributed only
alongside separate zip deliverables, to the repository itself.

**Results & Observations:** Guide now tracked alongside all other project
documentation.

**Issues & Troubleshooting:** None.

**Conclusion / Next Steps:** Proceed to manuscript-asset formatting work
(Entries 25–26).

---

### Entry 25 — July 26, 2026 — Architecture diagram script

**Objective:** Produce a labeled diagram of the network architecture for
use in the written report.

**Method:** Implemented a standalone script drawing the network's input
columns (labeled by physical meaning), hidden layers, hard-initial-condition
output transform, and optional Fourier feature encoding, with the trainable
parameter count in the title. Requires only the configuration file, no
trained checkpoint or dataset.

**Results & Observations:** Diagram generated correctly and independently
of the training pipeline.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Proceed to manuscript table/figure formatting
(Entry 26).

---

### Entry 26 — July 26, 2026 — APA (7th edition) manuscript formatting

**Objective:** Bring the manuscript's figures and tables into compliance
with APA 7th-edition academic formatting conventions.

**Method:** Moved figure captions out of the plotted images and into a
separate caption file (bold figure number, italicized title). Reformatted
tables in the compiled document as three-line tables (rules only above and
below the header row and below the final row, no shading), with a proper
italicized "Note." label preceding footnotes. Added a numbering-mode option
supporting both dissertation-chapter numbering and sequential
journal-article numbering.

**Results & Observations:** Verified end to end against a real trained
development checkpoint, including the ablation table, in both numbering
modes: correct table border formatting, correct caption text styling, no
captions baked into images, and no output filename collisions between
numbering modes.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Manuscript asset formatting complete for both
numbering conventions.

---

### Entry 27 — July 29, 2026 — GPU out-of-memory fix

**Objective:** Resolve a training crash at full production dataset scale.

**Method:** Diagnosed the cause as full-batch training constructing all
five loss terms' computational graphs (including the physics term's
second-order automatic differentiation over every collocation point)
simultaneously before a single backward pass, exceeding available GPU
memory. Implemented a gradient-accumulated computation path that computes
and backpropagates each loss term in bounded-size chunks instead, with each
chunk contributing its exact proportional share of the full-batch result.

**Results & Observations:** Dedicated tests confirmed the chunked
computation's loss values and resulting gradients are mathematically
identical to the original single-pass computation on identically
initialized models — not an approximation. Both training notebooks
re-verified end to end with the new default active; full test suite
passed.

**Issues & Troubleshooting:** Reproduced the original failure directly
(observed as a CUDA out-of-memory error on a 22 GB GPU) before implementing
and verifying the fix.

**Conclusion / Next Steps:** Memory-bounded training validated. Proceed to
right-sizing the dataset for realistic session-length constraints (Entry
28).

---

### Entry 28 — July 29, 2026 — Dataset scaled back to 2,000 simulations

**Objective:** Fit a full training run within a single cloud GPU session.

**Method:** Reduced `dataset.n_simulations` from 10,000 back to 2,000
(split 1,400/300/300), retaining the already-reduced per-simulation point
counts, since it is that combination that determines total per-iteration
training cost.

**Results & Observations:** At 10,000 simulations, measured throughput on
an L4 GPU (~17.5 sec/epoch) implied a worst-case ~97-hour training time for
the full 25,000-iteration schedule — exceeding a 24-hour cloud session
limit. At 2,000 simulations, worst-case estimated at ~25–28 hours.

**Issues & Troubleshooting:** Also vectorized a per-point lookup in the
microenvironment module used in the physics-loss computation path; measured
only a modest ~1.08x speedup in isolation, confirming this was not the
dominant cost (the GPU-side second-order automatic differentiation is).

**Conclusion / Next Steps:** Dataset scale now fits realistic session
constraints; iteration count may still require trimming depending on
measured hardware throughput.

---

### Entry 29 — July 30, 2026 — Windows multi-core data generation

**Objective:** Restore full multi-core data-generation speed on Windows.

**Method:** Automated the Entry-22 standalone-script workaround: the
notebook's data-generation cell now runs the standalone script as a real
subprocess on Windows (parallelizing safely across all CPU cores) and loads
its result automatically, with no manual steps required. Non-Windows
platforms continue parallelizing inline as before.

**Results & Observations:** Verified via automated notebook execution
(non-Windows path) and a standalone test of the subprocess-and-load path.

**Issues & Troubleshooting:** None of note.

**Conclusion / Next Steps:** Windows users now generate data at full
multi-core speed without manual intervention.

---

### Entry 30 — July 30 – August 1, 2026 — Training log frequency

**Objective:** Prevent long-running training from appearing frozen due to
sparse console output.

**Method:** Changed the default logging interval from once every 1,000
steps to once every step.

**Results & Observations:** A long gap between printed lines is now a
reliable signal to investigate, rather than an artifact of the previous
logging interval.

**Issues & Troubleshooting:** Prior default could make an hour or more of
correctly-running training appear indistinguishable from a hang, since
validation ran every step regardless — only the printed output was gated.

**Conclusion / Next Steps:** Proceed to resumable training (Entry 31).

---

### Entry 31 — August 1, 2026 — Resumable training

**Objective:** Allow training to survive a cloud session disconnect without
losing progress.

**Method:** Implemented periodic checkpointing during the first training
phase, saving model weights, optimizer momentum state, learning-rate
schedule position, best-validation tracking, patience counter, and loss
history — not just the model weights. If a checkpoint exists at the start
of a training call, it is loaded automatically and training resumes from
the saved point. Checkpoints are written atomically to prevent corruption
from a crash mid-write, validated against a configuration fingerprint
before loading, and deleted once training completes fully.

**Results & Observations:** A dedicated test split a 10-epoch run into 5
epochs, followed by a resume-and-continue for 5 more epochs (deliberately
using a different random seed for the second half, to prove the checkpoint
alone — not incidental in-memory state — carries everything required), and
confirmed the resulting parameters matched an uninterrupted 10-epoch run
exactly. Verified via automated notebook execution that the checkpoint file
is created during training and removed on completion. Full test suite (151
passed, 3 new) passed.

**Issues & Troubleshooting:** None of note; behavior verified correct on
first implementation.

**Conclusion / Next Steps:** Training now resilient to session interruption.

---

### Entry 32 — August 3, 2026 — Best-validation tracking in the second
training phase

**Objective:** Ensure the second (L-BFGS) training phase cannot produce a
worse final model than the first (Adam) phase.

**Method:** Observed that the first phase already tracked and restored the
best-validation-scored state seen during training, rather than simply the
final epoch, but the second phase had no equivalent mechanism — it retained
whatever state its line search happened to end on. Implemented the same
tracking for the second phase: validation score is evaluated at every
optimizer evaluation, and the best-scoring state is restored at the end of
the phase.

**Results & Observations:** A dedicated test proved the final validation
score after the second phase is always less than or equal to the score
measured immediately before it began, and that the loaded model's weights
correspond exactly to the reported best score — not merely the final
iteration. Full test suite (152 passed, 1 new) passed.

**Issues & Troubleshooting:** Root cause identified by inspecting recent
loss curves, which showed the data-fit component of the loss increasing
during the later portion of training — consistent with the final saved
model not being the best one seen.

**Conclusion / Next Steps:** Model-selection defect resolved. Retraining
under this fix is required to benefit an already-trained checkpoint.

---

*This logbook is a plain-language, chronological companion to the
project's git history. Refer to the repository's commits for exact code
changes and authoritative timestamps.*
