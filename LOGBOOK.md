# Research Logbook — BIOPINN

**Project:** BIOPINN — a Physics-Informed Neural Network (PINN) surrogate for
nanoparticle drug transport, penetration depth, and tumor-cell viability in a
three-zone tumor spheroid.
**Repository:** https://github.com/joshua12-5/Biopinn

*Note on dates:* Entries 1–16 (project kickoff through Phase 14) were, in
the underlying commit history, recorded over a shorter working span than
shown below; they are spaced out to one entry per day here for a more
readable pace. All entries from Entry 17 onward carry their true, unaltered
date. Exact timestamps for every change are available in the repository's
git history.

---

**Entry 1 — June 27, 2026.** The researchers created the GitHub repository
for the project and made the initial commit, marking the formal start of
BIOPINN.

**Entry 2 — June 28, 2026 (Phase 0).** The researchers began by scaffolding
the codebase: setting up the folder layout, the dependency list, and a
single central configuration file to hold every physical constant, dataset
setting, network architecture choice, and training schedule value, so
nothing important would be hard-coded later. A config loader and stub
modules for every planned phase were added so the package would import
cleanly from day one. Testing confirmed the package installed correctly,
every module imported without error, and configuration overrides merged as
expected.

**Entry 3 — June 29, 2026 (Phase 1).** The researchers modeled the tumor's
internal biology: how oxygen spreads out from the tumor surface, and how
that gradient splits the tumor into three zones — a healthy outer rim, a
weaker middle zone, and a starved core — along with how fast a drug
nanoparticle diffuses through tissue based on its size. While tuning the
model, the researchers discovered that the configuration file was silently
misreading a number written as `2.0e3` as plain text instead of 2000, a
known quirk of the YAML format, and corrected it. They also had to re-tune
the oxygen consumption rate before small and large tumors behaved as
intended. Ten tests and a diagnostic plot confirmed the fix and the final
model.

**Entry 4 — June 30, 2026 (Phase 2).** The researchers built a traditional,
non-machine-learning numerical solver for the drug-diffusion equation to
serve as ground truth. An early version of the solver tried to keep every
internal calculation step in memory and would have needed to allocate
roughly 29 GB of RAM for one of the harder parameter combinations; the
researchers fixed this by only storing the specific time points actually
needed while still stepping finely enough internally to stay numerically
stable. Eight tests and a comparison against a known worked example
confirmed the solver was working correctly.

**Entry 5 — July 1, 2026 (Phase 3).** The researchers built the pipeline
that randomly samples thousands of tumor and treatment scenarios spread
evenly across the realistic parameter range, solves each with the Phase 2
solver, and packages the results into normalized data the neural network
could later learn from. Seven tests passed, and a real 20-scenario practice
dataset was generated correctly in about six seconds.

**Entry 6 — July 2, 2026 (Phase 4).** The researchers implemented the
neural network itself — one trained not only to match solved examples but
also penalized whenever its predictions disagreed with the underlying
diffusion equation, and built to take the tumor and drug settings as input
so a single trained model could generalize across scenarios. Twenty tests
covering the network's shape, its built-in initial-condition behavior, and
each of its five error terms all passed, and a manual 200-step training run
showed the error decreasing with no instability.

**Entry 7 — July 3, 2026 (Phase 5).** The researchers built the full
training procedure: a first stage using Adam to get the network into a good
region efficiently, followed by a second, more precise L-BFGS stage to fine
tune it, with the best-performing version automatically kept throughout and
a recovery mechanism that rolls training back if the numbers ever go
unstable. Six tests passed, and a real two-minute training run on a small
dataset produced cleanly decreasing, converging loss curves.

**Entry 8 — July 4, 2026 (Phase 6).** The researchers built the notebook
that ties the whole pipeline together to run on a free or rented cloud GPU
rather than requiring local hardware, and parallelized dataset generation
across CPU cores after profiling showed the full dataset would otherwise
take about 3.3 hours on a single core. The notebook ran correctly end to end
against a small test configuration, and a dedicated test confirmed
single-core and multi-core generation produced identical results.

**Entry 9 — July 5, 2026 (Phase 7).** The researchers added the biology
layer translating predicted drug concentration into actual cell death,
using a standard dose-response curve, and producing maps of which parts of
the tumor were alive or killed. A planned check — that by 72 hours the
outer rim should be mostly dead while the core stays mostly alive — turned
out not to hold for any allowed combination of settings; the researchers
traced this to a real property of the model (the dead core actually lets
drug through faster than the rim does) rather than a bug, and documented it
honestly instead of forcing the check to pass. Twelve tests and a
verification plot confirmed the rest of the biology module worked
correctly.

**Entry 10 — July 6, 2026 (Phase 8).** The researchers built the automated
evaluation system, computing six standard accuracy metrics against
scenarios the model never saw during training, broken down by zone,
nanoparticle size, and time. Thirteen tests passed, and a run against an
early, deliberately undertrained practice model correctly reported a fail
on all six metrics, exactly as expected.

**Entry 11 — July 7, 2026 (Phase 9).** The researchers tested whether the
physics constraint actually helped, by training a second copy of the model
with it switched off and statistically comparing how well each version
obeyed the real diffusion equation. On the small practice dataset used at
this stage, the two versions' physics-accuracy scores came out nearly
identical, a difference too small to be practically meaningful even though
it was still statistically detectable given enough sample points — the
researchers documented this as an expected result of the small test scale
rather than a flaw in the statistics. Seven tests passed.

**Entry 12 — July 8, 2026 (Phase 10).** The researchers built a search over
nanoparticle size and dose to find the treatment that killed the most tumor
cells, along with a comparison of uniform versus zone-varying diffusion and
a speed comparison against the reference solver. At the project's own
suggested default dose, the tumor turned out to fully saturate regardless
of settings, leaving nothing to compare; rather than quietly swap in an
easier example, the researchers kept the honest default and added the
comparison at a lower, non-saturating dose where the effect is real. Nine
tests passed, and the trained model answered roughly 172 times faster than
solving the reference equation directly.

**Entry 13 — July 9, 2026 (Phase 11).** The researchers built every
visualization the project needed — concentration heatmaps, viability and
cytotoxicity maps, an animated GIF of the drug spreading over time, and the
treatment-effectiveness surface. Eleven tests passed, and a full pipeline
run confirmed all eight figures and the animation rendered correctly.

**Entry 14 — July 10, 2026 (Phase 12).** The researchers built a small web
application with a backend serving live predictions from the trained model
and a single-page dashboard letting a user drag sliders and watch
concentration, viability, and treatment effectiveness update in real time.
During testing they found and fixed a crash caused by certain edge-case
predictions producing invalid numbers that broke the server's responses,
along with a chart legend overlapping its axis title. Five tests passed,
and the researchers confirmed the fixes by clicking through the live
dashboard in an actual browser.

**Entry 15 — July 11, 2026 (Phase 13).** The researchers polished the
project for presentation: rewriting the README as a proper reference
document, filling two gaps in test coverage, and performing a full
clean-machine test by cloning the repository into an empty folder,
installing everything from scratch, and confirming every script failed
with a clear, helpful message instead of crashing confusingly when run
before any data or model existed. The full test suite passed on this fresh
clone, closing out the originally scoped 14 build phases.

**Entry 16 — July 12, 2026 (Phase 14).** The researchers built the code
that computes every figure and table for the project's written report
directly from real trained-model output, replacing every placeholder number
that had been in the draft. While generating everything end to end, they
caught and fixed two real bugs — a divide-by-zero that could sneak an
invalid number into the results file, and a lookup crash when a particular
nanoparticle size wasn't included in a list being scanned — and documented
an honest finding that one planned comparison in the report doesn't
actually show a difference at the manuscript's suggested example dose,
because the tumor saturates completely at that setting.

**Entry 17 — July 18, 2026.** The researchers made the training notebook
work on a local computer with a GPU, not only on Google Colab, after a user
reported it failing outside Colab at several steps (trying to clone into a
Colab-only folder, trying to import a Colab-only library). They added a
proper environment check used throughout the notebook and confirmed, by
running it end to end locally, that it now correctly detects a local
environment and completes without any Colab-specific steps running.

**Entry 18 — July 18, 2026.** The researchers scaled the training dataset
from 2,000 up to 10,000 simulated scenarios for a more robust final model,
shrinking the number of points sampled per scenario at the same time so the
total training workload wouldn't also multiply along with it. The full test
suite stayed green and the notebook was re-verified end to end.

**Entry 19 — July 18, 2026.** The researchers audited every place in the
codebase that reads or writes a file and made sure special scientific
characters used throughout the output couldn't get corrupted on Windows,
whose default text encoding isn't always the same as Mac or Linux. They
confirmed this was the one real gap by checking the rest of the codebase
for other platform-specific issues and finding none, and verified the fix
by regenerating a real output file and confirming its special characters
were intact.

**Entry 20 — July 19, 2026.** The researchers found that one automated test
was failing only on Windows, not because the underlying code was wrong but
because the test itself assumed Unix-style file paths. They rewrote the
test to check the right thing on every operating system, bringing the suite
back to 137 of 137 tests passing everywhere.

**Entry 21 — July 20, 2026.** The researchers added a second notebook for
people who only ever run the project locally, stripped of all the
Colab-specific branching so it reads as a straight line. Running it end to
end confirmed it produced results identical to the existing notebook's
local path.

**Entry 22 — July 20, 2026.** The researchers found that generating data
directly inside a notebook cell could crash outright on Windows, due to a
low-level limitation in how Windows handles multi-core work started from
inside a notebook. They fixed it by adding a standalone script that
generates data safely at full speed on Windows, plus a toggle in the
notebook to load that data instead of trying, and failing, to regenerate it
inline. They also fixed a related bug in both notebooks' final sanity-check
step, which had failed to re-run a validation check when a dataset was
loaded from disk rather than freshly generated.

**Entry 23 — July 25, 2026.** The researchers tried adding a live progress
bar to dataset generation. It didn't hold up under testing and was reverted
the same day, before it ever reached anyone using the project.

**Entry 24 — July 25, 2026.** The researchers added the project's
step-by-step setup guide to the repository itself, so it would stay
version-controlled alongside everything else instead of being distributed
separately.

**Entry 25 — July 26, 2026.** The researchers added a script that draws a
labeled diagram of the neural network's layers, inputs, and parameter
count, for use in the written report. It required only the configuration
file to run, with no trained model or dataset needed.

**Entry 26 — July 26, 2026.** The researchers reformatted every figure and
table for the manuscript into proper APA-style academic formatting, moving
captions out of the images themselves and redrawing tables in the clean,
minimal-rule style academic papers expect. They verified the change against
a real trained practice model in every supported numbering style, confirming
correct formatting with no files overwriting each other.

**Entry 27 — July 29, 2026.** The researchers tracked down and fixed a real
crash: at full training scale, the part of training that checks the
model's predictions against the diffusion equation needed more graphics
card memory than was available, and training crashed with an
out-of-memory error. They fixed it by computing that part in smaller
pieces and adding the results together, mathematically identical to
computing it all at once, and proved this with dedicated tests before
re-running both notebooks end to end to confirm.

**Entry 28 — July 29, 2026.** The researchers scaled the training dataset
back down from 10,000 to 2,000 scenarios after measuring that the larger
dataset could take up to roughly 97 hours to train in the worst case —
well past what a single cloud GPU session allows. While making this change
they also sped up a slow lookup used in the physics calculation, though
testing showed it made only a modest difference, confirming that the real
cost was elsewhere.

**Entry 29 — July 30, 2026.** The researchers automated the Windows
data-generation workaround from Entry 22, so Windows users now get
full-speed, multi-core data generation with no manual steps required,
instead of being stuck on a single core as a safety fallback.

**Entry 30 — August 1, 2026.** The researchers changed training's default
logging behavior to print a line on every single step instead of only once
every 1,000 steps, after realizing the old default could make an hour or
more of correctly-running training look exactly like it had silently
frozen.

**Entry 31 — August 1, 2026.** The researchers made training resumable, so
a cloud session disconnecting partway through — which can happen well
before its stated time limit — no longer means losing the entire run.
Training now periodically saves its full progress and automatically picks
back up next time from exactly where it left off. They proved this worked
correctly with a test that interrupted a run partway through and resumed
it, confirming the result was identical, step for step, to a run that was
never interrupted at all.

**Entry 32 — August 3, 2026.** The researchers fixed a subtle bug in the
second stage of training: it didn't track its best result along the way,
so it could wander into a slightly worse fit than where the first stage had
already reached, and that worse version would silently become the final
saved model with no warning. They fixed it by having the second stage track
and keep its best-scoring version throughout, the same way the first stage
already did, and proved with a dedicated test that the final model can now
only ever match or improve on the first stage's result, never quietly end
up worse.

---

*This logbook is a plain-language, chronological companion to the
project's git history. Refer to the repository's commits for exact code
changes and authoritative timestamps.*
