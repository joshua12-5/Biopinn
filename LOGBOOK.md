# BIOPINN Development Logbook

A running diary of how this project was built, in plain language, based on the
project's actual commit history. BIOPINN trains a neural network (a "PINN," or
Physics-Informed Neural Network) to predict how a cancer drug spreads through a
tumor over time — fast enough to explore thousands of treatment scenarios,
while still being constrained to obey the real diffusion physics during
training, not just pattern-match to examples.

Each entry below covers one real, committed piece of work: what was built,
why, how it was tested, and — just as importantly — what broke, what didn't
work as hoped, and what got fixed. Nothing here is smoothed over: a few
entries below are honest "this didn't pass" or "this got reverted" results,
because that's what actually happened.

*A note on dates:* the first fifteen entries (project kickoff through Phase
13) were, in reality, committed over the course of a single long working
session. They're spaced out below, one per day, to reflect the actual amount
of work in each phase in a more readable pace. Every entry from Phase 14
onward uses its real commit date. The exact timestamps for everything are
public in the repository's git history if you want the unvarnished record.

---

## June 27, 2026 — Project kickoff

Created the GitHub repository and made the first commit to mark the start of
the project.

---

## June 28, 2026 — Phase 0: Laying the foundation

Scaffolded the whole repo: folder layout, dependency list, and one central
configuration file that holds every physical constant, dataset setting,
network architecture choice, and training schedule in one place, so nothing
important is hard-coded and buried in code later. Every module planned for
the rest of the project got a stub file that at least imports cleanly.

**Testing:** confirmed the package installs cleanly, every module imports
without error, and the configuration file loads and merges overrides
correctly.

---

## June 29, 2026 — Phase 1: Modeling the tumor's microenvironment

Built the biology/physics groundwork: how oxygen spreads out from a tumor's
surface, and how that splits the tumor into three zones — a well-oxygenated
outer rim, a middle zone with less oxygen, and a starved, dead core in the
middle. Also modeled how fast a drug nanoparticle moves through tissue based
on its size, using a standard physics formula (Stokes–Einstein diffusion).

**Hiccups:** found a subtle bug where writing a number like `2.0e3` in a
settings file was being silently read as plain text instead of the number
2000 — a known quirk of the YAML file format used for configuration. Also had
to tune the oxygen-use rate so small tumors stay fully healthy while large
ones develop all three zones, matching the intended biology.

**Testing:** 10 automated tests plus a diagnostic plot of the diffusion
field.

---

## June 30, 2026 — Phase 2: Building the reference "ground truth" solver

Implemented a classic numerical solver (no machine learning involved) that
directly solves the drug-diffusion equation. This gives a trustworthy
reference answer that the neural network's predictions can later be checked
against.

**Hiccups:** a first, naive version tried to keep every single internal
calculation step in memory and would have needed to allocate 29 GB of RAM for
one of the trickier scenarios (a small tumor with a small nanoparticle).
Fixed by only keeping the specific time points that are actually needed,
while still internally taking small enough steps to stay numerically stable.

**Testing:** 8 tests covering boundary behavior and stability, plus a
sanity check against a known worked example — the computed result landed
within the expected range.

---

## July 1, 2026 — Phase 3: Generating training data

Built the pipeline that randomly samples thousands of different scenarios
(tumor size, nanoparticle size, drug dose, decay rate, treatment length),
spread evenly across the realistic range of each (a sampling technique called
Latin Hypercube Sampling), solves each one with the Phase 2 solver, and
packages the results into the numeric format the neural network will learn
from.

**Testing:** 7 tests plus a real run generating a small 20-scenario practice
dataset in about 6 seconds.

---

## July 2, 2026 — Phase 4: Building the neural network

Implemented the actual prediction model — a neural network that's trained not
only to match solved examples, but also penalized if its own predictions
disagree with the diffusion equation itself. It also takes the tumor and drug
parameters as input, not just position and time, so one trained model can
generalize across many different scenarios instead of needing to be retrained
for each one.

**Testing:** 20 tests covering the network's output shapes, its built-in
initial-condition behavior, and each of the five separate error terms that
measure how well it's satisfying the data, the physics equation, and the
boundary conditions individually. Also ran a manual 200-step training check
to confirm the error trends downward with no instability.

---

## July 3, 2026 — Phase 5: Building the training loop

Implemented the two-stage training process: first Adam (a widely used,
robust way to train neural networks from a random starting point), then
L-BFGS (a more precise but pickier method) to fine-tune from there. Added
safety nets: the best-performing version seen so far is automatically kept,
and if the numbers ever go unstable mid-training, it automatically rolls back
and recovers instead of crashing.

**Testing:** 6 tests plus a real ~2-minute training run on a small dataset,
confirming the error actually decreases and that a saved-and-reloaded model
makes identical predictions to the one that was saved.

---

## July 4, 2026 — Phase 6: Making it runnable in the cloud

Built the notebook that ties the whole pipeline together — install, generate
data, train, sanity-check — so the project can run on a free or rented cloud
GPU (Google Colab) instead of requiring a personal computer with a graphics
card. Also parallelized data generation across multiple CPU cores, since
early testing showed the full dataset would otherwise take about 3.3 hours on
a single core.

**Testing:** ran every cell end-to-end against a small test dataset, and
confirmed single-core and multi-core generation produce identical results.

---

## July 5, 2026 — Phase 7: Modeling the drug's effect on cells

Added the biology layer that turns a predicted drug concentration over time
into an actual outcome: how many cells die (via a standard dose-response
curve, the Hill equation), and from there, maps of which parts of the tumor
are still alive versus killed off.

**Hiccups:** a planned check — "by 72 hours, the outer rim should be mostly
dead and the core should be mostly alive" — turned out not to hold for any
allowed combination of settings in this model. Traced it to a real, intended
detail of the biology rather than a bug: the dead core actually lets drug
diffuse through it *faster* than the healthy rim does, so once the drug gets
past the rim, the interior catches up quickly. Documented this honestly
instead of quietly forcing the check to pass.

**Testing:** 12 tests plus a verification plot of the affected/unaffected
regions.

---

## July 6, 2026 — Phase 8: Automated grading of the model

Built the evaluation system: six standard accuracy metrics (including RMSE,
MAE, and R²) computed against scenarios the model never saw during training,
broken down by tumor zone, nanoparticle size, and time.

**Testing:** 13 tests plus a real run against an early, deliberately small
and undertrained practice model — it correctly reported "fail" on all six
metrics, exactly as expected for a model that undertrained.

---

## July 7, 2026 — Phase 9: Does the physics constraint actually help?

Built a comparison study: trained a second copy of the model with the
physics constraint switched off, then statistically compared how well each
version actually obeys the real diffusion equation.

**Hiccups / honest result:** on the small practice dataset used for testing,
the two versions' physics-accuracy scores came out nearly identical (about a
0.02% difference) — technically still a statistically detectable difference
given enough sample points, but not a practically meaningful one at that
tiny scale. Documented this as an expected result of the small test
configuration, not a bug in the statistics.

---

## July 8, 2026 — Phase 10: Finding the best treatment settings

Built a search that scans nanoparticle size and drug dose to find the
combination that kills the most tumor cells, plus a comparison of tumors with
uniform vs. zone-varying diffusion, and a speed comparison against the
reference solver.

**Hiccups:** at the project's own suggested default dose, the tumor turned
out to fully saturate with drug no matter the other settings, leaving nothing
meaningful to compare. Rather than quietly swapping in an easier example to
force a pass, the honest default was kept and the comparison was also run at
a lower, non-saturating dose where the effect is real and measurable.

**Testing:** 9 tests plus a real end-to-end run — the trained model answered
in roughly 1/172nd the time it takes to solve the reference equation
directly.

---

## July 9, 2026 — Phase 11: Drawing the figures

Built every visualization the project needs: concentration heatmaps,
viability/cytotoxicity maps, an animated GIF of the drug spreading through
the tumor over time, and the treatment-effectiveness surface across
nanoparticle sizes and doses.

**Testing:** 11 tests plus a full pipeline run confirming all 8 figures and
the animation render correctly.

---

## July 10, 2026 — Phase 12: An interactive dashboard

Built a small web app: a backend that serves live predictions from the
trained model, and a single-page dashboard where you can drag sliders (tumor
size, nanoparticle size, dose, etc.) and watch the predicted drug
concentration, cell viability, and treatment effectiveness update in real
time.

**Hiccups:** found and fixed a crash where certain edge-case predictions
produced invalid numbers that broke the web server's responses, plus a chart
legend that overlapped its own axis title.

**Testing:** launched the server, tested every API endpoint directly, and
clicked through the live dashboard in an actual browser to confirm the
sliders correctly update every chart.

---

## July 11, 2026 — Phase 13: Polishing the project

Rewrote the README as a proper reference document instead of a running build
log, filled in a couple of missing test cases, and did a full "clean
machine" test: cloned the repository into an empty folder from scratch,
installed everything fresh, and confirmed every script fails with a clear,
helpful message — instead of a confusing crash — when run before any data or
trained model exists yet.

**Testing:** full test suite green end-to-end on the fresh clone.

---

## July 12, 2026 — Phase 14: Generating the manuscript's results

Built the code that computes every figure and table for the project's
written report directly from real trained-model output, replacing every
placeholder number that had been in the draft until now.

**Hiccups:** while generating everything end-to-end, caught and fixed two
real bugs — one where a divide-by-zero could sneak an invalid number into the
results file and corrupt it, and one where a particular nanoparticle size
crashed a lookup because it wasn't included in the list being scanned. Also
flagged an honesty note in the results themselves: one planned comparison
(uniform vs. zone-varying tumor diffusion) doesn't actually show a difference
at the report's suggested example dose, because the tumor saturates
completely at that dose — rather than swap in more flattering numbers, this
was documented plainly in the results file and README.

---

## July 18, 2026 — Running locally, and scaling up

Three changes landed today:

**Local GPU support.** The training notebook previously assumed it was
always running on Google Colab. A user tried running it on their own
computer and it failed at several steps — trying to clone into a Colab-only
folder, trying to use a Colab-only library. Fixed by adding a proper
"am I running on Colab or locally?" check used throughout, so the same
notebook now works cleanly in both places.

**Bigger dataset.** Scaled the training dataset from 2,000 up to 10,000
simulated scenarios for a more robust final model, while reducing how many
data points are sampled per scenario, so the total training workload doesn't
also multiply 5x along with it.

**Windows text safety.** Audited every place the project reads or writes a
file, and made sure special scientific characters (µm, °, etc., used
throughout the output) can't get corrupted — Windows computers don't always
default to the same text encoding as Mac or Linux.

**Testing:** full test suite green after each change; the training notebook
was actually re-run end-to-end locally to confirm it works outside Colab, and
a real output file was checked to confirm its special characters survived
correctly on the Windows text-encoding path.

---

## July 19, 2026 — A Windows-only test failure

One automated test assumed Unix-style file paths and failed specifically on
Windows — not because the actual code was wrong, but because the test itself
made an assumption that doesn't hold on that platform. Fixed the test to
check the right thing on every operating system.

**Testing:** 137/137 tests passing.

---

## July 20, 2026 — A local-only notebook, and a Windows crash fix

**Streamlined local notebook.** Added a second notebook for people who only
ever run the project locally, with all the Colab-specific steps removed so
it reads as a straight line instead of a dual-mode script.

**Windows data-generation crash.** Found that generating data directly
inside a notebook cell on Windows could crash outright, due to a low-level
limitation in how Windows handles running work across multiple CPU cores
from inside a notebook. Fixed by adding a standalone script that generates
data safely at full multi-core speed on Windows, plus a toggle in the
notebook to load that already-generated data instead of trying (and
crashing) to regenerate it inline.

**Testing:** ran the new notebook end-to-end and confirmed it produces
identical results to the existing notebook's local path; full test suite
green.

---

## July 25, 2026 — A progress bar that didn't make the cut, and a setup guide

Tried adding a live progress bar to dataset generation — it didn't hold up
under testing and was reverted the same day, before it ever reached anyone
using the project. Also added the step-by-step setup guide to the repository
itself, so it stays version-controlled alongside everything else instead of
being handed out separately.

---

## July 26, 2026 — An architecture diagram, and academic formatting

**Architecture diagram.** Added a script that draws a labeled diagram of the
neural network's layers and parameter count, for use in the written report.

**APA-style formatting.** Reformatted every figure and table for the
manuscript into proper academic (APA 7th edition) style — captions moved out
of the images themselves and into a separate file, and tables redrawn in the
clean, minimal-rule style academic papers expect.

**Testing:** verified against a real trained practice model, confirming
correct formatting in every supported numbering style with no output files
overwriting each other.

---

## July 29, 2026 — Running out of GPU memory, and scaling the dataset back down

**Fixed a real crash.** At full training scale, the part of the training
process that checks the model's predictions against the diffusion equation
needed more graphics-card memory than was available, and training crashed
with an out-of-memory error. Fixed by computing that part in smaller pieces
and adding the results together — mathematically the exact same outcome, just
spread across less memory at any one moment, not an approximation.

**Scaled the dataset back down.** The 10,000-scenario dataset from July 18
was measured to take up to ~97 hours to fully train in the worst case — well
past what a single free/rented cloud GPU session allows. Scaled back down to
2,000 scenarios to comfortably fit in one session.

**Testing:** proved mathematically (and with dedicated automated tests) that
computing the physics check in smaller pieces gives the exact same result as
computing it all at once. Both training notebooks were re-run end-to-end to
confirm.

---

## July 30, 2026 — Fixing slow data generation on Windows

Windows users were stuck generating training data on a single CPU core (a
safety fallback added earlier to avoid a crash), even on machines with many
cores available. Automated the safe multi-core workaround from July 20, so
Windows users now get full-speed data generation with no manual steps
required.

---

## August 1, 2026 — Clearer progress logging, and resumable training

**Print every step.** Training used to print a progress line only once every
1,000 steps — meaning an hour of real, correctly-running training could look
exactly like it had silently frozen. Changed the default to print a line on
every single step.

**Resumable training.** Cloud GPU sessions can disconnect well before their
stated time limit, and until now, that meant losing the entire training run.
Training now automatically saves its progress periodically and, if
interrupted, picks back up next time from exactly where it left off, instead
of starting over from scratch.

**Testing:** proved that a training run split in two — interrupted, then
resumed — produces byte-for-byte identical results to a run that was never
interrupted at all, not just "close enough." Verified with a dedicated
automated test plus a real end-to-end run of both notebooks.

---

## August 3, 2026 — Fixing a subtle model-selection bug

The second stage of training (L-BFGS) didn't keep track of its best result
along the way — it could wander into a slightly worse fit than where the
first stage had already reached, and that worse version would silently
become the final saved model with no warning. Fixed by having it track and
keep the best-scoring version it sees throughout the whole stage, the same
way the first stage already did.

**Testing:** added a dedicated automated test proving the final model can
now only ever match or improve on the first stage's result — never quietly
end up worse. Full 152-test suite passing.

---

*This logbook is a plain-language companion to the project's git history —
see the repository's commits for exact code changes and timestamps.*
