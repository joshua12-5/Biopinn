#!/usr/bin/env python3
"""CLI: generate the synthetic FDM dataset (Latin Hypercube sampling + FDM
solves) and save it to paths.processed / paths.raw_simulations. Never trains
-- pairs with notebooks/biopinn_train.ipynb or biopinn_train_local.ipynb for
the training step.

This exists as a standalone script (not just a notebook cell) because
Windows' spawn-based multiprocessing needs a real `if __name__ == "__main__":`
guard to reliably parallelize FDM solves. Calling
concurrent.futures.ProcessPoolExecutor with n_jobs > 1 directly from a
Jupyter cell on Windows can crash with BrokenProcessPool -- each spawned
worker tries to re-import the Jupyter kernel launcher as "__main__" and
fails to bootstrap, since there's no real script file with a __main__ guard
to protect it. Running generation as a real script sidesteps that entirely
(this is exactly why every scripts/*.py in this repo already ends with
`if __name__ == "__main__": main()`).

After this finishes, open the training notebook, set
DATA_ALREADY_GENERATED = True near the QUICK_TEST toggle, and run the rest
of the notebook to train -- it will load this script's output instead of
regenerating it.

Usage:
    python scripts/generate_dataset.py [--experiment NAME] [--seed N] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data_pipeline import build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None, help="Parallel worker processes (default: all CPU cores)")
    args = parser.parse_args()

    config = load_config(args.experiment)
    n_jobs = args.n_jobs or os.cpu_count() or 1

    n_total = sum(config["dataset"]["split"].values())
    print(
        f"Generating {n_total} simulations across {n_jobs} worker process(es) "
        f"(experiment: {config['experiment_name']})..."
    )

    t0 = time.time()
    dataset = build_dataset(config, seed=args.seed, save=True, n_jobs=n_jobs, show_progress=True)
    elapsed = time.time() - t0

    print(f"\nDataset generation complete in {elapsed / 60:.1f} min.")
    for split_name, tensors in dataset["splits"].items():
        print(
            f"  {split_name}: {len(dataset['sims'][split_name])} sims, "
            f"data_X {tensors['data_X'].shape}, collocation_X {tensors['collocation_X'].shape}"
        )
    print(
        "\nNow open the training notebook, set DATA_ALREADY_GENERATED = True, "
        "and run the rest of it to train on this dataset."
    )


if __name__ == "__main__":
    main()
