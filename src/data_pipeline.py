"""LHS synthetic dataset generation + normalization pipeline. [Phase 3 — not
yet implemented]

Will provide: Latin Hypercube Sampling over the 5D parameter space
(R, d_NP, C0, k_d, t_max), FDM label generation via src/fdm_solver.py,
collocation/BC/IC point sampling for the PINN losses, min-max normalization
to [0,1], and serialization to train/val/test .npy files plus
normalization_stats.json. Runs on Colab; the resulting artifacts are the
handoff into the local pipeline.
"""

from __future__ import annotations

import numpy as np


def latin_hypercube_sample(n_samples: int, config: dict, seed: int) -> np.ndarray:
    """Sample n_samples points over the 5D parameter space via LHS.

    Returns an array of shape [n_samples, 5]: (R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr).
    """
    raise NotImplementedError("Phase 3")


def generate_dataset(n_simulations: int, config: dict, seed: int) -> list[dict]:
    """LHS-sample parameters, solve each with src/fdm_solver.solve_fdm, return raw sims."""
    raise NotImplementedError("Phase 3")


def sample_training_points(sim: dict, config: dict) -> dict:
    """Sample data/collocation/bc_surface/bc_center/ic points from one FDM solution."""
    raise NotImplementedError("Phase 3")


def compute_normalization_stats(sims: list[dict], config: dict) -> dict:
    """Compute min/max normalization stats across the training split."""
    raise NotImplementedError("Phase 3")


def save_processed_dataset(splits: dict, stats: dict, config: dict) -> None:
    """Write train/val/test .npy files and normalization_stats.json to paths.processed."""
    raise NotImplementedError("Phase 3")
