"""LHS synthetic dataset generation + normalization pipeline.

Latin Hypercube Sampling over the 5D parameter space (R, d_NP, C0, k_d,
t_max), FDM label generation via src/fdm_solver.py, collocation/BC/IC point
sampling for the PINN losses, and serialization of normalized train/val/test
tensors + normalization_stats.json. Runs on Colab; the resulting artifacts
in paths.processed are the handoff into the local pipeline.

Point tensor layout
--------------------
Every sampled point is stored as a 7-column feature row:

    [r_norm, t_norm, R_norm, d_NP_norm, C0_norm, k_d_norm, t_max_norm]

`r_norm`/`t_norm` are normalized by *that point's own simulation* (r/R_um,
t/t_max_hr) as required by the PINN's hard-IC output transform. The
remaining five columns are the simulation's physical parameters min-max
normalized against the *training-split* statistics in
normalization_stats.json, so the network can condition its prediction on
which point in the 5D parameter space it is being asked about (needed for
Phase 10's optimization surrogate to generalize across d_NP/C0/R without
retraining). Downstream code is free to use only the first two columns if a
simulation-specific (unconditioned) model is desired instead.

`*_y` targets (where present) are C_norm = C / C0_uM.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import qmc

from src.config import resolve_path
from src.fdm_solver import solve_fdm

PARAM_ORDER = ("R_um", "d_NP_nm", "C0_uM", "k_d_per_hr", "t_max_hr")


def _normalize(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return (value - lo) / (hi - lo)


def latin_hypercube_sample(n_samples: int, config: dict, seed: int | None) -> np.ndarray:
    """Sample n_samples points over the 5D parameter space via LHS.

    Returns an array of shape [n_samples, 5]: (R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr).
    """
    ranges = config["dataset"]["parameter_ranges"]
    l_bounds = [ranges[p][0] for p in PARAM_ORDER]
    u_bounds = [ranges[p][1] for p in PARAM_ORDER]

    sampler = qmc.LatinHypercube(d=len(PARAM_ORDER), seed=seed)
    sample = sampler.random(n=n_samples)
    return qmc.scale(sample, l_bounds, u_bounds)


def generate_dataset(n_simulations: int, config: dict, seed: int | None, n_jobs: int = 1) -> list[dict]:
    """LHS-sample parameters, solve each with src.fdm_solver.solve_fdm, return raw sims.

    Args:
        n_jobs: number of worker processes for parallel FDM solving. 1 (the
            default) solves sequentially in-process. > 1 uses a
            ProcessPoolExecutor -- each simulation is independent, so this
            scales close to linearly with core count. The full 2000-sim
            production dataset takes ~3 hours single-threaded (some
            LHS-sampled small-R/small-d_NP combinations need heavy CFL
            sub-stepping, see src/fdm_solver.py); pass n_jobs=os.cpu_count()
            on Colab to cut that down substantially.
    """
    params = latin_hypercube_sample(n_simulations, config, seed)

    if n_jobs == 1:
        results = [_solve_one((i, params[i], config)) for i in range(n_simulations)]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            results = list(pool.map(_solve_one, [(i, params[i], config) for i in range(n_simulations)]))

    return sorted(results, key=lambda sim: sim["sim_id"])


def _solve_one(args: tuple[int, np.ndarray, dict]) -> dict:
    """Module-level (picklable) worker: solve one LHS-sampled parameter row."""
    sim_id, param_row, config = args
    R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr = param_row
    fdm_result = solve_fdm(R_um, d_NP_nm, C0_uM, k_d_per_hr, t_max_hr, config)
    return {
        "sim_id": sim_id,
        "R_um": R_um,
        "d_NP_nm": d_NP_nm,
        "C0_uM": C0_uM,
        "k_d_per_hr": k_d_per_hr,
        "t_max_hr": t_max_hr,
        "r": fdm_result["r"],
        "t": fdm_result["t"],
        "C": fdm_result["C"],
    }


def sample_training_points(sim: dict, config: dict, rng: np.random.Generator | None = None) -> dict:
    """Sample data/collocation/bc_surface/bc_center/ic points from one FDM solution.

    All coordinates are returned in physical units (um, hr, uM); normalization
    happens downstream once the sim's own R_um/t_max_hr/C0_uM are known
    alongside the dataset-wide parameter statistics.
    """
    rng = rng if rng is not None else np.random.default_rng()
    points_cfg = config["dataset"]["points_per_sim"]

    r, t, C = sim["r"], sim["t"], sim["C"]
    N_t, N_r = C.shape
    R_um, t_max_hr, C0_uM = sim["R_um"], sim["t_max_hr"], sim["C0_uM"]
    r_min = r[0]

    n_data = points_cfg["data"]
    r_idx = rng.integers(0, N_r, size=n_data)
    t_idx = rng.integers(0, N_t, size=n_data)
    data = {"r": r[r_idx], "t": t[t_idx], "C": C[t_idx, r_idx]}

    n_coll = points_cfg["collocation"]
    collocation = {
        "r": rng.uniform(r_min, R_um, size=n_coll),
        "t": rng.uniform(0.0, t_max_hr, size=n_coll),
    }

    n_bc_s = points_cfg["bc_surface"]
    bc_surface = {
        "r": np.full(n_bc_s, R_um),
        "t": rng.uniform(0.0, t_max_hr, size=n_bc_s),
        "C": np.full(n_bc_s, C0_uM),
    }

    n_bc_c = points_cfg["bc_center"]
    bc_center = {
        "r": np.full(n_bc_c, r_min),
        "t": rng.uniform(0.0, t_max_hr, size=n_bc_c),
    }

    n_ic = points_cfg["ic"]
    ic = {
        "r": rng.uniform(r_min, R_um, size=n_ic),
        "t": np.zeros(n_ic),
        "C": np.zeros(n_ic),
    }

    return {
        "data": data,
        "collocation": collocation,
        "bc_surface": bc_surface,
        "bc_center": bc_center,
        "ic": ic,
    }


def compute_normalization_stats(sims: list[dict], config: dict) -> dict:
    """Compute min/max normalization stats across the training split."""
    stats = {}
    for p in PARAM_ORDER:
        values = np.array([sim[p] for sim in sims], dtype=float)
        stats[p] = {"min": float(values.min()), "max": float(values.max())}
    return stats


def _param_norm_row(sim: dict, stats: dict) -> np.ndarray:
    return np.array(
        [_normalize(sim[p], stats[p]["min"], stats[p]["max"]) for p in PARAM_ORDER],
        dtype=np.float32,
    )


def build_split_tensors(
    sims: list[dict], stats: dict, config: dict, rng: np.random.Generator | None = None
) -> dict:
    """Sample + normalize training points from every sim in a split, concatenated per category."""
    rng = rng if rng is not None else np.random.default_rng()
    categories = ("data", "collocation", "bc_surface", "bc_center", "ic")
    X_parts: dict[str, list[np.ndarray]] = {cat: [] for cat in categories}
    y_parts: dict[str, list[np.ndarray]] = {"data": [], "bc_surface": [], "ic": []}

    for sim in sims:
        points = sample_training_points(sim, config, rng=rng)
        param_row = _param_norm_row(sim, stats)
        R_um, t_max_hr, C0_uM = sim["R_um"], sim["t_max_hr"], sim["C0_uM"]

        for cat in categories:
            pts = points[cat]
            r_norm = pts["r"] / R_um
            t_norm = pts["t"] / t_max_hr
            n = len(r_norm)
            X = np.column_stack([r_norm, t_norm, np.tile(param_row, (n, 1))]).astype(np.float32)
            X_parts[cat].append(X)
            if "C" in pts:
                y_parts[cat].append((pts["C"] / C0_uM).reshape(-1, 1).astype(np.float32))

    tensors = {f"{cat}_X": np.concatenate(X_parts[cat], axis=0) for cat in categories}
    for cat, parts in y_parts.items():
        tensors[f"{cat}_y"] = np.concatenate(parts, axis=0)
    return tensors


def load_processed_dataset(config: dict) -> dict:
    """Load the train/val/test tensors + normalization stats written by
    save_processed_dataset, without regenerating or re-solving anything.

    Returns the same {"splits": ..., "stats": ...} shape build_dataset does
    (minus "sims", which isn't persisted) -- e.g. for src/ablation.py to
    train a baseline on the exact same data as an already-trained checkpoint.
    """
    processed_dir = resolve_path(config, "processed")

    splits = {}
    for split_name in ("train", "val", "test"):
        with np.load(processed_dir / f"{split_name}.npz") as data:
            splits[split_name] = {key: data[key] for key in data.files}

    with open(processed_dir / "normalization_stats.json") as f:
        stats = json.load(f)

    return {"splits": splits, "stats": stats}


def save_processed_dataset(splits: dict, stats: dict, config: dict, sims_by_split: dict | None = None) -> None:
    """Write train/val/test .npz files, normalization_stats.json, and (if
    `sims_by_split` is given) sim_params.json to paths.processed.

    sim_params.json stores each split's exact (sim_id, R_um, d_NP_nm, C0_uM,
    k_d_per_hr, t_max_hr) 5-tuples -- lightweight enough to keep alongside
    the point tensors, and lets src/evaluate.py deterministically re-solve
    the full FDM reference field for the held-out test set without needing
    to persist every raw (r, t, C) array to disk.
    """
    processed_dir = resolve_path(config, "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, tensors in splits.items():
        np.savez_compressed(processed_dir / f"{split_name}.npz", **tensors)

    with open(processed_dir / "normalization_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if sims_by_split is not None:
        sim_params = {
            split_name: [{k: (float(sim[k]) if k != "sim_id" else int(sim[k])) for k in ("sim_id",) + PARAM_ORDER} for sim in sims]
            for split_name, sims in sims_by_split.items()
        }
        with open(processed_dir / "sim_params.json", "w") as f:
            json.dump(sim_params, f, indent=2)


def build_dataset(config: dict, seed: int | None = None, save: bool = True, n_jobs: int = 1) -> dict:
    """End-to-end pipeline: LHS sample -> solve FDM -> split -> normalize -> save.

    Returns a dict with the raw sims and processed tensors per split plus the
    normalization stats, so callers (and tests) can inspect the result without
    re-reading it from disk. See generate_dataset's docstring for `n_jobs`.
    """
    ds_cfg = config["dataset"]
    split_cfg = ds_cfg["split"]
    n_total = split_cfg["train"] + split_cfg["val"] + split_cfg["test"]

    sims = generate_dataset(n_total, config, seed, n_jobs=n_jobs)

    n_train, n_val = split_cfg["train"], split_cfg["val"]
    sims_by_split = {
        "train": sims[:n_train],
        "val": sims[n_train : n_train + n_val],
        "test": sims[n_train + n_val :],
    }

    stats = compute_normalization_stats(sims_by_split["train"], config)

    rng = np.random.default_rng(seed)
    splits = {
        name: build_split_tensors(split_sims, stats, config, rng)
        for name, split_sims in sims_by_split.items()
    }

    if save:
        save_processed_dataset(splits, stats, config, sims_by_split=sims_by_split)

    return {"sims": sims_by_split, "splits": splits, "stats": stats}
