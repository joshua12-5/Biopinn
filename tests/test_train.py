"""Phase 5 tests: two-phase Adam -> L-BFGS training engine.

Uses a small dataset (via the same fast fdm/dataset overrides as
tests/test_data_pipeline.py) and short iteration counts so the suite runs
quickly while still exercising the real training loop end-to-end.
"""

import copy
import json

import numpy as np
import torch

from src.config import load_config
from src.data_pipeline import build_dataset
from src.model import BIOPINN, load_checkpoint
from src.train import (
    HISTORY_KEYS,
    save_checkpoint,
    to_tensors,
    train,
    train_adam_phase,
    train_lbfgs_phase,
)

BASE_CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(BASE_CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 25
FAST_CONFIG["fdm"]["N_t_initial"] = 15
FAST_CONFIG["dataset"]["split"] = {"train": 4, "val": 2, "test": 2}
FAST_CONFIG["dataset"]["points_per_sim"] = {
    "data": 60,
    "collocation": 100,
    "bc_surface": 15,
    "bc_center": 15,
    "ic": 15,
}
FAST_CONFIG["training"]["adam"]["iters"] = 30
FAST_CONFIG["training"]["adam"]["step_lr_step_size"] = 10
FAST_CONFIG["training"]["lbfgs"]["iters"] = 10
FAST_CONFIG["training"]["convergence"]["patience_epochs"] = 10_000  # effectively disabled by default


def _dataset():
    return build_dataset(FAST_CONFIG, seed=9, save=False)


def test_train_adam_phase_runs_and_logs_all_components():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    adam_result = train_adam_phase(
        model, train_batch, val_batch, FAST_CONFIG, result["stats"], log_every=0
    )

    assert adam_result["epochs_run"] == FAST_CONFIG["training"]["adam"]["iters"]
    for key in HISTORY_KEYS:
        assert len(adam_result["history"][key]) == adam_result["epochs_run"]
        assert all(np.isfinite(v) for v in adam_result["history"][key])
    assert len(adam_result["history"]["val_data"]) == adam_result["epochs_run"]
    assert len(adam_result["history"]["val_phys"]) == adam_result["epochs_run"]


def test_train_adam_phase_stops_early_on_convergence():
    config = copy.deepcopy(FAST_CONFIG)
    config["training"]["adam"]["iters"] = 200
    # Deliberately lax thresholds + short patience so convergence triggers
    # well before the full 200-iteration budget.
    config["training"]["convergence"]["phys_val_threshold"] = 1e6
    config["training"]["convergence"]["data_val_threshold"] = 1e6
    config["training"]["convergence"]["patience_epochs"] = 3

    result = build_dataset(config, seed=9, save=False)
    model = BIOPINN(config)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    adam_result = train_adam_phase(model, train_batch, val_batch, config, result["stats"], log_every=0)
    assert adam_result["epochs_run"] < config["training"]["adam"]["iters"]


def test_train_adam_phase_survives_nan_batch_without_raising():
    # A batch with a NaN target makes the loss non-finite from epoch 0 --
    # the NaN-recovery safeguard must catch this and return cleanly rather
    # than raising or corrupting the returned model.
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    poisoned = dict(train_batch)
    poisoned["data_y"] = poisoned["data_y"].clone()
    poisoned["data_y"][0, 0] = float("nan")

    config = copy.deepcopy(FAST_CONFIG)
    config["training"]["adam"]["iters"] = 5

    adam_result = train_adam_phase(model, poisoned, val_batch, config, result["stats"], log_every=0)
    # Every epoch saw a non-finite loss and was skipped (not appended).
    assert adam_result["history"]["total"] == []
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))


def test_train_lbfgs_phase_runs_and_returns_finite_history():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    # Warm up briefly with Adam first, as the real pipeline does.
    train_adam_phase(model, train_batch, val_batch, FAST_CONFIG, result["stats"], log_every=0)

    lbfgs_result = train_lbfgs_phase(
        model, train_batch, val_batch, FAST_CONFIG, result["stats"], log_every=0
    )
    assert lbfgs_result["closure_evaluations"] > 0
    assert len(lbfgs_result["history"]["total"]) == lbfgs_result["closure_evaluations"]
    assert all(np.isfinite(v) for v in lbfgs_result["history"]["total"])
    assert np.isfinite(lbfgs_result["val_data"])
    assert np.isfinite(lbfgs_result["val_phys"])


def test_save_and_load_checkpoint_round_trip(tmp_path):
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)

    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])

    import src.config as cfg_module

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["model_checkpoint"] = "artifacts/biopinn_model.pt"
        config["paths"]["normalization_stats"] = "artifacts/normalization_stats.json"

        artifacts = save_checkpoint(model, config, result["stats"])
        assert artifacts["checkpoint_path"].exists()
        assert artifacts["normalization_stats_path"].exists()

        with open(artifacts["normalization_stats_path"], encoding="utf-8") as f:
            stats_on_disk = json.load(f)
        assert stats_on_disk == result["stats"]

        loaded_model = load_checkpoint(str(artifacts["checkpoint_path"]), config)
        X = torch.rand(5, config["model"]["input_dim"])
        with torch.no_grad():
            original_out = model(X)
            loaded_out = loaded_model(X)
        assert torch.allclose(original_out, loaded_out)
    finally:
        cfg_module.REPO_ROOT = original_root


def test_train_end_to_end_short_run_produces_decreasing_loss_and_artifacts(tmp_path):
    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])

    import src.config as cfg_module

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["model_checkpoint"] = "artifacts/biopinn_model.pt"
        config["paths"]["normalization_stats"] = "artifacts/normalization_stats.json"

        dataset = build_dataset(config, seed=9, save=False)
        result = train(config, dataset, save=True)

        total_history = result["history"]["total"]
        assert len(total_history) > 0
        assert all(np.isfinite(v) for v in total_history)
        # A full Adam+L-BFGS run on this tiny dataset should meaningfully
        # reduce the loss from its initial value.
        assert total_history[-1] < total_history[0]

        assert result["artifacts"]["checkpoint_path"].exists()
        assert result["artifacts"]["normalization_stats_path"].exists()
        assert np.isfinite(result["final_val_data"])
        assert np.isfinite(result["final_val_phys"])
    finally:
        cfg_module.REPO_ROOT = original_root
