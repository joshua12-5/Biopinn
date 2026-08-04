"""Phase 5 tests: two-phase Adam -> L-BFGS training engine.

Uses a small dataset (via the same fast fdm/dataset overrides as
tests/test_data_pipeline.py) and short iteration counts so the suite runs
quickly while still exercising the real training loop end-to-end.
"""

import copy
import json

import numpy as np
import torch

import pytest

from src.config import load_config
from src.data_pipeline import build_dataset
from src.model import BIOPINN, load_checkpoint
from src.train import (
    HISTORY_KEYS,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    to_tensors,
    train,
    train_adam_phase,
    train_lbfgs_phase,
    _val_score,
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


def test_val_score_uses_the_configs_loss_weights():
    config = copy.deepcopy(FAST_CONFIG)
    config["loss"]["w_data"] = 3.0
    config["loss"]["w_phys"] = 0.5

    assert _val_score(2.0, 4.0, config) == pytest.approx(3.0 * 2.0 + 0.5 * 4.0)


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


def test_train_adam_phase_with_forced_multi_chunk_matches_unchunked():
    """config["training"]["max_points_per_chunk"] routes train_adam_phase
    through composite_loss_chunked (see src/losses.py); with a chunk size
    far smaller than FAST_CONFIG's per-category point counts (data=240,
    collocation=400, bc_surface/bc_center/ic=60 each, for 4 train sims),
    every term genuinely loops over multiple chunks each epoch. Starting
    from identical weights, this must produce the same trained parameters
    (within float tolerance) as the unchunked path after the same number of
    epochs -- proving the chunking config knob doesn't change what's
    learned, only peak memory."""
    result = _dataset()
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    model_unchunked = BIOPINN(FAST_CONFIG)
    model_chunked = BIOPINN(FAST_CONFIG)
    model_chunked.load_state_dict(copy.deepcopy(model_unchunked.state_dict()))

    config_unchunked = copy.deepcopy(FAST_CONFIG)
    config_unchunked["training"]["max_points_per_chunk"] = None
    config_chunked = copy.deepcopy(FAST_CONFIG)
    config_chunked["training"]["max_points_per_chunk"] = 7

    train_adam_phase(model_unchunked, train_batch, val_batch, config_unchunked, result["stats"], log_every=0)
    train_adam_phase(model_chunked, train_batch, val_batch, config_chunked, result["stats"], log_every=0)

    for p_unchunked, p_chunked in zip(model_unchunked.parameters(), model_chunked.parameters()):
        assert torch.allclose(p_chunked, p_unchunked, atol=1e-4, rtol=1e-3)


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


def test_train_lbfgs_phase_with_forced_multi_chunk_runs_and_returns_finite_history():
    config = copy.deepcopy(FAST_CONFIG)
    config["training"]["max_points_per_chunk"] = 7

    result = build_dataset(config, seed=9, save=False)
    model = BIOPINN(config)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    train_adam_phase(model, train_batch, val_batch, config, result["stats"], log_every=0)
    lbfgs_result = train_lbfgs_phase(model, train_batch, val_batch, config, result["stats"], log_every=0)

    assert lbfgs_result["closure_evaluations"] > 0
    assert all(np.isfinite(v) for v in lbfgs_result["history"]["total"])
    assert np.isfinite(lbfgs_result["val_data"])
    assert np.isfinite(lbfgs_result["val_phys"])
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))


def test_train_lbfgs_phase_never_regresses_below_pre_phase_validation_score():
    """L-BFGS's line search can wander through many closures with no
    guarantee the last one generalizes best -- this proves the phase always
    leaves the model at least as good, by validation score, as it started:
    it must track and restore the best-scoring state seen, not just
    whatever L-BFGS happened to end on. The guaranteed quantity is the same
    loss.w_data/w_phys-weighted score _val_score uses for selection, not the
    raw unweighted val_data + val_phys sum -- a weighted score can improve
    even on a run where the raw sum gets worse, since a big physics-residual
    gain can outweigh a small data-fit regression once weighted the same way
    the training objective itself is."""
    from src.train import _evaluate_validation, _val_score

    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    train_adam_phase(model, train_batch, val_batch, FAST_CONFIG, result["stats"], log_every=0)
    pre_val_data, pre_val_phys = _evaluate_validation(model, val_batch, FAST_CONFIG, result["stats"])
    pre_score = _val_score(pre_val_data, pre_val_phys, FAST_CONFIG)

    lbfgs_result = train_lbfgs_phase(model, train_batch, val_batch, FAST_CONFIG, result["stats"], log_every=0)

    post_score = _val_score(lbfgs_result["val_data"], lbfgs_result["val_phys"], FAST_CONFIG)
    assert post_score <= pre_score + 1e-6
    assert len(lbfgs_result["history"]["val_data"]) == lbfgs_result["closure_evaluations"]
    assert len(lbfgs_result["history"]["val_phys"]) == lbfgs_result["closure_evaluations"]

    # The model left loaded is the best-scoring state, not just the final closure's.
    post_val_data, post_val_phys = _evaluate_validation(model, val_batch, FAST_CONFIG, result["stats"])
    assert post_val_data == pytest.approx(lbfgs_result["val_data"])
    assert post_val_phys == pytest.approx(lbfgs_result["val_phys"])


def test_train_adam_phase_with_chunking_survives_nan_batch_without_raising():
    """Same as test_train_adam_phase_survives_nan_batch_without_raising, but
    through the chunked path -- confirms backward-as-you-go chunking doesn't
    leave corrupted gradients lying around when a chunk produces a
    non-finite loss: optimizer.step() is still skipped every epoch (the
    NaN branch continues past it), and the next epoch's zero_grad() clears
    whatever partial gradient a prior chunk accumulated."""
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    poisoned = dict(train_batch)
    poisoned["data_y"] = poisoned["data_y"].clone()
    poisoned["data_y"][0, 0] = float("nan")

    config = copy.deepcopy(FAST_CONFIG)
    config["training"]["adam"]["iters"] = 5
    config["training"]["max_points_per_chunk"] = 7

    adam_result = train_adam_phase(model, poisoned, val_batch, config, result["stats"], log_every=0)
    assert adam_result["history"]["total"] == []
    for p in model.parameters():
        assert torch.all(torch.isfinite(p))


def test_train_adam_phase_resume_matches_uninterrupted_run(tmp_path):
    """The whole point of resumable checkpointing is that it doesn't change
    what gets learned, only how much progress a disconnect can lose. Adam's
    updates are deterministic given the same starting weights, optimizer
    momentum state, and gradients -- so splitting a 10-epoch run into 5
    epochs (save) + resume for 5 more must produce parameters bit-identical
    to a single uninterrupted 10-epoch run, if (and only if) resume is
    restoring the optimizer/scheduler state correctly and not just the model
    weights. The second half deliberately starts from a differently-seeded
    model to prove the checkpoint itself carries everything needed, rather
    than relying on the model object happening to already hold the right
    state."""
    result = _dataset()
    train_batch = to_tensors(result["splits"]["train"])
    val_batch = to_tensors(result["splits"]["val"])

    config = copy.deepcopy(FAST_CONFIG)
    config["training"]["adam"]["iters"] = 10

    torch.manual_seed(0)
    model_straight = BIOPINN(config)
    straight_result = train_adam_phase(model_straight, train_batch, val_batch, config, result["stats"], log_every=0)

    checkpoint_path = tmp_path / "resume.pt"
    config_part1 = copy.deepcopy(config)
    config_part1["training"]["adam"]["iters"] = 5

    torch.manual_seed(0)
    model_part1 = BIOPINN(config)
    train_adam_phase(
        model_part1, train_batch, val_batch, config_part1, result["stats"], log_every=0,
        checkpoint_path=checkpoint_path, checkpoint_every=1,
    )
    assert checkpoint_path.exists()

    torch.manual_seed(123)  # deliberately different init than model_straight/model_part1
    model_part2 = BIOPINN(config)
    split_result = train_adam_phase(
        model_part2, train_batch, val_batch, config, result["stats"], log_every=0,
        checkpoint_path=checkpoint_path, checkpoint_every=1,
    )

    assert straight_result["epochs_run"] == 10
    assert split_result["epochs_run"] == 10
    for p_straight, p_split in zip(model_straight.parameters(), model_part2.parameters()):
        assert torch.equal(p_straight, p_split)
    assert straight_result["history"]["total"] == pytest.approx(split_result["history"]["total"])


def test_load_training_checkpoint_raises_on_config_mismatch(tmp_path):
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    checkpoint_path = tmp_path / "resume.pt"

    save_training_checkpoint(
        checkpoint_path, model, optimizer, scheduler, epoch=3,
        best_val_score=1.0, best_state=model.state_dict(), patience_counter=0,
        history={k: [] for k in HISTORY_KEYS}, ramp_state={"active": False}, config=FAST_CONFIG,
    )

    mismatched_config = copy.deepcopy(FAST_CONFIG)
    mismatched_config["dataset"]["n_simulations"] = 999999

    with pytest.raises(ValueError, match="doesn't match the current config"):
        load_training_checkpoint(checkpoint_path, mismatched_config)


def test_train_resumes_automatically_and_cleans_up_resume_file(tmp_path):
    """End-to-end through the public train() entry point. train() always
    runs to completion (or convergence) and cleans up its resume file
    afterward, so calling it with a shorter adam.iters wouldn't simulate an
    interruption -- it'd just be a complete short run. Instead, simulate a
    real mid-run kill directly: call train_adam_phase on its own and stop
    it short of the intended budget, leaving a checkpoint on disk exactly
    like a killed process would (train()'s end-of-run cleanup never gets a
    chance to run). Then call the real train() with the full budget and
    confirm it picks up from that checkpoint automatically and deletes it
    once the full run (both phases) actually completes."""
    import src.config as cfg_module
    from src.config import resolve_path

    config = copy.deepcopy(FAST_CONFIG)
    config["paths"] = dict(config["paths"])
    config["training"] = copy.deepcopy(config["training"])
    config["training"]["checkpoint_every"] = 1
    config["training"]["adam"]["iters"] = 10

    original_root = cfg_module.REPO_ROOT
    try:
        cfg_module.REPO_ROOT = tmp_path
        config["paths"]["model_checkpoint"] = "artifacts/biopinn_model.pt"
        config["paths"]["normalization_stats"] = "artifacts/normalization_stats.json"

        dataset = build_dataset(config, seed=9, save=False)
        train_batch = to_tensors(dataset["splits"]["train"])
        val_batch = to_tensors(dataset["splits"]["val"])

        model_checkpoint_path = resolve_path(config, "model_checkpoint")
        resume_path = model_checkpoint_path.with_name(model_checkpoint_path.stem + "_resume" + model_checkpoint_path.suffix)

        interrupted_model = BIOPINN(config)
        short_config = copy.deepcopy(config)
        short_config["training"]["adam"]["iters"] = 4
        train_adam_phase(
            interrupted_model, train_batch, val_batch, short_config, dataset["stats"], log_every=0,
            checkpoint_path=resume_path, checkpoint_every=1,
        )
        assert resume_path.exists(), "interrupted run should have left a resume checkpoint behind"

        full_result = train(config, dataset, save=True)
        assert full_result["adam_epochs_run"] == 10
        assert not resume_path.exists(), "resume checkpoint should be cleaned up after a full completion"
    finally:
        cfg_module.REPO_ROOT = original_root


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
