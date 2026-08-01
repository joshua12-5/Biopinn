"""Two-phase Adam -> L-BFGS training engine.

Phase-1 Adam training (StepLR schedule, gradient clipping, w_phys warmup
ramp, NaN-recovery safeguard), Phase-2 L-BFGS fine-tuning (strong Wolfe line
search) seeded from the Adam-trained state, per-component loss logging,
best-checkpoint saving, and the validation convergence/patience criterion.
Runs on Colab; used unchanged by src/ablation.py locally to train the
w_phys=0 baseline on CPU for the ablation study.

Training is full-batch, not mini-batched -- every iteration processes the
entire train split's points at once (see src/data_pipeline.py). At full
production scale, physics_loss's second-order autograd for the PDE residual
over millions of collocation points, computed alongside the other four loss
terms before a single backward() call, can exceed GPU memory. Set
config["training"]["max_points_per_chunk"] to switch to
src.losses.composite_loss_chunked, which computes and backward()s each loss
term in point-count-bounded chunks instead -- the resulting gradient (and
therefore what gets learned) is mathematically identical to the unchunked
path; only peak memory differs. Unset (the default) preserves the original
single-shot behavior exactly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from src.config import resolve_path
from src.losses import composite_loss, composite_loss_chunked, data_loss, physics_loss, physics_loss_chunked
from src.model import BIOPINN

HISTORY_KEYS = ("total", "data", "phys", "bc", "neu", "ic")


def _resume_fingerprint(config: dict) -> dict:
    """Config fields that must match between a training checkpoint and the run
    resuming it. The checkpoint's optimizer/scheduler state (momentum
    buffers, LR schedule position) is only meaningful for the exact dataset
    size and architecture it was computed against -- resuming with a
    different one wouldn't just be wrong, it would silently corrupt training
    rather than continue it. training.adam.iters is deliberately NOT
    included: it's just this call's loop bound, not a state-compatibility
    requirement -- extending or shortening the remaining budget on resume
    (e.g. deciding partway through that 20,000 iterations should really be
    15,000) is a legitimate, supported thing to do."""
    return {
        "n_simulations": config["dataset"]["n_simulations"],
        "n_layers": config["model"]["n_layers"],
        "n_neurons": config["model"]["n_neurons"],
    }


def save_training_checkpoint(
    path: Path | str,
    model: BIOPINN,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_score: float,
    best_state: dict,
    patience_counter: int,
    history: dict,
    ramp_state: dict,
    config: dict,
) -> None:
    """Save everything needed to resume train_adam_phase from immediately
    after `epoch` -- not just the model weights, but the optimizer's Adam
    momentum buffers and the LR scheduler's step count too, so a resumed run
    continues with the exact same optimization trajectory a single
    uninterrupted run would have taken, rather than restarting Adam with a
    cold optimizer partway through training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fingerprint": _resume_fingerprint(config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_score": best_val_score,
        "best_state": best_state,
        "patience_counter": patience_counter,
        "history": history,
        "ramp_state": ramp_state,
    }
    # Write to a temp file and rename over the old checkpoint atomically, so a
    # crash/disconnect mid-save can never leave a half-written, unloadable
    # checkpoint behind -- the previous good one stays intact until the new
    # one has fully landed on disk.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_training_checkpoint(path: Path | str, config: dict, device: str = "cpu") -> dict:
    """Load a checkpoint saved by save_training_checkpoint, after verifying
    it was produced by a compatible config (see _resume_fingerprint) --
    raises rather than silently resuming into a mismatched run."""
    checkpoint = torch.load(Path(path), map_location=device)
    expected = _resume_fingerprint(config)
    if checkpoint["fingerprint"] != expected:
        raise ValueError(
            f"Training checkpoint at {path} doesn't match the current config "
            f"(checkpoint: {checkpoint['fingerprint']}, current: {expected}) -- "
            "refusing to resume from a mismatched run. Delete the checkpoint file "
            "to start fresh, or fix the config to match what generated it."
        )
    return checkpoint


def to_tensors(split_tensors: dict, device: str = "cpu") -> dict:
    """Convert a dict of numpy arrays (as loaded from a processed .npz split) to tensors."""
    return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in split_tensors.items()}


def _evaluate_validation(model: BIOPINN, val_batch: dict, config: dict, norm_stats: dict) -> tuple[float, float]:
    """Return (val_data_loss, val_phys_loss). Physics loss needs its own autograd
    leaves even in eval mode, so it cannot run under torch.no_grad() -- its
    graph is chunked the same way as training (see max_points_per_chunk)
    since it's exactly as memory-heavy per point at eval time."""
    max_points_per_chunk = config["training"].get("max_points_per_chunk")
    model.eval()
    with torch.no_grad():
        val_data = data_loss(model, val_batch["data_X"], val_batch["data_y"]).item()
    if max_points_per_chunk:
        val_phys = physics_loss_chunked(model, val_batch["collocation_X"], config, norm_stats, max_points_per_chunk)
    else:
        val_phys = physics_loss(model, val_batch["collocation_X"], config, norm_stats).item()
    model.train()
    return val_data, val_phys


def train_adam_phase(
    model: BIOPINN,
    batch: dict,
    val_batch: dict,
    config: dict,
    norm_stats: dict,
    log_every: int = 1,
    checkpoint_path: Path | str | None = None,
    checkpoint_every: int = 0,
) -> dict:
    """Run Phase-1 Adam optimization. Returns a dict with loss history and the
    best-validation model state (also left loaded into `model`).

    If `checkpoint_path` is given and a checkpoint already exists there
    (from a previous, interrupted run of this exact function against a
    compatible config -- see _resume_fingerprint), this resumes from
    immediately after the saved epoch: model weights, Adam's momentum
    buffers, the LR scheduler's step count, best-validation tracking,
    patience counter, and loss history are all restored, so the resumed run
    continues the same optimization trajectory an uninterrupted run would
    have taken. If `checkpoint_every` is also set (> 0), a checkpoint is
    saved every that-many epochs (and once more at the end of the phase),
    so a disconnect loses at most `checkpoint_every` epochs of progress
    instead of the whole run.
    """
    adam_cfg = config["training"]["adam"]
    conv_cfg = config["training"]["convergence"]
    nan_cfg = config["training"]["nan_recovery"]
    ramp_cfg = config["loss"].get("w_phys_ramp", {})
    max_points_per_chunk = config["training"].get("max_points_per_chunk")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=adam_cfg["lr"], betas=tuple(adam_cfg["betas"]), eps=adam_cfg["eps"]
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=adam_cfg["step_lr_step_size"], gamma=adam_cfg["step_lr_gamma"]
    )

    history = {k: [] for k in HISTORY_KEYS} | {"val_data": [], "val_phys": []}
    best_val_score = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    # w_phys warmup ramp state (local, not mutated into `config`). Activated
    # from the start if configs/*.yaml enables it, or triggered on the fly by
    # the NaN-recovery safeguard below.
    ramp_state = {
        "active": ramp_cfg.get("enabled", False),
        "start_epoch": 0,
        "start_w": ramp_cfg.get("start", 0.001),
        "end_w": config["loss"]["w_phys"],
        "steps": max(ramp_cfg.get("ramp_steps", 5000), 1),
    }

    start_epoch = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        device = next(model.parameters()).device
        checkpoint = load_training_checkpoint(checkpoint_path, config, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_score = checkpoint["best_val_score"]
        best_state = checkpoint["best_state"]
        patience_counter = checkpoint["patience_counter"]
        history = checkpoint["history"]
        ramp_state.update(checkpoint["ramp_state"])
        print(f"[adam] resumed from checkpoint at epoch {checkpoint['epoch']} ({checkpoint_path})")

    def current_w_phys(epoch: int) -> float | None:
        if not ramp_state["active"]:
            return None
        frac = min(1.0, (epoch - ramp_state["start_epoch"]) / ramp_state["steps"])
        return ramp_state["start_w"] + (ramp_state["end_w"] - ramp_state["start_w"]) * frac

    def save_checkpoint_now(at_epoch: int) -> None:
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path, model, optimizer, scheduler, at_epoch,
                best_val_score, best_state, patience_counter, history, ramp_state, config,
            )

    epoch = start_epoch - 1
    for epoch in range(start_epoch, adam_cfg["iters"]):
        w_phys = current_w_phys(epoch)
        weight_overrides = {"w_phys": w_phys} if w_phys is not None else None

        optimizer.zero_grad()
        if max_points_per_chunk:
            # Chunked path already backward()-ed (once per chunk, per term) as
            # a side effect -- see composite_loss_chunked. Any partial/NaN
            # gradients from a diverging chunk are harmless: the NaN branch
            # below never calls optimizer.step(), and the next epoch's
            # zero_grad() (top of this loop) discards them before the next
            # backward pass.
            losses = composite_loss_chunked(model, batch, config, norm_stats, max_points_per_chunk, weight_overrides=weight_overrides)
        else:
            losses = composite_loss(model, batch, config, norm_stats, weight_overrides=weight_overrides)
        total = losses["total"]

        if not torch.isfinite(total):
            # NaN-recovery safeguard: revert to the last good checkpoint, drop
            # the learning rate, and (re)start a w_phys ramp from a small
            # value so the physics term doesn't immediately destabilize
            # training again.
            model.load_state_dict(best_state)
            for group in optimizer.param_groups:
                group["lr"] = nan_cfg["fallback_lr"]
            if nan_cfg.get("ramp_w_phys", True):
                ramp_state.update(active=True, start_epoch=epoch, start_w=0.001)
            continue

        if not max_points_per_chunk:
            total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), adam_cfg["grad_clip_norm"])
        optimizer.step()
        scheduler.step()

        for k in HISTORY_KEYS:
            history[k].append(losses[k].item())

        val_data, val_phys = _evaluate_validation(model, val_batch, config, norm_stats)
        history["val_data"].append(val_data)
        history["val_phys"].append(val_phys)

        val_score = val_data + val_phys
        if val_score < best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())

        converged_this_epoch = (
            val_phys < conv_cfg["phys_val_threshold"] and val_data < conv_cfg["data_val_threshold"]
        )
        patience_counter = patience_counter + 1 if converged_this_epoch else 0

        if checkpoint_every and epoch % checkpoint_every == 0:
            save_checkpoint_now(epoch)

        if log_every and epoch % log_every == 0:
            print(
                f"[adam] epoch {epoch:6d} | total {total.item():.4e} | "
                f"data {losses['data'].item():.4e} | phys {losses['phys'].item():.4e} | "
                f"bc {losses['bc'].item():.4e} | neu {losses['neu'].item():.4e} | "
                f"ic {losses['ic'].item():.4e} | val_data {val_data:.4e} | val_phys {val_phys:.4e}"
            )

        if patience_counter >= conv_cfg["patience_epochs"]:
            print(
                f"[adam] converged at epoch {epoch}: val_phys < {conv_cfg['phys_val_threshold']} "
                f"and val_data < {conv_cfg['data_val_threshold']} for "
                f"{conv_cfg['patience_epochs']} consecutive epochs"
            )
            break

    # Always save once more at the end -- whether the loop broke early on
    # convergence or ran out its full iteration budget -- so a resume after
    # a clean finish (e.g. to redo L-BFGS after that phase specifically was
    # interrupted) picks up from the true final state, not a stale periodic
    # snapshot from up to checkpoint_every epochs earlier.
    save_checkpoint_now(epoch)

    model.load_state_dict(best_state)
    return {"history": history, "best_state": best_state, "epochs_run": epoch + 1}


def train_lbfgs_phase(
    model: BIOPINN,
    batch: dict,
    val_batch: dict,
    config: dict,
    norm_stats: dict,
    log_every: int = 1,
) -> dict:
    """Run Phase-2 L-BFGS fine-tuning from the current (Adam-trained) state.

    Returns a dict with per-closure-evaluation loss history and final
    validation losses. Reverts to the pre-L-BFGS state if the optimizer
    diverges to a non-finite loss.
    """
    lbfgs_cfg = config["training"]["lbfgs"]
    max_points_per_chunk = config["training"].get("max_points_per_chunk")

    pre_lbfgs_state = copy.deepcopy(model.state_dict())

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=lbfgs_cfg["iters"],
        history_size=lbfgs_cfg["history_size"],
        line_search_fn=lbfgs_cfg["line_search_fn"],
        tolerance_grad=lbfgs_cfg["tolerance_grad"],
        tolerance_change=lbfgs_cfg["tolerance_change"],
    )

    history = {k: [] for k in HISTORY_KEYS}
    call_count = {"n": 0}

    def closure():
        optimizer.zero_grad()
        if max_points_per_chunk:
            losses = composite_loss_chunked(model, batch, config, norm_stats, max_points_per_chunk)
        else:
            losses = composite_loss(model, batch, config, norm_stats)
            losses["total"].backward()
        for k in HISTORY_KEYS:
            history[k].append(losses[k].item())
        call_count["n"] += 1
        if log_every and call_count["n"] % log_every == 0:
            print(f"[lbfgs] closure {call_count['n']:5d} | total {losses['total'].item():.4e}")
        return losses["total"]

    optimizer.step(closure)

    final_total = history["total"][-1] if history["total"] else None
    if final_total is not None and not np.isfinite(final_total):
        print("[lbfgs] diverged to a non-finite loss; reverting to the pre-L-BFGS state.")
        model.load_state_dict(pre_lbfgs_state)

    val_data, val_phys = _evaluate_validation(model, val_batch, config, norm_stats)
    return {
        "history": history,
        "val_data": val_data,
        "val_phys": val_phys,
        "closure_evaluations": call_count["n"],
    }


def save_checkpoint(model: BIOPINN, config: dict, norm_stats: dict, extra: dict | None = None) -> dict:
    """Save the trained model + normalization stats to paths.model_checkpoint
    / paths.normalization_stats (artifacts/ by default) -- the local handoff
    consumed by src/biology.py, src/evaluate.py, src/ablation.py,
    src/optimize.py, and app/server.py."""
    checkpoint_path = resolve_path(config, "model_checkpoint")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict(), "config": config, "normalization_stats": norm_stats}
    if extra:
        payload.update(extra)
    torch.save(payload, checkpoint_path)

    stats_path = resolve_path(config, "normalization_stats")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(norm_stats, f, indent=2)

    return {"checkpoint_path": checkpoint_path, "normalization_stats_path": stats_path}


def train(config: dict, dataset: dict, device: str = "cpu", save: bool = True) -> dict:
    """Full two-phase training run: Adam -> L-BFGS.

    Args:
        config: full config dict.
        dataset: as returned by src.data_pipeline.build_dataset (keys
            "splits" -> {"train","val","test"} and "stats"), or an
            equivalent dict loaded from paths.processed on disk.
        device: torch device string.
        save: if True, saves the best checkpoint + normalization stats via
            save_checkpoint().

    If config["training"]["checkpoint_every"] is set (> 0), the Adam phase
    periodically saves a resumable checkpoint to
    "<model_checkpoint>_resume.pt" -- if that file already exists when
    train() is called again (e.g. after a Colab disconnect), Adam resumes
    from it automatically instead of restarting from epoch 0, restoring the
    optimizer's momentum state and LR schedule position along with the
    model weights. The resume file is deleted once training fully completes
    (both phases), so a later fresh call to train() doesn't accidentally
    pick up a stale finished run's state.

    Returns a dict with the trained model, combined loss history, and (if
    save=True) the artifact paths written.
    """
    train_batch = to_tensors(dataset["splits"]["train"], device)
    val_batch = to_tensors(dataset["splits"]["val"], device)
    norm_stats = dataset["stats"]

    model = BIOPINN(config).to(device)

    # Resume checkpointing is a disk side effect, so it respects `save` the
    # same way the final artifact write does -- a save=False (dry-run) call
    # shouldn't leave files behind either.
    checkpoint_every = config["training"].get("checkpoint_every", 0) if save else 0
    resume_path = None
    if checkpoint_every:
        model_checkpoint_path = resolve_path(config, "model_checkpoint")
        resume_path = model_checkpoint_path.with_name(model_checkpoint_path.stem + "_resume" + model_checkpoint_path.suffix)

    adam_result = train_adam_phase(
        model, train_batch, val_batch, config, norm_stats,
        checkpoint_path=resume_path, checkpoint_every=checkpoint_every,
    )
    lbfgs_result = train_lbfgs_phase(model, train_batch, val_batch, config, norm_stats)

    combined_history = {k: adam_result["history"][k] + lbfgs_result["history"][k] for k in HISTORY_KEYS}
    combined_history["val_data"] = adam_result["history"]["val_data"]
    combined_history["val_phys"] = adam_result["history"]["val_phys"]

    result = {
        "model": model,
        "history": combined_history,
        "adam_epochs_run": adam_result["epochs_run"],
        "lbfgs_closure_evaluations": lbfgs_result["closure_evaluations"],
        "final_val_data": lbfgs_result["val_data"],
        "final_val_phys": lbfgs_result["val_phys"],
    }

    if save:
        result["artifacts"] = save_checkpoint(model, config, norm_stats)

    if resume_path is not None and resume_path.exists():
        resume_path.unlink()

    return result
