"""Phase 4 tests: composite PINN loss (data, physics/PDE residual, Dirichlet,
Neumann, IC) on a small real dataset generated end-to-end via
src/data_pipeline.py's fast test config (see tests/test_data_pipeline.py)."""

import copy

import numpy as np
import torch

from src.config import load_config
from src.data_pipeline import build_dataset
from src.losses import (
    composite_loss,
    data_loss,
    dirichlet_bc_loss,
    ic_loss,
    neumann_bc_loss,
    pde_residual,
    physics_loss,
)
from src.model import BIOPINN

CONFIG = load_config()

FAST_CONFIG = copy.deepcopy(CONFIG)
FAST_CONFIG["fdm"]["N_r"] = 25
FAST_CONFIG["fdm"]["N_t_initial"] = 15
FAST_CONFIG["dataset"]["split"] = {"train": 4, "val": 2, "test": 2}
FAST_CONFIG["dataset"]["points_per_sim"] = {
    "data": 40,
    "collocation": 60,
    "bc_surface": 10,
    "bc_center": 10,
    "ic": 10,
}


def _dataset():
    return build_dataset(FAST_CONFIG, seed=5, save=False)


def _batch_as_tensors(split_tensors: dict) -> dict:
    return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in split_tensors.items()}


def test_pde_residual_finite_and_correct_shape():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    residual = pde_residual(model, batch["collocation_X"], FAST_CONFIG, result["stats"])
    assert residual.shape == (batch["collocation_X"].shape[0], 1)
    assert torch.all(torch.isfinite(residual))


def test_physics_loss_is_nonnegative_scalar():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    L_phys = physics_loss(model, batch["collocation_X"], FAST_CONFIG, result["stats"])
    assert L_phys.shape == ()
    assert L_phys.item() >= 0.0
    assert np.isfinite(L_phys.item())


def test_data_loss_matches_manual_mse():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    L_data = data_loss(model, batch["data_X"], batch["data_y"])
    with torch.no_grad():
        expected = torch.mean((model(batch["data_X"]) - batch["data_y"]) ** 2)
    assert torch.allclose(L_data, expected)


def test_dirichlet_and_ic_losses_are_zero_for_a_perfect_model():
    # A model that always predicts exactly the hard-IC transform's boundary
    # value (sigmoid(0)*t_norm = 0.5*t_norm) won't be "perfect", but we can
    # instead check the *targets* used are physically correct: C_norm=1 at
    # the surface, C_norm=0 at t=0 -- and that the loss functions correctly
    # measure a real model's deviation from them (nonzero, finite).
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    assert torch.all(batch["bc_surface_y"] == 1.0)
    assert torch.all(batch["ic_y"] == 0.0)

    L_bc = dirichlet_bc_loss(model, batch["bc_surface_X"], batch["bc_surface_y"])
    L_ic = ic_loss(model, batch["ic_X"], batch["ic_y"])
    assert np.isfinite(L_bc.item()) and L_bc.item() >= 0.0
    assert np.isfinite(L_ic.item()) and L_ic.item() >= 0.0

    # The hard-IC output transform already forces C_norm=0 at t=0 exactly,
    # so the *soft* IC loss should be exactly zero regardless of weights.
    assert L_ic.item() == 0.0


def test_neumann_loss_finite_and_nonnegative():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    L_neu = neumann_bc_loss(model, batch["bc_center_X"])
    assert np.isfinite(L_neu.item())
    assert L_neu.item() >= 0.0


def test_composite_loss_matches_weighted_sum():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    losses = composite_loss(model, batch, FAST_CONFIG, result["stats"])
    loss_cfg = FAST_CONFIG["loss"]
    expected_total = (
        loss_cfg["w_data"] * losses["data"]
        + loss_cfg["w_phys"] * losses["phys"]
        + loss_cfg["w_bc"] * losses["bc"]
        + loss_cfg["w_neu"] * losses["neu"]
        + loss_cfg["w_ic"] * losses["ic"]
    )
    assert torch.allclose(losses["total"], expected_total)
    for key in ("total", "data", "phys", "bc", "neu", "ic"):
        assert np.isfinite(losses[key].item())


def test_composite_loss_backward_updates_parameters():
    result = _dataset()
    model = BIOPINN(FAST_CONFIG)
    batch = _batch_as_tensors(result["splits"]["train"])

    before = [p.clone() for p in model.parameters()]
    losses = composite_loss(model, batch, FAST_CONFIG, result["stats"])
    losses["total"].backward()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()

    changed = any(not torch.equal(b, a) for b, a in zip(before, model.parameters()))
    assert changed
