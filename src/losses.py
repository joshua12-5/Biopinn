"""Composite PINN loss: L_data + L_phys + L_bc + L_neu + L_ic.

Implements the five loss components (data fidelity, PDE residual via
torch.autograd.grad, Dirichlet surface, Neumann symmetry at r=0, and
initial condition), combined with the configured weights (w_data, w_phys,
w_bc, w_neu, w_ic). The point tensor layout (see src/data_pipeline.py and
src/model.py) is:

    [r_norm, t_norm, R_norm, d_NP_norm, C0_norm, k_d_norm, t_max_norm]

`physics_loss`/`pde_residual` denormalize the trailing five columns back to
physical units (using normalization_stats.json) to evaluate the governing
PDE with the correct per-point D_eff(r) and k_d(r) from
src/microenvironment.py, then convert the network's normalized derivatives
back to physical derivatives before forming the residual.

Governing PDE (matches the radial term src/fdm_solver.py actually solves,
NOT the idealized "2/r" spherical form quoted at the top of the build
prompt's scientific-model section -- the FDM-solver spec and the guide's own
reference code both use "1/r"; using anything else here would make the
physics loss fight the FDM-generated data loss over what PDE is even being
solved):

    dC/dt = D_eff(r) * (d2C/dr2 + (1/r)*dC/dr) - k_d(r) * C
"""

from __future__ import annotations

import torch

from src.data_pipeline import PARAM_ORDER
from src.microenvironment import decay_rate_field, effective_diffusivity
from src.model import BIOPINN

R_COLUMN, T_COLUMN = 0, 1
PARAM_COLUMNS_START = 2  # R_norm, d_NP_norm, C0_norm, k_d_norm, t_max_norm


def _denormalize_params(param_cols: torch.Tensor, norm_stats: dict) -> dict[str, torch.Tensor]:
    """Denormalize the five trailing parameter columns back to physical units.

    Returns a dict keyed by src.data_pipeline.PARAM_ORDER, each an [N,1] tensor.
    """
    physical = {}
    for i, key in enumerate(PARAM_ORDER):
        lo, hi = norm_stats[key]["min"], norm_stats[key]["max"]
        physical[key] = param_cols[:, i : i + 1] * (hi - lo) + lo
    return physical


def pde_residual(model: BIOPINN, X: torch.Tensor, config: dict, norm_stats: dict) -> torch.Tensor:
    """PDE residual dC/dt - D_eff*(d2C/dr2 + (1/r)*dC/dr) + k_d*C at each point in X."""
    r_norm = X[:, R_COLUMN : R_COLUMN + 1].clone().detach().requires_grad_(True)
    t_norm = X[:, T_COLUMN : T_COLUMN + 1].clone().detach().requires_grad_(True)
    param_cols = X[:, PARAM_COLUMNS_START:].detach()

    inputs = torch.cat([r_norm, t_norm, param_cols], dim=1)
    C_norm = model(inputs)

    grad_r = torch.autograd.grad(
        C_norm, r_norm, grad_outputs=torch.ones_like(C_norm), create_graph=True
    )[0]
    grad_t = torch.autograd.grad(
        C_norm, t_norm, grad_outputs=torch.ones_like(C_norm), create_graph=True
    )[0]
    grad_rr = torch.autograd.grad(
        grad_r, r_norm, grad_outputs=torch.ones_like(grad_r), create_graph=True
    )[0]

    physical = _denormalize_params(param_cols, norm_stats)
    R_um, d_NP_nm = physical["R_um"], physical["d_NP_nm"]
    C0_uM, k_d_per_hr, t_max_hr = physical["C0_uM"], physical["k_d_per_hr"], physical["t_max_hr"]

    r_phys = r_norm * R_um
    C_phys = C_norm * C0_uM

    dC_dr = grad_r * (C0_uM / R_um)
    d2C_dr2 = grad_rr * (C0_uM / R_um**2)
    dC_dt = grad_t * (C0_uM / t_max_hr)

    # D_eff(r) and k_d(r) are piecewise-constant zone coefficients, not
    # differentiated through -- evaluate them pointwise via NumPy (detached).
    r_phys_np = r_phys.detach().cpu().numpy().ravel()
    R_um_np = R_um.detach().cpu().numpy().ravel()
    d_NP_np = d_NP_nm.detach().cpu().numpy().ravel()
    k_d_np = k_d_per_hr.detach().cpu().numpy().ravel()

    D_eff_np = effective_diffusivity(r_phys_np, d_NP_np, config, R=R_um_np)
    k_d_field_np = decay_rate_field(r_phys_np, k_d_np, config, R=R_um_np)

    D_eff = torch.as_tensor(D_eff_np, dtype=X.dtype, device=X.device).reshape(-1, 1)
    k_d_field = torch.as_tensor(k_d_field_np, dtype=X.dtype, device=X.device).reshape(-1, 1)

    r_safe = torch.clamp(r_phys, min=1e-6)
    laplacian = d2C_dr2 + (1.0 / r_safe) * dC_dr
    return dC_dt - D_eff * laplacian + k_d_field * C_phys


def data_loss(model: BIOPINN, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(X) - y) ** 2)


def physics_loss(model: BIOPINN, X: torch.Tensor, config: dict, norm_stats: dict) -> torch.Tensor:
    residual = pde_residual(model, X, config, norm_stats)
    return torch.mean(residual**2)


def dirichlet_bc_loss(model: BIOPINN, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(X) - y) ** 2)


def neumann_bc_loss(model: BIOPINN, X: torch.Tensor) -> torch.Tensor:
    r_norm = X[:, R_COLUMN : R_COLUMN + 1].clone().detach().requires_grad_(True)
    rest = X[:, R_COLUMN + 1 :].detach()

    inputs = torch.cat([r_norm, rest], dim=1)
    C_norm = model(inputs)
    grad_r = torch.autograd.grad(
        C_norm, r_norm, grad_outputs=torch.ones_like(C_norm), create_graph=True
    )[0]
    return torch.mean(grad_r**2)


def ic_loss(model: BIOPINN, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((model(X) - y) ** 2)


def composite_loss(
    model: BIOPINN,
    batch: dict,
    config: dict,
    norm_stats: dict,
    weight_overrides: dict | None = None,
) -> dict:
    """Weighted sum of all five loss terms.

    `batch` uses the same keys src/data_pipeline.py's processed .npz files
    use: data_X/data_y, collocation_X, bc_surface_X/bc_surface_y,
    bc_center_X, ic_X/ic_y.

    `weight_overrides` optionally replaces individual config["loss"] weights
    (e.g. {"w_phys": 0.1}) without mutating the shared config dict -- used by
    src/train.py's w_phys warmup ramp and NaN-recovery safeguard.

    All five terms' full computational graphs are built before a single
    total.backward() call -- fine at dev scale, but at full production scale
    (10.5M collocation points needing second-order autograd for the PDE
    residual, alive in memory alongside the other four terms) this can
    exceed GPU memory. See composite_loss_chunked for a memory-bounded
    alternative with an identical (not approximate) gradient.
    """
    weights = {**config["loss"], **(weight_overrides or {})}

    L_data = data_loss(model, batch["data_X"], batch["data_y"])
    L_phys = physics_loss(model, batch["collocation_X"], config, norm_stats)
    L_bc = dirichlet_bc_loss(model, batch["bc_surface_X"], batch["bc_surface_y"])
    L_neu = neumann_bc_loss(model, batch["bc_center_X"])
    L_ic = ic_loss(model, batch["ic_X"], batch["ic_y"])

    total = (
        weights["w_data"] * L_data
        + weights["w_phys"] * L_phys
        + weights["w_bc"] * L_bc
        + weights["w_neu"] * L_neu
        + weights["w_ic"] * L_ic
    )

    return {"total": total, "data": L_data, "phys": L_phys, "bc": L_bc, "neu": L_neu, "ic": L_ic}


def _chunk_bounds(n_total: int, max_points_per_chunk: int):
    start = 0
    while start < n_total:
        end = min(start + max_points_per_chunk, n_total)
        yield start, end
        start = end


def _chunked_term(loss_fn, tensors: tuple[torch.Tensor, ...], weight: float, max_points_per_chunk: int) -> torch.Tensor:
    """Computes an unweighted mean-squared term (loss_fn(*tensors), exactly --
    not approximately -- the same value composite_loss would compute) in
    chunks of at most `max_points_per_chunk` rows, backward()-ing each
    chunk's weighted contribution immediately so only one chunk's forward +
    autograd graph is ever alive on the GPU at a time.

    Each chunk contributes chunk_n/n_total of the full-batch mean -- summing
    those contributions reconstructs the exact full-batch value, and summing
    their (already-scaled) per-chunk gradients reconstructs the exact
    full-batch gradient, since d(sum of scaled chunk terms)/d(theta) =
    d(full-batch term)/d(theta) by linearity. Mathematically identical to
    computing loss_fn(*tensors) once and backward()-ing the whole thing --
    only peak memory differs.

    Returns the DETACHED, unweighted term value (matching composite_loss's
    per-key convention, where only "total" is weighted) -- gradients are
    already accumulated into model.parameters()[i].grad as a side effect, so
    callers must NOT call .backward() again on the returned value or on
    "total" built from it.
    """
    n_total = tensors[0].shape[0]
    total_value = torch.zeros((), device=tensors[0].device, dtype=tensors[0].dtype)
    for start, end in _chunk_bounds(n_total, max_points_per_chunk):
        chunk = tuple(t[start:end] for t in tensors)
        chunk_term = loss_fn(*chunk) * ((end - start) / n_total)
        (weight * chunk_term).backward()
        total_value = total_value + chunk_term.detach()
    return total_value


def composite_loss_chunked(
    model: BIOPINN,
    batch: dict,
    config: dict,
    norm_stats: dict,
    max_points_per_chunk: int,
    weight_overrides: dict | None = None,
) -> dict:
    """Memory-bounded equivalent of composite_loss: identical "total"/per-term
    values and identical resulting gradients, computed and backward()-ed in
    chunks of at most `max_points_per_chunk` points per term instead of one
    full-batch forward + single backward. Needed once the collocation set is
    large enough that physics_loss's second-order autograd graph (plus the
    other four terms' graphs, all alive together in composite_loss) exceeds
    GPU memory -- see src/train.py's OOM-avoidance notes.

    Unlike composite_loss, backward() has already run (once per chunk, per
    term) by the time this returns -- callers must NOT call
    losses["total"].backward() again; "total" here is a detached sum of
    already-backward()-ed pieces, kept as a tensor only so callers can still
    do losses["total"].item() / torch.isfinite(...) the same way as before.
    """
    weights = {**config["loss"], **(weight_overrides or {})}

    L_data = _chunked_term(lambda X, y: data_loss(model, X, y), (batch["data_X"], batch["data_y"]), weights["w_data"], max_points_per_chunk)
    L_phys = _chunked_term(
        lambda X: physics_loss(model, X, config, norm_stats), (batch["collocation_X"],), weights["w_phys"], max_points_per_chunk
    )
    L_bc = _chunked_term(
        lambda X, y: dirichlet_bc_loss(model, X, y), (batch["bc_surface_X"], batch["bc_surface_y"]), weights["w_bc"], max_points_per_chunk
    )
    L_neu = _chunked_term(lambda X: neumann_bc_loss(model, X), (batch["bc_center_X"],), weights["w_neu"], max_points_per_chunk)
    L_ic = _chunked_term(lambda X, y: ic_loss(model, X, y), (batch["ic_X"], batch["ic_y"]), weights["w_ic"], max_points_per_chunk)

    total = (
        weights["w_data"] * L_data
        + weights["w_phys"] * L_phys
        + weights["w_bc"] * L_bc
        + weights["w_neu"] * L_neu
        + weights["w_ic"] * L_ic
    )

    return {"total": total, "data": L_data, "phys": L_phys, "bc": L_bc, "neu": L_neu, "ic": L_ic}


def physics_loss_chunked(model: BIOPINN, X: torch.Tensor, config: dict, norm_stats: dict, max_points_per_chunk: int) -> float:
    """Memory-bounded, backward-free equivalent of physics_loss(...).item(),
    for validation-time evaluation (src/train.py's _evaluate_validation) --
    physics_loss still needs autograd-tracked leaves internally for the PDE
    residual's derivatives even when no gradient w.r.t. model parameters is
    needed, so its graph is exactly as memory-heavy per point at eval time
    as at train time; chunking avoids that graph ever covering the whole
    validation collocation set at once."""
    n_total = X.shape[0]
    if n_total <= max_points_per_chunk:
        return physics_loss(model, X, config, norm_stats).item()
    total = 0.0
    for start, end in _chunk_bounds(n_total, max_points_per_chunk):
        chunk_value = physics_loss(model, X[start:end], config, norm_stats).item()
        total += chunk_value * ((end - start) / n_total)
    return total
