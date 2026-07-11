"""Phase 4 tests: BIOPINN network (forward shapes, hard-IC transform, init)."""

import copy

import torch

from src.config import load_config
from src.model import BIOPINN

CONFIG = load_config()


def _random_batch(n: int, input_dim: int) -> torch.Tensor:
    torch.manual_seed(0)
    X = torch.rand(n, input_dim)
    return X


def test_forward_pass_shape():
    model = BIOPINN(CONFIG)
    X = _random_batch(16, CONFIG["model"]["input_dim"])
    out = model(X)
    assert out.shape == (16, CONFIG["model"]["output_dim"])


def test_hard_ic_transform_zero_at_t0():
    model = BIOPINN(CONFIG)
    X = _random_batch(32, CONFIG["model"]["input_dim"])
    X[:, 1] = 0.0  # t_norm = 0 for every point
    out = model(X)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_output_bounded_in_zero_one():
    model = BIOPINN(CONFIG)
    X = _random_batch(200, CONFIG["model"]["input_dim"])
    out = model(X)
    assert torch.all(out >= 0.0)
    assert torch.all(out <= 1.0)


def test_hard_ic_transform_can_be_disabled():
    config = copy.deepcopy(CONFIG)
    config["model"]["hard_ic_transform"] = False
    model = BIOPINN(config)
    X = _random_batch(16, config["model"]["input_dim"])
    X[:, 1] = 0.0
    out = model(X)
    # Without the hard-IC transform, t=0 no longer forces output to zero.
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-6)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_xavier_init_produces_nonzero_weights():
    model = BIOPINN(CONFIG)
    linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == CONFIG["model"]["n_layers"] + 1
    for layer in linear_layers:
        assert torch.any(layer.weight != 0.0)
        assert torch.all(layer.bias == 0.0)


def test_fourier_features_change_effective_input_width():
    config = copy.deepcopy(CONFIG)
    config["model"]["fourier_features"]["enabled"] = True
    config["model"]["fourier_features"]["n_features"] = 8
    model = BIOPINN(config)
    first_linear = next(m for m in model.net if isinstance(m, torch.nn.Linear))
    assert first_linear.in_features == 16  # 2 * n_features (sin + cos)

    X = _random_batch(8, config["model"]["input_dim"])
    out = model(X)
    assert out.shape == (8, config["model"]["output_dim"])
    assert torch.all(torch.isfinite(out))


def test_gradients_flow_to_all_parameters():
    model = BIOPINN(CONFIG)
    X = _random_batch(8, CONFIG["model"]["input_dim"])
    out = model(X)
    loss = out.sum()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None
        assert torch.all(torch.isfinite(p.grad))
