"""BIOPINN network architecture.

The 5x96 tanh fully-connected network with Xavier init and the hard-IC
output transform C_NN = sigmoid(f_theta(x)) * t_norm. Input `x` is the
7-column point layout produced by src/data_pipeline.py:

    [r_norm, t_norm, R_norm, d_NP_norm, C0_norm, k_d_norm, t_max_norm]

(r_norm, t_norm) are normalized per-simulation; the remaining five columns
condition the prediction on which point in the 5D physical parameter space
is being queried, so one trained model generalizes across tumor radius,
nanoparticle size, dose, decay rate, and duration -- needed for Phase 10's
optimization surrogate. Shared module: constructed identically on Colab (for
training) and locally (for loading a checkpoint into src/biology.py,
src/evaluate.py, and app/server.py).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

T_NORM_COLUMN = 1  # index of t_norm within the model's input feature vector


class RandomFourierFeatures(nn.Module):
    """Random Fourier feature encoding gamma(x) = [sin(2*pi*B^T*x), cos(2*pi*B^T*x)].

    Optional input encoding to help the network learn high-frequency spatial
    variation (configs/default_config.yaml -> model.fourier_features).
    """

    def __init__(self, in_dim: int, n_features: int, sigma: float):
        super().__init__()
        B = torch.randn(in_dim, n_features) * sigma
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projection = 2 * math.pi * (x @ self.B)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class BIOPINN(nn.Module):
    """Physics-informed network mapping the 7-column input to C_norm."""

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config["model"]
        self.input_dim = model_cfg["input_dim"]
        self.hard_ic_transform = model_cfg.get("hard_ic_transform", True)

        fourier_cfg = model_cfg.get("fourier_features", {})
        self.use_fourier = fourier_cfg.get("enabled", False)
        if self.use_fourier:
            self.fourier = RandomFourierFeatures(
                self.input_dim, fourier_cfg["n_features"], fourier_cfg["sigma"]
            )
            net_input_dim = 2 * fourier_cfg["n_features"]
        else:
            self.fourier = None
            net_input_dim = self.input_dim

        n_layers = model_cfg["n_layers"]
        n_neurons = model_cfg["n_neurons"]
        output_dim = model_cfg["output_dim"]

        layers = [nn.Linear(net_input_dim, n_neurons), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_neurons, n_neurons), nn.Tanh()]
        layers += [nn.Linear(n_neurons, output_dim)]
        self.net = nn.Sequential(*layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [N, input_dim] (see module docstring for column layout)."""
        features = self.fourier(x) if self.use_fourier else x
        raw = self.net(features)
        if not self.hard_ic_transform:
            return torch.sigmoid(raw)
        t_norm = x[:, T_NORM_COLUMN : T_NORM_COLUMN + 1]
        return torch.sigmoid(raw) * t_norm


def load_checkpoint(checkpoint_path: str, config: dict) -> BIOPINN:
    """Load a trained BIOPINN model from a Colab-produced .pt checkpoint."""
    model = BIOPINN(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model
