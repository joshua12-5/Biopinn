"""Phase 0 smoke test: config loading works and the package imports cleanly."""

from src.config import load_config


def test_default_config_loads():
    config = load_config()
    assert config["model"]["n_layers"] == 5
    assert config["model"]["n_neurons"] == 64
    assert config["loss"]["w_bc"] == 10.0
    assert config["dataset"]["n_simulations"] == 2000


def test_experiment_override_merges():
    config = load_config("experiment_1")
    # Overridden field
    assert config["dataset"]["n_simulations"] == 20
    # Untouched field still present from defaults
    assert config["model"]["n_layers"] == 5
    assert config["microenvironment"]["f_zone"]["necrotic_core"] == 0.50


def test_ablation_experiment_disables_physics_loss():
    config = load_config("experiment_2")
    assert config["loss"]["w_phys"] == 0.0
    assert config["loss"]["w_bc"] == 10.0
