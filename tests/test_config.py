"""Phase 0 smoke test: config loading works and the package imports cleanly."""

from src.config import REPO_ROOT, load_config, resolve_path


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


def test_resolve_path_joins_relative_paths_under_repo_root():
    config = load_config()
    resolved = resolve_path(config, "model_checkpoint")
    assert resolved == REPO_ROOT / config["paths"]["model_checkpoint"]
    assert resolved.is_absolute()


def test_resolve_path_lets_an_absolute_override_win(tmp_path):
    # pathlib's / operator discards the left operand when the right is
    # absolute -- this is exactly what notebooks/biopinn_train.ipynb relies
    # on to redirect paths.* at a Google Drive mount without any special
    # casing in resolve_path itself. Use pytest's tmp_path (a real,
    # platform-appropriate absolute path) rather than a hardcoded POSIX
    # literal like "/absolute/..." -- on Windows, a leading "/" is
    # drive-relative rather than truly absolute, so pathlib normalizes it
    # to "C:\...", which isn't what this test is actually checking.
    config = load_config()
    absolute_override = tmp_path / "drive" / "path" / "biopinn_model.pt"
    config["paths"]["model_checkpoint"] = str(absolute_override)
    resolved = resolve_path(config, "model_checkpoint")
    assert resolved == absolute_override
