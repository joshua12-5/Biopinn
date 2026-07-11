"""Phase 0 smoke test: every src/ module (+ the dashboard app) imports without error."""

import importlib

MODULES = [
    "src.config",
    "src.microenvironment",
    "src.fdm_solver",
    "src.data_pipeline",
    "src.model",
    "src.losses",
    "src.train",
    "src.biology",
    "src.evaluate",
    "src.ablation",
    "src.optimize",
    "src.visualize",
    "app.server",
]


def test_all_modules_import():
    for module_name in MODULES:
        importlib.import_module(module_name)
