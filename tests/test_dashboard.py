"""Phase 12 tests: the results dashboard's pure helper functions.

The dashboard (app/server.py) loads a trained checkpoint in a background
thread at FastAPI startup and serves live/cached endpoints from it -- that
full lifecycle is exercised manually (see the phase's verification run
against a dev checkpoint), not spun up here. These tests cover the two
helpers most likely to break silently: NaN/Infinity sanitization (Starlette's
JSONResponse uses allow_nan=False, so an unsanitized NaN r2/l2_relative from
a zero-variance decomposition bucket would 500 the endpoint) and histogram
binning for the residual-distribution panels.
"""

from __future__ import annotations

import math

import numpy as np

from app.server import _histogram, _sanitize


def test_sanitize_replaces_nan_and_infinity_with_none():
    assert _sanitize(float("nan")) is None
    assert _sanitize(float("inf")) is None
    assert _sanitize(float("-inf")) is None
    assert _sanitize(1.5) == 1.5


def test_sanitize_recurses_through_nested_dicts_and_lists():
    payload = {
        "global": {"r2": float("nan"), "rmse": 0.01},
        "per_sim": [{"rim_viability_pct": float("nan"), "core_viability_pct": 62.0}, {"ok": 1.0}],
        "plain": "unchanged",
        "n": 5,
    }
    result = _sanitize(payload)

    assert result["global"]["r2"] is None
    assert result["global"]["rmse"] == 0.01
    assert result["per_sim"][0]["rim_viability_pct"] is None
    assert result["per_sim"][0]["core_viability_pct"] == 62.0
    assert result["plain"] == "unchanged"
    assert result["n"] == 5


def test_sanitize_leaves_finite_values_and_other_types_untouched():
    payload = {"a": [1, 2, 3], "b": True, "c": None, "d": "text"}
    assert _sanitize(payload) == payload


def test_histogram_counts_and_bin_edges_are_consistent():
    residuals = np.abs(np.random.default_rng(0).normal(1e-3, 5e-4, size=500))
    hist = _histogram(residuals, bins=40)

    assert len(hist["bin_edges"]) == 41
    assert len(hist["counts"]) == 40
    assert sum(hist["counts"]) == len(residuals)
    assert all(math.isfinite(edge) for edge in hist["bin_edges"])


def test_histogram_is_json_serializable_via_sanitize():
    import json

    residuals = np.array([0.0, 1e-4, 2e-4, 3e-4])
    hist = _sanitize(_histogram(residuals, bins=5))
    json.dumps(hist)  # must not raise
