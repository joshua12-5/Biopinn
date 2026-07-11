#!/usr/bin/env python3
"""CLI: launch the BIOPINN results dashboard (local web UI).

Loads artifacts/biopinn_model.pt (+ optionally a w_phys=0 ablation baseline)
once and serves an interactive dashboard: live parameter controls query the
PINN surrogate, plus cached panels for the six-metric evaluation, the
per-radius optimization grid search, and the ablation comparison. Never
retrains -- requires paths.model_checkpoint and a processed dataset
(paths.processed) to already exist.

Usage:
    python scripts/run_dashboard.py [--host HOST] [--port PORT]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on source changes (development only)")
    args = parser.parse_args()

    if args.experiment:
        os.environ["BIOPINN_EXPERIMENT"] = args.experiment

    print(f"Starting BIOPINN dashboard at http://{args.host}:{args.port}")
    uvicorn.run("app.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
