#!/usr/bin/env python3
"""CLI: draw a layer-by-layer diagram of the BIOPINN network architecture.

Reads only configs/*.yaml -- no trained checkpoint or dataset needed, so this
can run any time, including before the Colab training notebook has finished
(or even started). Useful as a paper's Methods/Architecture figure.

Usage:
    python scripts/visualize_architecture.py [--experiment NAME] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve_path
from src.visualize import plot_architecture_diagram


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None, help="Experiment config name (default: default_config.yaml)")
    parser.add_argument("--out", default=None, help="Output image path (default: paths.results/figures/architecture_diagram.png)")
    args = parser.parse_args()

    config = load_config(args.experiment)
    out_path = Path(args.out) if args.out else resolve_path(config, "results") / "figures" / "architecture_diagram.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_architecture_diagram(config, save_path=str(out_path))
    print("saved:", out_path)


if __name__ == "__main__":
    main()
