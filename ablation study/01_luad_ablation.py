"""Fit the full, no-single-cell, no-VAE and no-diffusion NicheBridge LUAD models.

Install NicheBridge first. Inputs and options are described by --help. Defaults
use all supplied observations, 200 pretraining and 800 joint updates for each
of three seeds. Outcomes are projected and evaluated only after fitting.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _luad import main

if __name__ == "__main__":
    main("ablation")
