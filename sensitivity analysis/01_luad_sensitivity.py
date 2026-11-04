"""Fit NicheBridge LUAD models over a consistency-weight sensitivity grid.

Install NicheBridge first. Inputs and options are described by --help. Defaults
use all supplied observations, 200 pretraining and 800 joint updates per weight.
Outcomes are projected and evaluated only after fitting; no values are replaced
by plotting targets or results from earlier runs.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _luad import main

if __name__ == "__main__":
    main("sensitivity")
