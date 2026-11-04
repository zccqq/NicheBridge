"""Fit five NicheBridge mouse-brain ablations and evaluate Leiden ARI after fitting.

Use --data-root to locate data2. No external model source is imported.
Run --help for settings and explicit optional observation subsetting.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _mouse_brain import main


if __name__ == "__main__":
    main("ablation")
