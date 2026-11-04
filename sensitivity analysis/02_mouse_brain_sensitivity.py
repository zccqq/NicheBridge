"""Sweep the actual NicheBridge cross-modal loss coefficient on paired mouse data.

--modal-weights changes modal_weight in the trained NicheBridge objective.
These are new-model experiments, not a promise of identical earlier scores.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _mouse_brain import main


if __name__ == "__main__":
    main("sensitivity")
