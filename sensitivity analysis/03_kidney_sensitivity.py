"""Sweep the current NicheBridge slice_weight coefficient on kidney sections.

Required: --input kidney.h5ad. Supply --slice-weights 0 0.05 0.1 0.5 to change
the actual cross-slice loss coefficient. This is a fresh current-model study;
it does not claim equivalence to a different model's gated objective.
Use --reference domains.h5ad for fixed-domain cross-section AUROC.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _kidney_common import run


if __name__ == "__main__":
    run("sensitivity")
