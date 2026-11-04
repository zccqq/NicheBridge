"""Fit full/no_batch_removal/no_vae/no_diffusion NicheBridge kidney models.

Required: --input kidney.h5ad. Use --reference domains.h5ad for fixed-domain
cross-section AUROC. Relative input paths use --data-root DATA_ROOT/data3.
The default samples 128 spots per section without using domain labels and fits
at most 512 genes. --spots-per-section 0 requests all spots and still enforces
the explicit dense model limits. Evaluation annotations never enter fitting.

The no-diffusion ablation is a separately fitted variational autoencoder with
latent correspondence losses. Its variance sampling and KL remain enabled;
only the bridge, score network, reverse rollout and ODE are removed.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproducibility"))

from _kidney_common import run


if __name__ == "__main__":
    run("ablation")
