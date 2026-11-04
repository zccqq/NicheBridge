"""Fresh NicheBridge multi-slice fitting on a reproducible kidney subset.

This tutorial helper is separate from the public model API. The supplied data's
log1p layer stores log1p integer counts; inversion is validated before fitting.
No supplied embedding, reconstruction, or anatomical domain enters training.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import torch

import nichebridge as nb


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sparse_rows(group, rows):
    if group.attrs.get("encoding-type") != "csr_matrix":
        raise ValueError("The tutorial input requires a CSR log1p layer")
    pointers = np.asarray(group["indptr"])
    values, indices, output_ptr = [], [], [0]
    for row in rows:
        start, stop = pointers[row:row + 2]
        values.append(np.asarray(group["data"][start:stop], dtype=np.float64))
        indices.append(np.asarray(group["indices"][start:stop]))
        output_ptr.append(output_ptr[-1] + stop - start)
    return sparse.csr_matrix((np.concatenate(values), np.concatenate(indices),
                              np.asarray(output_ptr)),
                             shape=(len(rows), int(group.attrs["shape"][1])))


def run_fresh_fit(input_path, output_dir, *, seed=42, spots_per_section=32,
                  n_genes=256, pretrain_epochs=20, epochs=40,
                  coupling_steps=16, ode_steps=64):
    """Train current NicheBridge and return saved results and figure paths.

    The default 256-spot fit demonstrates execution on all eight sections. It
    does not replace the complete dataset's supplied results or claim recovery
    of manuscript effect sizes. Sampling uses section IDs only.
    """
    started = time.perf_counter()
    input_path, output_dir = Path(input_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if isinstance(spots_per_section, bool) or spots_per_section < 2:
        raise ValueError("spots_per_section must be at least two")
    rng = np.random.default_rng(seed)
    torch.set_num_threads(2)
    with h5py.File(input_path, "r") as handle:
        obs = ad.io.read_elem(handle["obs"])
        var = ad.io.read_elem(handle["var"])
        required = ["section", "age", "sex"]
        if not set(required).issubset(obs) or obs[required].isna().any().any():
            raise ValueError("Complete section, age and sex annotations are required")
        sections = obs["section"].astype(str)
        rows = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(sections.to_numpy() == section),
                       min(spots_per_section, int((sections == section).sum())), replace=False)
            for section in sorted(sections.unique())]))
        logged = _sparse_rows(handle["layers"]["log1p"], rows)
        recovered = np.rint(np.expm1(logged.data))
        error = float(np.max(np.abs(np.log1p(recovered) - logged.data)))
        if not np.isfinite(recovered).all() or (recovered < 0).any() or error > 1e-5:
            raise ValueError("Input layer cannot be validated as log1p integer counts")
        counts = sparse.csr_matrix((recovered.astype(np.float32), logged.indices, logged.indptr),
                                   shape=logged.shape)
        coordinates = np.asarray(handle["obsm"]["spatial"])[rows]
    training = ad.AnnData(counts, obs=obs.iloc[rows][required].copy(),
                         var=pd.DataFrame(index=var.index.copy()))
    training.obsm["spatial"] = coordinates
    config = nb.TrajectoryConfig(
        seed=seed, device="cpu", latent_dim=16, hidden_dim=64,
        n_top_genes=n_genes, pretrain_epochs=pretrain_epochs, epochs=epochs,
        coupling_steps=coupling_steps, ode_steps=ode_steps)
    model = nb.NicheBridgeTrajectory(config).fit(
        {"spatial": training}, slice_key="section", verbose=False)
    latent = model.transform("spatial")
    reconstruction = model.reconstruct("spatial")
    if not np.isfinite(latent).all() or not np.isfinite(reconstruction.to_numpy()).all():
        raise FloatingPointError("Fresh model returned nonfinite outputs")
    if not np.any(reconstruction.to_numpy() > 0):
        raise RuntimeError("Fresh reconstruction is entirely zero")
    features = reconstruction.columns
    result = ad.AnnData(reconstruction.to_numpy(dtype=np.float32),
                        obs=training.obs.copy(), var=pd.DataFrame(index=features))
    result.layers["nichebridge_reconstructed"] = result.X.copy()
    result.layers["counts"] = training[:, features].X.copy()
    result.layers["observed_log1p"] = model.data_["spatial"].cpu().numpy() * config.log_scale
    result.obsm["X_nichebridge"] = latent
    result.obsm["spatial"] = coordinates.copy()
    domains = nb.analysis.identify_domains(pd.DataFrame(latent, index=result.obs_names), seed=seed)
    result.obs["nichebridge_domain"] = domains.astype(str).astype("category")
    result.uns["nichebridge"] = {
        "version": nb.__version__, "kind": "fresh_subset_fit", "seed": seed,
        "training_source": "validated inverse of supplied log1p counts",
        "output_scale": "library-normalized log1p", "domain_identity": "unsupervised numeric clusters"}
    history = pd.DataFrame(model.history_)
    history.to_csv(output_dir / "training_history.csv", index=False)
    checkpoint = output_dir / "model"
    model.save(checkpoint)
    result_path = output_dir / "kidney_nichebridge_fresh.h5ad"
    result.write_h5ad(result_path)
    embedding = pd.DataFrame(latent, index=result.obs_names)
    embedding.to_csv(output_dir / "latent.csv")
    reconstruction.to_csv(output_dir / "reconstruction.csv")
    domains.to_csv(output_dir / "domains.csv")
    figures = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for field in ("bridge", "slice", "slice_distribution"):
        axes[0].plot(np.arange(1, len(history) + 1), history[field], label=field)
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].set(xlabel="Optimization update", ylabel="Objective component", title="Fresh NicheBridge training")
    axes[0].axvline(pretrain_epochs + 0.5, color="0.6", linestyle="--")
    axes[0].legend(frameon=False)
    for section in sorted(training.obs.section.astype(str).unique()):
        mask = training.obs.section.astype(str).to_numpy() == section
        axes[1].scatter(latent[mask, 0], latent[mask, 1], s=12, label=section, alpha=0.8)
    axes[1].set(xlabel="Latent dimension 1", ylabel="Latent dimension 2", title="Newly fitted subset embedding")
    axes[1].legend(fontsize=6, frameon=False)
    for extension in ("png", "pdf"):
        path = output_dir / f"fresh_fit_overview.{extension}"
        fig.savefig(path, dpi=180)
        if extension == "png": figures.append(path)
    plt.close(fig)
    summary = {
        "status": "completed", "kind": "fresh_fit", "package_version": nb.__version__,
        "input_sha256": _sha256(input_path), "seed": seed, "config": asdict(config),
        "n_input_spots": int(len(obs)), "n_selected_spots": int(training.n_obs),
        "n_selected_genes": int(result.n_vars), "n_sections": int(sections.nunique()),
        "sampling": "Seeded uniform sampling within each section; no anatomical labels used",
        "selected_obs_names": training.obs_names.tolist(), "selected_features": features.tolist(),
        "raw_count_recovery_max_log_error": error, "elapsed_seconds": time.perf_counter() - started,
        "source_annotations_used_for_training": ["section"],
        "fit_uses_supplied_latent_or_reconstruction": False,
        "reconstruction_nonzero_fraction": float(np.mean(reconstruction.to_numpy() > 0)),
        "limitations": "Small subset demonstration; no full-cohort or biological accuracy claim",
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                     for name in ("torch", "anndata", "scanpy", "numpy", "pandas")}}}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "figures": figures, "adata": result_path, "checkpoint": checkpoint}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run_fresh_fit(args.input, args.output, seed=args.seed)["summary"], indent=2))
