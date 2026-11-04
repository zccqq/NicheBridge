"""AnnData estimator, training loop, and portable fitted bundles."""

import json
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ._bridge import VPBridge
from ._config import NicheBridgeConfig
from ._data import count_matrix, make_pseudospots, reference_profiles
from ._network import MixtureBridge, negative_binomial_loss, poisson_loss
from ._mixtures import make_target_matched_pseudospots


def _spatial_graph(coords, k):
    """Build a fixed symmetric kNN graph from observed coordinates."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or len(coords) < 2:
        return None
    distances = ((coords[:, None, :2] - coords[None, :, :2]) ** 2).sum(axis=2)
    np.fill_diagonal(distances, np.inf)
    k = min(int(k), len(coords) - 1)
    edges = set()
    for i in range(len(coords)):
        neighbours = np.argpartition(distances[i], k)[:k]
        for j in neighbours:
            if i != j:
                edges.add(tuple(sorted((int(i), int(j)))))
    if not edges:
        return None
    pairs = np.asarray(sorted(edges), dtype=int)
    d = np.sqrt(distances[pairs[:, 0], pairs[:, 1]])
    scale = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    weights = np.exp(-((d / max(scale, 1e-08)) ** 2)).astype(np.float32)
    return (pairs[:, 0], pairs[:, 1], weights)


def _spatial_smoothness(probabilities, graph, device):
    if graph is None:
        return probabilities.new_tensor(0.0)
    left, right, weights = graph
    left = torch.as_tensor(left, device=device)
    right = torch.as_tensor(right, device=device)
    weights = torch.as_tensor(weights, device=device)
    difference = probabilities[left] - probabilities[right]
    return (weights[:, None] * difference.square()).sum() / (
        weights.sum().clamp_min(1e-08) * probabilities.shape[1]
    )


_FITTED_LAYER = object()


def _cross_entropy(target, prediction):
    return -(target * prediction.clamp_min(1e-08).log()).sum(dim=1).mean()


class NicheBridgeDeconvolver:
    """Fit a reference-constrained composition model using annotated counts.

    Cell fractions are conditional on fixed reference-derived relative RNA
    yields. They are not absolute cell counts. The batch covariate has two
    levels: reference (0) and the fitted spatial platform (1). Additional
    unseen platforms require a new fit. Evaluation labels are never read.
    """

    def __init__(self, config=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Pass a config or keyword settings, not both")
        self.config = config or NicheBridgeConfig(**kwargs)

    def _device(self):
        name = self.config.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(name)

    def _batch_metadata(self):
        """Describe the fitted correction without assuming perfect alignment."""
        offset = self.network_.platform_log().detach().cpu()
        enabled = self.config.batch_correction
        return {
            "batch_correction_enabled": enabled,
            "batch_correction_active": bool(
                enabled and torch.any(offset.abs() > 1e-08)
            ),
            "batch_correction_method": (
                "reference_anchored_gene_scaling" if enabled else "none"
            ),
            "batch_correction_scope": "reference and one fitted spatial platform",
            "platform_log_rms": float(offset.square().mean().sqrt()),
            "bridge_domain_conditioning_enabled": bool(
                enabled and self.config.bridge_weight > 0
            ),
        }

    def fit(
        self,
        reference,
        spatial,
        cell_type_key="cell_type",
        counts_layer=None,
        verbose=True,
    ):
        """Fit signatures and model without accessing spatial annotations."""
        started = time.perf_counter()
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        torch.set_num_threads(min(4, torch.get_num_threads()))
        if cell_type_key not in reference.obs:
            raise KeyError(f"Missing reference annotation: {cell_type_key}")
        labels = reference.obs[cell_type_key]
        if labels.isna().any():
            raise ValueError("Every reference cell requires a type label")
        labels = labels.astype(str).to_numpy()
        self.genes_ = [str(x) for x in spatial.var_names if x in reference.var_names]
        if len(self.genes_) < 2:
            raise ValueError("At least two shared genes are required")
        ref = count_matrix(reference, counts_layer, self.genes_)
        target = count_matrix(spatial, counts_layer, self.genes_)
        self.types_ = sorted(np.unique(labels).tolist())
        signatures, yields = reference_profiles(
            ref, labels, self.types_, mode=cfg.profile_mode
        )
        target_dense = target.toarray()
        valid_spots = np.flatnonzero(target_dense.sum(axis=1) > 0)
        if len(valid_spots) == 0:
            raise ValueError("At least one spatial observation must contain counts")
        n_pseudospots = cfg.pseudospots
        if cfg.pseudo_strategy == "balanced":
            n_pseudospots = min(
                n_pseudospots, max(10 * len(valid_spots), len(valid_spots))
            )
        if cfg.pseudo_library_match:
            synthetic, synthetic_cell, synthetic_rna = make_target_matched_pseudospots(
                ref,
                labels,
                self.types_,
                target_dense[valid_spots].sum(axis=1),
                n_pseudospots,
                cfg.pseudo_min_cells,
                cfg.pseudo_max_cells,
                cfg.seed,
                strategy=cfg.pseudo_strategy,
                active_max=cfg.pseudo_active_max,
                single_fraction=cfg.pseudo_single_fraction,
            )
        else:
            synthetic, synthetic_cell, synthetic_rna = make_pseudospots(
                ref,
                labels,
                self.types_,
                n_pseudospots,
                cfg.pseudo_min_cells,
                cfg.pseudo_max_cells,
                cfg.seed,
                strategy=cfg.pseudo_strategy,
                active_max=cfg.pseudo_active_max,
                single_fraction=cfg.pseudo_single_fraction,
            )
        graph = None
        if cfg.spatial_weight > 0 and "spatial" in spatial.obsm:
            graph = _spatial_graph(
                np.asarray(spatial.obsm["spatial"])[valid_spots], cfg.spatial_k
            )
        device = self._device()
        self.network_ = MixtureBridge(signatures, yields, cfg).to(device)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        bridge = VPBridge(time_mode=cfg.bridge_time_mode)
        rng = np.random.default_rng(cfg.seed + 1)
        history = []
        total_steps = cfg.pretrain_epochs + cfg.epochs
        self.network_.train()
        for step in range(total_steps):
            pretrain = step < cfg.pretrain_epochs
            joint_step = step - cfg.pretrain_epochs + 1
            pick = rng.choice(
                len(synthetic), size=min(cfg.batch_size, len(synthetic)), replace=False
            )
            ref_count = torch.as_tensor(synthetic[pick], device=device)
            ref_cell = torch.as_tensor(synthetic_cell[pick], device=device)
            ref_rna = torch.as_tensor(synthetic_rna[pick], device=device)
            ref_batch = torch.zeros(len(pick), device=device)
            if pretrain:
                counts, batch = (ref_count, ref_batch)
            else:
                spatial_pick = (
                    valid_spots
                    if graph is not None
                    else rng.choice(
                        valid_spots,
                        size=min(cfg.batch_size, len(valid_spots)),
                        replace=False,
                    )
                )
                spatial_count = torch.as_tensor(
                    target_dense[spatial_pick], device=device
                )
                counts = torch.cat([ref_count, spatial_count], dim=0)
                batch = torch.cat(
                    [ref_batch, torch.ones(len(spatial_pick), device=device)]
                )
            p, z, mu_z, logvar, observed = self.network_.encode(
                counts, batch, mask=True, sample=True
            )
            p_mean = torch.softmax(
                self.network_.composition(mu_z) / cfg.composition_temperature, dim=-1
            )
            expected, q, _ = self.network_.decode(p, counts.sum(dim=1), batch)
            n_ref = len(ref_count)
            likelihood_loss = (
                poisson_loss if cfg.likelihood == "poisson" else negative_binomial_loss
            )
            if cfg.likelihood == "poisson":
                ref_loss = likelihood_loss(
                    counts[:n_ref], expected[:n_ref], self.network_.gene_weights
                )
                st_loss = likelihood_loss(
                    counts[n_ref:], expected[n_ref:], self.network_.gene_weights
                )
            else:
                ref_loss = likelihood_loss(
                    counts[:n_ref],
                    expected[:n_ref],
                    self.network_.log_dispersion,
                    self.network_.gene_weights,
                )
                st_loss = likelihood_loss(
                    counts[n_ref:],
                    expected[n_ref:],
                    self.network_.log_dispersion,
                    self.network_.gene_weights,
                )
            nb_ref = ref_loss
            nb = nb_ref if pretrain else 0.5 * (ref_loss + st_loss)
            supervised = _cross_entropy(ref_cell, p[:n_ref]) + 0.5 * _cross_entropy(
                ref_rna, q[:n_ref]
            )
            kl = 0.5 * (mu_z.square() + logvar.exp() - 1 - logvar).sum(dim=1).mean()
            x0, xT = (self.network_.features(counts), self.network_.features(expected))
            hidden = 1 - observed
            masked = ((xT - x0).square() * hidden).sum() / hidden.sum().clamp_min(1)
            platform = self.network_.platform_log().square().mean()
            bridge_ramp = (
                0.0 if pretrain else min(1.0, joint_step / max(cfg.bridge_warmup, 1))
            )
            bridge_loss = (
                bridge.loss(self.network_, x0, xT, z, batch)
                if cfg.bridge_weight > 0 and bridge_ramp > 0
                else counts.new_tensor(0.0)
            )
            if not pretrain and cfg.spatial_weight > 0 and (graph is not None):
                smooth_loss = _spatial_smoothness(p_mean[n_ref:], graph, device)
            else:
                smooth_loss = counts.new_tensor(0.0)
            if not pretrain and cfg.entropy_weight > 0:
                spatial_entropy = (
                    -(p[n_ref:] * p[n_ref:].clamp_min(1e-08).log()).sum(dim=1).mean()
                )
            else:
                spatial_entropy = counts.new_tensor(0.0)
            kl_ramp = min(1.0, (step + 1) / 100)
            loss = nb + cfg.pseudo_weight * supervised + cfg.kl_weight * kl_ramp * kl
            loss = loss + cfg.mask_weight * masked
            loss = (
                loss
                + cfg.bridge_weight * bridge_ramp * bridge_loss
                + cfg.platform_penalty * platform
            )
            loss = (
                loss
                + cfg.spatial_weight * smooth_loss
                + cfg.entropy_weight * spatial_entropy
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss at step {step + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.network_.parameters(), 10.0
            )
            optimizer.step()
            if step == 0 or (step + 1) % cfg.log_every == 0 or step == total_steps - 1:
                record = {
                    "step": step + 1,
                    "phase": "pretrain" if pretrain else "joint",
                    "loss": float(loss.detach()),
                    "nb": float(nb.detach()),
                    "pseudo": float(supervised.detach()),
                    "kl": float(kl.detach()),
                    "masked": float(masked.detach()),
                    "bridge": float(bridge_loss.detach()),
                    "bridge_scale": cfg.bridge_weight * bridge_ramp,
                    "spatial": float(smooth_loss.detach()),
                    "spatial_scale": cfg.spatial_weight,
                    "entropy": float(spatial_entropy.detach()),
                    "platform_penalty": float(platform.detach()),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                history.append(record)
                if verbose:
                    print(
                        f"step={step + 1}/{total_steps} phase={record['phase']} loss={record['loss']:.4f} nb={record['nb']:.4f} bridge={record['bridge']:.4f}",
                        flush=True,
                    )
        self.network_.eval()
        self.history_ = pd.DataFrame(history)
        self.metadata_ = {
            "format_version": 1,
            "version": "0.0.1",
            "training_seconds": time.perf_counter() - started,
            "n_reference_cells": ref.shape[0],
            "n_spots": target.shape[0],
            "n_genes": len(self.genes_),
            "n_types": len(self.types_),
            "zero_count_spots": int(target.shape[0] - len(valid_spots)),
            "counts_layer": counts_layer,
            "cell_type_key": cell_type_key,
            "rna_yield_assumption": (
                "Reference-derived RNA weights; size-factor profiles use raw-library scaling, so RNA and cell outputs are not independently calibrated"
                if cfg.profile_mode == "size_factor"
                else "Fixed mean reference counts per cell on the modeled gene panel; not externally calibrated"
            ),
            "batch_vocabulary": {"reference": 0, "fitted_spatial_platform": 1},
            "inference": "Clean student posterior mean; uniform cell fractions for zero-count spots",
            "device": str(device),
            "torch_version": torch.__version__,
            "pseudo_strategy": cfg.pseudo_strategy,
            "n_pseudospots": int(len(synthetic)),
            "spatial_weight": cfg.spatial_weight,
            "spatial_graph_edges": 0 if graph is None else int(len(graph[0])),
            "marker_weight": cfg.marker_weight,
            "bridge_time_mode": cfg.bridge_time_mode,
            "profile_mode": cfg.profile_mode,
            "composition_temperature": cfg.composition_temperature,
            "entropy_weight": cfg.entropy_weight,
            "pseudo_library_match": cfg.pseudo_library_match,
            "likelihood": cfg.likelihood,
            "reference_anchor_weight": cfg.reference_anchor_weight,
        }
        self.metadata_.update(self._batch_metadata())
        return self

    @torch.no_grad()
    def predict(self, spatial, counts_layer=_FITTED_LAYER, batch="spatial"):
        """Return named fractions, latent means, and count reconstructions."""
        if not hasattr(self, "network_"):
            raise RuntimeError("Fit or load a model before prediction")
        if batch not in ("reference", "spatial"):
            raise ValueError(
                "batch must be reference or spatial; unseen platforms require refitting"
            )
        if counts_layer is _FITTED_LAYER:
            counts_layer = self.metadata_["counts_layer"]
        matrix = count_matrix(spatial, counts_layer, self.genes_)
        self.network_.eval()
        device = next(self.network_.parameters()).device
        outputs = {
            key: []
            for key in (
                "cell_fraction",
                "rna_fraction",
                "reference_rna_fraction",
                "latent",
                "expected_counts",
            )
        }
        for start in range(0, matrix.shape[0], self.config.batch_size):
            counts = torch.as_tensor(
                matrix[start : start + self.config.batch_size].toarray(), device=device
            )
            covariate = torch.full(
                (len(counts),), float(batch == "spatial"), device=device
            )
            p, _, latent, _, _ = self.network_.encode(counts, covariate)
            zero = counts.sum(dim=1) == 0
            p[zero] = 1.0 / len(self.types_)
            expected, q, q_observed = self.network_.decode(
                p, counts.sum(dim=1), covariate
            )
            for key, value in (
                ("cell_fraction", p),
                ("rna_fraction", q_observed),
                ("reference_rna_fraction", q),
                ("latent", latent),
                ("expected_counts", expected),
            ):
                outputs[key].append(value.cpu().numpy())
        outputs = {key: np.concatenate(value, axis=0) for key, value in outputs.items()}
        for key in ("cell_fraction", "rna_fraction", "reference_rna_fraction"):
            outputs[key] = pd.DataFrame(
                outputs[key], index=spatial.obs_names.copy(), columns=self.types_
            )
        outputs["low_information"] = np.asarray(matrix.sum(axis=1)).ravel() == 0
        return outputs

    def transform(self, spatial, counts_layer=_FITTED_LAYER, copy=False):
        """Write composition and latent results into an AnnData object."""
        target = spatial.copy() if copy else spatial
        result = self.predict(target, counts_layer)
        for key in ("cell_fraction", "rna_fraction", "reference_rna_fraction"):
            target.obsm[f"nichebridge_{key}"] = result[key]
        target.obsm["X_nichebridge"] = result["latent"]
        target.obs["nichebridge_low_information"] = result["low_information"]
        target.uns["nichebridge"] = {
            "version": "0.0.1",
            "types": self.types_,
            "genes": self.genes_,
            "rna_yield": self.network_.rna_yield.detach().cpu().numpy(),
            "cell_fraction_assumption": self.metadata_["rna_yield_assumption"],
        }
        target.uns["nichebridge"].update(self._batch_metadata())
        return target

    def save(self, directory):
        """Save inference weights and ordered feature/type metadata."""
        if not hasattr(self, "network_"):
            raise RuntimeError("Cannot save an unfitted model")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in self.network_.state_dict().items()
            },
            directory / "weights.pt",
        )
        bundle = {
            "config": self.config.to_dict(),
            "metadata": self.metadata_,
            "genes": self.genes_,
            "types": self.types_,
        }
        (directory / "model.json").write_text(
            json.dumps(bundle, indent=2), encoding="utf-8"
        )
        self.history_.to_csv(directory / "history.csv", index=False)

    @classmethod
    def load(cls, directory, device="auto"):
        """Restore a fitted model from a versioned bundle."""
        directory = Path(directory)
        info = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        if info["metadata"].get("format_version") != 1:
            raise ValueError("Unsupported bundle format")
        config = info["config"]
        config["device"] = device
        model = cls(NicheBridgeConfig(**config))
        state = torch.load(
            directory / "weights.pt", map_location="cpu", weights_only=True
        )
        model.network_ = MixtureBridge(
            state["signatures"], state["rna_yield"], model.config
        )
        model.network_.load_state_dict(state, strict=True)
        model.network_.to(model._device()).eval()
        model.genes_, model.types_ = (info["genes"], info["types"])
        metadata_keys = {
            "format_version",
            "version",
            "training_seconds",
            "n_reference_cells",
            "n_spots",
            "n_genes",
            "n_types",
            "zero_count_spots",
            "counts_layer",
            "cell_type_key",
            "batch_vocabulary",
            "inference",
            "device",
            "torch_version",
            "pseudo_strategy",
            "n_pseudospots",
            "spatial_weight",
            "spatial_graph_edges",
            "marker_weight",
            "bridge_time_mode",
            "profile_mode",
            "composition_temperature",
            "entropy_weight",
            "pseudo_library_match",
            "likelihood",
            "reference_anchor_weight",
        }
        model.metadata_ = {
            key: value
            for key, value in info["metadata"].items()
            if key in metadata_keys
        }
        model.metadata_["bridge_time_mode"] = model.config.bridge_time_mode
        model.metadata_["pseudo_strategy"] = model.config.pseudo_strategy
        model.metadata_["rna_yield_assumption"] = (
            "Reference-derived RNA weights; size-factor profiles use raw-library scaling, so RNA and cell outputs are not independently calibrated"
            if model.config.profile_mode == "size_factor"
            else "Fixed mean reference counts per cell on the modeled gene panel; not externally calibrated"
        )
        model.metadata_.update(model._batch_metadata())
        model.history_ = pd.read_csv(directory / "history.csv")
        return model


__all__ = ["NicheBridgeDeconvolver"]
