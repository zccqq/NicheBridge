"""Trajectory transport and generative reconstruction for NicheBridge.

This implementation uses the Brownian special case of the manuscript's general
diffusion: f=0, g=sigma, T=1. Observations are rows. A transport(target, source)
matrix has target observations as rows and source observations as columns.
The solver integrates the probability-flow ODE, not a decoder-only prediction.
Training is full batch; explicit matrix-size limits prevent accidental allocation
of atlas-sized dense relations. Use a representative subset for larger inputs.
"""

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import hashlib
import json
import math
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class TrajectoryConfig:
    """Settings for independent bridges followed by joint trajectory training."""

    latent_dim: int = 16
    hidden_dim: int = 128
    n_top_genes: int = 3000
    hvg_method: str = "scanpy"
    use_vae: bool = True
    use_diffusion: bool = True
    pretrain_epochs: int = 50
    epochs: int = 200
    learning_rate: float = 0.001
    diffusion_sigma: float = 0.5
    scale_weight: float = 1.0
    modal_weight: float = 1.0
    slice_weight: float = 0.1
    kl_weight: float = 0.001
    endpoint_weight: float = 0.1
    temperature: float = 0.2
    condition_dropout: float = 0.1
    target_sum: float = 10000.0
    log_scale: float = 4.0
    normalize_library: bool = True
    time_epsilon: float = 0.001
    trajectory_times: int = 2
    coupling_steps: int = 16
    ode_steps: int = 64
    max_relation_entries: int = 4_000_000
    max_matrix_entries: int = 20_000_000
    max_trajectory_entries: int = 25_000_000
    seed: int = 0
    device: str = "auto"

    def __post_init__(self):
        for name in ("latent_dim", "hidden_dim", "n_top_genes", "epochs",
                     "trajectory_times", "coupling_steps", "ode_steps", "max_relation_entries",
                     "max_matrix_entries", "max_trajectory_entries"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.pretrain_epochs, bool) or not isinstance(self.pretrain_epochs, int) or self.pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be a nonnegative integer")
        if self.ode_steps < 2:
            raise ValueError("ode_steps must be at least two")
        if self.coupling_steps < 2:
            raise ValueError("coupling_steps must be at least two")
        if self.trajectory_times > self.coupling_steps:
            raise ValueError("trajectory_times cannot exceed coupling_steps")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if not isinstance(self.device, str) or re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device) is None:
            raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")
        if self.hvg_method not in ("scanpy", "variance"):
            raise ValueError("hvg_method must be 'scanpy' or 'variance'")
        for name in ("use_vae", "use_diffusion"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("learning_rate", "diffusion_sigma", "temperature", "target_sum", "log_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("scale_weight", "modal_weight", "slice_weight", "kl_weight", "endpoint_weight"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not self.use_diffusion and self.endpoint_weight <= 0:
            raise ValueError("endpoint_weight must be positive for the no-diffusion autoencoder")
        if not math.isfinite(self.time_epsilon) or not 0 < self.time_epsilon < 0.1:
            raise ValueError("time_epsilon must lie between 0 and 0.1")
        if not math.isfinite(self.condition_dropout) or not 0 <= self.condition_dropout < 1:
            raise ValueError("condition_dropout must lie in [0, 1)")
        if not isinstance(self.normalize_library, bool):
            raise ValueError("normalize_library must be boolean")


class BrownianBridge:
    """Analytic bridge q(x_t|x_0,x_T) and its probability-flow drift."""

    def __init__(self, sigma=0.5):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("sigma must be finite and positive")
        self.sigma = float(sigma)

    def sample(self, x0, endpoint, time, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        variance = self.sigma ** 2 * time * (1 - time)
        return (1 - time) * x0 + time * endpoint + variance.sqrt() * noise

    def conditional_score(self, xt, x0, endpoint, time):
        return -(xt - (1 - time) * x0 - time * endpoint) / (self.sigma ** 2 * time * (1 - time))

    def drift(self, xt, x0_prediction, endpoint, time):
        # Algebraically identical to sigma^2*h - sigma^2*score/2,
        # h=(endpoint-xt)/(sigma^2*(1-time)); avoids cancellation.
        return (endpoint - xt) / (2 * (1 - time)) + (xt - x0_prediction) / (2 * time)


class _ModalityBridge(nn.Module):
    def __init__(self, n_features, n_slices, config):
        super().__init__()
        h, z = config.hidden_dim, config.latent_dim
        self.n_slices = n_slices
        self.variational = config.use_vae
        self.encoder = nn.Sequential(nn.Linear(n_features + n_slices, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.mean = nn.Linear(h, z)
        self.log_variance = nn.Linear(h, z) if self.variational else None
        self.decoder = nn.Sequential(nn.Linear(z + n_slices, h), nn.SiLU(), nn.Linear(h, n_features), nn.Softplus())
        self.denoiser = (nn.Sequential(nn.Linear(2 * n_features + 2 * z + n_slices + 3, h),
                                      nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, n_features))
                         if config.use_diffusion else None)

    def batch(self, codes):
        return F.one_hot(codes, self.n_slices).float()

    def encode(self, x, codes, sample=False):
        h = self.encoder(torch.cat((x, self.batch(codes)), dim=1))
        mean = self.mean(h)
        log_var = self.log_variance(h).clamp(-10, 8) if self.variational else torch.zeros_like(mean)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(mean) if sample and self.variational else mean
        return z, mean, log_var

    def endpoint(self, z, codes):
        return self.decoder(torch.cat((z, self.batch(codes)), dim=1))

    def denoise(self, xt, time, endpoint, z, codes, condition):
        if self.denoiser is None:
            raise RuntimeError("The no-diffusion model has no score network")
        t = time.expand(xt.shape[0], 1)
        time_features = torch.cat((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), dim=1)
        return self.denoiser(torch.cat((xt, endpoint, z, condition, self.batch(codes), time_features), dim=1))


class _Relation(nn.Module):
    def __init__(self, dim, temperature, fixed=None):
        super().__init__()
        self.temperature = temperature
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.query.weight)
        nn.init.eye_(self.key.weight)
        self.register_buffer("fixed", fixed)

    def forward(self, target, source):
        if self.fixed is not None:
            return self.fixed
        q = F.normalize(self.query(target), dim=1)
        k = F.normalize(self.key(source), dim=1)
        return torch.softmax(q @ k.T / self.temperature, dim=1)


def _normalise_reverse(weights):
    """Normalize transposed mass, assigning uniform rows to zero-mass columns.

    Exact zeros can arise from floating-point softmax underflow. Such a source
    observation has no identified reverse correspondence; a uniform row is the
    explicit fallback rather than returning an invalid all-zero transport row.
    """
    reverse = weights.T
    mass = reverse.sum(1, keepdim=True)
    positive = mass > 0
    normalized = reverse / torch.where(positive, mass, torch.ones_like(mass))
    return torch.where(positive, normalized, torch.full_like(reverse, 1 / reverse.shape[1]))


class NicheBridgeTrajectory:
    """Independent diffusion bridges with learned cross-dataset transport.

    Canonical modality names are ``spatial``, ``reference``, ``bulk``, ``rna``
    and ``atac``. The first three use a common gene panel; RNA/ATAC keep separate
    feature spaces. ``paired=True`` requires identical RNA/ATAC observation IDs
    and builds their exact correspondence, respecting different row orders.
    No phenotype or cell-type labels are read while fitting this engine.

    Reconstruction returns library-normalized log1p values (or log1p input values
    when normalize_library=False), not raw counts. This model is transductive for
    transport and generation: relations refer to the observations used in fit.
    """

    def __init__(self, config=None, **kwargs):
        if config is not None and kwargs:
            raise ValueError("Supply config or keyword settings, not both")
        if config is not None and not isinstance(config, TrajectoryConfig):
            raise TypeError("config must be a TrajectoryConfig instance")
        self.config = config or TrajectoryConfig(**kwargs)
        self.bridge = BrownianBridge(self.config.diffusion_sigma)

    def _device(self):
        requested = self.config.device
        return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)

    def _tensor(self, values, dtype=torch.float32):
        return torch.as_tensor(values, dtype=dtype, device=self.device_)

    def _matrix(self, adata, layer):
        if not adata.obs_names.is_unique or not adata.var_names.is_unique:
            raise ValueError("Observation and feature identifiers must be unique")
        if adata.n_obs < 1 or adata.n_vars < 1:
            raise ValueError("Each modality must contain observations and features")
        if layer is not None and layer not in adata.layers:
            raise ValueError(f"Missing layer {layer!r}")
        values = sparse.csr_matrix(adata.X if layer is None else adata.layers[layer], dtype=np.float64)
        if not np.isfinite(values.data).all() or (values.data < 0).any():
            raise ValueError("Input profiles must be finite, nonnegative, and unlogged")
        if self.config.normalize_library:
            total = np.asarray(values.sum(1)).ravel()
            values = sparse.diags(np.divide(self.config.target_sum, total, out=np.zeros_like(total), where=total > 0)) @ values
        values = values.tocsr()
        values.data = np.log1p(values.data) / self.config.log_scale
        return values

    def _check_relation_size(self, n_target, n_source):
        if n_target * n_source > self.config.max_relation_entries:
            raise ValueError("Dense transport exceeds max_relation_entries; subset observations explicitly or increase the configured limit")

    def _feature_indices(self, values):
        if values.shape[1] <= self.config.n_top_genes:
            return np.arange(values.shape[1])
        if self.config.hvg_method == "variance":
            means = np.asarray(values.mean(0)).ravel()
            scores = np.asarray(values.power(2).mean(0)).ravel() - means ** 2
        else:
            import anndata as ad
            import scanpy as sc
            panel = ad.AnnData(values * self.config.log_scale)
            try:
                sc.pp.highly_variable_genes(panel, flavor="seurat", n_top_genes=self.config.n_top_genes, inplace=True)
            except (ValueError, IndexError, ZeroDivisionError) as exc:
                raise ValueError("Scanpy HVG selection failed on a degenerate feature distribution; inspect the input or explicitly choose hvg_method='variance'") from exc
            scores = panel.var["dispersions_norm"].to_numpy(dtype=float)
            if not np.isfinite(scores).any():
                raise ValueError("Scanpy HVG scores are all nonfinite; inspect the input or explicitly choose hvg_method='variance'")
            scores = np.where(np.isfinite(scores), scores, -np.inf)
        # Stable sorting breaks tied dispersion scores by the original feature
        # order and prevents Scanpy's tie-inclusive mask from exceeding budget.
        return np.argsort(-scores, kind="stable")[:self.config.n_top_genes]

    def _prepare(self, modalities, layers, slice_key, paired, relations):
        if not modalities:
            raise ValueError("Provide at least one modality")
        if any(not isinstance(name, str) or not name.isidentifier() for name in modalities):
            raise ValueError("Modality names must be valid Python identifiers")
        self.modality_names_ = list(modalities)
        self.layers_ = {name: (layers or {}).get(name) for name in modalities}
        matrices = {name: self._matrix(adata, self.layers_[name]) for name, adata in modalities.items()}
        self.genes_, self.obs_names_, self.slice_categories_, self.slice_keys_ = {}, {}, {}, {}
        self.data_, self.codes_, self.input_hashes_ = {}, {}, {}
        common_names = [name for name in ("spatial", "reference", "bulk") if name in modalities]
        shared = None
        if len(common_names) > 1:
            shared = modalities[common_names[0]].var_names
            for name in common_names[1:]:
                shared = shared[shared.isin(modalities[name].var_names)]
            if len(shared) == 0:
                raise ValueError("Cross-scale modalities have no shared genes")
            pooled = sparse.vstack([matrices[name][:, modalities[name].var_names.get_indexer(shared)] for name in common_names], format="csr")
            chosen = self._feature_indices(pooled)
            shared = shared[chosen]
        for name, adata in modalities.items():
            values = matrices[name]
            if shared is not None and name in common_names:
                features = shared
            else:
                features = adata.var_names[self._feature_indices(values)]
            values = values[:, adata.var_names.get_indexer(features)]
            if values.shape[0] * values.shape[1] > self.config.max_matrix_entries:
                raise ValueError("Dense molecular matrix exceeds max_matrix_entries; subset observations or features")
            array = values.toarray().astype(np.float32)
            self.genes_[name] = list(map(str, features))
            self.obs_names_[name] = list(map(str, adata.obs_names))
            self.data_[name] = self._tensor(array)
            digest = hashlib.sha256(array.tobytes())
            digest.update(json.dumps([self.obs_names_[name], self.genes_[name]]).encode())
            self.input_hashes_[name] = digest.hexdigest()
            key = slice_key.get(name) if isinstance(slice_key, dict) else slice_key if name in ("spatial", "rna", "atac") else None
            self.slice_keys_[name] = key
            if key is not None:
                if key not in adata.obs or adata.obs[key].isna().any():
                    raise ValueError(f"Missing or incomplete slice column {key!r} in {name}")
                labels = adata.obs[key].astype(str)
                categories = list(pd.unique(labels))
                codes = pd.Categorical(labels, categories=categories).codes.copy()
            else:
                categories, codes = ["all"], np.zeros(adata.n_obs, dtype=np.int64)
            self.slice_categories_[name] = categories
            self.codes_[name] = self._tensor(codes, torch.long)
        self.branches_ = nn.ModuleDict({name: _ModalityBridge(len(self.genes_[name]), len(self.slice_categories_[name]), self.config) for name in modalities}).to(self.device_)
        if relations is None:
            relations = {}
            for source in ("reference", "bulk"):
                if "spatial" in modalities and source in modalities:
                    relations[("spatial", source)] = "scale"
            if "rna" in modalities and "atac" in modalities:
                relations[("rna", "atac")] = "modal"
        self.relation_specs_, relation_modules = [], {}
        for i, ((target, source), kind) in enumerate(relations.items()):
            if target not in modalities or source not in modalities or target == source or kind not in ("scale", "modal"):
                raise ValueError("Relations map distinct (target, source) modality names to 'scale' or 'modal'")
            if kind == "scale" and self.genes_[target] != self.genes_[source]:
                raise ValueError("Cross-scale relations require the identical ordered gene panel")
            self._check_relation_size(len(self.obs_names_[target]), len(self.obs_names_[source]))
            fixed = None
            if paired and kind == "modal":
                left, right = pd.Index(self.obs_names_[target]), pd.Index(self.obs_names_[source])
                if len(left) != len(right) or set(left) != set(right):
                    raise ValueError("Paired modalities must have identical observation identifiers")
                fixed = torch.zeros((len(left), len(right)), device=self.device_)
                fixed[torch.arange(len(left), device=self.device_), self._tensor(right.get_indexer(left), torch.long)] = 1
            key = f"relation_{i}"
            relation_modules[key] = _Relation(self.config.latent_dim, self.config.temperature, fixed)
            self.relation_specs_.append({"key": key, "target": target, "source": source, "kind": kind, "paired": fixed is not None})
        self.relations_ = nn.ModuleDict(relation_modules).to(self.device_)
        self.slice_pairs_ = []
        for name in modalities:
            for left, right in combinations(range(len(self.slice_categories_[name])), 2):
                a, b = torch.where(self.codes_[name] == left)[0], torch.where(self.codes_[name] == right)[0]
                self._check_relation_size(len(a), len(b))
                self.slice_pairs_.append((name, a, b))
        if self.config.use_diffusion and (self.relation_specs_ or self.slice_pairs_):
            reverse_entries = sum(values.numel() for values in self.data_.values()) * (self.config.coupling_steps + 1)
            if reverse_entries > self.config.max_trajectory_entries:
                raise ValueError("Differentiable reverse rollout exceeds max_trajectory_entries; reduce observations/features/coupling_steps or increase the explicit limit")

    def _encode(self, sample=False):
        result = {}
        for name, branch in self.branches_.items():
            z, mean, log_var = branch.encode(self.data_[name], self.codes_[name], sample)
            result[name] = (z, mean, log_var, branch.endpoint(z, self.codes_[name]))
        return result

    def _weights(self, encoded):
        return {spec["key"]: self.relations_[spec["key"]](encoded[spec["target"]][1], encoded[spec["source"]][1]) for spec in self.relation_specs_}

    def _project_states(self, states):
        return {name: self.branches_[name].encode(state, self.codes_[name])[1] for name, state in states.items()}

    def _state_conditions(self, projected, weights):
        conditions = {name: [] for name in projected}
        for spec in self.relation_specs_:
            target, source, relation = spec["target"], spec["source"], weights[spec["key"]]
            conditions[target].append(relation @ projected[source])
            conditions[source].append(_normalise_reverse(relation) @ projected[target])
        return {name: torch.stack(values).mean(0) if values else torch.zeros_like(projected[name]) for name, values in conditions.items()}

    def _reverse_drift(self, states, time, encoded, weights):
        if self.relation_specs_:
            conditions = self._state_conditions(self._project_states(states), weights)
        else:
            conditions = {name: torch.zeros_like(encoded[name][0]) for name in states}
        result = {}
        for name, state in states.items():
            z, _, _, endpoint = encoded[name]
            prediction = self.branches_[name].denoise(state, time, endpoint, z, self.codes_[name], conditions[name])
            result[name] = self.bridge.drift(state, prediction, endpoint, time)
        return result

    def _reverse_transport_losses(self, encoded, weights, slice_relations):
        """Differentiate transport penalties through actual reverse ODE states.

        Every modality advances on one common Heun grid. Learned relations are
        held fixed over that trajectory, while cross-modal score conditions use
        the transported states at the current ODE time. A shared subset of grid
        times is sampled per iteration for the molecular transport penalties.
        """
        losses = {key: torch.zeros((), device=self.device_) for key in ("scale", "modal", "slice")}
        if not self.relation_specs_ and not slice_relations:
            return losses
        grid = self._time_grid(self.config.coupling_steps)
        selected = set((torch.randperm(self.config.coupling_steps, device=self.device_)[:self.config.trajectory_times] + 1).tolist())
        states = {name: values[3] for name, values in encoded.items()}
        for step, (time, next_time) in enumerate(zip(grid[:-1], grid[1:]), start=1):
            dt = next_time - time
            drift = self._reverse_drift(states, time, encoded, weights)
            proposal = {name: state + dt * drift[name] for name, state in states.items()}
            next_drift = self._reverse_drift(proposal, next_time, encoded, weights)
            states = {name: state + 0.5 * dt * (drift[name] + next_drift[name]) for name, state in states.items()}
            if step not in selected:
                continue
            projected = self._project_states(states) if any(spec["kind"] == "modal" for spec in self.relation_specs_) else None
            for spec in self.relation_specs_:
                target, source, kind = spec["target"], spec["source"], spec["kind"]
                values = states if kind == "scale" else projected
                losses[kind] = losses[kind] + self._symmetric_loss(values[target], values[source], weights[spec["key"]]) / self.config.trajectory_times
            for name, a, b, relation in slice_relations:
                losses["slice"] = losses["slice"] + self._symmetric_loss(states[name][a], states[name][b], relation) / self.config.trajectory_times
        return losses

    @staticmethod
    def _symmetric_loss(target, source, weights):
        return 0.5 * (F.mse_loss(target, weights @ source) + F.mse_loss(source, _normalise_reverse(weights) @ target))

    @staticmethod
    def _posterior_alignment(target_mean, target_log_var, source_mean, source_log_var, weights):
        """Symmetric KL to moment-matched transported Gaussian posteriors.

        Each transported posterior is a mixture of source diagonal Gaussians.
        Its first two moments define a diagonal Gaussian approximation, retaining
        both within-cell variance and between-cell mean variation. This is an
        explicit operational choice for the manuscript's CVAE slice alignment.
        """
        def directed(mean, log_var, other_mean, other_log_var, relation):
            transported_mean = relation @ other_mean
            transported_second = relation @ (other_log_var.exp() + other_mean.square())
            transported_var = (transported_second - transported_mean.square()).clamp_min(1e-6)
            return 0.5 * (transported_var.log() - log_var +
                          (log_var.exp() + (mean - transported_mean).square()) / transported_var - 1).mean()
        return 0.5 * (directed(target_mean, target_log_var, source_mean, source_log_var, weights) +
                      directed(source_mean, source_log_var, target_mean, target_log_var, _normalise_reverse(weights)))

    def _loss(self, joint):
        encoded = self._encode(sample=True)
        weights = self._weights(encoded) if joint else {}
        terms = {key: torch.zeros((), device=self.device_) for key in ("bridge", "endpoint", "kl", "scale", "modal", "slice", "slice_distribution", "slice_latent")}
        for name, (_, mean, log_var, endpoint) in encoded.items():
            terms["endpoint"] = terms["endpoint"] + F.mse_loss(endpoint, self.data_[name])
            if self.config.use_vae:
                terms["kl"] = terms["kl"] - 0.5 * (1 + log_var - mean.square() - log_var.exp()).mean()
        slice_relations = []
        if joint:
            for name, a, b in self.slice_pairs_:
                mean, log_var = encoded[name][1], encoded[name][2]
                w = torch.softmax((mean[a] @ mean[b].T) / (math.sqrt(self.config.latent_dim) * self.config.temperature), dim=1)
                slice_relations.append((name, a, b, w))
                if self.config.use_vae:
                    terms["slice_distribution"] = terms["slice_distribution"] + self._posterior_alignment(mean[a], log_var[a], mean[b], log_var[b], w)
                if not self.config.use_diffusion:
                    terms["slice_latent"] = terms["slice_latent"] + self._symmetric_loss(mean[a], mean[b], w)
        for _ in range(self.config.trajectory_times if self.config.use_diffusion else 0):
            t = torch.rand((), device=self.device_) * (1 - 2 * self.config.time_epsilon) + self.config.time_epsilon
            states = {name: self.bridge.sample(self.data_[name], values[3], t) for name, values in encoded.items()}
            conditions = (self._state_conditions(self._project_states(states), weights)
                          if joint and self.relation_specs_ else {name: torch.zeros_like(encoded[name][0]) for name in states})
            for name, branch in self.branches_.items():
                condition = conditions[name]
                if joint and torch.rand((), device=self.device_) < self.config.condition_dropout:
                    condition = torch.zeros_like(condition)
                prediction = branch.denoise(states[name], t, encoded[name][3], encoded[name][0], self.codes_[name], condition)
                # variance-weighted conditional score matching, analytically
                # simplified for an x0-parameterized score network.
                score_loss = (1 - t) / (self.config.diffusion_sigma ** 2 * t) * F.mse_loss(prediction, self.data_[name])
                terms["bridge"] = terms["bridge"] + score_loss / self.config.trajectory_times
        if joint:
            if self.config.use_diffusion:
                for name, value in self._reverse_transport_losses(encoded, weights, slice_relations).items():
                    terms[name] = terms[name] + value
            for spec in self.relation_specs_:
                terms[spec["kind"]] = terms[spec["kind"]] + self._symmetric_loss(encoded[spec["target"]][1], encoded[spec["source"]][1], weights[spec["key"]])
        total = (terms["bridge"] + self.config.endpoint_weight * terms["endpoint"] + self.config.kl_weight * terms["kl"] +
                 self.config.scale_weight * terms["scale"] + self.config.modal_weight * terms["modal"] +
                 self.config.slice_weight * (terms["slice"] + terms["slice_distribution"] + terms["slice_latent"]))
        return total, terms

    def fit(self, modalities, *, layers=None, slice_key=None, paired=False, relations=None, verbose=True):
        """Fit bridges without reading phenotype, domain, or composition labels.

        ``layers`` maps modality to a nonnegative unlogged expression layer.
        HVGs use Scanpy Seurat normalized dispersion by default; cross-scale
        datasets use one pooled ordered intersection. RNA/ATAC use independent
        feature panels. ``hvg_method='variance'`` selects a simpler alternative.
        ``slice_key`` is an observation column or a modality-to-column mapping.
        """
        started = time.perf_counter()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        self.device_ = self._device()
        self._prepare(modalities, layers, slice_key, paired, relations)
        parameters = list(self.branches_.parameters()) + list(self.relations_.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)
        self.history_ = []
        self.branches_.train()
        self.relations_.train()
        for stage, epochs in (("pretrain", self.config.pretrain_epochs), ("joint", self.config.epochs)):
            for epoch in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss, terms = self._loss(stage == "joint")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite trajectory objective; inspect input scale and learning rate")
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 10)
                optimizer.step()
                record = {"stage": stage, "epoch": epoch + 1, "loss": float(loss.detach()), **{key: float(value.detach()) for key, value in terms.items()}}
                self.history_.append(record)
                if verbose and (epoch == 0 or (epoch + 1) % 25 == 0 or epoch == epochs - 1):
                    print(f"NicheBridge {stage} {epoch + 1}/{epochs}: loss={record['loss']:.5g}")
        self.branches_.eval()
        self.relations_.eval()
        self.fitted_ = True
        self.metadata_ = {"model": "NicheBridgeTrajectory", "diffusion": "Brownian bridge f=0, g=sigma, T=1" if self.config.use_diffusion else "disabled",
                          "model_mode": "diffusion_bridge" if self.config.use_diffusion else "variational_autoencoder" if self.config.use_vae else "deterministic_autoencoder",
                          "variational_enabled": self.config.use_vae,
                          "diffusion_enabled": self.config.use_diffusion,
                          "reconstruction_method": "reverse_probability_flow_ode" if self.config.use_diffusion else "posterior_mean_endpoint_decoder",
                          "config": asdict(self.config), "runtime_seconds": time.perf_counter() - started,
                          "input_hashes": self.input_hashes_, "preprocessing": f"{self.config.hvg_method} HVG; optional library normalization; log1p",
                          "hvg_flavor": "seurat dispersion" if self.config.hvg_method == "scanpy" else "pooled log-expression variance",
                          "output_scale": "library-normalized log1p" if self.config.normalize_library else "log1p",
                          "transport_orientation": "target rows, source columns; each target row sums to one",
                          "slice_distribution_alignment": "symmetric Gaussian KL against moment-matched transported CVAE posteriors" if self.config.use_vae else "disabled",
                          "trajectory_coupling": "differentiable common-grid Heun reverse rollout; shared randomly selected grid times; direct score-network gradients" if self.config.use_diffusion else "disabled; latent correspondence losses only",
                          "solver": "Heun probability-flow ODE on logit-spaced times, epsilon-truncated endpoints" if self.config.use_diffusion else "none"}
        return self

    def _require_fitted(self):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("Fit the trajectory model first")

    def _name(self, modality):
        self._require_fitted()
        if modality not in self.branches_:
            raise ValueError(f"Unknown fitted modality: {modality}")

    @torch.no_grad()
    def transform(self, modality="spatial", adata=None):
        """Return posterior means; optional new observations use the fitted panel.

        New inputs must follow the fitted unlogged input-scale convention. With
        library normalization enabled, supply the same normalization feature
        universe as during fit: normalization precedes fitted-panel selection,
        so supplying only a cropped panel would change library totals. Already
        normalized inputs can instead be fitted with normalize_library=False.
        """
        self._name(modality)
        if adata is None:
            values, codes = self.data_[modality], self.codes_[modality]
        else:
            missing = pd.Index(self.genes_[modality]).difference(adata.var_names)
            if len(missing):
                raise ValueError(f"Missing {len(missing)} fitted features")
            matrix = self._matrix(adata, self.layers_[modality])
            if matrix.shape[0] * len(self.genes_[modality]) > self.config.max_matrix_entries:
                raise ValueError("Dense molecular matrix exceeds max_matrix_entries; transform fewer observations per call or increase the explicit limit")
            values = self._tensor(matrix[:, adata.var_names.get_indexer(self.genes_[modality])].toarray())
            key = self.slice_keys_[modality]
            if key is None:
                codes = torch.zeros(adata.n_obs, dtype=torch.long, device=self.device_)
            else:
                if key not in adata.obs or adata.obs[key].isna().any():
                    raise ValueError(f"Missing slice column {key!r}")
                raw = pd.Categorical(adata.obs[key].astype(str), categories=self.slice_categories_[modality]).codes.copy()
                if (raw < 0).any():
                    raise ValueError("New slice identities require refitting")
                codes = self._tensor(raw, torch.long)
        return self.branches_[modality].encode(values, codes)[1].cpu().numpy()

    def _transport_tensor(self, target, source):
        encoded = self._encode()
        for spec in self.relation_specs_:
            if spec["target"] == target and spec["source"] == source:
                return self.relations_[spec["key"]](encoded[target][1], encoded[source][1])
            if spec["target"] == source and spec["source"] == target:
                return _normalise_reverse(self.relations_[spec["key"]](encoded[source][1], encoded[target][1]))
        raise ValueError(f"No fitted relation between {target!r} and {source!r}")

    @torch.no_grad()
    def transport(self, target="spatial", source="reference"):
        """Return nonnegative row-normalized weights with explicit identifiers."""
        self._name(target)
        self._name(source)
        return pd.DataFrame(self._transport_tensor(target, source).cpu().numpy(), index=self.obs_names_[target], columns=self.obs_names_[source])

    def _time_grid(self, steps=None):
        steps = self.config.ode_steps if steps is None else steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
            raise ValueError("ODE steps must be an integer of at least two")
        bound = math.log((1 - self.config.time_epsilon) / self.config.time_epsilon)
        return torch.sigmoid(torch.linspace(bound, -bound, steps + 1, device=self.device_))

    def _integrate(self, modality, endpoint, z, codes, grid, conditions=None, collect_states=False):
        if collect_states and endpoint.numel() * len(grid) > self.config.max_trajectory_entries:
            raise ValueError("Stored ODE history exceeds max_trajectory_entries; reduce observations/features/steps or increase the explicit limit")
        branch, state = self.branches_[modality], endpoint.clone()
        states = [state.clone()] if collect_states else None
        zero = torch.zeros_like(z)
        for i, (t, next_t) in enumerate(zip(grid[:-1], grid[1:])):
            condition = zero if conditions is None else conditions(i)
            denoised = branch.denoise(state, t, endpoint, z, codes, condition)
            drift = self.bridge.drift(state, denoised, endpoint, t)
            dt = next_t - t
            proposal = state + dt * drift
            next_condition = zero if conditions is None else conditions(i + 1)
            next_denoised = branch.denoise(proposal, next_t, endpoint, z, codes, next_condition)
            next_drift = self.bridge.drift(proposal, next_denoised, endpoint, next_t)
            state = state + dt * 0.5 * (drift + next_drift)
            if not torch.isfinite(state).all():
                raise FloatingPointError("ODE integration diverged; increase ode_steps or retrain with a smaller learning rate")
            if collect_states:
                states.append(state.clone())
        return state, states

    def _frame(self, values, modality):
        return pd.DataFrame(values.clamp_min(0).cpu().numpy() * self.config.log_scale,
                            index=self.obs_names_[modality], columns=self.genes_[modality])

    @torch.no_grad()
    def reconstruct(self, modality="spatial", *, steps=None, return_trajectory=False):
        """Integrate the reverse bridge; optionally return intermediate states.

        All returned matrices have the fitted feature panel and original row IDs.
        Final values are projected to the nonnegative log-expression domain.
        With use_diffusion=False, this explicitly decodes the posterior mean
        and rejects a request for a diffusion trajectory. Variational training
        remains controlled independently by use_vae.
        """
        self._name(modality)
        branch, codes = self.branches_[modality], self.codes_[modality]
        z = branch.encode(self.data_[modality], codes)[1]
        if not self.config.use_diffusion:
            if return_trajectory:
                raise ValueError("The no-diffusion model has no diffusion trajectory")
            return self._frame(branch.endpoint(z, codes), modality)
        endpoint, grid = branch.endpoint(z, codes), self._time_grid(steps)
        values, states = self._integrate(modality, endpoint, z, codes, grid, collect_states=return_trajectory)
        frame = self._frame(values, modality)
        if return_trajectory:
            return frame, {"times": grid.cpu().numpy(), "states": np.stack([value.cpu().numpy() * self.config.log_scale for value in states])}
        return frame

    @torch.no_grad()
    def generate(self, source, target, *, steps=None):
        """Condition a target reverse bridge on a transported source trajectory.

        The target endpoint and latent input are obtained from transported source
        representations. The target score receives the transported source state
        at every integration time. Output rows are the fitted target locations.
        With use_diffusion=False, generation is explicitly a transported latent
        decoder prediction; no score network or ODE is available in that mode.
        """
        self._name(source)
        self._name(target)
        weights = self._transport_tensor(target, source)
        source_branch, source_codes = self.branches_[source], self.codes_[source]
        source_z = source_branch.encode(self.data_[source], source_codes)[1]
        if not self.config.use_diffusion:
            target_z = weights @ source_z
            return self._frame(self.branches_[target].endpoint(target_z, self.codes_[target]), target)
        grid = self._time_grid(steps)
        source_endpoint = source_branch.endpoint(source_z, source_codes)
        stored_entries = len(grid) * (source_endpoint.numel() + weights.shape[0] * self.config.latent_dim)
        if stored_entries > self.config.max_trajectory_entries:
            raise ValueError("Conditional ODE histories exceed max_trajectory_entries; reduce observations/features/steps or increase the explicit limit")
        _, source_states = self._integrate(source, source_endpoint, source_z, source_codes, grid, collect_states=True)
        condition_states = [weights @ source_branch.encode(state, source_codes)[1] for state in source_states]
        target_z, target_codes = weights @ source_z, self.codes_[target]
        endpoint = self.branches_[target].endpoint(target_z, target_codes)
        generated, _ = self._integrate(target, endpoint, target_z, target_codes, grid, lambda i: condition_states[i])
        return self._frame(generated, target)

    def map_phenotypes(self, phenotypes, *, target="spatial", source="bulk"):
        """Project numeric traits or explicitly encoded categories into space.

        A Series of nonnumeric categories is one-hot encoded. A DataFrame must
        contain finite numeric columns. Index matching is by identifier, never
        row position. Missing values must be resolved explicitly before mapping.
        """
        weights = self.transport(target, source)
        if isinstance(phenotypes, pd.Series):
            if not pd.api.types.is_numeric_dtype(phenotypes):
                if phenotypes.isna().any():
                    raise ValueError("Phenotype categories contain missing values")
                phenotypes = pd.get_dummies(phenotypes, dtype=float)
            else:
                phenotypes = phenotypes.to_frame()
        if not isinstance(phenotypes, pd.DataFrame) or not phenotypes.index.is_unique:
            raise ValueError("Phenotypes must be a uniquely indexed Series or DataFrame")
        if set(phenotypes.index.astype(str)) != set(weights.columns):
            raise ValueError("Phenotype IDs must match the fitted source observations")
        aligned = phenotypes.copy()
        aligned.index = aligned.index.astype(str)
        aligned = aligned.loc[weights.columns]
        try:
            values = aligned.to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("Encode categorical DataFrame columns explicitly before mapping") from exc
        if not np.isfinite(values).all():
            raise ValueError("Phenotypes must contain finite values")
        return pd.DataFrame(weights.to_numpy() @ values, index=weights.index, columns=aligned.columns)

    def map_cell_types(self, labels, *, target="spatial", source="reference"):
        """Aggregate transported cell identity indicators into cell fractions."""
        if not isinstance(labels, pd.Series) or labels.isna().any():
            raise ValueError("Cell identities must be an indexed Series without missing values")
        return self.map_phenotypes(pd.get_dummies(labels.astype(str), dtype=float), target=target, source=source)

    def save(self, folder):
        """Save fitted parameters, ordered axes and transformed training inputs."""
        self._require_fitted()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        metadata = {"format_version": 1, "config": asdict(self.config), "modality_names": self.modality_names_,
                    "genes": self.genes_, "obs_names": self.obs_names_, "layers": self.layers_,
                    "slice_categories": self.slice_categories_, "slice_keys": self.slice_keys_,
                    "relations": self.relation_specs_, "history": self.history_, "metadata": self.metadata_}
        (folder / "nichebridge_trajectory.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.save({"branches": self.branches_.state_dict(), "relations": self.relations_.state_dict(),
                    "data": {k: v.cpu() for k, v in self.data_.items()}, "codes": {k: v.cpu() for k, v in self.codes_.items()}}, folder / "nichebridge_trajectory.pt")

    @classmethod
    def load(cls, folder, *, device="auto"):
        """Load a checkpoint created by save with safe tensor-only loading."""
        folder = Path(folder)
        info = json.loads((folder / "nichebridge_trajectory.json").read_text(encoding="utf-8"))
        if info.get("format_version") != 1:
            raise ValueError("Unsupported trajectory checkpoint version")
        model = cls(TrajectoryConfig(**{**info["config"], "device": device}))
        model.device_ = model._device()
        for attr, key in (("modality_names_", "modality_names"), ("genes_", "genes"), ("obs_names_", "obs_names"),
                          ("layers_", "layers"), ("slice_categories_", "slice_categories"), ("slice_keys_", "slice_keys"),
                          ("relation_specs_", "relations"), ("history_", "history"), ("metadata_", "metadata")):
            setattr(model, attr, info[key])
        saved = torch.load(folder / "nichebridge_trajectory.pt", map_location=model.device_, weights_only=True)
        model.data_, model.codes_ = saved["data"], saved["codes"]
        model.input_hashes_ = model.metadata_["input_hashes"]
        model.slice_pairs_ = []
        for name in model.modality_names_:
            for left, right in combinations(range(len(model.slice_categories_[name])), 2):
                a = torch.where(model.codes_[name] == left)[0]
                b = torch.where(model.codes_[name] == right)[0]
                model._check_relation_size(len(a), len(b))
                model.slice_pairs_.append((name, a, b))
        model.branches_ = nn.ModuleDict({name: _ModalityBridge(len(model.genes_[name]), len(model.slice_categories_[name]), model.config) for name in model.modality_names_}).to(model.device_)
        modules = {}
        for spec in model.relation_specs_:
            fixed = saved["relations"].get(spec["key"] + ".fixed")
            modules[spec["key"]] = _Relation(model.config.latent_dim, model.config.temperature, fixed)
        model.relations_ = nn.ModuleDict(modules).to(model.device_)
        model.branches_.load_state_dict(saved["branches"])
        model.relations_.load_state_dict(saved["relations"])
        model.branches_.eval()
        model.relations_.eval()
        model.fitted_ = True
        return model
