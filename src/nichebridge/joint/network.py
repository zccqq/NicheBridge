"""Joint relations around the installed MixtureBridge encoder and VP denoiser."""

from dataclasses import replace
import torch
from torch import nn
from torch.nn import functional as F
from nichebridge._network import MixtureBridge


def normalize(x):
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class JointNetwork(nn.Module):
    def __init__(self, signatures, yields, projection, reference_keys, config):
        super().__init__()
        self.config = config
        self.core = MixtureBridge(signatures, yields, replace(config, batch_correction=True))
        self.core.platform_raw.requires_grad_(False)
        self.core.batch_condition = nn.Embedding(3, config.hidden_dim)
        n_genes = len(projection)
        self.offset_raw = nn.Parameter(torch.zeros(2, n_genes), requires_grad=config.batch_correction)
        self.register_buffer("projection", torch.as_tensor(projection, dtype=torch.float32))
        self.register_buffer("reference_keys", torch.as_tensor(reference_keys, dtype=torch.float32))
        self.query_adjustment = nn.Linear(config.latent_dim, projection.shape[1], bias=False)
        nn.init.zeros_(self.query_adjustment.weight)
        if config.use_vae:
            self.deterministic_latent = None
        else:
            self.deterministic_latent = nn.Linear(config.hidden_dim, config.latent_dim)
            self.core.mean = None
            self.core.logvar = None
        if not config.use_diffusion:
            # Remove the denoiser rather than leave untrained diffusion parameters.
            for name in ("denoise_input", "time_condition", "latent_condition",
                         "batch_condition", "blocks", "output"):
                setattr(self.core, name, None)

    def platform_logs(self):
        offsets = self.offset_raw - self.offset_raw.mean(dim=1, keepdim=True)
        bound = self.config.max_log_platform
        offsets = offsets * (bound / offsets.abs().amax(dim=1, keepdim=True).clamp_min(bound))
        if not self.config.batch_correction:
            offsets = offsets * 0
        return torch.cat([torch.zeros_like(offsets[:1]), offsets], dim=0)

    def aligned(self, x, domain):
        return x * torch.exp(-self.platform_logs()[domain].detach())

    def encode(self, x, domain, mask=False, sample=False):
        aligned = self.aligned(x, domain)
        if self.config.use_vae:
            return self.core.encode(aligned, torch.zeros_like(domain), mask, sample)
        features = self.core.features(aligned)
        observed = torch.ones_like(features)
        if mask and self.config.mask_fraction > 0:
            observed = (torch.rand_like(features) >= self.config.mask_fraction).to(features.dtype)
        hidden = self.core.encoder(features * observed)
        z = self.deterministic_latent(hidden)
        logits = self.core.composition(z)
        if self.config.reference_anchor_weight > 0:
            anchor = F.normalize(features * observed, dim=1) @ self.core.signature_features.T
            logits = logits + self.config.reference_anchor_weight * anchor
        p = torch.softmax(logits / self.config.composition_temperature, dim=-1)
        zeros = torch.zeros_like(z)
        return p, z, z, zeros, observed

    def profile_features(self, profile):
        return torch.log1p(self.config.target_sum * normalize(profile)) / self.config.log_scale

    def decode_types(self, p, domain):
        _, q, _ = self.core.decode(p, torch.ones(len(p), device=p.device), torch.zeros_like(domain))
        return normalize((q @ self.core.signatures) * torch.exp(self.platform_logs()[domain]))

    def keys(self, x, domain, mean):
        base = F.normalize(self.core.features(self.aligned(x, domain)) @ self.projection, dim=-1)
        return F.normalize(base + 0.25 * self.query_adjustment(mean), dim=-1)

    def cell_relation(self, x, domain, p, mean, ids, valid, dictionary):
        query = self.keys(x, domain, mean)
        logits = torch.einsum("nd,ntkd->ntk", query, self.reference_keys[ids])
        logits = (logits / self.config.attention_temperature).masked_fill(~valid, -torch.inf)
        within = torch.softmax(logits, dim=-1)
        weights = p[:, :, None] * within
        rna_weights = normalize(p * self.core.rna_yield)
        profile = torch.einsum("ntk,ntkg->ng", rna_weights[:, :, None] * within, dictionary)
        profile = normalize(profile * torch.exp(self.platform_logs()[domain]))
        return weights, profile

    def bulk_relation(self, x, domain, mean, bulk_x, bulk_mean):
        query = self.keys(x, domain, mean)
        bulk_domain = torch.full((len(bulk_x),), 2, dtype=torch.long, device=x.device)
        key = self.keys(bulk_x, bulk_domain, bulk_mean)
        weights = torch.softmax(query @ key.T / self.config.attention_temperature, dim=-1)
        aligned_bulk = normalize(bulk_x * torch.exp(-self.platform_logs()[2]))
        profile = normalize((weights @ aligned_bulk) * torch.exp(self.platform_logs()[domain]))
        return weights, profile

    def forward(self, xt, time, endpoint, z, batch):
        if not self.config.use_diffusion:
            raise RuntimeError("Diffusion is disabled in this ablation")
        return self.core(xt, time, endpoint, z, batch)
