"""Gaussian composition encoder and constrained batch-aware mixture decoder."""

import math
import torch
from torch import nn
import torch.nn.functional as F


def time_embedding(time, width):
    half = width // 2
    frequency = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=time.device, dtype=time.dtype)
        / max(half, 1)
    )
    phase = time[:, None] * frequency[None, :]
    return F.pad(torch.cat([phase.cos(), phase.sin()], dim=-1), (0, width - 2 * half))


class ConditionalBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = nn.Linear(width, 2 * width)
        self.residual = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)
        )

    def forward(self, x, condition):
        scale, shift = self.film(condition).chunk(2, dim=-1)
        return x + self.residual(self.norm(x) * (1 + scale) + shift)


class MixtureBridge(nn.Module):
    def __init__(self, signatures, rna_yield, config):
        super().__init__()
        self.config = config
        signatures = torch.as_tensor(signatures, dtype=torch.float32)
        self.register_buffer("signatures", signatures)
        self.register_buffer(
            "rna_yield", torch.as_tensor(rna_yield, dtype=torch.float32)
        )
        signature_features = (
            torch.log1p(config.target_sum * signatures) / config.log_scale
        )
        self.register_buffer(
            "signature_features", F.normalize(signature_features, dim=1)
        )
        n_types, n_genes = signatures.shape
        hidden = config.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden, config.latent_dim)
        self.logvar = nn.Linear(hidden, config.latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.0)
        self.composition = nn.Sequential(
            nn.Linear(config.latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_types)
        )
        global_profile = signatures.mean(dim=0)
        specificity = signatures / (global_profile[None, :] + 1e-8)
        gene_weights = specificity.max(dim=0).values
        gene_weights = gene_weights / gene_weights.mean().clamp_min(1e-8)
        gene_weights = 1.0 + config.marker_weight * (gene_weights - 1.0)
        self.register_buffer("gene_weights", gene_weights.clamp_min(0.1))
        self.platform_raw = nn.Parameter(
            torch.zeros(n_genes), requires_grad=config.batch_correction
        )
        self.log_dispersion = nn.Parameter(torch.full((n_genes,), 2.0))
        self.denoise_input = nn.Linear(2 * n_genes, hidden)
        self.time_condition = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.latent_condition = nn.Linear(config.latent_dim, hidden)
        self.batch_condition = nn.Embedding(2, hidden)
        self.blocks = nn.ModuleList([ConditionalBlock(hidden) for _ in range(2)])
        self.output = nn.Sequential(
            nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, n_genes)
        )
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def platform_log(self):
        if not self.config.batch_correction:
            return self.platform_raw * 0
        centered = self.platform_raw - self.platform_raw.mean()
        bound = self.config.max_log_platform
        return centered * (bound / centered.abs().max().clamp_min(bound))

    def features(self, counts):
        normalized = (
            counts
            / counts.sum(dim=1, keepdim=True).clamp_min(1)
            * self.config.target_sum
        )
        return torch.log1p(normalized) / self.config.log_scale

    def encode(self, counts, batch, mask=False, sample=False):
        # The fixed reference domain has zero offset. The biological head sees no batch embedding.
        aligned = counts * torch.exp(
            -batch[:, None] * self.platform_log().detach()[None, :]
        )
        features = self.features(aligned)
        observed = torch.ones_like(features)
        if mask and self.config.mask_fraction > 0:
            observed = (torch.rand_like(features) >= self.config.mask_fraction).to(
                features.dtype
            )
        h = self.encoder(features * observed)
        mu, logvar = self.mean(h), self.logvar(h).clamp(-8, 4)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if sample else mu
        logits = self.composition(z)
        if self.config.reference_anchor_weight > 0:
            anchor = F.normalize(features * observed, dim=1) @ self.signature_features.T
            logits = logits + self.config.reference_anchor_weight * anchor
        p = torch.softmax(logits / self.config.composition_temperature, dim=-1)
        return p, z, mu, logvar, observed

    def decode(self, cell_fraction, library_size, batch):
        q = cell_fraction * self.rna_yield[None, :]
        q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        gamma = torch.exp(batch[:, None] * self.platform_log()[None, :])
        expected_profile = (q @ self.signatures) * gamma
        expected_profile = expected_profile / expected_profile.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        mu = library_size[:, None] * expected_profile
        type_panel_mass = gamma @ self.signatures.T
        q_observed = q * type_panel_mass
        q_observed = q_observed / q_observed.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return mu, q, q_observed

    def forward(self, xt, time, endpoint, z, batch):
        condition = self.time_condition(
            time_embedding(time, self.config.hidden_dim)
        ) + self.latent_condition(z)
        if self.config.batch_correction:
            condition = condition + self.batch_condition(batch.long())
        h = self.denoise_input(torch.cat([xt, endpoint], dim=-1))
        for block in self.blocks:
            h = block(h, condition)
        return self.output(h)


def poisson_loss(counts, mean, gene_weights=None):
    mean = mean.clamp_min(1e-6)
    log_prob = counts * mean.log() - mean - torch.lgamma(counts + 1)
    if gene_weights is not None:
        log_prob = log_prob * gene_weights[None, :]
        return -log_prob.sum() / (
            log_prob.shape[0] * gene_weights.sum().clamp_min(1e-8)
        )
    return -log_prob.mean()


def negative_binomial_loss(counts, mean, log_dispersion, gene_weights=None):
    mean = mean.clamp_min(1e-6)
    theta = F.softplus(log_dispersion).clamp_min(1e-4)[None, :]
    log_sum = (theta + mean).log()
    log_prob = (
        torch.lgamma(counts + theta) - torch.lgamma(theta) - torch.lgamma(counts + 1)
    )
    log_prob = (
        log_prob + theta * (theta.log() - log_sum) + counts * (mean.log() - log_sum)
    )
    if gene_weights is not None:
        log_prob = log_prob * gene_weights[None, :]
        return -log_prob.sum() / (
            log_prob.shape[0] * gene_weights.sum().clamp_min(1e-8)
        )
    return -log_prob.mean()
