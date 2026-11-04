"""Conditional variance-preserving bridge in continuous expression space."""

from dataclasses import dataclass
import torch


@dataclass
class VPBridge:
    beta_min: float = 0.1
    beta_d: float = 2.0
    t_min: float = 0.001
    t_max: float = 0.999
    time_mode: str = "uniform"

    def coefficients(self, t):
        integrated = self.beta_min * t + 0.5 * self.beta_d * t.square()
        alpha = torch.exp(-0.5 * integrated)
        variance = -torch.expm1(-integrated)
        end = t.new_tensor(self.beta_min + 0.5 * self.beta_d)
        alpha_end = torch.exp(-0.5 * end)
        variance_end = -torch.expm1(-end)
        ratio = (alpha_end.square() * variance / (variance_end * alpha.square())).clamp(
            0, 1
        )
        return (
            ratio * alpha / alpha_end,
            (1 - ratio) * alpha,
            ((1 - ratio) * variance).clamp_min(0),
        )

    def sample(self, x0, xT, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a, b, variance = self.coefficients(t)
        return a[:, None] * xT + b[:, None] * x0 + variance.sqrt()[:, None] * noise

    def loss(self, denoiser, x0, xT, z, batch, t=None, noise=None):
        if t is None:
            t = (
                torch.rand(len(x0), device=x0.device) * (self.t_max - self.t_min)
                + self.t_min
            )
        a, b, variance = self.coefficients(t)
        marginal = (a.square() + b.square() + variance).clamp_min(1e-12)
        residual = (a.square() + variance).clamp_min(1e-12)
        xt = self.sample(x0, xT, t, noise)
        if self.time_mode == "logit":
            time_input = torch.logit(t).clamp(-8, 8) / 4.0
        else:
            time_input = 250 * t.log()
        raw = denoiser(xt * marginal.rsqrt()[:, None], time_input, xT, z, batch)
        prediction = (b / marginal)[:, None] * xt + (residual / marginal).sqrt()[
            :, None
        ] * raw
        mse = (prediction - x0).square().mean(dim=1)
        return (marginal / residual * mse).mean()
