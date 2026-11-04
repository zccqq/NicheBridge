"""Validated training settings for the composition model."""

from dataclasses import asdict, dataclass
import math
import re


@dataclass
class NicheBridgeConfig:
    """Configure model dimensions, mixture supervision, and training objectives."""

    latent_dim: int = 16
    hidden_dim: int = 128
    dropout: float = 0.05
    epochs: int = 1000
    pretrain_epochs: int = 200
    batch_size: int = 128
    pseudospots: int = 5000
    pseudo_min_cells: int = 15
    pseudo_max_cells: int = 30
    learning_rate: float = 0.0005
    weight_decay: float = 0.00001
    kl_weight: float = 0.001
    bridge_weight: float = 0.0
    bridge_warmup: int = 100
    mask_fraction: float = 0.05
    mask_weight: float = 0.1
    pseudo_weight: float = 2.0
    batch_correction: bool = False
    platform_penalty: float = 0.0
    max_log_platform: float = 2.0
    target_sum: float = 10000.0
    log_scale: float = 4.0
    seed: int = 42
    device: str = "auto"
    log_every: int = 50
    spatial_weight: float = 0.01
    spatial_k: int = 6
    marker_weight: float = 0.1
    pseudo_strategy: str = "balanced"
    pseudo_active_max: int = 8
    pseudo_single_fraction: float = 0.0
    bridge_time_mode: str = "uniform"
    composition_temperature: float = 0.5
    entropy_weight: float = 0.005
    profile_mode: str = "size_factor"
    pseudo_library_match: bool = True
    likelihood: str = "poisson"
    reference_anchor_weight: float = 0.25

    def __post_init__(self):
        for name in (
            "latent_dim",
            "hidden_dim",
            "epochs",
            "batch_size",
            "pseudospots",
            "pseudo_min_cells",
            "pseudo_max_cells",
            "log_every",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("pretrain_epochs", "bridge_warmup", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.device, str) or not re.fullmatch(
            r"auto|cpu|cuda(?::[0-9]+)?", self.device
        ):
            raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")
        if (
            not math.isfinite(self.pseudo_single_fraction)
            or not 0 <= self.pseudo_single_fraction <= 1
        ):
            raise ValueError("pseudo_single_fraction must be in [0, 1]")
        if self.pseudo_min_cells > self.pseudo_max_cells:
            raise ValueError("Minimum cell count exceeds maximum")
        if (
            isinstance(self.spatial_k, bool)
            or not isinstance(self.spatial_k, int)
            or self.spatial_k < 1
        ):
            raise ValueError("spatial_k must be a positive integer")
        if (
            isinstance(self.pseudo_active_max, bool)
            or not isinstance(self.pseudo_active_max, int)
            or self.pseudo_active_max < 1
        ):
            raise ValueError("pseudo_active_max must be a positive integer")
        if self.pseudo_strategy not in ("random", "balanced"):
            raise ValueError("pseudo_strategy must be random or balanced")
        if self.profile_mode not in ("raw", "size_factor"):
            raise ValueError("profile_mode must be raw or size_factor")
        if self.bridge_time_mode not in ("uniform", "logit"):
            raise ValueError("bridge_time_mode must be uniform or logit")
        if (
            not math.isfinite(self.composition_temperature)
            or self.composition_temperature <= 0
        ):
            raise ValueError("composition_temperature must be finite and positive")
        if (
            not math.isfinite(self.reference_anchor_weight)
            or self.reference_anchor_weight < 0
        ):
            raise ValueError("reference_anchor_weight must be finite and nonnegative")
        if self.likelihood not in ("nb", "poisson"):
            raise ValueError("likelihood must be nb or poisson")
        for name in ("pseudo_library_match", "batch_correction"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        for name in ("dropout", "mask_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("learning_rate", "target_sum", "log_scale", "max_log_platform"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "weight_decay",
            "kl_weight",
            "bridge_weight",
            "mask_weight",
            "pseudo_weight",
            "platform_penalty",
            "spatial_weight",
            "marker_weight",
            "entropy_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)
