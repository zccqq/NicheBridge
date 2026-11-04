"""Configuration extending the installed NicheBridgeDeconvolver API."""

from dataclasses import dataclass
import math
from .._config import NicheBridgeConfig


@dataclass
class NicheBridgeJointConfig(NicheBridgeConfig):
    epochs: int = 800
    profile_mode: str = "raw"
    batch_correction: bool = True
    platform_penalty: float = 0.01
    bridge_weight: float = 0.05
    candidates_per_type: int = 8
    relation_dim: int = 32
    attention_temperature: float = 0.25
    dictionary_pseudocount: float = 20.0
    bulk_weight: float = 1.0
    bulk_reference_weight: float = 0.5
    consistency_weight: float = 0.1
    use_single_cell: bool = True
    use_vae: bool = True
    use_diffusion: bool = True

    def __post_init__(self):
        super().__post_init__()
        if self.profile_mode != "raw":
            raise ValueError("The individual-cell dictionary requires profile_mode='raw'")
        if not self.pseudo_library_match:
            raise ValueError("Joint deconvolution requires pseudo_library_match=True")
        for name in ("candidates_per_type", "relation_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("attention_temperature", "dictionary_pseudocount"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("bulk_weight", "bulk_reference_weight", "consistency_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("use_single_cell", "use_vae", "use_diffusion"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if not self.use_diffusion and self.bridge_weight != 0:
            raise ValueError("bridge_weight must be zero when use_diffusion=False")
