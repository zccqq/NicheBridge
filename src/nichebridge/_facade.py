"""One entry point for NicheBridge's three explicitly selected estimators."""

import json
from pathlib import Path


class NicheBridge:
    """Select ``pair``, ``joint`` or ``trajectory`` without changing packages.

    Pair and joint modes preserve count-mixture deconvolution. Trajectory mode
    implements the manuscript's generative transport and reverse reconstruction.
    Each estimator retains its documented input contract and return types.
    """

    def __init__(self, mode="pair", config=None, **settings):
        from ._model import NicheBridgeDeconvolver
        from .joint.model import NicheBridgeJoint
        if mode not in {"pair", "joint", "trajectory"}:
            raise ValueError("mode must be pair, joint or trajectory")
        if config is not None and settings:
            raise ValueError("Supply a config or keyword settings, not both")
        if mode == "trajectory":
            from .trajectory import NicheBridgeTrajectory, TrajectoryConfig
            self.estimator_ = NicheBridgeTrajectory(config or TrajectoryConfig(**settings))
        else:
            constructor = NicheBridgeDeconvolver if mode == "pair" else NicheBridgeJoint
            self.estimator_ = constructor(config=config, **settings)
        self.mode = mode

    def fit(self, *args, **kwargs):
        self.estimator_.fit(*args, **kwargs)
        return self

    def __getattr__(self, name):
        estimator = self.__dict__.get("estimator_")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)

    def save(self, directory):
        directory = Path(directory)
        self.estimator_.save(directory)
        (directory / "nichebridge.json").write_text(
            json.dumps({"format": "nichebridge-facade-v1", "mode": self.mode}, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, directory, device="auto"):
        directory = Path(directory)
        metadata = json.loads((directory / "nichebridge.json").read_text(encoding="utf-8"))
        if metadata.get("format") != "nichebridge-facade-v1":
            raise ValueError("Unsupported NicheBridge checkpoint")
        self = cls(mode=metadata["mode"])
        self.estimator_ = type(self.estimator_).load(directory, device=device)
        return self
