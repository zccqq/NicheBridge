"""NicheBridge spatial deconvolution and generative diffusion transport."""
from ._config import NicheBridgeConfig
from ._model import NicheBridgeDeconvolver
from .joint.config import NicheBridgeJointConfig
from .joint.model import NicheBridgeJoint
from . import pp, tl, pl

__version__ = "0.0.1"
__all__ = ["NicheBridge", "NicheBridgeConfig", "NicheBridgeDeconvolver",
           "NicheBridgeJoint", "NicheBridgeJointConfig",
           "NicheBridgeTrajectory", "TrajectoryConfig",
           "pp", "tl", "pl", "analysis"]

def __getattr__(name):
    if name in {"NicheBridgeTrajectory", "TrajectoryConfig"}:
        from .trajectory import NicheBridgeTrajectory, TrajectoryConfig
        return {"NicheBridgeTrajectory": NicheBridgeTrajectory,
                "TrajectoryConfig": TrajectoryConfig}[name]
    if name == "NicheBridge":
        from ._facade import NicheBridge
        return NicheBridge
    if name == "analysis":
        import importlib
        return importlib.import_module(".analysis", __name__)
    raise AttributeError(name)
