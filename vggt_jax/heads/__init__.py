from importlib import import_module

from vggt_jax.heads.head_act import activate_head, activate_pose
from vggt_jax.heads.track_ops import run_tracker_predictor

_HEAD_EXPORTS = {
    "CameraHeadConfig",
    "CameraHead",
    "DPTHeadConfig",
    "DPTHead",
    "TrackHeadConfig",
    "TrackHead",
}

__all__ = [
    "activate_head",
    "activate_pose",
    "run_tracker_predictor",
    "CameraHeadConfig",
    "CameraHead",
    "DPTHeadConfig",
    "DPTHead",
    "TrackHeadConfig",
    "TrackHead",
]


def __getattr__(name: str):
    if name in _HEAD_EXPORTS:
        module = import_module("vggt_jax.models.vggt.modeling")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
