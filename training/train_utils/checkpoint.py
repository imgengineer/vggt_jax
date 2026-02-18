from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp


@dataclass
class DDPCheckpointSaver:
    save_dir: str

    def __post_init__(self):
        self.save_path = Path(self.save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())

    def save(self, checkpoint: dict[str, Any], name: str = "latest.ckpt") -> str:
        path = str(self.save_path / name)
        self.checkpointer.save(path, checkpoint, force=True)
        return path

    def load(self, name: str = "latest.ckpt") -> dict[str, Any]:
        path = str(self.save_path / name)
        return self.checkpointer.restore(path)


__all__ = ["DDPCheckpointSaver"]
