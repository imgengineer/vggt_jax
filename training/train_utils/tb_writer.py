from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SummaryWriter:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "events.jsonl"

    def add_scalar(self, tag: str, scalar_value: float, global_step: int):
        record = {"type": "scalar", "tag": tag, "value": float(scalar_value), "step": int(global_step)}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def add_images(self, tag: str, data: Any, global_step: int):
        record = {"type": "images", "tag": tag, "shape": getattr(data, "shape", None), "step": int(global_step)}
        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def flush(self):
        return None

    def close(self):
        return None


class TensorBoardLogger(SummaryWriter):
    def __init__(self, path: str):
        super().__init__(log_dir=path)


__all__ = ["SummaryWriter", "TensorBoardLogger"]
