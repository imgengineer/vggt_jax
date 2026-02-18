# Training (JAX/Flax NNX)

[中文文档](README.zh-CN.md)

This folder contains a pure JAX training pipeline for VGGT.

## 1) Install

```bash
bash scripts/setup_uv.sh train
```

Install quick reference:

- Training only: `bash scripts/setup_uv.sh train`
- Training + data: `bash scripts/setup_uv.sh --extra data --extra train`
- Full development: `bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- Lightweight (no `dev`): `bash scripts/setup_uv.sh --no-dev train`

## 2) Configure dataset paths

Edit `training/config/default.yaml`:

- `CO3D_DIR`
- `CO3D_ANNOTATION_DIR`

## 3) Run training

```bash
uv run python -m training.launch --config default
```

## 4) Notes

- Data tensors use `NHWC` (`B,T,H,W,C` for image sequences).
- Checkpoints are saved with Orbax **Refactored `CheckpointManager` API** into `checkpoint.save_dir`.
- Device selection follows JAX runtime behavior (GPU if available, otherwise CPU).
