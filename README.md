# VGGT: Visual Geometry Grounded Transformer (JAX/Flax NNX)

[中文文档](README.zh-CN.md)

This repository is a pure JAX ecosystem reimplementation of VGGT based on `JAX + Flax NNX + Grain + Orbax`, with a structure aligned to `facebookresearch/vggt`.

- Hugging Face public weights are the default for inference
- Orbax Refactored CheckpointManager is used for training save/restore
- Multi-image inference, COLMAP export, and Gradio/Viser demos are included

---

## Overview

VGGT is a feed-forward 3D geometry model that predicts from single or multiple images:

- Camera parameters (pose/intrinsics encoding)
- Depth map and confidence
- Point map and confidence
- Point tracks (given query points)

This implementation is pure JAX/Flax and uses `NHWC` by default (`B,T,H,W,C` for sequences).

---

## Quick Start

### 1) Install

```bash
bash scripts/setup_uv.sh data
```

For demo + training:

```bash
bash scripts/setup_uv.sh --extra data --extra demo --extra train
```

Custom setup example (without `dev`):

```bash
bash scripts/setup_uv.sh --no-dev train
```

Script details: `scripts/README.md`

Install quick reference:

- Inference (recommended): `bash scripts/setup_uv.sh data`
- Inference + demo: `bash scripts/setup_uv.sh --extra data --extra demo`
- Inference + training: `bash scripts/setup_uv.sh --extra data --extra train`
- Full development: `bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- Lightweight (no `dev`): `bash scripts/setup_uv.sh --no-dev data`

Key dependencies:

- `jax[cuda13]>=0.9.0.1`
- `flax>=0.12.4`

JAX selects devices automatically: GPU when available, CPU fallback otherwise.

### 2) Minimal inference (default HF weights)

```python
from flax import nnx
import jax.numpy as jnp

from vggt.models import VGGT
from vggt.utils import load_and_preprocess_images

image_paths = ["path/to/a.jpg", "path/to/b.jpg", "path/to/c.jpg"]
images = load_and_preprocess_images(image_paths, target_size=518)   # [T,H,W,C]

model = VGGT.from_pretrained(
    repo_id="facebook/VGGT-1B",
    cache_dir="./weights",
    rngs=nnx.Rngs(0),
)
model.eval()

pred = model(images[None, ...])  # [B,T,H,W,C]
print(pred["pose_enc"].shape)        # [B,T,9]
print(pred["depth"].shape)           # [B,T,H,W,1]
print(pred["world_points"].shape)    # [B,T,H,W,3]
```

---

## Weight Loading Strategy

### Inference defaults

Inference defaults to Hugging Face public weights (`model.safetensors`):

- API: `VGGT.from_pretrained(...)`
- default behavior: `prefer_hf_weights=True`
- Orbax cache writing is optional

### Orbax usage

Orbax is primarily used for training checkpoint save/restore, plus optional converted-weight caching.

Convert HF weights to local Orbax (for warm start, etc.):

```bash
uv run python scripts/download_weights.py \
  --repo-id facebook/VGGT-1B \
  --filename model.safetensors \
  --cache-dir ./weights

uv run python scripts/convert_weights.py \
  --format orbax \
  --repo-id facebook/VGGT-1B \
  --filename model.safetensors \
  --cache-dir ./weights \
  --output ./weights/orbax/facebook__VGGT-1B_model_full \
  --report ./weights/vggt_convert_report_full.json
```

Verify mapping coverage:

```bash
uv run python scripts/verify_weights.py --checkpoint-path weights/model.safetensors
```

---

## Inference CLI

More script examples: `scripts/README.md`

### Standard inference (HF default)

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images
```

### Prefer local Orbax/NPZ

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images --prefer-orbax
```

### Grain streaming inference

```bash
uv run python scripts/run_infer_grain.py --image-dir /path/to/images --limit-batches 1
```

---

## Detailed Usage

Common output keys:

- `pose_enc`, `pose_enc_list`
- `depth`, `depth_conf`
- `world_points`, `world_points_conf`
- `track`, `vis`, `conf` (when `query_points` is provided)

Track prediction example:

```python
import jax.numpy as jnp

query_points = jnp.array([[[100.0, 200.0], [60.0, 260.0]]], dtype=jnp.float32)  # [B,N,2]
pred = model(images[None, ...], query_points=query_points)
print(pred["track"].shape, pred["vis"].shape, pred["conf"].shape)
```

---

## Interactive Demo

Install demo dependencies first:

```bash
bash scripts/setup_uv.sh --extra data --extra demo
```

### Gradio

```bash
uv run python demo_gradio.py
```

### Viser

```bash
uv run python demo_viser.py --image_folder /path/to/images
```

---

## Export to COLMAP

Export VGGT predictions to COLMAP format:

```bash
uv run python demo_colmap.py --scene-dir /path/to/scene
```

Input structure:

```text
scene/
└── images/
```

Output structure:

```text
scene/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

---

## Training

Training entrypoint:

```bash
uv run python -m training.launch --config default
```

Training checkpoints are saved/restored with Orbax Refactored `CheckpointManager`.

Training guide: `training/README.md`

---

## Repository Layout

```text
vggt/                 # model and utility implementation
training/             # training framework (JAX/Flax)
scripts/              # inference/conversion/verification scripts
demo_gradio.py        # Gradio demo
demo_viser.py         # Viser demo
demo_colmap.py        # COLMAP export entry
```
