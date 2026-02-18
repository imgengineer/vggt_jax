# VGGT: Visual Geometry Grounded Transformer (JAX/Flax NNX)

[English](README.md)

基于 `JAX + Flax NNX + Grain + Orbax` 的 VGGT 复刻实现，目录结构与官方 `facebookresearch/vggt` 保持一致风格，并提供：

- Hugging Face 公共权重默认加载（推理）
- Orbax Refactored CheckpointManager（训练保存/恢复）
- 多图推理、COLMAP 导出、Gradio/Viser 可视化

---

## Overview

VGGT 是一个前馈式 3D 视觉几何模型，可从单张到多张图像直接预测：

- 相机参数（外参/内参编码）
- 深度图与置信度
- 点图（point map）与置信度
- 点轨迹（给定 query points）

本仓库是纯 JAX 生态实现，输入布局默认采用 `NHWC`，序列输入为 `B,T,H,W,C`。

---

## Quick Start

### 1) 安装

```bash
bash scripts/setup_uv.sh data
```

如需 demo / 训练：

```bash
bash scripts/setup_uv.sh --extra data --extra demo --extra train
```

如果需要自定义组合（例如去掉 `dev`）：

```bash
bash scripts/setup_uv.sh --no-dev train
```

脚本详情见：`scripts/README.zh-CN.md`

安装组合速查：

- 推理（推荐）：`bash scripts/setup_uv.sh data`
- 推理 + Demo：`bash scripts/setup_uv.sh --extra data --extra demo`
- 推理 + 训练：`bash scripts/setup_uv.sh --extra data --extra train`
- 全功能开发：`bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- 轻量环境（无 dev）：`bash scripts/setup_uv.sh --no-dev data`

默认关键依赖：

- `jax[cuda13]>=0.9.0.1`
- `flax>=0.12.4`

JAX 会自动选择设备：优先 GPU，无 GPU 自动回退 CPU。

### 2) 最小推理示例（默认加载 HF 开源权重）

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

### 推理默认策略

默认优先加载 Hugging Face 公共权重（`model.safetensors`）：

- API：`VGGT.from_pretrained(...)`
- 默认行为：`prefer_hf_weights=True`
- 不强制写 Orbax 缓存（可选）

### Orbax 用途

Orbax 主要用于训练过程中的 checkpoint 保存与恢复，以及可选的权重预转换缓存。

如果需要将 HF 权重转换为本地 Orbax（例如训练 warm-start）：

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

校验映射：

```bash
uv run python scripts/verify_weights.py --checkpoint-path weights/model.safetensors
```

---

## Inference CLI

脚本参数和更多示例见：`scripts/README.zh-CN.md`

### 常规推理（默认 HF）

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images
```

### 强制优先本地 Orbax/NPZ

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images --prefer-orbax
```

### Grain 数据流推理

```bash
uv run python scripts/run_infer_grain.py --image-dir /path/to/images --limit-batches 1
```

---

## Detailed Usage

模型前向输出字典常用字段：

- `pose_enc`, `pose_enc_list`
- `depth`, `depth_conf`
- `world_points`, `world_points_conf`
- `track`, `vis`, `conf`（提供 `query_points` 时）

轨迹预测示例：

```python
import jax.numpy as jnp

query_points = jnp.array([[[100.0, 200.0], [60.0, 260.0]]], dtype=jnp.float32)  # [B,N,2]
pred = model(images[None, ...], query_points=query_points)
print(pred["track"].shape, pred["vis"].shape, pred["conf"].shape)
```

---

## Interactive Demo

先安装 demo 依赖：

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

支持将 VGGT 预测结果导出为 COLMAP 格式：

```bash
uv run python demo_colmap.py --scene-dir /path/to/scene
```

输入目录要求：

```text
scene/
└── images/
```

输出目录：

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

训练入口：

```bash
uv run python -m training.launch --config default
```

训练 checkpoint 使用 Orbax Refactored `CheckpointManager` API 保存/恢复。

训练相关说明见：`training/README.zh-CN.md`

---

## Repository Layout

```text
vggt/                 # 模型与工具实现
training/             # 训练框架（JAX/Flax）
scripts/              # 推理/转换/校验脚本
demo_gradio.py        # Gradio demo
demo_viser.py         # Viser demo
demo_colmap.py        # COLMAP 导出入口
```
