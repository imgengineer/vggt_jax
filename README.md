# VGGT JAX (Flax NNX)

基于 `JAX + Flax NNX` 的 VGGT 实现，参考：

- `jax-ml/bonsai` 的工程组织（`modeling.py + params.py` 风格）
- `facebookresearch/vggt` 的模型结构（Aggregator + Camera/Depth/Point/Track heads）

同时提供 `bonsai` 风格的模型入口：

- `vggt_jax.models.vggt.modeling`
- 核心实现文件位于 `vggt_jax/models/vggt/`

## 1) 用 uv 配置环境

```bash
cd vggt_jax
bash scripts/setup_uv.sh
```

如果需要 GPU（CUDA 13）+ Grain 数据加载：

```bash
bash scripts/setup_uv.sh cuda13 data
```

并执行：

- `uv venv .venv`
- `uv sync --extra dev [--extra cuda13] [--extra data] [--extra convert]`

## 2) 下载官方权重

```bash
uv run python scripts/download_weights.py \
  --repo-id facebook/VGGT-1B \
  --filename model.pt \
  --cache-dir ./weights
```

## 3) 官方 `model.pt` 转 NNX `npz`

先安装转换依赖（`torch`，用于读取 `.pt`）：

```bash
uv sync --extra dev --extra convert
```

执行转换：

```bash
uv run python scripts/convert_weights.py \
  --repo-id facebook/VGGT-1B \
  --filename model.pt \
  --cache-dir ./weights \
  --output ./weights/vggt_jax_weights_full.npz \
  --report ./weights/vggt_jax_convert_report_full.json
```

可选：

- `--checkpoint-path /path/to/model.pt`：使用本地权重
- `--strict`：只要有未映射/形状不匹配即失败
- `--no-include-track`：跳过 `track_head.tracker.*` 映射

当前版本已支持官方 `VGGT-1B` checkpoint 的**全键映射加载**（`mapped=loaded=1797`）。

可用下面命令做一次严格校验：

```bash
uv run python scripts/verify_bonsai_structure_and_weights.py
```

## 4) 运行最小前向示例

```python
import jax
from flax import nnx
from vggt_jax.models import ModelConfig, VGGT

cfg = ModelConfig.vggt_tiny()
model = VGGT(cfg, rngs=nnx.Rngs(0))

images = jax.random.uniform(jax.random.key(1), (1, 3, 3, 56, 56))
pred = model(images, deterministic=True)

print(pred["pose_enc"].shape)
print(pred["depth"].shape)
print(pred["world_points"].shape)
```

也支持便捷构造方式（不需要先建 `ModelConfig`）：

```python
from flax import nnx
from vggt_jax.models import VGGT

model = VGGT(rngs=nnx.Rngs(0))  # 等价于 base 配置
model.eval()
```

预训练加载（会从 HF 下载 `model.pt` 并转换加载；需要安装 `torch`）：

```python
from flax import nnx
from vggt_jax.models import VGGT

model = VGGT.from_pretrained(rngs=nnx.Rngs(0))
model.eval()
```

如果你已经有转换后的 `npz`，可以直接加载（不需要 `torch`）：

```python
from flax import nnx
from vggt_jax.models.vggt import create_vggt_from_nnx_npz

model = create_vggt_from_nnx_npz("weights/vggt_jax_weights_full.npz", rngs=nnx.Rngs(0))
model.eval()
```

## 5) 当前实现范围

- ✅ NNX 版 Aggregator（frame/global 交替注意力）
- ✅ Aggregator `patch_embed` 已切换为官方 DINO-ViT holder 运行路径
- ✅ DPT Head 已按官方结构实现（`projects/resize/scratch/refinenet/custom_interpolate/pos_embed`）
- ✅ Camera / Track 头（Track 为 JAX 迭代更新推理路径）
- ✅ `model.pt -> NNX` 转换入口、键映射与转换报告
- ✅ 官方 `VGGT-1B` 已实现全键映射加载（含 `track_head.tracker`）

## 6) 用 Grain 加载图片并跑一次前向（GPU/CPU 都可）

需要先安装 `grain`（`uv sync --extra data` 或 `bash scripts/setup_uv.sh data`）。

```bash
uv run python scripts/run_infer_grain.py \
  --image-dir /path/to/images \
  --batch-size 1 \
  --limit-batches 1
```

如果已经有转换后的 `npz`（默认会尝试 `weights/vggt_jax_weights_full.npz`），也可以显式指定：

```bash
uv run python scripts/run_infer_grain.py \
  --image-dir /path/to/images \
  --weights-npz weights/vggt_jax_weights_full.npz
```

## 7) 速度基准（JAX）

```bash
uv run python scripts/bench_speed.py --mode forward --tiny --seq-len 3 --height 294 --width 224
```
