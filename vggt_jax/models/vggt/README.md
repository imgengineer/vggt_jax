# VGGT in JAX (Flax NNX)

本目录包含 **VGGT（Visual Geometry Grounded Transformer）** 的 **JAX + Flax NNX** 实现，并刻意采用 `jax-ml/bonsai` 的模型组织方式（`modeling.py + params.py`）：

- `modeling.py`
  - 模型结构定义（`Aggregator + Camera/DPT/Track heads`）
  - 预训练权重下载、`model.pt -> NNX` 键映射与加载
  - `npz` 权重保存 / 加载
- `params.py`
  - 兼容导出层：将 bonsai 风格的对外 API 转发到 `modeling.py`
- `__init__.py`
  - 统一导出（便于 `from vggt_jax.models.vggt import ...`）

## Model Configuration Support Status

| Variant | `ModelConfig` | 权重加载 | 说明 |
| :--- | :--- | :--- | :--- |
| VGGT-1B (base) | `ModelConfig.vggt_base()` | ✅ 支持 | 对齐 `facebook/VGGT-1B` 的 `model.pt`（全键映射） |
| VGGT-tiny | `ModelConfig.vggt_tiny()` | ⚠️ 仅随机初始化 | 用于单测/调试与速度基准（无官方 tiny 权重） |

## Running this model

### 1) 最小前向（随机权重）

```python
import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.models import ModelConfig, VGGT

cfg = ModelConfig.vggt_tiny()
model = VGGT(cfg, rngs=nnx.Rngs(0)).eval()

images = jax.random.uniform(jax.random.key(0), (1, 3, 3, 56, 56), dtype=jnp.float32)
pred = model(images, deterministic=True)

print(pred["depth"].shape, pred["world_points"].shape, pred["pose_enc"].shape)
```

### 2) 加载预训练权重（HF `model.pt` -> 现场转换并加载）

该路径需要 `torch`（用于读取 `model.pt`）。安装转换依赖：

```bash
uv sync --extra dev --extra convert
```

加载预训练：

```python
from flax import nnx
from vggt_jax.models import VGGT

model = VGGT.from_pretrained(rngs=nnx.Rngs(0))
model.eval()
```

### 3) 直接加载转换后的 `npz`（不需要 `torch`）

```python
from flax import nnx
from vggt_jax.models.vggt import create_vggt_from_nnx_npz

model = create_vggt_from_nnx_npz("weights/vggt_jax_weights_full.npz", rngs=nnx.Rngs(0), strict=True).eval()
```

## Weight Loading / Verification

严格校验 `.pt` 权重是否 **全键映射且全部成功加载**（需要 `--extra convert`）：

```bash
uv run python scripts/verify_bonsai_structure_and_weights.py
```

说明：

- “结构对齐”在这里指 **模块层级 / 参数路径** 对齐，以便做到稳定的键映射与权重加载。
- `weights/vggt_jax_convert_report_full.json` 记录了全量映射统计（例如 `total_source_keys == loaded_keys`）。

## Determinism / train & eval

- NNX 模块支持 `.train()` / `.eval()` 切换；本实现也提供同名方法以对齐原版 API 的使用习惯。
- 推理建议使用：`model.eval(); pred = model(images, deterministic=True)`
- 为对齐原版行为：仅在 `deterministic=True`（即 eval/inference）时返回 `pred["images"]`。

## Benchmarking

- JAX（NNX）速度：`scripts/bench_speed.py`

## Notes / Known Gaps

- 本仓库目标是 **结构与权重路径的严格对齐**，并尽可能对齐数值行为。
- 由于浮点数与后端实现差异，`track_head.tracker` 的多迭代推理可能出现数值漂移；这不影响权重映射与加载的“结构一致性”目标。
