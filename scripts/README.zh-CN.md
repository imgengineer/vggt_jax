# Scripts

[English](README.md)

`scripts/` 目录按功能拆分为三类：

- 推理：`run_infer.py`、`run_infer_grain.py`、`demo_colmap.py`
- 权重：`download_weights.py`、`convert_weights.py`、`verify_weights.py`
- 环境：`setup_uv.sh`

其中 `scripts/_common.py` 为内部公共模块，统一模型加载参数与默认行为（HF 优先、可切换本地 Orbax/NPZ）。

## 0) 环境脚本

推荐先执行：

```bash
bash scripts/setup_uv.sh data
```

说明：

- 默认包含 `dev` extra。
- 支持位置参数或 `--extra`：`bash scripts/setup_uv.sh --extra data --extra demo`。
- 如需去掉 `dev`：`bash scripts/setup_uv.sh --no-dev train`。
- 可用 extras：`data demo dev convert track train`。

安装组合速查：

- 推理（推荐）：`bash scripts/setup_uv.sh data`
- 推理 + Demo：`bash scripts/setup_uv.sh --extra data --extra demo`
- 推理 + 训练：`bash scripts/setup_uv.sh --extra data --extra train`
- 全功能开发：`bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- 轻量环境（无 `dev`）：`bash scripts/setup_uv.sh --no-dev data`

## 1) 推理脚本

### 单次推理（默认 HF）

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images
```

### Grain 数据流推理

```bash
uv run python scripts/run_infer_grain.py \
  --image-dir /path/to/images \
  --batch-size 1 \
  --limit-batches 1
```

### 导出 COLMAP

```bash
uv run python scripts/demo_colmap.py --scene-dir /path/to/scene
```

输入目录结构：

```text
scene/
└── images/
```

## 2) 权重脚本

### 下载官方权重

```bash
uv run python scripts/download_weights.py \
  --repo-id facebook/VGGT-1B \
  --filename model.safetensors \
  --cache-dir ./weights
```

### 转换为 Orbax 或 NPZ

```bash
uv run python scripts/convert_weights.py \
  --format orbax \
  --output ./weights/orbax/facebook__VGGT-1B_model_full
```

### 校验映射覆盖率

```bash
uv run python scripts/verify_weights.py
```

## 3) 常用模型加载参数

推理脚本统一支持：

- `--from-pretrained/--no-from-pretrained`
- `--prefer-orbax`
- `--weights-orbax`
- `--weights-npz`
- `--repo-id`
- `--filename`
- `--cache-dir`

说明：

- 默认行为：在 Base 配置下若未指定本地权重，自动走 HF 公共权重。
- `--prefer-orbax`：优先尝试本地 Orbax/NPZ（存在时优先）。
