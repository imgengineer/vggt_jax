# Training (JAX/Flax NNX) / 训练说明

[English](README.md)

该目录包含 VGGT 的纯 JAX 训练流程。

## 1) 安装

```bash
bash scripts/setup_uv.sh train
```

安装组合速查：

- 仅训练：`bash scripts/setup_uv.sh train`
- 训练 + 数据：`bash scripts/setup_uv.sh --extra data --extra train`
- 全功能开发：`bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- 轻量环境（无 `dev`）：`bash scripts/setup_uv.sh --no-dev train`

## 2) 配置数据路径

修改 `training/config/default.yaml`：

- `CO3D_DIR`
- `CO3D_ANNOTATION_DIR`

## 3) 启动训练

```bash
uv run python -m training.launch --config default
```

## 4) 说明

- 数据张量采用 `NHWC`（图像序列为 `B,T,H,W,C`）。
- Checkpoint 使用 Orbax **Refactored `CheckpointManager` API** 保存到 `checkpoint.save_dir`。
- 设备由 JAX 运行时自动选择（有 GPU 用 GPU，否则回退 CPU）。
