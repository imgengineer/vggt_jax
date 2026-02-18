# Scripts

[中文文档](README.zh-CN.md)

The `scripts/` directory is grouped into three categories:

- Inference: `run_infer.py`, `run_infer_grain.py`, `demo_colmap.py`
- Weights: `download_weights.py`, `convert_weights.py`, `verify_weights.py`
- Environment: `setup_uv.sh`

`scripts/_common.py` is the shared internal module for unified model-loading arguments and defaults (HF-first with optional local Orbax/NPZ preference).

## 0) Environment

Recommended first step:

```bash
bash scripts/setup_uv.sh data
```

Notes:

- `dev` extra is included by default.
- Supports positional extras or `--extra`: `bash scripts/setup_uv.sh --extra data --extra demo`.
- To drop `dev`: `bash scripts/setup_uv.sh --no-dev train`.
- Available extras: `data demo dev convert track train`.

Install quick reference:

- Inference (recommended): `bash scripts/setup_uv.sh data`
- Inference + demo: `bash scripts/setup_uv.sh --extra data --extra demo`
- Inference + training: `bash scripts/setup_uv.sh --extra data --extra train`
- Full development: `bash scripts/setup_uv.sh --extra data --extra demo --extra train --extra convert --extra track`
- Lightweight (no `dev`): `bash scripts/setup_uv.sh --no-dev data`

## 1) Inference scripts

### Single-run inference (HF default)

```bash
uv run python scripts/run_infer.py --image-dir /path/to/images
```

### Grain streaming inference

```bash
uv run python scripts/run_infer_grain.py \
  --image-dir /path/to/images \
  --batch-size 1 \
  --limit-batches 1
```

### COLMAP export

```bash
uv run python scripts/demo_colmap.py --scene-dir /path/to/scene
```

Expected input structure:

```text
scene/
└── images/
```

## 2) Weight scripts

### Download official weights

```bash
uv run python scripts/download_weights.py \
  --repo-id facebook/VGGT-1B \
  --filename model.safetensors \
  --cache-dir ./weights
```

### Convert to Orbax or NPZ

```bash
uv run python scripts/convert_weights.py \
  --format orbax \
  --output ./weights/orbax/facebook__VGGT-1B_model_full
```

### Verify conversion mapping coverage

```bash
uv run python scripts/verify_weights.py
```

## 3) Common model-loading arguments

Inference scripts share:

- `--from-pretrained/--no-from-pretrained`
- `--prefer-orbax`
- `--weights-orbax`
- `--weights-npz`
- `--repo-id`
- `--filename`
- `--cache-dir`

Notes:

- Base config defaults to HF public weights when no local weights are provided.
- `--prefer-orbax` prioritizes local Orbax/NPZ when available.
