import argparse
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.models import ModelConfig, VGGT
from vggt_jax.utils.io import load_and_preprocess_images_np, resolve_image_paths

try:
    import grain  # type: ignore
except ImportError:  # pragma: no cover
    grain = None


def _block_until_ready(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)


def _build_model(*, tiny: bool) -> VGGT:
    cfg = ModelConfig.vggt_tiny() if tiny else ModelConfig.vggt_base()
    model = VGGT(cfg, rngs=nnx.Rngs(0))
    model.eval()
    return model


def _python_loader(image_dir: str, *, seq_len: int, target_size: int) -> Callable[[], dict[str, Any]]:
    paths = resolve_image_paths(image_dir)
    if len(paths) < seq_len:
        raise ValueError(f"Need at least {seq_len} images in {image_dir}, got {len(paths)}")
    fixed = paths[:seq_len]

    def next_batch() -> dict[str, Any]:
        images = load_and_preprocess_images_np(fixed, target_size=target_size)
        return {"images": images, "paths": fixed}

    return next_batch


def _grain_loader(
    image_dir: str,
    *,
    seq_len: int,
    target_size: int,
    num_threads: int,
    prefetch: int,
) -> Callable[[], dict[str, Any]]:
    if grain is None:
        raise SystemExit("Missing dependency: grain. Install with: uv sync --extra data")

    paths = resolve_image_paths(image_dir)
    if len(paths) < seq_len:
        raise ValueError(f"Need at least {seq_len} images in {image_dir}, got {len(paths)}")
    fixed = paths[:seq_len]

    def batch_fn(indices: list[int]) -> dict[str, Any]:
        selected = [fixed[int(i)] for i in indices]
        images = load_and_preprocess_images_np(selected, target_size=target_size)
        return {"images": images, "paths": selected}

    ds = grain.MapDataset.range(0, seq_len).repeat(num_epochs=None).batch(seq_len, batch_fn=batch_fn)
    it = iter(ds.to_iter_dataset(read_options=grain.ReadOptions(num_threads=num_threads, prefetch_buffer_size=prefetch)))

    def next_batch() -> dict[str, Any]:
        return next(it)

    return next_batch


def _time_loop(fn: Callable[[], Any], *, iters: int) -> tuple[float, Any]:
    start = time.perf_counter()
    last = None
    for _ in range(iters):
        last = fn()
    elapsed = time.perf_counter() - start
    return elapsed, last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("forward", "data", "e2e"), default="e2e")
    parser.add_argument("--loader", choices=("python", "grain"), default="grain")
    parser.add_argument("--image-dir", type=str, default=None)

    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--height", type=int, default=None, help="Only used for mode=forward.")
    parser.add_argument("--width", type=int, default=None, help="Only used for mode=forward.")

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=16)
    args = parser.parse_args()

    if args.target_size is None:
        args.target_size = 224 if args.tiny else 518

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())
    print(f"mode={args.mode} loader={args.loader} tiny={args.tiny} seq_len={args.seq_len} target_size={args.target_size}")

    if args.mode in ("data", "e2e") and args.image_dir is None:
        raise SystemExit("--image-dir is required for mode=data/e2e")

    if args.mode == "forward":
        model = _build_model(tiny=args.tiny)
        patch_size = model.cfg.patch_size
        h = args.height if args.height is not None else args.target_size
        w = args.width if args.width is not None else args.target_size
        if h % patch_size != 0 or w % patch_size != 0:
            raise SystemExit(f"--height/--width must be divisible by patch_size={patch_size} (got {h}x{w})")
        images = jax.random.uniform(jax.random.key(0), (1, args.seq_len, 3, h, w), dtype=jnp.float32)

        @nnx.jit
        def step(model: VGGT, images: jax.Array):
            return model(images, deterministic=True)

        # compile + warmup
        for _ in range(args.warmup):
            out = step(model, images)
            _block_until_ready(out)

        elapsed, out = _time_loop(lambda: _block_until_ready(step(model, images)), iters=args.iters)
        print("images shape:", images.shape, "device:", images.device)
        print("one output keys:", sorted(out.keys()))
        print(f"avg step: {elapsed / args.iters * 1000:.2f} ms")
        print(f"throughput: {(args.seq_len * args.iters) / elapsed:.2f} frames/s")
        return

    # Data / e2e loaders.
    if args.loader == "python":
        next_batch = _python_loader(args.image_dir, seq_len=args.seq_len, target_size=args.target_size)
    else:
        next_batch = _grain_loader(
            args.image_dir,
            seq_len=args.seq_len,
            target_size=args.target_size,
            num_threads=args.num_threads,
            prefetch=args.prefetch,
        )

    if args.mode == "data":
        for _ in range(args.warmup):
            _ = next_batch()

        elapsed, batch = _time_loop(next_batch, iters=args.iters)
        images = batch["images"]
        print("batch images shape:", images.shape, "dtype:", getattr(images, "dtype", None))
        print(f"avg batch: {elapsed / args.iters * 1000:.2f} ms")
        print(f"throughput: {(args.seq_len * args.iters) / elapsed:.2f} frames/s (load+preprocess)")
        return

    # e2e
    model = _build_model(tiny=args.tiny)

    @nnx.jit
    def step(model: VGGT, images: jax.Array):
        return model(images, deterministic=True)

    def e2e_step() -> dict[str, Any]:
        batch = next_batch()
        images = jnp.asarray(batch["images"])[None, ...]
        preds = step(model, images)
        _block_until_ready(preds)
        return preds

    for _ in range(args.warmup):
        _ = e2e_step()

    elapsed, _ = _time_loop(e2e_step, iters=args.iters)
    print(f"avg step: {elapsed / args.iters * 1000:.2f} ms")
    print(f"throughput: {(args.seq_len * args.iters) / elapsed:.2f} frames/s (load+preprocess+forward)")


if __name__ == "__main__":
    main()
