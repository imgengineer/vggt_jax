import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
from flax import nnx

from vggt_jax.data.grain import create_image_folder_iter_dataset
from vggt_jax.models import ModelConfig, VGGT
from vggt_jax.models.vggt import create_vggt_from_nnx_npz


def _load_model(
    *,
    tiny: bool,
    weights_npz: str | None,
    from_pretrained: bool,
    proxy_port: int | None,
) -> VGGT:
    config = ModelConfig.vggt_tiny() if tiny else ModelConfig.vggt_base()
    rngs = nnx.Rngs(0)

    if from_pretrained:
        model = VGGT.from_pretrained(proxy_port=proxy_port, config=config, rngs=rngs)
    elif weights_npz is not None:
        model = create_vggt_from_nnx_npz(weights_npz, config, rngs=rngs, strict=True)
    else:
        model = VGGT(config, rngs=rngs)

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--prefetch", type=int, default=16)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--drop-remainder", action="store_true")
    parser.add_argument("--limit-batches", type=int, default=1)

    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--weights-npz", type=str, default=None)
    parser.add_argument("--from-pretrained", action="store_true")
    parser.add_argument("--proxy-port", type=int, default=7890)
    args = parser.parse_args()

    if args.target_size is None:
        args.target_size = 224 if args.tiny else 518

    weights_npz = args.weights_npz
    if weights_npz is None and not args.tiny:
        default_npz = Path("weights/vggt_jax_weights_full.npz")
        if default_npz.exists():
            weights_npz = str(default_npz)

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    model = _load_model(
        tiny=args.tiny,
        weights_npz=weights_npz,
        from_pretrained=args.from_pretrained,
        proxy_port=args.proxy_port,
    )

    ds = create_image_folder_iter_dataset(
        args.image_dir,
        batch_size=args.batch_size,
        target_size=args.target_size,
        shuffle=args.shuffle,
        seed=args.seed,
        num_epochs=args.epochs,
        drop_remainder=args.drop_remainder,
        num_threads=args.num_threads,
        prefetch_buffer_size=args.prefetch,
    )

    for step, batch in enumerate(ds):
        images_np = batch["images"]
        paths = batch["paths"]

        images = jnp.asarray(images_np)
        preds = model(images, deterministic=True)

        depth = preds["depth"]
        points = preds["world_points"]
        print(f"[batch {step}] images={images.shape} device={images.device}")
        print(f"  depth={depth.shape} device={getattr(depth, 'device', None)}")
        print(f"  world_points={points.shape} device={getattr(points, 'device', None)}")
        print(f"  paths[0]={paths[0]}")

        if step + 1 >= args.limit_batches:
            break


if __name__ == "__main__":
    main()
