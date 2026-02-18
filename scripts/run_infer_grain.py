import argparse

import jax
import jax.numpy as jnp

from vggt.data.grain import create_image_folder_iter_dataset

try:
    from scripts._common import (
        add_model_loading_args,
        load_vggt_model,
        resolve_model_config,
        resolve_model_load_spec,
        resolve_target_size,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _common import (
        add_model_loading_args,
        load_vggt_model,
        resolve_model_config,
        resolve_model_load_spec,
        resolve_target_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT inference with a Grain image folder iterator.")
    parser.add_argument("--image-dir", type=str, required=True, help="Input image directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for Grain iterator.")
    parser.add_argument("--target-size", type=int, default=None, help="Inference resolution.")
    parser.add_argument("--num-threads", type=int, default=16, help="Grain loading threads.")
    parser.add_argument("--prefetch", type=int, default=16, help="Grain prefetch buffer size.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle image order.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for Grain.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs for iterator.")
    parser.add_argument("--drop-remainder", action="store_true", help="Drop final incomplete batch.")
    parser.add_argument("--limit-batches", type=int, default=1, help="Stop after N batches.")
    add_model_loading_args(parser, include_tiny=True)
    args = parser.parse_args()

    target_size = resolve_target_size(args.target_size, tiny=args.tiny)
    cfg = resolve_model_config(tiny=args.tiny)
    load_spec = resolve_model_load_spec(
        tiny=args.tiny,
        weights_orbax=args.weights_orbax,
        weights_npz=args.weights_npz,
        from_pretrained=args.from_pretrained,
        prefer_orbax=args.prefer_orbax,
    )

    print("backend:", jax.default_backend())
    print("devices:", jax.devices())

    model = load_vggt_model(
        config=cfg,
        load_spec=load_spec,
        repo_id=args.repo_id,
        filename=args.filename,
        cache_dir=args.cache_dir,
    )

    ds = create_image_folder_iter_dataset(
        args.image_dir,
        batch_size=args.batch_size,
        target_size=target_size,
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
        preds = model(images)

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
