import argparse

from vggt.utils import load_and_preprocess_images, resolve_image_paths

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
    parser = argparse.ArgumentParser(description="Run VGGT inference on an image directory.")
    parser.add_argument("--image-dir", type=str, required=True, help="Input image directory.")
    parser.add_argument("--target-size", type=int, default=None, help="Inference resolution.")
    add_model_loading_args(parser, include_tiny=True)
    args = parser.parse_args()

    image_paths = resolve_image_paths(args.image_dir)
    target_size = resolve_target_size(args.target_size, tiny=args.tiny)
    images = load_and_preprocess_images(image_paths, target_size=target_size)
    images = images[None, ...]

    cfg = resolve_model_config(tiny=args.tiny)
    load_spec = resolve_model_load_spec(
        tiny=args.tiny,
        weights_orbax=args.weights_orbax,
        weights_npz=args.weights_npz,
        from_pretrained=args.from_pretrained,
        prefer_orbax=args.prefer_orbax,
    )
    model = load_vggt_model(
        config=cfg,
        load_spec=load_spec,
        repo_id=args.repo_id,
        filename=args.filename,
        cache_dir=args.cache_dir,
    )

    preds = model(images)
    for key, value in preds.items():
        if isinstance(value, list):
            print(key, len(value), value[-1].shape)
        else:
            print(key, value.shape)


if __name__ == "__main__":
    main()
