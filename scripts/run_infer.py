import argparse

from flax import nnx

from vggt_jax.models import ModelConfig, VGGT
from vggt_jax.utils import load_and_preprocess_images, resolve_image_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--tiny", action="store_true")
    args = parser.parse_args()

    image_paths = resolve_image_paths(args.image_dir)
    images = load_and_preprocess_images(image_paths)
    images = images[None, ...]

    cfg = ModelConfig.vggt_tiny() if args.tiny else ModelConfig.vggt_base()
    model = VGGT(cfg, rngs=nnx.Rngs(0))

    preds = model(images, deterministic=True)
    for key, value in preds.items():
        if isinstance(value, list):
            print(key, len(value), value[-1].shape)
        else:
            print(key, value.shape)


if __name__ == "__main__":
    main()
