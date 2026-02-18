# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse


def main():
    try:
        from hydra import compose, initialize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Missing `hydra-core`. Install with `bash scripts/setup_uv.sh train`.") from exc
    try:
        from .trainer import Trainer
    except ImportError:  # pragma: no cover
        from trainer import Trainer

    parser = argparse.ArgumentParser(description="Train model with configurable YAML file")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Name of the config file (without .yaml extension, default: default)"
    )
    args = parser.parse_args()

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
