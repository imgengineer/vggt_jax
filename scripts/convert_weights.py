import argparse
import dataclasses
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from vggt_jax.models.vggt.modeling import (
    configure_proxy_env,
    create_vggt_from_torch_checkpoint,
    save_nnx_weights_npz,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert official VGGT PyTorch checkpoint to Flax NNX npz")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to local .pt checkpoint")
    parser.add_argument("--repo-id", type=str, default="facebook/VGGT-1B", help="HF repo id when downloading")
    parser.add_argument("--filename", type=str, default="model.pt", help="HF filename")
    parser.add_argument("--cache-dir", type=str, default="./weights", help="Directory for downloaded files")
    parser.add_argument("--output", type=str, default="./weights/vggt_jax_weights.npz", help="Output .npz path")
    parser.add_argument(
        "--report", type=str, default="./weights/vggt_jax_convert_report.json", help="Output report path"
    )
    parser.add_argument("--proxy-port", type=int, default=None, help="Local proxy port, e.g. 7890")
    parser.add_argument("--strict", action="store_true", help="Fail if any weight is unmapped or mismatched")
    parser.add_argument(
        "--include-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to map track_head.tracker.* keys",
    )
    args = parser.parse_args()

    configure_proxy_env(args.proxy_port)

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.cache_dir,
            local_dir_use_symlinks=False,
        )

    model, report = create_vggt_from_torch_checkpoint(
        checkpoint_path,
        strict=args.strict,
        include_track=args.include_track,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nnx_weights_npz(model, str(output_path))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")

    print(f"checkpoint: {checkpoint_path}")
    print(f"weights npz: {output_path}")
    print(f"report json: {report_path}")
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
