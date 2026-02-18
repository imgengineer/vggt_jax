import argparse
import dataclasses
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from vggt.models.vggt import create_vggt_from_checkpoint, save_nnx_weights_npz, save_nnx_weights_orbax

try:
    from scripts._common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID
except ModuleNotFoundError:  # pragma: no cover
    from _common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official VGGT checkpoint (.safetensors/.npz) to Flax NNX weights (Orbax or npz)"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to local source checkpoint (`.safetensors` or `.npz`).",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HF repo id when downloading.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="HF filename when downloading.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory for downloaded files.")
    parser.add_argument("--format", choices=("orbax", "npz"), default="orbax")
    parser.add_argument(
        "--output",
        type=str,
        default="./weights/orbax/facebook__VGGT-1B_model_full",
        help="Output path (dir for orbax, file for npz)",
    )
    parser.add_argument(
        "--report", type=str, default="./weights/vggt_convert_report.json", help="Output report path"
    )
    parser.add_argument("--strict", action="store_true", help="Fail if any weight is unmapped or mismatched")
    parser.add_argument(
        "--include-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to map track_head.tracker.* keys",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.cache_dir,
            local_dir_use_symlinks=False,
        )

    model, report = create_vggt_from_checkpoint(
        checkpoint_path,
        strict=args.strict,
        include_track=args.include_track,
    )

    output_path = Path(args.output)
    if args.format == "npz":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_nnx_weights_npz(model, str(output_path))
    else:
        save_nnx_weights_orbax(model, str(output_path), force=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dataclasses.asdict(report), indent=2), encoding="utf-8")

    print(f"checkpoint: {checkpoint_path}")
    print(f"weights ({args.format}): {output_path}")
    print(f"report json: {report_path}")
    print(json.dumps(dataclasses.asdict(report), indent=2))


if __name__ == "__main__":
    main()
