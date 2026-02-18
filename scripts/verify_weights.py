import argparse
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from huggingface_hub import hf_hub_download

from vggt.models.vggt import create_vggt_from_checkpoint

try:
    from scripts._common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID
except ModuleNotFoundError:  # pragma: no cover
    from _common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify checkpoint -> JAX mapping coverage for official VGGT weights")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Local checkpoint path (`.safetensors` or `.npz`). If omitted, download from HF.",
    )
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HF repo id used when downloading.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="HF filename used when downloading.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Download destination directory.")
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--include-track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to include track_head.tracker.* keys",
    )
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.cache_dir,
        )

    _, report = create_vggt_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=args.strict,
        include_track=args.include_track,
    )

    print("total_source_keys", report.total_source_keys)
    print("mapped_keys", report.mapped_keys)
    print("loaded_keys", report.loaded_keys)
    print("skipped_unmapped", report.skipped_unmapped)
    print("skipped_missing_target", report.skipped_missing_target)
    print("skipped_shape_mismatch", report.skipped_shape_mismatch)
    print("missing_target_leaves", report.missing_target_leaves)

    if report.loaded_keys != report.total_source_keys:
        raise RuntimeError(
            f"Expected all source keys loaded, got loaded={report.loaded_keys}, total={report.total_source_keys}"
        )
    if report.skipped_unmapped or report.skipped_missing_target or report.skipped_shape_mismatch:
        raise RuntimeError(
            "Strict loading incomplete: "
            f"unmapped={report.skipped_unmapped}, "
            f"missing_target={report.skipped_missing_target}, "
            f"shape_mismatch={report.skipped_shape_mismatch}"
        )

    print("status PASS")


if __name__ == "__main__":
    main()
