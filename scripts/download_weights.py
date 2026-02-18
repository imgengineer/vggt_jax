import argparse

from huggingface_hub import hf_hub_download

try:
    from scripts._common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID
except ModuleNotFoundError:  # pragma: no cover
    from _common import DEFAULT_CACHE_DIR, DEFAULT_FILENAME, DEFAULT_REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VGGT public weights from Hugging Face.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Weight filename in repo.")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Download destination directory.")
    args = parser.parse_args()

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.cache_dir,
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()
