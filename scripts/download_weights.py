import argparse
import os

from huggingface_hub import hf_hub_download


def set_proxy(proxy_port: int | None) -> None:
    if proxy_port is None:
        return
    http_proxy = f"http://127.0.0.1:{proxy_port}"
    socks_proxy = f"http://127.0.0.1:{proxy_port}"
    os.environ["HTTP_PROXY"] = http_proxy
    os.environ["HTTPS_PROXY"] = http_proxy
    os.environ["ALL_PROXY"] = socks_proxy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, default="facebook/VGGT-1B")
    parser.add_argument("--filename", type=str, default="model.pt")
    parser.add_argument("--cache-dir", type=str, default="./weights")
    parser.add_argument("--proxy-port", type=int, default=None)
    args = parser.parse_args()

    set_proxy(args.proxy_port)
    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.cache_dir,
        local_dir_use_symlinks=False,
    )
    print(path)


if __name__ == "__main__":
    main()
