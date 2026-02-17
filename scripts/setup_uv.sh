#!/usr/bin/env bash
set -euo pipefail

PROXY_PORT=""
EXTRAS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --proxy-port)
      PROXY_PORT="${2:-}"
      shift 2
      ;;
    *)
      EXTRAS+=("$1")
      shift
      ;;
  esac
done

if [ ${#EXTRAS[@]} -eq 0 ]; then
  EXTRAS=("dev")
else
  EXTRAS=("dev" "${EXTRAS[@]}")
fi

if [ -n "${PROXY_PORT}" ]; then
  export HTTP_PROXY="http://127.0.0.1:${PROXY_PORT}"
  export HTTPS_PROXY="http://127.0.0.1:${PROXY_PORT}"
  export ALL_PROXY="http://127.0.0.1:${PROXY_PORT}"
  echo "[setup] proxy set to 127.0.0.1:${PROXY_PORT}"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, please install uv first"
  exit 1
fi

if [ ! -d ".venv" ]; then
  uv venv .venv
fi

SYNC_CMD=(uv sync)
for extra in "${EXTRAS[@]}"; do
  SYNC_CMD+=(--extra "$extra")
done
echo "[setup] running: ${SYNC_CMD[*]}"
"${SYNC_CMD[@]}"

echo "[setup] done. activate env with: source .venv/bin/activate"
