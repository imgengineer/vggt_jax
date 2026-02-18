#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KNOWN_EXTRAS=(data demo dev convert track train)
EXTRAS=(dev)

print_usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_uv.sh [extra ...]
  bash scripts/setup_uv.sh --extra <name> [--extra <name> ...]
  bash scripts/setup_uv.sh --no-dev [extra ...]

Examples:
  bash scripts/setup_uv.sh
  bash scripts/setup_uv.sh data
  bash scripts/setup_uv.sh --extra data --extra demo
  bash scripts/setup_uv.sh --no-dev train

Known extras:
  data demo dev convert track train
EOF
}

contains_extra() {
  local target="$1"
  shift
  local item
  for item in "$@"; do
    if [ "$item" = "$target" ]; then
      return 0
    fi
  done
  return 1
}

append_extra() {
  local extra="$1"
  if ! contains_extra "$extra" "${KNOWN_EXTRAS[@]}"; then
    echo "[setup] unknown extra: $extra"
    echo "[setup] use --help to see supported extras"
    exit 1
  fi
  if ! contains_extra "$extra" "${EXTRAS[@]}"; then
    EXTRAS+=("$extra")
  fi
}

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      print_usage
      exit 0
      ;;
    --no-dev)
      NEXT_EXTRAS=()
      extra_item=
      for extra_item in "${EXTRAS[@]}"; do
        if [ "$extra_item" != "dev" ]; then
          NEXT_EXTRAS+=("$extra_item")
        fi
      done
      EXTRAS=("${NEXT_EXTRAS[@]}")
      shift
      ;;
    --extra)
      if [ $# -lt 2 ]; then
        echo "[setup] --extra requires a value"
        exit 1
      fi
      append_extra "$2"
      shift 2
      ;;
    *)
      append_extra "$1"
      shift
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] uv not found, install uv first"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "[setup] creating virtual env: .venv"
  uv venv .venv
fi

if [ ${#EXTRAS[@]} -eq 0 ]; then
  echo "[setup] syncing base dependencies only"
else
  echo "[setup] syncing with extras: ${EXTRAS[*]}"
fi

SYNC_CMD=(uv sync)
for extra in "${EXTRAS[@]}"; do
  SYNC_CMD+=(--extra "$extra")
done
echo "[setup] running: ${SYNC_CMD[*]}"
"${SYNC_CMD[@]}"

echo "[setup] done"
echo "[setup] activate with: source .venv/bin/activate"
