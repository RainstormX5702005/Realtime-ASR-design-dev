#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"

export PYTHONPATH="${BACKEND_DIR}/src:${BACKEND_DIR}/wenet:${PYTHONPATH:-}"
exec python "${SCRIPT_DIR}/wenet_serve.py" "$@"
