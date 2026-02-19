#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOK_DIR="${ROOT_DIR}/notebooks"
OUT_DIR="${ROOT_DIR}/artifacts/checkpoints/$(date +%Y%m%d_%H%M%S)"

mkdir -p "${OUT_DIR}"

cp "${NOTEBOOK_DIR}"/week*.ipynb "${OUT_DIR}"/

echo "Checkpoint created:"
echo "${OUT_DIR}"
