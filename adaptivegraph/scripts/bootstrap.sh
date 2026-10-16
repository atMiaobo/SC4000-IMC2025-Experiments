#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH. On NixOS, enter the project with: nix develop"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARNING: nvidia-smi is not on PATH. This project is intended for an NVIDIA GPU."
else
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
fi

if [[ ! -d .venv ]]; then
  uv venv --python python3.11 .venv
fi
source .venv/bin/activate

# Install a CUDA-enabled wheel before timm; otherwise a resolver may pull a CPU torch.
uv pip install --upgrade pip setuptools wheel
uv pip install \
  torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128

uv pip install -e .

mkdir -p third_party
if [[ ! -d third_party/LightGlue/.git ]]; then
  git clone --depth 1 https://github.com/cvg/LightGlue.git third_party/LightGlue
else
  git -C third_party/LightGlue pull --ff-only || true
fi
uv pip install -e third_party/LightGlue

git -C third_party/LightGlue rev-parse HEAD > third_party/LIGHTGLUE_COMMIT.txt

python scripts/fetch_metric.py

python - <<'PY'
import torch
print("\nEnvironment check")
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
else:
    raise SystemExit("CUDA is not available. Fix the NVIDIA driver / Nix shell before running matching.")

import pycolmap, timm, cv2, networkx
from lightglue import ALIKED, LightGlue
print("pycolmap:", pycolmap.__version__)
print("imports: OK")
PY

echo
echo "Bootstrap complete. Next: python scripts/cache_models.py"
