#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo 'Run: nix develop'; exit 1; }
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install kaggle pytest nbformat nbclient numpy pandas scikit-learn
if ! command -v codex >/dev/null; then
  echo 'Codex missing: installing into ~/.local (not your NixOS config)'
  npm install --prefix "$HOME/.local" @openai/codex
fi
command -v codex >/dev/null || { echo 'Codex still not found; ensure ~/.local/bin is on PATH'; exit 1; }
echo 'Installed tools:'
python --version
kaggle --version
codex --version
printf '\nRun Codex login once: codex login\n'
printf 'Run Kaggle login once: kaggle auth login\n'
printf 'Then: bash scripts/preflight.sh\n'
