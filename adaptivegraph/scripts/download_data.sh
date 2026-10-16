#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
mkdir -p downloads data

# Kaggle CLI handles authentication (legacy token or modern auth).

kaggle competitions download -c image-matching-challenge-2025 -p downloads
ZIP=$(find downloads -maxdepth 1 -type f -name '*image-matching-challenge-2025*.zip' | head -n1)
if [[ -z "$ZIP" ]]; then
  echo "Could not find downloaded zip in downloads/"
  exit 1
fi
unzip -o "$ZIP" -d data

echo "Data extracted to $ROOT/data"
find data -maxdepth 2 -type d | head -40
