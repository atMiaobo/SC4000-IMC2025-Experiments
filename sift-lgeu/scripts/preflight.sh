#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
command -v codex >/dev/null
command -v kaggle >/dev/null
printf '[1/5] Codex version: '; codex --version
printf '[2/5] Kaggle version: '; kaggle --version
printf '[3/5] Official competition files (must succeed):\n'
kaggle competitions files image-matching-challenge-2025
printf '[4/5] Download official sample only (no other competition):\n'
mkdir -p contract
kaggle competitions download image-matching-challenge-2025 \
  -f sample_submission.csv -p contract
python scripts/check_contract.py
printf '[5/5] Verify clean-room boundaries:\n'
if find . -maxdepth 2 \( -iname '*yiji*' -o -iname '*moon*' -o -iname '*approach*' \) -print | grep -q .; then
  echo 'FAIL: teammate artifacts found in workspace; remove before running Codex'
  exit 1
fi
if [[ ! -s PROMPT.md || ! -s AGENTS.md || ! -s CONTRACT.md ]]; then
  echo 'FAIL: missing required project instructions'
  exit 1
fi
printf '\nPRE-FLIGHT PASSED (CLI access and official sample schema).\n'
printf 'This does NOT yet prove training data availability or Kaggle GPU execution.\n'
