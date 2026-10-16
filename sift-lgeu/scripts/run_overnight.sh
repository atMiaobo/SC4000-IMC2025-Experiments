#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
python scripts/check_contract.py
mkdir -p logs
set -o pipefail
# Non-interactive, workspace write only. Can fail if Kaggle credentials/caches are blocked.
# network access is enabled for installation/official Kaggle interaction, not a full-access sandbox.
# systemd-inhibit may fail if logind is unavailable; fail visibly instead of silently suspending.
systemd-inhibit --what=sleep:idle --mode=block \
  --why='SC4000 clean-room overnight run' \
  codex --ask-for-approval never --sandbox workspace-write \
  -c 'model_reasoning_effort="high"' \
  -c 'sandbox_workspace_write.network_access=true' \
  exec - < PROMPT.md 2>&1 | tee logs/codex-overnight.log
