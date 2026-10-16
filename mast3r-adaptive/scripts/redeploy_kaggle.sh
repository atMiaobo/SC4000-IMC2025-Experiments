#!/usr/bin/env bash
# One-shot redeploy of BOTH candidates with the finite-pose scorer fix.
# PREREQUISITE: valid Kaggle auth (the OAuth token expired 2026-10-15 10:14 UTC).
#   Re-auth first, e.g.:  kaggle auth login     (or place a classic ~/.kaggle/kaggle.json)
# Run OUTSIDE nix develop (network/ssl). Bundles already contain the fixed submission.py.
set -e
MAST=/home/mihao/Projects/SC4000-MASt3R-candidate
AG=/home/mihao/Projects/SC4000-IMC2025-2/IMC2025_AdaptiveGraph
source $AG/.venv/bin/activate

echo "### 0. sanity: auth works"; kaggle kernels list --mine >/dev/null && echo "auth OK"

echo "### 1. MASt3R bundle (new version, finite poses)"
kaggle datasets version -p $MAST/kaggle_bundle -m "finite identity/zero poses (scorer fix)" --dir-mode zip
echo "   waiting for bundle to process..."
for i in $(seq 1 30); do kaggle datasets status miaobo/mast3r-imc25-bundle 2>/dev/null | grep -qi ready && { echo ready; break; }; sleep 30; done

echo "### 2. push MASt3R kernel (new version)"
kaggle kernels push -p $MAST/kaggle
echo "   kernel: https://www.kaggle.com/code/miaobo/mast3r-adaptive-orig  (poll status, then verify submission has NO 'nan')"

echo "### 3. AdaptiveGraph backup bundle + kernel"
kaggle datasets version -p $AG/kaggle_bundle -m "finite identity/zero poses (scorer fix)" --dir-mode zip
for i in $(seq 1 30); do kaggle datasets status miaobo/adaptive-graph-imc25-bundle 2>/dev/null | grep -qi ready && { echo ready; break; }; sleep 30; done
kaggle kernels push -p $AG/kaggle
echo "   kernel: https://www.kaggle.com/code/miaobo/adaptivegraph-fast-ref (ongoing) — for the ORIGINAL use $AG/kaggle_orig"
kaggle kernels push -p $AG/kaggle_orig
echo "   kernel: https://www.kaggle.com/code/miaobo/adaptivegraph-fast-orig"

echo "### DONE. Poll each kernel to COMPLETE, confirm submission.csv has 0 rows containing 'nan', then submit manually."
