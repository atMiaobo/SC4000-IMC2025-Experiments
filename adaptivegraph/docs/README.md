# IMC2025 Adaptive Scene-Graph Matching

An Image Matching Challenge 2025 pipeline designed around novel routing / graph logic on top of
pretrained components.

> **On scores:** no code package can guarantee a hidden Kaggle score before it is executed. This
> repository is set up to make `40` a *validation gate*: run the local train metric first; if
> `balanced` is below 40 on local train evaluation, switch to `high_score` and tune from the
> diagnostics before making a Kaggle submission. The aggressive preset deliberately spends much
> more compute on tiling.

## Architecture

```text
DINOv2 retrieval
      ↓
ALIKED + LightGlue base pairs
      ↓
MAGSAC pair-quality vector
      ↓
┌──────────────── adaptive router ────────────────┐
│ easy → accept                                  │
│ bad  → reject                                  │
│ hard → tiles / rotation only when justified    │
└────────────────────────────────────────────────┘
      ↓
triangle consistency + low-degree graph rescue
      ↓
COLMAP multi-model reconstruction
      ↓
graph scene-label propagation
      ↓
submission.csv
```

The full method description is in [`METHOD.md`](METHOD.md).

## Folder layout

```text
IMC2025_AdaptiveGraph/
├── configs/
│   ├── fast.yaml          # installation/smoke test
│   ├── balanced.yaml      # recommended method
│   └── high_score.yaml    # aggressive fallback if CV < 40
├── notebooks/
│   └── AdaptiveSceneGraph_IMC2025.ipynb
├── scripts/
│   ├── bootstrap.sh
│   ├── cache_models.py
│   ├── download_data.sh
│   ├── run_pipeline.py
│   ├── score_local.py
│   └── check_submission.py
├── src/adaptive_imc/
├── flake.nix
├── pyproject.toml
├── METHOD.md
└── THIRD_PARTY_NOTICES.md
```

## Full NixOS workflow

### 1. Unzip and enter

```bash
unzip IMC2025_AdaptiveGraph.zip
cd IMC2025_AdaptiveGraph
```

Check that your NVIDIA driver works *outside* the shell:

```bash
nvidia-smi
```

If that fails, fix your NixOS NVIDIA configuration first.

### 2. Enter the reproducible shell

```bash
nix develop
```

The flake supplies Python 3.11, `uv`, Git and native runtime libraries. It does **not** replace
your host NVIDIA driver.

### 3. Install the Python/CUDA environment

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

The bootstrap installs PyTorch 2.5.1 + CUDA 12.4 wheels, the project dependencies, and the
official LightGlue repository. It finishes by asserting that `torch.cuda.is_available()` is true.

### 4. Pre-download model weights

```bash
python scripts/cache_models.py
```

This downloads/caches DINOv2, ALIKED and LightGlue so a long run does not die halfway because of
a network/model-cache surprise.

### 5. Download the competition data

First accept the competition / late-submission rules on Kaggle and configure your Kaggle API
credentials. With the classic API token, the expected location is:

```bash
mkdir -p ~/.kaggle
# copy kaggle.json there
chmod 600 ~/.kaggle/kaggle.json
```

Then:

```bash
bash scripts/download_data.sh
```

You should end up with approximately:

```text
data/
├── train/
│   ├── ETs/
│   ├── stairs/
│   └── ...
├── test/
├── train_labels.csv
├── train_thresholds.csv
└── sample_submission.csv
```

If Kaggle changes the archive layout, move those files/folders into the layout above.

### 6. Smoke test one dataset

Do not start with a multi-hour full run. First prove that retrieval → matching → COLMAP works:

```bash
python scripts/run_pipeline.py \
  --mode train \
  --dataset ETs \
  --config configs/fast.yaml \
  --run-name smoke_ets
```

Then score it:

```bash
python scripts/score_local.py outputs/smoke_ets/submission.csv
```

The fast preset is an installation test, **not** the final model.

### 7. Run the proposed method on ETs

```bash
python scripts/run_pipeline.py \
  --mode train \
  --dataset ETs \
  --config configs/balanced.yaml \
  --run-name balanced_ets

python scripts/score_local.py outputs/balanced_ets/submission.csv
```

Inspect the routing decisions:

```bash
column -s, -t < outputs/balanced_ets/ETs/pair_diagnostics.csv | less -S
cat outputs/balanced_ets/ETs/summary.json
```

### 8. Full local validation

This is the score that should drive configuration decisions:

```bash
python scripts/run_pipeline.py \
  --mode train \
  --config configs/balanced.yaml \
  --run-name balanced_full

python scripts/score_local.py outputs/balanced_full/submission.csv
```

**Decision rule:**

- `>= 40`: keep `balanced`; this is the cleanest novelty/performance trade-off.
- `36–40`: acceptable, but try `high_score` for margin.
- `< 36`: do not submit yet; use the diagnostic CSVs to see whether retrieval, pair filtering, or
  reconstruction is the failure.

### 9. Aggressive fallback if balanced is under 40

```bash
python scripts/run_pipeline.py \
  --mode train \
  --config configs/high_score.yaml \
  --run-name high_score_full

python scripts/score_local.py outputs/high_score_full/submission.csv
```

`high_score.yaml` raises retrieval coverage and applies the 5-view base+quadrants matcher to many
more valid pairs. It is much slower, but gives a score-oriented fallback without switching to a
MASt3R backbone.

### 10. Generate the test submission

After choosing the config based on train CV:

```bash
python scripts/run_pipeline.py \
  --mode test \
  --config configs/balanced.yaml \
  --run-name final_test
```

or replace `balanced.yaml` with `high_score.yaml`.

Validate the file:

```bash
python scripts/check_submission.py \
  outputs/final_test/submission.csv \
  --sample data/sample_submission.csv
```

The file to upload is:

```text
outputs/final_test/submission.csv
```

## Notebook workflow

Start Jupyter from the same shell and venv:

```bash
nix develop
source .venv/bin/activate
jupyter lab notebooks/AdaptiveSceneGraph_IMC2025.ipynb
```

The notebook is a readable experiment driver around the package. The reusable algorithm lives in
`src/adaptive_imc/`, which keeps the project easier to test, version and explain than burying
several thousand lines in notebook cells.

## Reporting results

The pretrained components (DINOv2, ALIKED, LightGlue, MAGSAC, COLMAP) are used as-is and are not
claimed as original. The designed parts are the pair-quality vector, spatial-coverage criterion,
adaptive compute router, triangle-supported edges, topology-aware rescue and post-reconstruction
graph propagation. Report the ablations from `METHOD.md` with the **measured** score/runtime for
each variant.

## GPU / runtime notes

- `balanced` targets a normal single NVIDIA GPU and caches features across pairs.
- `high_score` can be dramatically slower because a rescued pair may execute many tile
  combinations.
- Watch VRAM with `watch -n 1 nvidia-smi`.
- You can stop after one dataset, adjust YAML, and rerun; outputs are isolated by `--run-name`.

## Reproducibility

After bootstrap, record:

```bash
python --version
python -c 'import torch, pycolmap; print(torch.__version__, torch.version.cuda, pycolmap.__version__)'
cat third_party/LIGHTGLUE_COMMIT.txt
nvidia-smi --query-gpu=name,driver_version --format=csv
```

Keep these with the final report and output logs.
