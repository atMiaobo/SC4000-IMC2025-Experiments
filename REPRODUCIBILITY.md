# Reproducibility

This repository is **source-only**. Pretrained weights, competition data, caches and large
reconstruction outputs are intentionally not committed. This document explains how to set up
each pipeline and where to obtain the external assets.

> **Note:** The three pipelines were developed in separate working trees on a NixOS machine with
> an RTX 5080 (sm_120), Python 3.11 locally and Python 3.13 on Kaggle. Some scripts contain
> machine-specific paths and cross-project dependencies, documented per pipeline below. They are
> preserved as-is.

## External assets (obtain separately — never committed)

| Asset | Where to get it | Used by |
|---|---|---|
| IMC2025 competition images + `train_labels.csv` | Kaggle: `image-matching-challenge-2025` (competition data; requires accepting rules) | all pipelines (local eval) |
| DINOv2 `vit_base_patch14_dinov2.lvd142m` | Hugging Face / `timm` (auto-cached) | AdaptiveGraph, MASt3R |
| ALIKED `aliked-n16` + LightGlue `aliked_lightglue` | `github.com/cvg/LightGlue` (weights auto-downloaded at runtime) | AdaptiveGraph |
| MASt3R `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` (2.6 GB, CC-BY-NC-SA-4.0) | HF `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | MASt3R |
| `pycolmap` wheel (Kaggle Py3.13) | PyPI / build; the Kaggle bundles ship a `pycolmap-3.13.0-cp313` wheel | all (Kaggle) |
| IMC25 scoring metric | fetched by `scripts/fetch_metric.py` from the public competition-solution repo | AdaptiveGraph, MASt3R (local scoring only) |

Kaggle credentials (`~/.kaggle/kaggle.json`) are **user-supplied** and must never be committed.

---

## `sift-lgeu/` — SIFT + COLMAP + trained LGEU

Self-contained; the only trained artifact (`models/lgeu.joblib`) is included.

```bash
cd sift-lgeu
uv sync                       # or: pip install -e . (see pyproject.toml / uv.lock)
uv run pytest -q              # offline, no GPU required

# Retrain the LGEU classifier (needs train images + train_labels.csv placed under data/):
uv run python training/train_lgeu.py       # writes models/lgeu.joblib + experiments/lgeu_metrics*.json

# Local smoke/eval (needs the official train datasets locally):
uv run python scripts/gate_b_smoke.py
uv run python scripts/eval_v4_vs_v5.py
```

- Data layout expected by the local scripts: `data/official/train/<dataset>/...` and
  `data/raw/train_labels.csv`. These are **not** shipped — download from the competition.
- The `pycolmap` wheel for Kaggle is **not** committed; install `pycolmap` from PyPI locally,
  or attach the Kaggle asset dataset `miaobo/imc2025-lgeu-assets` when running the notebook.
- Best notebook: `notebooks/final_imc2025_lgeu_v7.py` → Kaggle `miaobo/imc2025-lgeu-v7` (v2).

## `adaptivegraph/` — DINOv2 + ALIKED/LightGlue + adaptive scene graph

```bash
cd adaptivegraph
# First-time setup (clones LightGlue at the pinned commit, builds env):
bash scripts/bootstrap.sh             # re-clones third_party/LightGlue @ eb42fee...
python scripts/cache_models.py        # pre-downloads DINOv2 / ALIKED / LightGlue weights
python scripts/download_data.sh       # Kaggle competition data (requires kaggle.json)
python scripts/fetch_metric.py        # fetches the IMC25 metric into third_party/ for scoring

# Reproduce the ablation ladder:
python scripts/run_ablation.py \
  --config configs/ablation_A0.yaml configs/ablation_A1.yaml configs/ablation_A2.yaml \
           configs/ablation_A3_naive.yaml configs/ablation_A3.yaml \
  --dataset ETs amy_gardens fbk_vineyard imc2023_haiper stairs --tag ladder

# Score a submission locally:
python scripts/score_local.py outputs/<run>/submission.csv
```

- The vendored LightGlue tree is **not** committed; `third_party/LIGHTGLUE_COMMIT.txt` pins the
  exact commit (`eb42fee2d71449efb0aa5c10549752b5d75384d8`) and `bootstrap.sh` re-clones it.
- Deployed config: `configs/final.yaml` (= A2, rescue off). Kaggle:
  `miaobo/adaptivegraph-fast-orig` (v3, original competition) and
  `miaobo/adaptivegraph-fast-ref` (v5, `-ongoing` mirror); bundle
  `miaobo/adaptive-graph-imc25-bundle`.
- GPU: torch 2.7.1+cu128, pycolmap 3.13.0. On NixOS:
  `export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH`.

## `mast3r-adaptive/` — MASt3R dense matching + adaptive budget

> **Cross-project dependency:** this pipeline **reuses the AdaptiveGraph package**
> (`adaptive_imc`: retrieval, MAGSAC, COLMAP wrapper, graph ops, submission writer) and the
> IMC25 metric. To run it locally you need the `adaptivegraph/` project importable on
> `PYTHONPATH` and the MASt3R weights downloaded. The original tree invoked it via
> `nix develop ~/Projects/SC4000-IMC2025-2/IMC2025_AdaptiveGraph -c bash` — adjust that path to
> wherever you place the `adaptivegraph/` project.

```bash
cd mast3r-adaptive
# Make the AdaptiveGraph package importable (adjust the path to your checkout):
export PYTHONPATH="$PWD/../adaptivegraph/src:$PYTHONPATH"
# Download MASt3R weights to weights/mast3r_hf/ from
#   https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/main/
# (config.json + model.safetensors). Run network ops outside nix develop on NixOS.

# Baseline vs adaptive ablation on local train datasets:
python scripts/run_mast3r.py --dataset ETs imc2023_haiper --mode baseline adaptive
```

- Kaggle: notebook `kaggle/MASt3R_Adaptive_Kaggle.ipynb` → kernel
  `miaobo/mast3r-adaptive-orig` (v4); bundle `miaobo/mast3r-imc25-bundle` (weights + repo +
  DINOv2 cache + wheels). MASt3R uses the pure-PyTorch RoPE fallback (no `curope`/`nvcc`).
- **License:** MASt3R weights are CC-BY-NC-SA-4.0 (non-commercial, research use only).
- The vendored NAVER MASt3R source tree is **not** committed; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the exact commit to clone.

## `a4r045-optimization-proposal/`

No execution environment is provided or needed here — it is a patch + analysis. The A4R034
asset datasets are required to run the parent pipeline and are private. See the directory's
`README.md` and `OPTIMIZATION_REPORT.md`.

---

## What is intentionally missing (and why)

| Omitted | Why | How to restore |
|---|---|---|
| All pretrained weights (DINOv2/ALIKED/LightGlue/MASt3R) | redistribution not necessary; large; licensing | download per table above / `cache_models.py` |
| Competition images + `train_labels.csv` | private competition data — must not be committed | download from Kaggle competition |
| `pycolmap` wheels, `.venv/`, caches | build artifacts | `uv sync` / `pip install` |
| COLMAP `.db` reconstruction outputs | large generated artifacts | regenerated by running the pipelines |
| Vendored LightGlue / MASt3R source trees | avoid vendoring third-party repos | clone at the pinned commits |
| IMC25 metric source | third-party; fetched not vendored | `scripts/fetch_metric.py` |
| A4R034 implementation + assets | not part of this repository; private | the team's own repository |
