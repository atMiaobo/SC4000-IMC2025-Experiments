# SIFT/LGEU — clean-room SIFT + COLMAP with a trained edge-utility model

A from-scratch IMC2025 pipeline using classical features (OpenCV SIFT + FLANN + MAGSAC) and
COLMAP incremental SfM, with a **trained gradient-boosting classifier (LGEU)** that predicts
useful image-pair edges *before* local matching, to cut the O(N²) matching cost on large
datasets.

## What's here

```
src/imc2025_lgeu/      core package: scene clustering, SfM wrapper, graph union-find,
                       pair features, submission contract, versioned orchestrators (V4–V26)
training/train_lgeu.py training code for the LGEU classifier
models/lgeu.joblib     the trained LGEU classifier (GBM, 274 KB) — OUR model
notebooks/             Kaggle kernel scripts, final_imc2025_lgeu*.py (V4 … V26)
tests/                 28 offline tests (pytest -q)
scripts/               build/eval/smoke/research-automation scripts
experiments/           lightweight results: research_state.json, lgeu_metrics*.json,
                       ablation_results.csv, v*_hypothesis.txt, research logs
results/               per-run submission_diagnostics.json + the best V7 submission.csv
docs/                  REPORT.md, PROVENANCE.md, CONTRACT.md, README_SETUP.md
```

## Method summary

- **LGEU features** (7-dim, inference-time only): global color/texture descriptor cosine
  similarity, mutual retrieval ranks, aspect/area metadata, filename numeric gap, token overlap.
- **Label** (train only): positive = same non-outlier scene with translation distance in
  [0.05, 5.0] m. No match counts, RANSAC inliers, test labels, or future info are used.
- **Pipeline:** retrieval/LGEU pair selection → SIFT+MAGSAC edges → union-find scene clusters →
  COLMAP incremental SfM per scene → submission CSV. Versions add crash isolation (V6),
  exhaustive COLMAP for short video (V7), and a larger GBM-driven pair budget (V8).

## Results (summary)

- **Best:** V7 registers **27/73** cameras on the public test (ETs 19/22 + stairs 8/51).
- **Kaggle private leaderboard:** **11.81** (V7), from team competition records.
- **LGEU model:** GBM ROC-AUC 0.690 (vs LR 0.614), Recall@20% 0.320.
- **Finding:** better classifier quality bought throughput, not more cameras; the larger V8 pair
  budget *regressed* by over-merging video sub-scenes.

Full tables and provenance: [../RESULTS.md](../RESULTS.md) · ablations:
[../docs/ABLATIONS.md](../docs/ABLATIONS.md) · deep detail: [docs/REPORT.md](docs/REPORT.md).

## Run

```bash
uv sync
uv run pytest -q        # offline, no GPU. Verified here: 108 passed, 1 skipped.
                        # 2 contract tests need contract/sample_submission.csv (the
                        # competition template — download from Kaggle; not shipped here).
# retrain (needs train images + train_labels.csv under data/ — download from Kaggle):
uv run python training/train_lgeu.py
```

See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for data layout and the Kaggle assets.

## Kaggle

- Best kernel: `miaobo/imc2025-lgeu-v7` (v2) — https://www.kaggle.com/code/miaobo/imc2025-lgeu-v7
- Asset dataset: `miaobo/imc2025-lgeu-assets` (src, model, pycolmap wheel)
- Other versions: `imc2025-lgeu-v5`, `imc2025-lgeu-v6`, `imc2025-lgeu-v8`,
  `imc2025-lgeu-clean-room` (V4)
