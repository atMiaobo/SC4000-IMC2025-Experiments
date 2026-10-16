# AdaptiveGraph — DINOv2 + ALIKED/LightGlue with an adaptive scene-graph layer

A clean-room IMC2025 pipeline built from pretrained components (DINOv2 retrieval, ALIKED +
LightGlue matching, COLMAP SfM) with an independently designed decision/graph layer on top:
quality-vector edge gating, triangle-support edge rehabilitation, graph label propagation, and a
scene-preserving adaptive rescue. The research question is whether these graph heuristics add
value over the pretrained backbone — answered by a controlled A0→A3 ablation.

## What's here

```
src/adaptive_imc/      config, retrieval (DINOv2), matching (ALIKED/LightGlue),
                       geometry (MAGSAC), graph_ops, reconstruction (COLMAP), pipeline, submission
configs/               final.yaml (deployed=A2), ablation_A0..A3*.yaml, balanced/fast/high_score/proposed_B
scripts/               run_pipeline, run_ablation, score_local, cache_models, download_data,
                       fetch_metric, check_submission, make_kaggle_bundle, bootstrap.sh
notebooks/             AdaptiveSceneGraph_IMC2025.ipynb (interactive driver)
kaggle/, kaggle_orig/  Kaggle kernel notebooks + metadata (ongoing / original competition)
results/               ablation results.json (fullset, ladder, a3cmp, smoke), run summaries,
                       per-scene summary.json, submissions, and a3cmp pair_diagnostics.csv
third_party/           LIGHTGLUE_COMMIT.txt (pin only — LightGlue is NOT vendored here)
docs/                  README, METHOD.md, REPORT.md, FINAL_WORKFLOW.md, THIRD_PARTY_NOTICES.md
```

## Method summary

```
DINOv2 retrieval → ALIKED+LightGlue matching → MAGSAC verification (inliers/ratio/coverage)
  → quality-vector edge gating → triangle-supported edge rehab
  → [optional] scene-preserving adaptive tile/rotation rescue
  → COLMAP multi-model SfM → graph label propagation → submission.csv
```

## Results (summary)

- **Local, full 13-scene official metric:** A0 backbone **61.26**, A2 deployed **60.89**,
  proposed_B 60.81 — the graph layer is net-neutral in aggregate.
- **Label propagation** gives a narrow clusterness benefit (stairs 50 → 88.9) at zero pose cost;
  **adaptive rescue is a negative result** (A2 54.17 → A3 44.01) and ships **off** in
  `final.yaml`.
- **Kaggle private leaderboard:** **22.63**, from team competition records.

Full tables: [../RESULTS.md](../RESULTS.md) · ablation detail: [../docs/ABLATIONS.md](../docs/ABLATIONS.md)
· method/ablation writeup: [docs/REPORT.md](docs/REPORT.md), [docs/METHOD.md](docs/METHOD.md).

## Run

```bash
bash scripts/bootstrap.sh            # clones LightGlue @ eb42fee..., builds env
python scripts/cache_models.py       # DINOv2 / ALIKED / LightGlue weights
python scripts/fetch_metric.py       # IMC25 scoring metric (not vendored)
python scripts/run_ablation.py --config configs/ablation_A0.yaml ... --tag ladder
```

See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md). GPU required (torch 2.7.1+cu128, pycolmap 3.13).

## Kaggle

- Original competition: `miaobo/adaptivegraph-fast-orig` (v3)
- `-ongoing` mirror: `miaobo/adaptivegraph-fast-ref` (v5)
- Bundle: `miaobo/adaptive-graph-imc25-bundle`
