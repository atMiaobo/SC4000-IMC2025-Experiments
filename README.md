# SC4000 — Image Matching Challenge 2025 Experiments

A consolidated, reproducible record of experiments on the Kaggle
[Image Matching Challenge 2025](https://www.kaggle.com/competitions/image-matching-challenge-2025)
(IMC2025). The task is multi-scene structure-from-motion: given an unlabelled bag of images
spanning several scenes, cluster them by scene and recover each camera's 6-DoF pose.
Submissions are scored by a harmonic combination of pose **mAA** and scene **clusterness**.

The repository packages three standalone reconstruction pipelines, each built and ablated
independently, plus a small single-parameter optimization proposal.

## Repository layout

| Path | What it is |
|---|---|
| [`sift-lgeu/`](sift-lgeu/) | Clean-room SIFT + COLMAP pipeline with a **trained** gradient-boosting edge-utility model (LGEU). V4–V26 experiments. |
| [`adaptivegraph/`](adaptivegraph/) | DINOv2 retrieval + ALIKED/LightGlue matching + an **adaptive scene-graph** decision layer. A0–A3 ablations. |
| [`mast3r-adaptive/`](mast3r-adaptive/) | MASt3R dense matching + a **reconstruction-aware adaptive matching budget**. Baseline-vs-adaptive ablation. |
| [`a4r045-optimization-proposal/`](a4r045-optimization-proposal/) | A one-line, evidence-grounded parameter proposal (A4R045) on top of the team's A4R034 baseline. Patch + analysis only. |
| [`RESULTS.md`](RESULTS.md) | All scores, separated by kind (Kaggle private / local official metric / camera counts) with provenance. |
| [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Authorship boundaries and attribution. |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Environment setup, how to obtain the external weights/data, and how to run each pipeline. |
| [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) | Pretrained models, libraries, licenses and their provenance. |
| [`docs/ABLATIONS.md`](docs/ABLATIONS.md) | The most important ablation experiments, consolidated. |

## The three pipelines at a glance

| Pipeline | Backbone (pretrained) | Contribution | Kaggle private | Local dev metric |
|---|---|---|---:|---|
| **SIFT/LGEU** | OpenCV SIFT, COLMAP | Trained LGEU pair-edge classifier; crash-isolated versioned pipeline (V4–V26) | **11.81** (V7) | 27/73 cameras registered on public test |
| **AdaptiveGraph** | DINOv2, ALIKED, LightGlue, COLMAP | Adaptive scene-graph: quality gating, triangle rehab, label propagation, scene-preserving rescue | **22.63** | 60.89 (A2, full 13-scene official metric) |
| **MASt3R Adaptive** | MASt3R, DINOv2, COLMAP | Reconstruction-aware, scene-preserving adaptive matching budget | **22.97** | 51.44 (adaptive, 3-scene official metric) |

Kaggle private scores are from the original `image-matching-challenge-2025` competition,
taken from the team's competition records. Local metrics are **training-set** evaluations
with the official IMC25 metric (or raw camera counts for SIFT/LGEU) and are **not**
leaderboard scores — see [RESULTS.md](RESULTS.md) for the full separation by result kind.

## Findings

- **SIFT/LGEU:** The trained LGEU classifier (GBM, ROC-AUC 0.690) primarily improved
  *throughput* (fewer pairs to match) rather than registering more cameras. A larger pair
  budget (V8) regressed by over-merging video sub-scenes. The best version is V7 (exhaustive
  COLMAP for short video sequences).
- **AdaptiveGraph:** On a strong ALIKED+LightGlue+COLMAP backbone the graph heuristics are
  net-neutral in aggregate (A0 backbone 61.26 ≈ A2 60.89). Graph **label propagation** gives a
  narrow but real clusterness gain on fragmented scenes; the **adaptive rescue** component is a
  negative result and ships **off**.
- **MASt3R Adaptive:** The adaptive budget beats the MASt3R baseline on the small subset
  (+6.4 overall, driven by clusterness) but regresses on large scenes (fbk_vineyard) and does
  not beat AdaptiveGraph in aggregate on the shared subset.

## Repository guide

1. [RESULTS.md](RESULTS.md) — the numbers, with provenance.
2. [docs/ABLATIONS.md](docs/ABLATIONS.md) — the experiments behind those numbers.
3. [a4r045-optimization-proposal/](a4r045-optimization-proposal/) — the single-parameter
   proposal on top of the A4R034 baseline.

## Provenance of this package

Consolidated on 2026-10-16 from four working repositories. Source code, lightweight result
files (JSON/CSV), notebooks and documentation were copied; pretrained weights, competition
images, caches, virtual environments and large reconstruction databases were deliberately
excluded (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for where to obtain them). The original
working repositories were left unchanged.
