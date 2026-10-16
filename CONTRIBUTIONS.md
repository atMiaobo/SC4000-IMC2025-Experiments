# Contributions & attribution

This document draws boundaries between (1) code and models developed here, (2) pretrained
vision models, (3) third-party libraries, (4) public baseline ideas, (5) the team's A4R034
implementation, and (6) the single-parameter optimization proposal built on top of A4R034.

## 1. Developed in this repository

**SIFT/LGEU** (`sift-lgeu/`)
- The clean-room pipeline in `src/imc2025_lgeu/` (scene clustering, SfM wrapper, graph
  union-find, pair feature engineering, submission contract, versioned orchestration V4–V26).
- The **LGEU gradient-boosting edge-utility classifier**: training code
  (`training/train_lgeu.py`) and the trained checkpoint `models/lgeu.joblib`, trained on
  official IMC2025 train labels (see provenance below). It is small (274 KB) and is the only
  trained weight redistributed in this repo.
- The versioned Kaggle notebooks, the test suite, and the research-automation scripts.

**AdaptiveGraph** (`adaptivegraph/`)
- The adaptive scene-graph decision layer in `src/adaptive_imc/`: quality-vector edge gating,
  triangle-support edge rehabilitation, topology-aware low-degree rescue, graph label
  propagation, and the **scene-preserving** adaptive tile/rotation rescue.
- The ablation methodology and configs (`configs/ablation_A0..A3*.yaml`) and drivers.

**MASt3R Adaptive** (`mast3r-adaptive/`)
- The reconstruction-aware, scene-preserving **adaptive matching budget** in `src/mast3r_imc/`
  (base-6 trusted pass + low-degree routing + evidence-gated cross-scene merges), plus the
  crash-isolated Kaggle orchestrator.

**A4R045 proposal** (`a4r045-optimization-proposal/`)
- The ledger analysis, the one-line patch, the candidate spec, and the report.

## 2. Pretrained vision models (not trained here)

| Model | Used in | Role |
|---|---|---|
| **DINOv2** (`vit_base_patch14_dinov2.lvd142m`, timm / Meta) | AdaptiveGraph, MASt3R | global image retrieval |
| **ALIKED** (`aliked-n16`) + **LightGlue** (`aliked_lightglue`) | AdaptiveGraph | local feature detection + matching |
| **MASt3R** (`MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`, NAVER) | MASt3R Adaptive | dense correspondence |

These are downloaded pretrained weights and are **not** committed here (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md)).

## 3. Third-party libraries

COLMAP / pycolmap (SfM + geometric verification), OpenCV (SIFT/MAGSAC), scikit-learn (the LGEU
classifier family), PyTorch, timm. Standard dependencies; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## 4. Public baseline ideas (credited, not reproduced verbatim)

- The IMC25 **evaluation metric** implementation originates from a public competition solution
  repository; it is used **only for local scoring** and is fetched via `scripts/fetch_metric.py`
  rather than vendored here.
- The idea that **image tiling** can rescue scale/viewpoint failures was motivated by a public
  3rd-place write-up; these pipelines route tiling *selectively* by geometry/graph topology
  rather than reproducing a universal multi-combination pipeline.

## 5. The team's A4R034 implementation (not included here)

Yiji (Kaggle `yijisuk`) built the team's A4R034 submission (DINOv2 + MASt3R + RDD + ALIKED
fusion over a COLMAP pipeline), its pretrained assets, and the A4R001–A4R044 empirical sweep.
That implementation is **not copied into this repository.** Where its numbers are needed (e.g.
the 42.25 private score, the inlier-count sweep) they are cited; the authoritative copy lives
in its own repository and in the team reference snapshot. The only A4R034-derived file
referenced here is the single overlay file the proposal modifies, included solely as the patch
target (see §6).

## 6. A4R045 optimization proposal

`a4r045-optimization-proposal/` contains a one-line change to `abs_pose_min_num_inliers`
(15 → 12), grounded in the A4R034 monotonic sweep, packaged as a drop-in candidate for that
pipeline. It is **unvalidated** — it was not run (the required asset datasets are private) and
no claim is made that it beats 42.25. It is a candidate with an explicit falsification criterion
documented in its report.

## LGEU model provenance (the one trained artifact)

- **Trained by:** `training/train_lgeu.py`.
- **Data:** 127 official IMC2025 train images across ETs, imc2023_haiper, stairs; labels derived
  only from `train_labels.csv` (positive = same non-outlier scene with translation distance in
  [0.05, 5.0] m).
- **No test labels, match counts, or future information** are used as features.
- **Redistribution:** the checkpoint is a model over a 7-dim hand-crafted feature vector.
  (Note: trained with scikit-learn 1.9.1; Kaggle ran 1.6.1, which raises a benign
  version-mismatch warning on load.)

## Summary

The three standalone pipelines are independent engineering and, in AdaptiveGraph/MASt3R, a
novel algorithmic combination on top of pretrained backbones. None of the three standalone
pipelines outperformed A4R034; the AdaptiveGraph graph heuristics did not raise the aggregate
official metric. These results are documented in [RESULTS.md](RESULTS.md) and
[docs/ABLATIONS.md](docs/ABLATIONS.md).
