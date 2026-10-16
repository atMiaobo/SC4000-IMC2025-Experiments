# Third-party notices & attribution

This repository contains original source code plus one trained model
(`sift-lgeu/models/lgeu.joblib`). All other models and the evaluation metric are third-party and
are **not** redistributed here — they are downloaded/fetched at setup time. This file records
what the pipelines depend on, where it comes from, and its license.

## Pretrained models (downloaded separately; weights NOT in this repo)

| Model | Source | License | Where used |
|---|---|---|---|
| **DINOv2** `vit_base_patch14_dinov2.lvd142m` | Meta AI; via `timm` / Hugging Face | Apache-2.0 (code); see model card | AdaptiveGraph, MASt3R — global retrieval |
| **ALIKED** `aliked-n16` | ALIKED authors (Alibaba) | model repo terms | AdaptiveGraph — local features |
| **LightGlue** `aliked_lightglue` | `github.com/cvg/LightGlue` @ `eb42fee2d71449efb0aa5c10549752b5d75384d8` | Apache-2.0 | AdaptiveGraph — matching |
| **MASt3R** `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` | NAVER; `github.com/naver/mast3r` @ `f5209af` (submodules dust3r `3cc8c88`, croco `d7de070`) | **CC-BY-NC-SA-4.0 (non-commercial)** | MASt3R Adaptive — dense matching |

Pinned commits are preserved in `adaptivegraph/third_party/LIGHTGLUE_COMMIT.txt` and in the
MASt3R notes above so the exact versions can be re-obtained. The MASt3R license restricts use to
**non-commercial** (research/competition) contexts.

## Libraries

| Library | Role | License |
|---|---|---|
| COLMAP / **pycolmap** | geometric verification + incremental SfM | COLMAP: BSD; pycolmap wheels per release |
| **OpenCV** (`opencv-python-headless`) | SIFT, FLANN, MAGSAC (SIFT/LGEU pipeline) | Apache-2.0 |
| **scikit-learn** | `HistGradientBoostingClassifier` / `LogisticRegression` for LGEU | BSD-3-Clause |
| **PyTorch**, **timm** | model inference backends | BSD-3 / Apache-2.0 |

## Evaluation metric

The IMC25 scoring metric (harmonic combination of pose mAA and clusterness) is a **third-party
public competition-solution implementation**. It is used **only for local scoring** and is
fetched at setup time via `adaptivegraph/scripts/fetch_metric.py` — it is **not** vendored into
this repository. The originating public solution repository should be cited in any write-up.

## Public ideas credited

- **Selective image tiling** for scale/viewpoint rescue was motivated by a public 3rd-place
  write-up; these pipelines route tiling selectively (by geometry/graph topology) rather than
  reproducing a universal multi-combination pipeline.

## Trained model (redistributed)

- `sift-lgeu/models/lgeu.joblib` — a scikit-learn gradient-boosting classifier trained on
  official IMC2025 train labels over a 7-dim hand-crafted feature vector (274 KB). Training code:
  `sift-lgeu/training/train_lgeu.py`. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for full provenance.

## Team A4R034 work

The team's A4R034 implementation (by Yiji / `yijisuk`) and its assets are **not** included or
redistributed here; they are private. The `a4r045-optimization-proposal/` directory contains
only the patch/analysis plus the single overlay file it modifies, included solely as the patch
target.

---
*Per-project third-party notices are also preserved under each project's `docs/` directory
(`adaptivegraph/docs/THIRD_PARTY_NOTICES.md`, `mast3r-adaptive/docs/THIRD_PARTY_NOTICES.md`).*
