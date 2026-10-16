# Third-party components and attribution

This project uses public pretrained building blocks while keeping the decision/routing pipeline
in `src/adaptive_imc/` original to this project.

- **DINOv2 / timm** — used only for global image retrieval. See Meta DINOv2 and the timm model
  implementation/model card.
- **ALIKED + LightGlue** — official LightGlue repository (`https://github.com/cvg/LightGlue`,
  Apache-2.0). A copy is **vendored** in `third_party/LightGlue/` pinned to commit
  `eb42fee2d71449efb0aa5c10549752b5d75384d8` (recorded in `third_party/LIGHTGLUE_COMMIT.txt`);
  `scripts/bootstrap.sh` can re-clone it at that commit. The ALIKED detector and LightGlue matcher
  weights are downloaded from their official sources at runtime. These are used as-is; no
  LightGlue/ALIKED source is claimed as original.
- **COLMAP / pycolmap** — standard geometric verification and incremental structure-from-motion
  backend.
- **IMC2025 public metric** — `scripts/fetch_metric.py` fetches the public metric implementation
  from the public 7th-place repository solely for local scoring.
- **IMC2025 solution literature** — the idea that image tiling can rescue scale/viewpoint failures
  was motivated by a public 3rd-place write-up. This project does *not* reproduce its universal
  25-combination pipeline: tiling is routed selectively using geometry and graph topology.

Before a write-up, include the exact package versions / LightGlue commit recorded by
`scripts/bootstrap.sh`, and cite the papers/repositories above.
