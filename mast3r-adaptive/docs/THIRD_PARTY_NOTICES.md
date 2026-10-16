# Third-party components and attribution

This project combines public pretrained models. The original contribution is the
reconstruction-aware adaptive matching / scene-preserving graph logic in `src/mast3r_imc/`; the
pretrained models and their APIs are not claimed as novel.

- **MASt3R** — NAVER, https://github.com/naver/mast3r (commit `f5209af`), with submodules dust3r
  (`3cc8c88`) and croco (`d7de070`). **License: CC-BY-NC-SA-4.0 (NON-COMMERCIAL).** Checkpoint
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric` (HF repo
  `naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`, `config.json`+`model.safetensors`,
  sha256 of the .pth mirror `e28f91b4…`). Non-commercial (research/competition) use only. See the
  repo's `CHECKPOINTS_NOTICE`. The curope CUDA extension is not built; the pure-PyTorch RoPE
  fallback is used.
- **The MASt3R matching recipe** (encode → decode → `fast_reciprocal_NNs` → map patch coords back
  to original resolution with centre-crop/scale) is standard MASt3R usage. The coordinate rescale
  mirrors the public MASt3R demo and the A4R034 reference `mast3r_colmap/match.py`. The A4R034
  Top24-MASt3R + Top8-RDD + Top8-ALIKED fusion recipe is not reproduced here.
- **DINOv2 / timm** — global image retrieval only.
- **COLMAP / pycolmap** — geometric verification + incremental SfM.
- **AdaptiveGraph infrastructure** — the `adaptivegraph/` project (`IMC2025_AdaptiveGraph`):
  retrieval, MAGSAC verification, COLMAP wrapper, graph label propagation, submission writer are
  reused.
- **IMC2025 public metric** — used locally for scoring only (fetched from the public repo).

Cite MASt3R, DUSt3R, CroCo, DINOv2, COLMAP, LightGlue papers in any write-up.
