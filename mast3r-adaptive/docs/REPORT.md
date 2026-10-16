# MASt3R Adaptive Reconstruction — Technical Report

> **Scope of numbers.** All numbers below are **local training-set** scores with the official
> IMC25 metric on legitimate `train_labels.csv` scenes (single run, no seed averaging, small
> dataset subsets). They are not hidden Kaggle leaderboard scores. The Kaggle private score for
> this pipeline (22.97) is in [../../RESULTS.md](../../RESULTS.md).

## 1. System overview

```
DINOv2 (timm) retrieval  →  MASt3R semi-dense matching (per candidate pair)
  →  MAGSAC fundamental-matrix verification  →  graph (triangle-support rehab)
  →  COLMAP multi-model SfM  →  graph label propagation  →  submission.csv
```

Pretrained, not trained here: **MASt3R** `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`
(NAVER, CC-BY-NC-SA-4.0, commit `f5209af`), **DINOv2** (timm), **COLMAP/pycolmap 3.13**. The
MASt3R matcher is standard API usage (encode→decode→`fast_reciprocal_NNs`→rescale), attributed in
`THIRD_PARTY_NOTICES.md`. Retrieval / MAGSAC / COLMAP / submission reuse the AdaptiveGraph
infrastructure.

## 2. Feasibility (RTX 5080, 16 GB)

MASt3R runs on sm_120 with torch 2.7.1+cu128 using the **pure-PyTorch RoPE fallback** (no
`curope`/`nvcc` needed). Measured: model load ~20 s; encode ~0.65 s/img (cached per image);
**match ~0.7 s/pair**; **peak VRAM 3.74 GB**. Encoder features cached on CPU → O(N) encode,
O(pairs) decode.

## 3. Contribution — reconstruction-aware adaptive matching

**Problem.** MASt3R's dense matching produces geometrically-consistent correspondences even
between visually near-identical but physically distinct scenes (doppelgängers, e.g. ETs'
`ET`/`another_ET`). A fixed top-K retrieval + accept-all-verified baseline therefore merges such
scenes → clusterness collapses. (This is the same failure the ALIKED AdaptiveGraph project
documented.)

**Method.** Instead of a fixed top-K budget:
1. A cheap **base pass** uses a *small* retrieval budget (top-6) and MAGSAC-verifies it to build a
   *trusted* scene graph with few cross-scene edges.
2. **Reconstruction-aware routing:** extra MASt3R matching is spent only on **low-degree /
   under-connected nodes** (those at risk of being stranded), not uniformly.
3. **Scene-preservation:** a rescued edge may densify within a trusted component or attach a
   singleton, but a cross-component merge of two established scenes requires **multi-view
   (triangle) evidence** — never a single high-match-count edge.

This applies the AdaptiveGraph observation ("aggressive rescue improves mAA but wrecks
clusterness") by making cross-scene merging conservative and evidence-gated.

## 4. Experiments (official IMC25 metric, local train)

### Baseline (A) vs Proposed adaptive (B), same MASt3R backbone

| Dataset | A combined | A mAA | A clust | B combined | B mAA | B clust |
|---|---|---|---|---|---|---|
| ETs            | 43.7 | 44.2 | 43.2 | **55.0** | 40.4 | 86.4 |
| imc2023_haiper | 67.8 | 72.8 | 63.5 | **82.7** | 70.6 | 100.0 |
| fbk_vineyard   | 23.7 | 16.1 | 44.8 | 16.6 | 9.1 | 94.5 |
| **mean (3 ds)**| **45.06** | 44.33 | 50.70 | **51.44** | 40.03 | 93.57 |

The adaptive method beats the MASt3R baseline overall: 51.44 vs 45.06 (+6.4) across 3 datasets
(and +13.1 on ETs+haiper). It consistently raises clusterness (50.7→93.6) by stopping the
doppelgänger over-merge, and matches far fewer pairs (base-6 vs top-20) so it is faster.

**Regression (fbk_vineyard).** On this large 3-sub-scene dataset the adaptive method regresses
(23.7→16.6): the conservative base-6 budget under-connects a big scene, so only 62/163 cameras
register (vs 162) — clusterness rises (45→95) but mAA collapses (16→9). The budget is too sparse
when N is large. The principled fix (not yet validated) is a **size-aware budget** (neighbours ∝
scene size / connectivity) so large scenes get more edges while small scenes stay conservative —
see §6.

### Comparison vs AdaptiveGraph (ALIKED+LightGlue), same 3 datasets

| | ETs | haiper | fbk | mean |
|---|---|---|---|---|
| AdaptiveGraph A2 | 61.3 | 61.1 | 42.8 | **55.1** |
| MASt3R baseline  | 43.7 | 67.8 | 23.7 | 45.1 |
| MASt3R adaptive  | 55.0 | **82.7** | 16.6 | 51.4 |

Mixed: MASt3R gives **much better poses** on well-matched scenes (haiper mAA 70.6; aggregate mAA
favours MASt3R on ETs+haiper) and the adaptive method clearly beats the MASt3R baseline, **but**
MASt3R does not beat AdaptiveGraph in aggregate on this subset (51.4 vs 55.1) — it loses on the
repetitive/large fbk_vineyard and the ETs doppelgänger.

## 5. Kaggle deployment

- Bundle: `miaobo/mast3r-imc25-bundle` (MASt3R weights + repo + dino cache + adaptive_imc + cp313
  wheels). Notebook: `kaggle/MASt3R_Adaptive_Kaggle.ipynb` (offline, GPU, adaptive mode).
- Target: **original `image-matching-challenge-2025`**, kernel `miaobo/mast3r-adaptive-orig` (v4).
  The orchestrator writes a finite fallback submission first and runs each dataset in an isolated
  subprocess with a wall-clock timeout, so a native OOM/segfault on one dataset leaves the
  fallback in place and the notebook still finishes.
- Runtime note: MASt3R (RoPE fallback) is slower than ALIKED; on Kaggle T4 the full hidden test
  may be time-constrained. The adaptive base-6 budget keeps the pair count (and thus runtime)
  low, which helps.

## 6. Caveats and open items

- Ablation is on small subsets (2–3 datasets), single run, in-sample; not scene-disjoint CV. The
  +13 adaptive gain is consistent and mechanism-explained (clusterness), but should be confirmed
  on more scenes before strong claims.
- MASt3R-adaptive still under-separates ETs (clust 86, not 100) — the doppelgänger problem is
  mitigated, not solved.
- Size-aware budget (neighbours ∝ scene size) is the proposed fix for the fbk_vineyard regression;
  not yet validated.
- CC-BY-NC-SA MASt3R license restricts to non-commercial use.
