# IMC2025 Adaptive Scene-Graph Matching — Technical Report

> **Scope of numbers.** The ~60.89 figure quoted below is a **local training-set** mean using
> the official IMC25 metric on legitimate `train_labels.csv` scenes. It is not a hidden Kaggle
> leaderboard score. The Kaggle private score for this pipeline (22.63) is in
> [../../RESULTS.md](../../RESULTS.md).

## 1. System overview

A clean-room IMC2025 pipeline built from public pretrained components, with an independently
designed decision/graph layer:

```
DINOv2 (timm) global retrieval  →  ALIKED + LightGlue base matching
  →  MAGSAC fundamental-matrix verification (inliers / ratio / spatial coverage)
  →  quality-vector edge gating
  →  triangle-supported borderline edge rehabilitation
  →  [optional] adaptive tile/rotation rescue  (scene-preserving, §4)
  →  COLMAP multi-model incremental SfM
  →  graph-based scene-label propagation
  →  submission.csv
```

Pretrained, not trained here: **DINOv2** (`vit_base_patch14_dinov2.lvd142m`, timm),
**ALIKED-n16 + LightGlue** (cvg/LightGlue, Apache-2.0, vendored at commit `eb42fee`),
**COLMAP/pycolmap 3.13.0**. See `THIRD_PARTY_NOTICES.md`.

## 2. Originality audit (provenance)

- **Independent code.** `src/adaptive_imc/*.py` (~1.45k lines) shares no identifiers, structure,
  or constants with the team's A4R034 (Yiji) MASt3R+RDD+ALIKED solution or the first-place
  MASt3R-hybrid. Cross-check: of 52 top-level names, the only one also present in the A4R034 tree
  is the generic `write_submission`.
- **Different backbone.** Both references are MASt3R-centric (2.75 GB weights, dual-GPU). This
  system uses only ALIKED+LightGlue and does not use MASt3R or the Top-K fusion recipes.
- **Reference-absent components.** Triangle-support edge gating, topology-aware low-degree rescue,
  graph label propagation, and *selective* (vs uniform) tile rescue do not appear in either
  reference.

**Claim separation.** The evidence supports (a) *independent engineering* and (b) a *novel
algorithmic combination*. Claim (c) *measurable contribution over the pretrained backbone* is
addressed by the ablations in §3–§5 — not assumed.

## 3. Ablation methodology

Controlled ladder, identical backbone/data/scorer, one component added per rung
(`configs/ablation_A0..A3.yaml`, driver `scripts/run_ablation.py`). Scored with the official
IMC25 metric (harmonic combination of pose mAA and clusterness) via `third_party/imc25metric.py`.

| Rung | Added component |
|---|---|
| **A0** | Backbone: DINOv2 + ALIKED/LightGlue + MAGSAC (inlier-count gate) + COLMAP |
| **A1** | + quality gate (inlier-ratio + spatial-coverage) and quality ranking |
| **A2** | + triangle-support rehab + graph label propagation (**= scored `fast`/`final`**) |
| **A3** | + adaptive tile/rotation rescue with scene-preservation (§4) |

Evaluation subset (legitimate train scenes, chosen to be discriminative rather than saturated):
**ETs, amy_gardens, fbk_vineyard, imc2023_haiper, stairs**. All numbers below are measured on
this development subset unless stated; the full 13-scene `fast`/A2 mean is 60.89 (dominated by
near-saturated phototourism scenes).

### Results — development subset (ETs, amy_gardens, fbk_vineyard, imc2023_haiper, stairs)

| Config | combined | pose mAA | clusterness | notes |
|---|---|---|---|---|
| A0 backbone        | 41.35 | 32.09 | 80.21 | loose inlier-count gate |
| A1 + quality gate  | 36.82 | 27.90 | 80.58 | stricter gate drops recall here |
| A2 + graph (=fast) | 37.47 | 30.16 | 80.75 | scored reference config |

Rescue comparison — subset (ETs, fbk_vineyard, imc2023_haiper):

| Config | combined | pose mAA | clusterness | notes |
|---|---|---|---|---|
| A2 (no rescue)      | 54.17 | 43.14 | 82.50 | reference |
| A3-naive rescue     | 41.59 | 37.77 | 47.37 | rescue over-merges (ETs → 1 component) |
| A3 scene-preserving | 44.01 | 40.57 | 51.36 | fix beats naive (+ETs mAA 28.8→46.2) but still < A2 |

### Results — FULL 13-scene training set (official metric)

| Config | combined | pose mAA | clusterness |
|---|---|---|---|
| **A0 backbone**                 | **61.26** | 49.37 | 92.44 |
| A2 graph (= scored `fast`/`final`, deployed) | 60.89 | 49.25 | 95.69 |
| proposed_B (backbone + propagation) | 60.81 | 49.05 | 95.31 |

All within ~0.5 (single-run, no seed averaging). Numbers from `outputs/ablation/*/results.json`
and the preserved `outputs/fast_full` (A2).

## 4. Adaptive rescue: failure diagnosis and repair

**Observed failure.** Enabling the original (naive) adaptive rescue raised pose accuracy but
collapsed clustering. On ETs: `fast` → 61.33 (mAA 44.2 / clusterness 100.0); `balanced` (rescue
on) → 49.17 (mAA 53.9 / clusterness 45.2).

**Root cause (measured).** Of 150 accepted edges under rescue, **72 were cross-scene, and all 72
were rescued** tile/rotation edges — several with very high inlier counts (217, 202, 230). ETs'
two near-identical statue scenes (`ET`/`another_ET`) are a doppelgänger case: high-resolution
tile matching finds strong *local* correspondences between physically distinct scenes and merges
them. High local match count is therefore an unreliable merge signal.

**Fix (`scene_preserving`).** Snapshot the trusted components formed by the base (pre-rescue)
strong edges. A rescued edge may then:
- densify *within* a trusted component (improves pose coverage), or
- attach a singleton image to a scene,

but may **not** merge two already-established components (size ≥ 2) on rescued local-match
evidence. Base (non-rescued) edges are never blocked, so legitimate cross-scene links are
preserved. This operationalizes "treat cross-component rescue as untrusted" without discarding
useful within-scene recovery.

## 5. Decision (Path A / Path B)

**Path A (adaptive rescue) is rejected.** Scene-preserving rescue (44.01) beats naive rescue
(41.59) and recovers within-scene pose accuracy (ETs mAA 28.8→46.2), but both are well below
no-rescue A2 (54.17). Rescue over-connects visually-similar/doppelgänger scenes; even constrained
to within-component densification it does not yield a net gain. → **rescue disabled** in the
shipped config (`final.yaml`). Negative result.

**Path B outcome.** On the full official metric the graph decision layer is net-neutral: A0
(61.26) ≈ A2 (60.89) ≈ proposed_B (60.81), all within single-run noise. Per-component, measured
effects:

- **Graph label propagation** — the one component with a clear directional benefit: raises
  clusterness (mean 92.4→95.3; stairs 50.0→88.9) at **zero pose cost** (mAA unchanged). It is
  net-neutral on the *combined* harmonic score only because (a) most scenes are already at 100%
  clusterness, (b) the scenes it helps (fragmented/video) have ≈0 pose mAA so the harmonic score
  stays ≈0, and (c) it mislabels some unregistered images on a few scenes (fbk_vineyard −7.8).
  Kept (it is the component with a measurable sub-metric effect) and retained in the shipped
  config.
- **Quality gate (inlier-ratio + coverage)** — reduces recall on sparse scenes (A0→A1 drop);
  marginal. Retained at mild thresholds in `final.yaml`; near-neutral.
- **Triangle-support rehab** — ≈neutral on full set, slightly over-merges multi-scene datasets
  (fbk). Retained (neutral) / optional.

**Shipped config:** `final.yaml` (= A2: DINOv2 + ALIKED/LightGlue + MAGSAC + triangle rehab +
label propagation, **rescue off**), validated on Kaggle (§6). A0 is marginally higher (61.26 vs
60.89) but within noise; the shipped config keeps the one component (propagation) with a
demonstrated directional benefit.

**Contribution.** The independent work is the *adaptive scene-graph framework* and an *ablation
methodology* that measures each component's effect on the official metric. The result is that,
on a strong ALIKED+LightGlue+COLMAP backbone, these graph heuristics are net-neutral in
aggregate, with graph label propagation providing a narrow clusterness benefit on fragmented
scenes.

## 6. Kaggle deployment

- Reference notebook: `kaggle/AdaptiveGraph_FAST_Kaggle.ipynb` (frozen `fast`/`final` config),
  offline, GPU, writes `/kaggle/working/submission.csv`.
- Bundle dataset: `miaobo/adaptive-graph-imc25-bundle` (code + weights + wheels).
- Deployment notes: Kaggle runs **Python 3.13** — the bundle ships a `pycolmap-3.13.0-cp313`
  wheel (an earlier cp312 wheel failed to install); the offline HF cache path is `models/hf`.
  Submissions emit finite identity/zero poses for unregistered images (the official scorer
  rejects non-finite values).
- Competition: deployed to the **original `image-matching-challenge-2025`** (kernel
  `miaobo/adaptivegraph-fast-orig`, v3) — comparable to A4R034's 38.16/42.25. The original is
  past its ranked deadline (2025-06-02) but accepts late submissions that score. A secondary
  kernel (`miaobo/adaptivegraph-fast-ref`, v5) targets the `-ongoing` mirror.

## 7. Reproduction

```bash
nix develop
source .venv/bin/activate          # or scripts/bootstrap.sh on first setup
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH   # NixOS GPU
# Ablation ladder:
python scripts/run_ablation.py \
  --config configs/ablation_A0.yaml configs/ablation_A1.yaml configs/ablation_A2.yaml \
           configs/ablation_A3_naive.yaml configs/ablation_A3.yaml \
  --dataset ETs amy_gardens fbk_vineyard imc2023_haiper stairs --tag ladder
# Score any submission:
python scripts/score_local.py outputs/<run>/submission.csv
```

Environment: Python 3.11 (local) / 3.13 (Kaggle), torch 2.7.1+cu128, pycolmap 3.13.0,
LightGlue @ `eb42fee`. GPU verified: RTX 5080 (sm_120).
