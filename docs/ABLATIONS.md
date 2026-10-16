# Key ablation experiments

Consolidated summary of the most important ablations across the three pipelines, with pointers
to the raw evidence. All local scores use the official IMC25 metric (harmonic of pose mAA and
scene clusterness) unless stated otherwise, and are **training-set** numbers, not leaderboard
scores (see [../RESULTS.md](../RESULTS.md)).

---

## 1. AdaptiveGraph — A0→A3 component ladder

**Question:** does each graph decision component add value over a strong
DINOv2 + ALIKED/LightGlue + MAGSAC + COLMAP backbone?

**Design:** identical backbone/data/scorer, one component added per rung
(`configs/ablation_A0..A3*.yaml`, driver `scripts/run_ablation.py`).

| Rung | Added component |
|---|---|
| A0 | Backbone only (inlier-count gate) |
| A1 | + quality gate (inlier-ratio + spatial coverage) |
| A2 | + triangle-support rehab + graph label propagation (**= shipped `final.yaml`**) |
| A3 | + adaptive tile/rotation rescue (scene-preserving) |

**Full 13-scene result** (`adaptivegraph/results/ablation/fullset/results.json`):

| Config | combined | mAA | clusterness |
|---|---:|---:|---:|
| A0 backbone | **61.26** | 49.37 | 92.44 |
| A2 graph (deployed) | 60.89 | 49.25 | 95.69 |
| proposed_B (backbone + propagation) | 60.81 | 49.05 | 95.31 |

**Finding:** the graph layer is net-neutral in aggregate (all within ~0.5, single run). The
graph heuristics do not raise the aggregate official metric on this backbone.

**Development-subset ladder** (`.../ladder/results.json`; discriminative, non-saturated scenes)
shows the gate can even *reduce* recall on sparse scenes: A0 = 41.35, A1 = 36.82, A2 = 37.47.

**Why propagation is still kept:** it is the one component with a clear *directional* benefit —
it raises clusterness (mean 92.4 → 95.3; `stairs` 50.0 → 88.9) at **zero pose cost**. It is
net-neutral on the *combined* harmonic score only because most scenes are already at 100%
clusterness, and the fragmented scenes it helps have ≈0 pose mAA (so the harmonic stays ≈0).

## 2. AdaptiveGraph — adaptive rescue (negative result)

**Question:** can tile/rotation "rescue" matching recover stranded images?

**Result** (`adaptivegraph/results/ablation/a3cmp/results.json`; ETs, fbk_vineyard,
imc2023_haiper):

| Config | combined | mAA | clusterness |
|---|---:|---:|---:|
| A2 (no rescue) — reference | **54.17** | 43.14 | 82.50 |
| A3 naive rescue | 41.59 | 37.77 | 47.37 |
| A3 scene-preserving rescue | 44.01 | 40.57 | 51.36 |

**Root cause (measured):** on ETs, of 150 accepted edges under naive rescue, 72 were cross-scene
and **all 72 were rescued** tile/rotation edges — some with very high inlier counts (217, 202,
230). ETs' two near-identical statue scenes (`ET`/`another_ET`) are a **doppelgänger** case:
high-resolution tiling finds strong *local* correspondences between physically distinct scenes
and merges them. High local match count is an unreliable merge signal.

**Fix attempted:** `scene_preserving` rescue forbids merging two already-established components
on rescued local-match evidence (within-component densification and singleton attachment are
still allowed). It beats naive rescue (44.01 vs 41.59, ETs mAA 28.8 → 46.2) but still loses to
no-rescue A2 (54.17). → **Rescue disabled in the shipped config.** Raw per-pair routing decisions
for this experiment are preserved in
`adaptivegraph/results/ablation/a3cmp/*/pair_diagnostics.csv`.

## 3. MASt3R — baseline vs reconstruction-aware adaptive budget

**Question:** does spending matching budget *adaptively* (only on under-connected nodes, with
scene-preserving merges) beat a fixed top-K accept-all baseline, on the same MASt3R backbone?

**Result** (`mast3r-adaptive/results/cmp/results.json`, `.../fbk/results.json`):

| Dataset | Baseline | Adaptive |
|---|---:|---:|
| ETs | 43.7 | **55.0** |
| imc2023_haiper | 67.8 | **82.7** |
| fbk_vineyard | **23.7** | 16.6 |
| **mean (3)** | **45.06** | **51.44** |

**Finding:** adaptive beats baseline overall (+6.4), driven by clusterness (50.7 → 93.6) from
stopping the doppelgänger over-merge. On the large 3-sub-scene `fbk_vineyard` it regresses
(23.7 → 16.6) — the conservative base-6 budget under-connects a big scene, registering only
62/163 cameras. Proposed (unvalidated) fix: a **size-aware budget** (neighbours ∝ scene size).
On the same 3 datasets, AdaptiveGraph A2 scores **55.1**, so MASt3R-adaptive does not beat
AdaptiveGraph in aggregate here.

## 4. SIFT/LGEU — pipeline version ladder (V4→V8)

**Question (engineering):** which pipeline changes increase registered cameras on the public
test (ETs + stairs, 73 images)? Camera counts, not IMC25 metric
(`sift-lgeu/results/*/submission_diagnostics.json`, `sift-lgeu/docs/REPORT.md`):

| Version | Total /73 | Key change | Verdict |
|---|---:|---|---|
| V4 | 21 | exhaustive SIFT baseline | — |
| V5 | 22 | LGEU pre-filter + MAGSAC floor + video detection | marginal +1 |
| V6 | 19 | crash isolation (fixes hidden-eval exceptions) | video regression |
| **V7** | **27** | exhaustive COLMAP for video N≤100 | **best** |
| V8 | 22 | GBM LGEU + TOPK 20→30 | **regression** (over-merge) |

**Finding:** robustness (V6) and video handling (V7) mattered more than classifier quality. V8
used a *better* classifier (ROC-AUC 0.690 vs 0.614) and more pairs but regressed by merging two
`stairs` video sub-scenes into one 39-image scene that COLMAP could only partially initialize.

## 5. SIFT/LGEU — LGEU classifier & pair-selection ablation

- **LR vs GBM** (`sift-lgeu/experiments/lgeu_metrics*.json`): GBM improves ROC-AUC
  0.614 → 0.690 and Recall@20% 0.242 → 0.320 (+32%). But — see V8 above — better pair ranking
  did not yield more registered cameras; its measured benefit was throughput.
- **Pair-selection strategies** (`sift-lgeu/experiments/ablation_results.csv`), selected-positive
  rate: fixed Top-K 0.550, retrieval-only 0.696, LGEU naive Top-K 0.613, full LGEU budget graph
  0.636.

## 6. A4R045 (pose12) — evidence-grounded, unvalidated proposal

Not an experiment that was run — an analysis of the A4R034 `abs_pose_min_num_inliers` sweep
showing a monotonic private-score gain as the floor drops (30 → 37.06, 20 → 40.37, 15 → 42.25),
and a proposal to try 12. No score is claimed; the falsification criterion (if pose12 ≤ 42.25,
keep 15 as the optimum) is documented in
[`../a4r045-optimization-proposal/OPTIMIZATION_REPORT.md`](../a4r045-optimization-proposal/OPTIMIZATION_REPORT.md).

---

## Cross-cutting observation

All three pipelines independently hit the **doppelgänger / repetitive-scene** failure: strong
local correspondence between physically distinct scenes causes over-merging and clusterness
collapse. The recurring (partial) remedy is to treat cross-scene merges as *untrusted* unless
supported by multi-view/graph evidence — but on strong backbones this buys clusterness without
a net aggregate-metric gain, which is the main result across AdaptiveGraph and MASt3R.
