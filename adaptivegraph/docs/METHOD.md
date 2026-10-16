# Method: Adaptive Scene-Graph Matching

## Research question

Can competitive IMC-2025 reconstruction quality be achieved without applying an expensive
matcher or exhaustive crop augmentation to every retrieved pair?

The project treats pair selection and matching as an **adaptive resource-allocation problem**. A
cheap first pass estimates both visual similarity and geometric quality. Expensive operations are
then routed only to pairs that are ambiguous or topologically important to the scene graph.

## Pipeline

1. **DINOv2 retrieval.** A single global descriptor creates a moderate candidate graph. A
   multi-retriever ensemble is intentionally not used.
2. **ALIKED + LightGlue base matching.** Every candidate edge receives one full-image
   local-feature match.
3. **Geometric edge signature.** MAGSAC estimates a fundamental matrix and computes inlier count,
   inlier ratio, 4x4 spatial coverage, and median Sampson error.
4. **Adaptive rescue router.** Only uncertain but promising edges receive extra compute:
   - 2x2 tile matching for poor spatial coverage / scale mismatch,
   - 90/180/270-degree retry for high semantic similarity with catastrophic local matching.
5. **Triangle consistency.** A borderline pair is accepted when it closes a triangle between two
   already-trusted edges. This injects a simple multi-view consistency prior without training
   another classifier.
6. **Topology-aware rescue.** Low-degree nodes are given a final expensive rescue attempt because
   one missed bridge can split a true scene component.
7. **COLMAP reconstruction.** Accepted correspondences are imported into a standard pycolmap
   incremental mapper. Multiple models represent multiple scenes.
8. **Graph label propagation.** When COLMAP cannot register an image, the verified graph can still
   assign it to a likely scene component, leaving the pose as NaN rather than discarding the
   clustering information.

## Original components

The pretrained components are not claimed as novel. The project contribution is the routing and
graph logic around them:

- match-quality vector rather than raw number of matches,
- spatial-coverage diagnostic,
- conditional rather than universal crop augmentation,
- conditional rotation retry,
- triangle-supported edge rehabilitation,
- topology-aware rescue of low-degree / bridge-like images,
- graph-based scene-label propagation after reconstruction.

## Ablations to report

Run these with the same train data and metric:

| Experiment | Config/change | Question |
|---|---|---|
| A | `configs/fast.yaml` | What does a simple DINO + ALIKED/LG baseline do? |
| B | balanced, disable tiles+rotation | Does graph filtering alone help? |
| C | balanced, tiles on / rotation off | Does spatially triggered tile rescue help? |
| D | balanced, tiles+rotation, triangle support off | Does multi-view graph consistency help? |
| E | `configs/balanced.yaml` | Full proposed method |
| F | `configs/high_score.yaml` | What happens when efficiency/routing sparsity is traded for more crop compute? |

Keep `outputs/*/submission.csv`, `pair_diagnostics.csv`, and `run_summary.json` so the report can
quote measured results and runtime.
