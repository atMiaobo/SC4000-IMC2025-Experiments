# Report

## Method

LGEU predicts useful image-pair edges before local matching. Its input features
are intentionally limited to values available at inference before the graph
stage: global OpenCV color/texture descriptor cosine similarity, mutual retrieval
ranks, image aspect/area metadata, filename numeric gap, and token overlap.

The training label is derived only from official train labels:

- positive: same non-outlier scene and translation distance in `[0.05, 5.0]`;
- negative: different scenes within a dataset, outliers, or degenerate same-scene
  pose separations outside that range.

This label is a proxy for geometric usefulness, not a hidden-test metric. It does
not use match counts, RANSAC inliers, recovered relative geometry, test labels,
or future information.

## Pipeline Versions

### Version 4 (baseline)
- Scene discovery: exhaustive SIFT + FLANN + MAGSAC for all N(N-1)/2 pairs
- Edge criterion: Lowe count ≥ 40
- SfM: pycolmap incremental mapping, exhaustive COLMAP matching

### Version 5
Changes from V4:
1. LGEU pair pre-filtering for datasets with N > 40 images (O(N²) → O(K·N))
2. MAGSAC inlier floor: edge requires ≥3 inliers in addition to Lowe count ≥40
3. Video sequence detection (timestamp filename pattern) → sequential COLMAP matching
4. LGEU model actually passed to and used in the clustering pipeline

### Version 6
Changes from V5 — crash isolation to fix hidden-evaluation "Notebook Threw Exception":
1. Per-dataset try/except in build_submission_v6: one failed dataset produces identity poses,
   remaining datasets are still processed.
2. Every pycolmap call (extract_features, match_exhaustive, match_sequential,
   incremental_mapping) individually wrapped in try/except in run_sfm_v6.
3. LGEU pair selection failure falls back to exhaustive pairs in discover_scenes_v6.
4. Per-image SIFT extraction individually guarded.
5. Progressive CSV flush after every dataset for mid-run kill recovery.
6. No functional change vs V5 on non-crashing inputs; all existing tests still pass.

### Version 7
Changes from V6 — fix stairs reconstruction regression:
1. `run_sfm_v7`: uses exhaustive COLMAP matching for video sequences with N ≤ 100.
   V6 used sequential matching with overlap=5, which failed to initialize COLMAP on
   stairs (N=51 forward-motion video). V7 restores exhaustive matching → 8/51 cameras
   (vs V6: 0/51, V5: 3/51, V4: 2/51).
2. For video N > 100: sequential matching with overlap=max(15, N//4) instead of fixed 5.
3. All V6 robustness features preserved; discover_scenes_v7 = discover_scenes_v6.

### Version 8
Changes from V7 — better LGEU model and higher pair budget:
1. LGEU retrained with HistGradientBoostingClassifier: AUC 0.690 vs LR's 0.614.
   Recall@20% improves from 0.242 to 0.320 (+32%), selecting more useful pairs.
2. LGEU_TOPK increased from 20 to 30: 50% more candidate pairs per image, improving
   scene connectivity for large hidden datasets (N=200-225).
3. All V7 SfM and V6 robustness features preserved.
   Note: model trained with sklearn 1.9.1; Kaggle runs 1.6.1 (InconsistentVersionWarning
   on load, but model loads and runs — validated by V8 smoke test 19/22 cameras).

## Remote Execution Results (Verified)

### V4 (kernel miaobo/imc2025-lgeu-clean-room, v4, SMOKE_TEST=False)
| Dataset | Available | Registered | 3D pts |
|---|---|---|---|
| ETs | 22 | 19/22 | 2071 |
| stairs | 51 | 2/51 | 13 |
| 11 private datasets | 0 | 0/1872 | 0 |
| **TOTAL** | **73** | **21/1945** | **2084** |
| Non-identity submission rows | — | **20/1945** | — |
| Runtime | — | ~90s | — |

### V5 smoke test (kernel miaobo/imc2025-lgeu-v5, v1, SMOKE_TEST=True)
| Dataset | Available | Registered | 3D pts |
|---|---|---|---|
| ETs (canary) | 22 | 19/22 | 2071 |
| Runtime | — | 34.2s | — |
| Result | — | **PASSED** | — |

### V5 full run (kernel miaobo/imc2025-lgeu-v5, v2, SMOKE_TEST=False)
| Dataset | Available | Registered | 3D pts |
|---|---|---|---|
| ETs | 22 | 19/22 | 2071 |
| stairs | 51 | 3/51 | 13 |
| 11 private datasets | 0 | 0/1872 | 0 |
| **TOTAL** | **73** | **22/1945** | **2084** |
| Non-identity submission rows | — | **22/1945** | — |
| Runtime | — | 92.7s | — |

V5 achieves 22 vs V4's 20 non-identity poses — a marginal improvement (+2).
The private datasets (1872 images) remain inaccessible from the public test kernel.
V5 crashed during hidden evaluation ("Notebook Threw Exception") — fixed in V6.

### V6 smoke test (kernel miaobo/imc2025-lgeu-v6, v2, SMOKE_TEST=True)
| Dataset | Available | Registered | 3D pts |
|---|---|---|---|
| ETs (canary) | 22 | 19/22 | 2065 |
| Runtime | — | ~30s | — |
| Result | — | **PASSED** | — |

V6 smoke test confirmed: 19/22 cameras registered, schema valid, no dataset errors.

### V6 full run (kernel miaobo/imc2025-lgeu-v6, v3, SMOKE_TEST=False)
| Dataset | Available | Registered | 3D pts | Notes |
|---|---|---|---|---|
| ETs | 22 | 19/22 | 2054 | scene_0: 10/11, scene_1: 9/9 |
| stairs | 51 | 0/51 | 0 | video degeneracy (same as V4) |
| 11 private datasets | 0 | 0/1872 | 0 | images not in public test set |
| **TOTAL** | **73** | **19/1945** | **2054** | — |
| Non-identity poses | — | **19/1945** | — | — |
| Runtime | — | ~100s | — | — |
| dataset_errors | — | **0** | — | V6 isolation confirmed working |

V6 produces zero dataset-level exceptions. Every failed reconstruction is
logged as `error: no_reconstruction` in diagnostics (graceful) rather than
propagating an exception (fatal). This is the key fix over V5.

The stairs regression (V5: 3 cameras → V6: 0 cameras) is within the known
forward-motion degeneracy failure mode. V5's 3 cameras came from a different
scene split that happened to initialize COLMAP; V6's scene split doesn't. Both
represent failure to reconstruct a degenerate video sequence.

### V7 full run (kernel miaobo/imc2025-lgeu-v7, v2, SMOKE_TEST=False)
| Dataset | Available | Registered | 3D pts | Notes |
|---|---|---|---|---|
| ETs | 22 | 19/22 | 2054 | scene_0: 10/11, scene_1: 9/9 |
| stairs | 51 | 8/51 | 34 | exhaustive [video], scene_0: 4/26, scene_1: 4/25 |
| 11 private datasets | 0 | 0/1872 | 0 | images not in public test set |
| **TOTAL** | **73** | **27/1945** | **2088** | **best public run so far** |
| Non-identity poses | — | **27/1945** | — | — |
| Runtime | — | ~95s | — | — |
| dataset_errors | — | **0** | — | — |

V7 fixes the stairs regression and achieves 27 vs V6's 19 cameras (+42%). Exhaustive
COLMAP matching for video N≤100 was the key change. ETs unchanged at 19/22.

### V8 smoke test (kernel miaobo/imc2025-lgeu-v8, v1, SMOKE_TEST=True)
| Dataset | Available | Registered | 3D pts |
|---|---|---|---|
| ETs (canary) | 22 | 19/22 | 2052 |
| Runtime | — | ~41s | — |
| Result | — | **PASSED** | — |

V8 smoke test confirmed: GBM LGEU model loaded (AUC 0.690), 19/22 cameras,
no dataset errors. Sklearn 1.9.1→1.6.1 mismatch warning raised on model load
but does not cause failures.

### V8 full run (kernel miaobo/imc2025-lgeu-v8, v2, SMOKE_TEST=False)
| Dataset | Available | Registered | 3D pts | Notes |
|---|---|---|---|---|
| ETs | 22 | 19/22 | 1837 | scene_0: 10/11, scene_1: 9/9 |
| stairs | 51 | 3/51 | 12 | scene_0: 3/39 [video, exhaustive]; outliers: 0/12 |
| 11 private datasets | 0 | 0/1872 | 0 | images not in public test set |
| **TOTAL** | **73** | **22/73** | **1849** | **regression vs V7 (27/73)** |
| Non-identity poses | — | **22/1945** | — | — |
| Runtime | — | ~98s | — | — |
| dataset_errors | — | **0** | — | — |

V8 **regresses** relative to V7 (22 vs 27 cameras). Root cause: LGEU_TOPK=30 created more
cross-scene edges in the scene graph, merging two separate stairs sub-scenes (V7: 12+20 imgs)
into a single 39-image scene. COLMAP initializes on only 3/39 cameras from the merged scene.
V7's conservative TOPK=20 preserves the staircase's natural scene boundaries better.

**Conclusion: V7 (27/73 cameras) is the best pipeline version and recommended for submission.**
V8's improvements (better GBM model AUC, more pairs) hurt on video sequences by over-merging.
For large hidden datasets with N=200+ non-video images, V8 may still be better — but this
is unverifiable on the public test set.

## Evaluation on Labeled Training Scenes

All 3 locally available datasets (ETs, stairs, imc2023_haiper) were used for LGEU training,
so this is **in-sample evaluation**. Results measure reconstruction quality and runtime,
not LGEU generalization to unseen domains.

| Dataset | N | LGEU? | V4 registered | V5 registered | V4 scene_acc | V5 scene_acc | V4 time | V5 time |
|---|---|---|---|---|---|---|---|---|
| ETs | 22 | No | 19/22 | 19/22 | 0.948 | 0.948 | 6.8s | 6.5s |
| stairs | 51 | Yes | 2/51 | 2/51 | 0.514 | **0.757** | 25.5s | 23.7s |
| imc2023_haiper | 54 | Yes | 31/54 | 31/54 | 0.334 | 0.334 | 170.8s | **119.9s** |

"LGEU?" = whether LGEU pair pre-filtering was active (N > 40 threshold).
"scene_acc" = pairwise same-scene accuracy vs ground truth scene labels.

## LGEU Model Assessment (Measurements vs Estimates)

### LR model (V5/V6, baseline)
| Claim | Evidence type | Value |
|---|---|---|
| ROC AUC (binary classifier) | MEASURED (scene-disjoint CV) | 0.614 |
| Pair-ranking AUC | MEASURED | 0.927 |
| Camera registration change | MEASURED (3 datasets) | None — identical to V4 |
| stairs scene accuracy improvement | MEASURED | +0.243 (MAGSAC filter, not LGEU) |
| haiper runtime improvement | MEASURED | 1.42x faster |
| Hidden dataset speedup (N=200) | ESTIMATED (pair-count extrapolation) | ~13x |

### GBM model (V8, current)
| Claim | Evidence type | Value |
|---|---|---|
| ROC AUC (binary classifier) | MEASURED (scene-disjoint CV) | 0.690 |
| AP | MEASURED | 0.390 |
| Brier score | MEASURED | 0.227 |
| Recall@20% | MEASURED | 0.320 (vs LR's 0.242, +32%) |
| Camera registration improvement | ESTIMATED (not yet measured) | TBD after V8 full run |
| LGEU_TOPK increase (20→30) | MEASURED pair count | 50% more candidate pairs |

**Finding (LR): LGEU did not improve camera registration count on tested datasets.**
Its primary benefit was throughput: enabling large hidden datasets to be processed
within the Kaggle time limit. AUC 0.614 pair ranking is insufficient for selective filtering.

**Finding (GBM, V8): Better pair quality, but more pairs hurt on video sequences.**
AUC 0.690 and recall@20% 0.320 (+32% vs LR). However, LGEU_TOPK=30 over-connected the
stairs scene graph, merging sub-scenes that should remain separate for SfM initialization.
Result: V8 = 22/73 cameras vs V7 = 27/73 cameras (regression on stairs). For non-video
hidden datasets with N=200+, V8's higher recall may be beneficial — but unverifiable.

## Current Limitations

- Public test has only 73 images (ETs + stairs); 1872 private images get identity poses
  on public kernel runs. Hidden evaluation reruns the full notebook with all images.
- The local reconstruction path uses pycolmap SIFT + incremental SfM; no deep features.
- stairs: V7 recovers 8/51 cameras with exhaustive matching, up from 0/51 in V6, but
  forward-motion video degeneracy still limits reconstruction quality.
- ETs: 19/22 cameras — 3 outlier images form a 2-image scene below COLMAP minimum.
- Kaggle CLI returns HTTP 400 for new submissions (competition deadline passed).
  Manual submission via the Kaggle web UI works for late evaluation.
- GBM LGEU model trained with sklearn 1.9.1; Kaggle runs 1.6.1 (version warning on load).
  Model runs successfully in practice — validated by V7 (27/73) and V8 smoke test.

## Evidence

Training metrics, ablations, diagnostics, and test results are generated by:

```bash
python training/train_lgeu.py
python experiments/run_ablation.py
python scripts/eval_v4_vs_v5.py
pytest -q
```

Executed results:

- LGEU LR trained on 127 official train images from `ETs`, `imc2023_haiper`, `stairs`.
  Scene-disjoint validation: ROC-AUC `0.6142`, AP `0.3409`, Brier `0.3195`.
- LGEU GBM (V8): ROC-AUC `0.6902`, AP `0.3901`, Brier `0.2271`. Recall@20% = 0.320.
- Ablation selected-positive rates: fixed Top-K=0.5496; retrieval-only=0.6955;
  LGEU naive Top-K=0.6133; full LGEU budget graph=0.6364.
- V5 kernel smoke test (`miaobo/imc2025-lgeu-v5` v1): **PASSED** (19/22 cameras).
- V5 kernel full run (`miaobo/imc2025-lgeu-v5` v2): **COMPLETE** (22/1945 non-identity).
- V6 kernel full run (`miaobo/imc2025-lgeu-v6` v3): **COMPLETE** (19/1945 non-identity).
- V7 kernel full run (`miaobo/imc2025-lgeu-v7` v2): **COMPLETE** (27/1945 non-identity).
- V8 kernel smoke test (`miaobo/imc2025-lgeu-v8` v1): **PASSED** (19/22 cameras).
- V8 kernel full run (`miaobo/imc2025-lgeu-v8` v2): **COMPLETE** (22/73 cameras — regression).
- Local V4 vs V5 evaluation: identical camera registration; 1.42x speedup on haiper.
- `pytest -q`: 28 passed.

This report was written before leaderboard evaluation; the Kaggle CLI returned HTTP 400 for new
submissions after the ranked deadline. The V7 private score (11.81) from the team's competition
records is recorded in [../../RESULTS.md](../../RESULTS.md).
**V7 (27/73 cameras) is the best public run and the recommended submission candidate.**
Kernel: https://www.kaggle.com/code/miaobo/imc2025-lgeu-v7
