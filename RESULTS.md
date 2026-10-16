# Results

This file separates result **kinds** so they are not conflated:

- **A. Kaggle private leaderboard** — hidden-test scores on the *original*
  `image-matching-challenge-2025` competition.
- **B. Local official-metric scores** — the official IMC25 metric computed on legitimate
  `train_labels.csv` scenes. These are **development** scores, not leaderboard scores.
- **C. Camera-registration counts** — SIFT/LGEU's primary local diagnostic (how many cameras
  COLMAP registered on the public test), which is *not* the IMC25 metric.
- **D. Hypotheses / negative results** — see [docs/ABLATIONS.md](docs/ABLATIONS.md).

The IMC25 metric is a harmonic combination of pose **mAA** and scene **clusterness**.

---

## A. Kaggle private leaderboard (original `image-matching-challenge-2025`)

| Submission | Owner | Public | Private | Provenance |
|---|---|---:|---:|---|
| **A4R034** (Top24, angle8, pose15) — team's selected final | Yiji | 38.16 | **42.25** | `result.json` (submission_ref 57156705, notebook v1) |
| MASt3R Adaptive | this repo | — | **22.97** | team competition records; kernel `miaobo/mast3r-adaptive-orig` v4 |
| AdaptiveGraph (A2 / `final.yaml`) | this repo | — | **22.63** | team competition records; kernel `miaobo/adaptivegraph-fast-orig` v3 |
| SIFT/LGEU V7 | this repo | — | **11.81** | team competition records; kernel `miaobo/imc2025-lgeu-v7` v2 |

Notes:
- The public-leaderboard values for the three standalone kernels are not recorded in the
  packaged artifacts; only the private values were provided for this consolidation.
- The in-repo technical reports (written before these evaluations were available) state "no
  hidden-test score claimed" and note the Kaggle CLI returned HTTP 400 for new submissions
  after the ranked deadline. The private scores above come from the **team's competition
  records**. They are consistent with the local development ordering
  (MASt3R ≈ AdaptiveGraph > SIFT/LGEU).
- All three standalone pipelines score below A4R034 (42.25).

---

## B. Local official-metric scores (training set — NOT leaderboard)

### AdaptiveGraph — full 13-scene training set
Source: `adaptivegraph/results/ablation/fullset/results.json`, `.../fast_full/run_summary.json`.

| Config | combined | pose mAA | clusterness |
|---|---:|---:|---:|
| A0 backbone (DINOv2+ALIKED/LightGlue+MAGSAC+COLMAP) | **61.26** | 49.37 | 92.44 |
| A2 graph (= deployed `final.yaml`) | **60.89** | 49.25 | 95.69 |
| proposed_B (backbone + label propagation) | 60.81 | 49.05 | 95.31 |

All three are within ~0.5 (single run, no seed averaging). The graph layer is net-neutral in
aggregate on this backbone; see [docs/ABLATIONS.md](docs/ABLATIONS.md).

### AdaptiveGraph — discriminative development subsets
Source: `adaptivegraph/results/ablation/ladder/results.json`, `.../a3cmp/results.json`.

Ladder (added one component per rung):

| Config | combined | pose mAA | clusterness |
|---|---:|---:|---:|
| A0 backbone | 41.35 | 32.09 | 80.21 |
| A1 + quality gate | 36.82 | 27.90 | 80.58 |
| A2 + graph (= `fast`) | 37.47 | 30.16 | 80.75 |

Adaptive rescue comparison (ETs, fbk_vineyard, imc2023_haiper):

| Config | combined | pose mAA | clusterness |
|---|---:|---:|---:|
| A2 (no rescue) — reference | **54.17** | 43.14 | 82.50 |
| A3 naive rescue | 41.59 | 37.77 | 47.37 |
| A3 scene-preserving rescue | 44.01 | 40.57 | 51.36 |

Rescue (A3) scores below no-rescue (A2) → rescue disabled in the shipped config.

### MASt3R Adaptive — 3-scene training subset
Source: `mast3r-adaptive/results/cmp/results.json`, `.../fbk/results.json`.

| Dataset | Baseline combined | Adaptive combined |
|---|---:|---:|
| ETs | 43.7 | **55.0** |
| imc2023_haiper | 67.8 | **82.7** |
| fbk_vineyard | **23.7** | 16.6 (regression) |
| **mean (3 datasets)** | **45.06** | **51.44** |

Adaptive beats the MASt3R baseline overall (+6.4), driven by clusterness (50.7 → 93.6), but
regresses on the large `fbk_vineyard`: the conservative base-6 budget under-connects a big scene
(only 62/163 cameras register). On the *same* 3 datasets the AdaptiveGraph A2 configuration
scores 55.1 mean, i.e. MASt3R-adaptive (51.4) does not beat AdaptiveGraph in aggregate here.

---

## C. SIFT/LGEU camera-registration counts (public test: 73 images)

Source: `sift-lgeu/results/*/submission_diagnostics.json`, `sift-lgeu/docs/REPORT.md`
(remote Kaggle execution tables).

| Version | ETs (of 22) | stairs (of 51) | Total (of 73) | Change |
|---|---:|---:|---:|---|
| V4 baseline (exhaustive SIFT) | 19 | 2 | 21 | — |
| V5 (LGEU pre-filter + MAGSAC) | 19 | 3 | 22 | +1 |
| V6 (crash isolation) | 19 | 0 | 19 | video regression |
| **V7 (exhaustive COLMAP for video N≤100)** | 19 | 8 | **27** | **best** |
| V8 (GBM LGEU + TOPK=30) | 19 | 3 | 22 | regression (over-merge) |

The 1,872 private images are absent from the public kernel, so they receive identity poses on
public runs; the hidden evaluation reruns the full notebook. The "27/73" figure is a camera
count on the public test, not an IMC25 metric score and not a leaderboard score.

### SIFT/LGEU trained model (LGEU) — measured quality
Source: `sift-lgeu/experiments/lgeu_metrics.json`, `.../lgeu_metrics_v8.json`.

| Model | ROC-AUC | AP | Brier | Recall@20% |
|---|---:|---:|---:|---:|
| LR (V5/V6 baseline) | 0.614 | 0.341 | 0.320 | 0.242 |
| **GBM (V8, shipped `models/lgeu.joblib`)** | **0.690** | 0.390 | 0.227 | 0.320 |

Scene-disjoint CV on 127 official train images (ETs, imc2023_haiper, stairs). Better classifier
quality did not translate into more registered cameras; its benefit was throughput, and a larger
pair budget hurt video scenes (see V8 above).

### SIFT/LGEU pair-selection ablation
Source: `sift-lgeu/experiments/ablation_results.csv`. Selected-positive rate by strategy:
fixed Top-K = 0.550, retrieval-only = 0.696, LGEU naive Top-K = 0.613, full LGEU budget graph
= 0.636.

---

## D. Hypotheses and negative results

- **AdaptiveGraph adaptive rescue** — rejected (negative result): over-merges doppelgänger /
  repetitive scenes; even the scene-preserving fix stays below no-rescue A2.
- **SIFT/LGEU V8 (bigger pair budget)** — regression: TOPK 20→30 merged video sub-scenes.
- **MASt3R on large scenes** — regression: fixed small budget under-connects; the proposed
  (unvalidated) fix is a size-aware budget.
- **A4R045 (pose12) proposal** — *unvalidated*: evidence-grounded continuation of the A4R034
  monotonic `abs_pose_min_num_inliers` trend (30→20→15), but never run; no score is claimed.
  See [`a4r045-optimization-proposal/`](a4r045-optimization-proposal/).

Full detail and the per-scene evidence are in [docs/ABLATIONS.md](docs/ABLATIONS.md) and the
per-project `results/` directories.
