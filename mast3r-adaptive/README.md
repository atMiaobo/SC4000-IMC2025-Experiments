# MASt3R Adaptive — dense matching with a reconstruction-aware matching budget

An IMC2025 reconstruction pipeline using **MASt3R** dense correspondence (pretrained, NAVER) in
place of sparse local features, with an independently designed, reconstruction-aware adaptive
matching budget: a cheap trusted base pass, routing of extra matching only to under-connected
nodes, and evidence-gated (scene-preserving) cross-scene merges.

## What's here

```
src/mast3r_imc/        matching_mast3r.py (MASt3R matcher wrapper), pipeline_mast3r.py (baseline + adaptive)
scripts/               run_mast3r.py (local eval), infer_kaggle.py + mast3r_worker.py
                       (crash-isolated Kaggle orchestrator), test_scorer_edgecases.py
kaggle/                MASt3R_Adaptive_Kaggle.ipynb, build_notebook.py, kernel-metadata.json
results/               cmp/fbk/repro_large/smoke results.json, per-scene summaries, submissions
docs/                  REPORT.md, THIRD_PARTY_NOTICES.md
```

> **Cross-project dependency:** this pipeline reuses the **AdaptiveGraph** package
> (`adaptive_imc`: retrieval, MAGSAC, COLMAP wrapper, graph ops, submission writer) and the IMC25
> metric. See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for how to make it importable.

## Method summary

```
DINOv2 retrieval → MASt3R semi-dense matching (per candidate pair) → MAGSAC verification
  → graph (triangle-support rehab) → COLMAP multi-model SfM → graph label propagation → submission.csv
```

The adaptive budget: (1) base pass with a *small* retrieval budget (top-6) to build a trusted
graph; (2) spend extra MASt3R matching only on low-degree nodes; (3) a cross-component merge of
two established scenes requires multi-view (triangle) evidence, never a single high-match edge.

## Results (summary)

- **Local, 3-scene official metric:** adaptive **51.44** vs baseline **45.06** (+6.4), driven by
  clusterness (50.7 → 93.6). It regresses on the large `fbk_vineyard` (23.7 → 16.6): the
  conservative budget under-connects a big scene. On the same 3 scenes AdaptiveGraph A2 = 55.1,
  so MASt3R-adaptive does not beat AdaptiveGraph in aggregate here.
- **Kaggle private leaderboard:** **22.97**, from team competition records.
- Feasibility: runs on a 16 GB GPU via the pure-PyTorch RoPE fallback (no `curope`), ~0.7 s/pair,
  peak ~3.7 GB VRAM.

Full tables: [../RESULTS.md](../RESULTS.md) · ablation detail: [../docs/ABLATIONS.md](../docs/ABLATIONS.md)
· full writeup: [docs/REPORT.md](docs/REPORT.md).

## License note

MASt3R weights are **CC-BY-NC-SA-4.0 (non-commercial)**. This pipeline is for non-commercial
research/competition use only. Weights are not committed here — see
[../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Kaggle

- Kernel: `miaobo/mast3r-adaptive-orig` (v4) — original competition
- Bundle: `miaobo/mast3r-imc25-bundle` (weights + repo + DINOv2 cache + wheels)
