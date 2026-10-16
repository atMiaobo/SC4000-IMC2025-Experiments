# Final minimal workflow

The validated local baseline is `configs/final.yaml` (same algorithm as the tested `fast.yaml`).
It achieved a 60.8912 mean local train score. This is a local-train figure, not a Kaggle score.

Do not use `balanced.yaml` for the final submission unless you intentionally want a much slower
research run.

For a hidden-test score, submit `kaggle/AdaptiveGraph_FAST_Kaggle.ipynb` to the **Image Matching
Challenge 2025 Ongoing** competition. It is essentially identical to the original challenge but
exposes one rolling leaderboard over the full test data.

The Kaggle notebook expects these inputs to be attached:
- Competition: Image Matching Challenge 2025 Ongoing
- Dataset: `imc2024-packages-pycolmap-lightglue-rerun-kornia`
- Model: DINO v2 / base / V1
- Model: ALIKED / aliked-n16 / V1
- Model: LightGlue / aliked / V1

Use a GPU, disable internet, Save Version -> Save & Run All. The notebook writes
`/kaggle/working/submission.csv`. Submit that output to the competition.
