# Competition I/O contract (verify against official Kaggle data)

The **only** competition is:

- Official page: https://www.kaggle.com/competitions/image-matching-challenge-2025/
- Exact slug: `image-matching-challenge-2025`
- Official Kaggle input: competition mounted under `/kaggle/input` with its real path discovered at runtime (do not assume an arbitrary location)
- Expected output: `/kaggle/working/submission.csv`

The *expected*, **not yet independently validated**, ordered submission header is:

`image_id,dataset,scene,image,rotation_matrix,translation_vector`

The original `sample_submission.csv` downloaded from the exact competition is authoritative. If it differs from the header here, stop and report a discrepancy; do **not** silently change competitions or schemas. Inspect actual example rows for matrix/vector serialization, fallback values, and pose conventions. At hidden evaluation, do not assume that visible test images or row counts match the local sample.

Use `train/`, `train_labels.csv`, and `test/` only when the official competition actually provides them. Never train using hidden test labels or the scoring output. Do not treat a sample CSV full of default poses as proof of successful reconstruction.

# Research goal

Build **Learned Geometric Edge Utility + Budgeted Graph Optimization (LGEU)**, independently:

1. Retrieve candidate pairs with licensed, independently sourced public pretrained features (global embeddings or simple hand-crafted features).
2. Train an original lightweight predictor using *training-only labels*, on **features available before the graph stage at inference**.
3. Use its pair scores in a connectivity / budget / redundancy-aware graph optimization; optionally perform local matching and geometric filtering in a second stage.
4. Reconstruct with independently integrated licensed matching/SfM libraries, and serialize actual poses to the competition contract.
5. Evaluate fixed Top-K, non-learned graph, learned ranking alone, and full learned graph, preferably on held-out scenes.

Proposed components are a hypothesis, **not guaranteed to improve Kaggle score**. The trained component should have a clear, measurable role and an actual saved checkpoint. "Unique" means an independently developed, explainable contribution with documented provenance, not guaranteed first-in-the-literature novelty.
