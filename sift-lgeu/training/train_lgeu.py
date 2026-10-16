#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imc2025_lgeu.lgeu import save_model, train_lgeu
from imc2025_lgeu.pairs import build_pair_records, compute_descriptor_table, records_to_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=Path("data/raw/train_labels.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/official"))
    parser.add_argument("--model-out", type=Path, default=Path("models/lgeu.joblib"))
    parser.add_argument("--pairs-out", type=Path, default=Path("experiments/pair_training_table.csv"))
    parser.add_argument("--metrics-out", type=Path, default=Path("experiments/lgeu_metrics.json"))
    parser.add_argument("--max-pairs-per-dataset", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model-type", choices=["gbm", "lr"], default="gbm",
                        help="Classifier type: gbm (HistGradientBoosting, default) or lr (LogisticRegression)")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    meta, descs = compute_descriptor_table(labels, args.image_root, split="train")
    if meta.empty:
        raise SystemExit("no official train images were available under --image-root")
    joined = meta.merge(labels, on=["dataset", "scene", "image"], how="left")
    records = build_pair_records(joined, descs, max_pairs_per_dataset=args.max_pairs_per_dataset, seed=args.seed)
    pair_df = records_to_frame(records)
    if pair_df["label"].nunique(dropna=True) < 2:
        raise SystemExit("pair labels do not contain both classes; cannot train LGEU honestly")
    model, metrics = train_lgeu(pair_df, seed=args.seed, model_type=args.model_type)
    metrics.update(
        {
            "available_train_images": int(len(meta)),
            "available_datasets": sorted(meta["dataset"].unique().tolist()),
            "label_definition": "positive iff same non-outlier train scene and translation distance in [0.05, 5.0]; features are image-only descriptors/name/metadata available before matching",
            "model_type": args.model_type,
        }
    )
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(args.pairs_out, index=False)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model(model, args.model_out, metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
