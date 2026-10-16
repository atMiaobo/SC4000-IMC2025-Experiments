#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

from adaptive_imc.config import load_config
from adaptive_imc.pipeline import find_images, process_dataset
from adaptive_imc.submission import rows_for_dataset, write_submission


def parse_args():
    p = argparse.ArgumentParser(description="Adaptive Scene-Graph IMC2025 runner")
    p.add_argument("--mode", choices=["train", "test"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--config", type=Path, default=Path("configs/balanced.yaml"))
    p.add_argument("--dataset", action="append", default=[], help="Dataset name; repeat to run several. Default=all.")
    p.add_argument("--run-name", default=None)
    p.add_argument("--output-root", type=Path, default=Path("outputs"))
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    run_name = args.run_name or f"{args.mode}_{cfg.name}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_root = args.data_dir / args.mode
    if not image_root.exists():
        raise SystemExit(f"Missing {image_root}. See README.md for the expected Kaggle data layout.")

    available = sorted([p.name for p in image_root.iterdir() if p.is_dir()])
    datasets = args.dataset or available
    missing = [d for d in datasets if d not in available]
    if missing:
        raise SystemExit(f"Unknown dataset(s): {missing}. Available: {available}")

    all_rows = []
    summaries = []
    for dataset in datasets:
        image_paths = find_images(image_root / dataset)
        if not image_paths:
            print(f"WARNING: no images in {image_root / dataset}; skipping")
            continue
        preds, diag, summary = process_dataset(
            dataset_name=dataset,
            image_paths=image_paths,
            cfg=cfg,
            work_dir=out_dir / dataset,
        )
        all_rows.extend(rows_for_dataset(dataset, image_paths, preds))
        summaries.append(summary)

    sample = args.data_dir / "sample_submission.csv" if args.mode == "test" else None
    submission = write_submission(
        all_rows,
        out_dir / "submission.csv",
        sample_submission=sample,
    )
    (out_dir / "run_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(f"\nSubmission written: {submission}")
    print(f"Run diagnostics: {out_dir}")


if __name__ == "__main__":
    main()
