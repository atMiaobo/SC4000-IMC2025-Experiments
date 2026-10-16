#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np


def load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("imc25metric", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description="Score a train-mode submission with the public IMC25 metric")
    p.add_argument("submission", type=Path)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    metric_path = root / "third_party" / "imc25metric.py"
    if not metric_path.exists():
        subprocess.check_call([sys.executable, str(root / "scripts" / "fetch_metric.py")])

    gt = args.data_dir / "train_labels.csv"
    thresholds = args.data_dir / "train_thresholds.csv"
    for required in (args.submission, gt, thresholds):
        if not required.exists():
            raise SystemExit(f"Missing required file: {required}")

    # If the user ran only ETs (or another subset), score just those datasets.
    sub = pd.read_csv(args.submission)
    datasets = set(sub["dataset"].unique())
    gt_df = pd.read_csv(gt)
    th_df = pd.read_csv(thresholds)
    gt_sub = gt_df[gt_df["dataset"].isin(datasets)].copy()
    th_sub = th_df[th_df["dataset"].isin(datasets)].copy()
    gt_datasets = set(gt_sub["dataset"].unique())
    if datasets != gt_datasets:
        raise SystemExit(f"Submission contains datasets not found in train labels: {sorted(datasets - gt_datasets)}")

    with tempfile.TemporaryDirectory(prefix="imc25_score_") as td:
        td = Path(td)
        gt_path = td / "gt.csv"
        th_path = td / "thresholds.csv"
        gt_sub.to_csv(gt_path, index=False)
        th_sub.to_csv(th_path, index=False)

        metric = load_metric(metric_path)
        result = metric.score(
            gt_csv=str(gt_path),
            user_csv=str(args.submission),
            thresholds_csv=str(th_path),
            mask_csv=None,
            combo_mode="harmonic",
            inl_cf=0,
            strict_cf=-1,
            verbose=True,
        )

    scores = list(result.get("score", {}).values())
    mean_score = float(np.mean(scores)) if scores else float("nan")
    print("\nPer-dataset result:", result)
    print(f"LOCAL IMC25 MEAN SCORE: {mean_score:.4f}")
    if mean_score >= 40:
        print("TARGET STATUS: >= 40 on this local split/profile")
    else:
        print("TARGET STATUS: < 40 on this local split/profile; try configs/high_score.yaml and inspect diagnostics.")


if __name__ == "__main__":
    main()
