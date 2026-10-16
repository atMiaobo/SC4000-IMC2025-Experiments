#!/usr/bin/env python3
"""Run an ablation ladder over a dataset subset and score each config with the
official IMC25 metric. Writes a consolidated results.json + prints a markdown table.

Usage:
  python scripts/run_ablation.py \
      --config configs/ablation_A0.yaml configs/ablation_A1.yaml configs/ablation_A2.yaml \
      --dataset ETs amy_gardens fbk_vineyard imc2023_haiper stairs \
      --tag ladder
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd

from adaptive_imc.config import load_config
from adaptive_imc.pipeline import find_images, process_dataset
from adaptive_imc.submission import rows_for_dataset, write_submission


def load_metric(path: Path):
    spec = importlib.util.spec_from_file_location("imc25metric", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def score_submission(root: Path, submission: Path, data_dir: Path) -> dict:
    metric = load_metric(root / "third_party" / "imc25metric.py")
    sub = pd.read_csv(submission)
    datasets = set(sub["dataset"].unique())
    gt = pd.read_csv(data_dir / "train_labels.csv")
    th = pd.read_csv(data_dir / "train_thresholds.csv")
    gt = gt[gt["dataset"].isin(datasets)].copy()
    th = th[th["dataset"].isin(datasets)].copy()
    with tempfile.TemporaryDirectory(prefix="abl_score_") as td:
        td = Path(td)
        gt.to_csv(td / "gt.csv", index=False)
        th.to_csv(td / "th.csv", index=False)
        return metric.score(
            gt_csv=str(td / "gt.csv"),
            user_csv=str(submission),
            thresholds_csv=str(td / "th.csv"),
            mask_csv=None,
            combo_mode="harmonic",
            inl_cf=0,
            strict_cf=-1,
            verbose=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--tag", default="ladder")
    ap.add_argument("--output-root", type=Path, default=Path("outputs/ablation"))
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    image_root = args.data_dir / "train"
    all_results = []

    for cfg_path in args.config:
        cfg = load_config(cfg_path)
        run_name = Path(cfg_path).stem
        out_dir = args.output_root / args.tag / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n########## CONFIG {run_name} ({cfg.name}) ##########")

        rows, summaries = [], {}
        t0 = time.time()
        for ds in args.dataset:
            paths = find_images(image_root / ds)
            preds, _, summary = process_dataset(
                dataset_name=ds, image_paths=paths, cfg=cfg, work_dir=out_dir / ds
            )
            rows.extend(rows_for_dataset(ds, paths, preds))
            summaries[ds] = summary
        wall = time.time() - t0
        sub_path = out_dir / "submission.csv"
        write_submission(rows, sub_path, sample_submission=None)

        scored = score_submission(root, sub_path, args.data_dir)
        per_ds = {}
        for ds in args.dataset:
            s = summaries[ds]
            colmap = s.get("colmap", {}) or {}
            per_ds[ds] = {
                "score": scored["score"].get(ds),
                "mAA": scored["mAA"].get(ds),
                "clusterness": scored["clusterness"].get(ds),
                "num_images": s.get("num_images"),
                "num_candidate_pairs": s.get("num_candidate_pairs"),
                "num_accepted_pairs": s.get("num_accepted_pairs"),
                "num_rescued_pairs": s.get("num_rescued_pairs"),
                "num_registered": colmap.get("num_registered"),
                "num_models": colmap.get("num_models"),
                "elapsed_s": s.get("elapsed_seconds"),
            }
        mean_score = float(np.mean([v["score"] for v in per_ds.values() if v["score"] is not None]))
        mean_maa = float(np.mean([v["mAA"] for v in per_ds.values() if v["mAA"] is not None]))
        mean_clu = float(np.mean([v["clusterness"] for v in per_ds.values() if v["clusterness"] is not None]))
        rec = {
            "config": run_name,
            "config_name": cfg.name,
            "mean_score": mean_score,
            "mean_mAA": mean_maa,
            "mean_clusterness": mean_clu,
            "wall_seconds": wall,
            "per_dataset": per_ds,
        }
        all_results.append(rec)
        print(f">>> {run_name}: mean_score={mean_score:.3f} mAA={mean_maa:.3f} "
              f"clusterness={mean_clu:.3f} wall={wall:.0f}s")

    out_json = args.output_root / args.tag / "results.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))

    # Markdown summary
    print("\n\n===== ABLATION SUMMARY (subset: %s) =====" % ", ".join(args.dataset))
    print(f"{'config':26s} {'score':>7} {'mAA':>7} {'clust':>7} {'wall_s':>7}")
    for r in all_results:
        print(f"{r['config']:26s} {r['mean_score']:7.2f} {r['mean_mAA']:7.2f} "
              f"{r['mean_clusterness']:7.2f} {r['wall_seconds']:7.0f}")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
