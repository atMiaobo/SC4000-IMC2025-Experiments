#!/usr/bin/env python3
"""Run the MASt3R pipeline (baseline and/or adaptive) on train datasets and score with the
official IMC25 metric. Reuses AdaptiveGraph's submission writer + vendored metric."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AG = Path("/home/mihao/Projects/SC4000-IMC2025-2/IMC2025_AdaptiveGraph")
sys.path.insert(0, str(AG / "src"))
sys.path.insert(0, str(ROOT / "src"))

from adaptive_imc.config import ColmapConfig
from adaptive_imc.submission import rows_for_dataset, write_submission
from adaptive_imc.pipeline import find_images
from mast3r_imc.pipeline_mast3r import process_dataset_mast3r, DEFAULT_CFG
from mast3r_imc.matching_mast3r import MASt3RMatcher


def load_metric():
    path = AG / "third_party" / "imc25metric.py"
    spec = importlib.util.spec_from_file_location("imc25metric", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def score_submission(sub_path: Path, data_dir: Path) -> dict:
    m = load_metric()
    sub = pd.read_csv(sub_path)
    ds = set(sub["dataset"].unique())
    gt = pd.read_csv(data_dir / "train_labels.csv"); gt = gt[gt.dataset.isin(ds)]
    th = pd.read_csv(data_dir / "train_thresholds.csv"); th = th[th.dataset.isin(ds)]
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); gt.to_csv(td / "g.csv", index=False); th.to_csv(td / "t.csv", index=False)
        return m.score(gt_csv=str(td / "g.csv"), user_csv=str(sub_path), thresholds_csv=str(td / "t.csv"),
                       mask_csv=None, combo_mode="harmonic", inl_cf=0, strict_cf=-1, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", choices=["baseline", "adaptive"], default=["baseline"])
    ap.add_argument("--dataset", nargs="+", required=True)
    ap.add_argument("--data-dir", type=Path, default=AG / "data")
    ap.add_argument("--model-dir", type=Path, default=ROOT / "weights" / "mast3r_hf")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    args = ap.parse_args()

    colmap_cfg = ColmapConfig()
    matcher = MASt3RMatcher(args.model_dir, ROOT / "third_party", device="cuda")
    results = []
    for mode in args.mode:
        adaptive = (mode == "adaptive")
        rows, summaries = [], {}
        for ds in args.dataset:
            paths = find_images(args.data_dir / "train" / ds)
            preds, summary = process_dataset_mast3r(
                dataset_name=ds, image_paths=paths, matcher=matcher, cfg=DEFAULT_CFG,
                colmap_cfg=colmap_cfg, work_dir=args.out / args.tag / mode / ds, adaptive=adaptive)
            rows.extend(rows_for_dataset(ds, paths, preds)); summaries[ds] = summary
        out_dir = args.out / args.tag / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = out_dir / "submission.csv"
        write_submission(rows, sub, sample_submission=None)
        scored = score_submission(sub, args.data_dir)
        per = {}
        for ds in args.dataset:
            s = summaries[ds]; c = s.get("colmap", {}) or {}
            per[ds] = {"score": scored["score"].get(ds), "mAA": scored["mAA"].get(ds),
                       "clusterness": scored["clusterness"].get(ds), "num_images": s.get("num_images"),
                       "matched": s.get("num_matched_pairs"), "accepted": s.get("num_accepted_pairs"),
                       "rescued": s.get("num_rescued_pairs"), "registered": c.get("num_registered"),
                       "elapsed_s": s.get("elapsed_seconds")}
        mean = lambda k: float(np.mean([v[k] for v in per.values() if v[k] is not None]))
        rec = {"mode": mode, "mean_score": mean("score"), "mean_mAA": mean("mAA"),
               "mean_clusterness": mean("clusterness"), "per_dataset": per}
        results.append(rec)
        print(f"\n>>> MASt3R {mode}: mean_score={rec['mean_score']:.2f} mAA={rec['mean_mAA']:.2f} "
              f"clust={rec['mean_clusterness']:.2f}")
        for ds, v in per.items():
            print(f"    {ds:16s} score={(v['score'] or -1):5.1f} mAA={(v['mAA'] or -1):5.1f} "
                  f"clust={(v['clusterness'] or -1):5.1f} reg={v['registered']}/{v['num_images']} "
                  f"matched={v['matched']} acc={v['accepted']} resc={v['rescued']} {v['elapsed_s']:.0f}s")

    (args.out / args.tag / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.out / args.tag / 'results.json'}")


if __name__ == "__main__":
    main()
