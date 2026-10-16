#!/usr/bin/env python3
"""Single-dataset MASt3R worker, run as an isolated subprocess so that any failure
(Python exception, CUDA OOM-kill, or native COLMAP segfault) dies WITH the subprocess
and cannot bring down the main inference. Writes predicted rows to a JSON file on success.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--adaptive-src", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["baseline", "adaptive"], default="adaptive")
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-rows", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(args.adaptive_src))
    sys.path.insert(0, str(args.project / "src"))
    import torch
    from adaptive_imc.config import ColmapConfig
    from adaptive_imc.submission import rows_for_dataset
    from adaptive_imc.pipeline import find_images
    from mast3r_imc.pipeline_mast3r import process_dataset_mast3r, DEFAULT_CFG
    from mast3r_imc.matching_mast3r import MASt3RMatcher

    paths = find_images(args.dataset_dir)
    if not paths:
        args.out_rows.write_text(json.dumps([]))
        return

    matcher = MASt3RMatcher(args.model_dir, args.project / "third_party", device="cuda")
    preds, summary = process_dataset_mast3r(
        dataset_name=args.dataset, image_paths=paths, matcher=matcher, cfg=DEFAULT_CFG,
        colmap_cfg=ColmapConfig(), work_dir=args.work_dir, adaptive=(args.mode == "adaptive"))
    rows = rows_for_dataset(args.dataset, paths, preds)
    args.out_rows.write_text(json.dumps(rows))
    # free GPU before exit (defensive)
    try:
        matcher.clear_cache(); del matcher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print(f"[worker] {args.dataset}: {len(rows)} rows, registered={summary.get('colmap',{}).get('num_registered')}")


if __name__ == "__main__":
    main()
