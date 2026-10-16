#!/usr/bin/env python3
"""Robust Kaggle test-mode inference for the MASt3R pipeline.

Robustness design (handles the hidden-eval 'Notebook Threw Exception'):
  * A COMPLETE, valid fallback submission is written from sample_submission BEFORE any
    reconstruction, so a well-formed /kaggle/working/submission.csv always exists.
  * Each dataset is reconstructed in an ISOLATED SUBPROCESS with a wall-clock timeout, so a
    native COLMAP segfault / CUDA OOM-kill / unhandled exception kills only that subprocess;
    the orchestrator keeps the fallback for that dataset and continues.
  * The submission is re-written after every dataset, so partial progress is never lost and
    successful datasets are preserved even if a later one fails.
  * Output is always sample-aligned, finite, exactly one row per required id.
No reconstruction algorithm is changed.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--adaptive-src", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["baseline", "adaptive"], default="adaptive")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/submission.csv"))
    ap.add_argument("--per-dataset-timeout", type=int, default=3000)
    ap.add_argument("--work-root", type=Path, default=Path("/kaggle/working/mast3r_out"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.adaptive_src))
    sys.path.insert(0, str(args.project / "src"))
    from adaptive_imc.submission import write_submission  # noqa
    from adaptive_imc.pipeline import find_images  # noqa

    sample = args.data_dir / "sample_submission.csv"
    test_root = args.data_dir / "test"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    # 1) Write a complete valid fallback submission up front (all identity/outlier via writer).
    all_rows: list[dict] = []
    write_submission(all_rows, args.out, sample_submission=sample)
    print(f"[infer] wrote initial fallback submission ({args.out})")

    datasets = sorted(p.name for p in test_root.iterdir() if p.is_dir()) if test_root.is_dir() else []
    # process smaller collections first -> maximise coverage before any time pressure
    datasets.sort(key=lambda d: len(find_images(test_root / d)))
    print("[infer] test datasets (small->large):", datasets)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([
        str(args.project / "src"), str(args.adaptive_src),
        str(args.project / "third_party" / "mast3r"),
        str(args.project / "third_party" / "mast3r" / "dust3r"),
        str(args.project / "third_party" / "mast3r" / "dust3r" / "croco"),
        env.get("PYTHONPATH", ""),
    ])

    for ds in datasets:
        out_rows = args.work_root / f"{ds}.rows.json"
        cmd = [sys.executable, str(args.project / "scripts" / "mast3r_worker.py"),
               "--dataset", ds, "--dataset-dir", str(test_root / ds),
               "--project", str(args.project), "--adaptive-src", str(args.adaptive_src),
               "--model-dir", str(args.model_dir), "--mode", args.mode,
               "--work-dir", str(args.work_root / ds), "--out-rows", str(out_rows)]
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, timeout=args.per_dataset_timeout)
            ok = (r.returncode == 0 and out_rows.exists())
        except subprocess.TimeoutExpired:
            ok = False; print(f"[infer] {ds}: TIMEOUT after {args.per_dataset_timeout}s -> fallback")
        except Exception as e:
            ok = False; print(f"[infer] {ds}: orchestration error {e!r} -> fallback")
        if ok:
            try:
                rows = json.loads(out_rows.read_text())
                all_rows.extend(rows)
                print(f"[infer] {ds}: OK {len(rows)} rows in {time.time()-t0:.0f}s")
            except Exception as e:
                print(f"[infer] {ds}: could not read rows ({e!r}) -> fallback")
        else:
            print(f"[infer] {ds}: FAILED (rc/segfault/timeout) in {time.time()-t0:.0f}s -> fallback kept")
        # re-write submission after every dataset (crash-safe partial progress)
        write_submission(all_rows, args.out, sample_submission=sample)

    # final validation
    import pandas as pd, numpy as np
    sub = pd.read_csv(args.out); smp = pd.read_csv(sample)
    assert len(sub) == len(smp), f"row count {len(sub)} != sample {len(smp)}"
    for c in ("rotation_matrix", "translation_vector"):
        bad = sub[c].astype(str).str.contains("nan|inf", case=False).sum()
        assert bad == 0, f"{bad} non-finite values in {c}"
    assert not sub[["dataset","scene","image","rotation_matrix","translation_vector"]].isna().any().any()
    print(f"[infer] FINAL submission OK: {len(sub)} rows, finite, sample-aligned -> {args.out}")


if __name__ == "__main__":
    main()
