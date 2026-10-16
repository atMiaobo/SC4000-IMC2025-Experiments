"""Evaluate geometric scene discovery on official training data.

Ground truth:
  ETs/      — scene prefix from filename before second underscore:
               another_et_* -> "another_et", et_* -> "et", outliers_* -> "outliers"
  imc2023_haiper/ — bike_* -> "bike", chairs_* -> "chairs", fountain_* -> "fountain"

Reports:
  - Per-dataset pairwise clustering accuracy (same-scene recall, diff-scene precision)
  - Predicted scene map vs. ground truth
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imc2025_lgeu.scene_clustering import evaluate_clustering, GOOD_MATCH_THRESHOLD


def _gt_scene_ETs(image_name: str) -> str:
    """Extract ground-truth scene from ETs image filename."""
    stem = Path(image_name).stem
    parts = stem.split("_")
    if parts[0] == "outliers":
        return "outliers"
    if parts[0] == "another":
        return "another_et"
    return "et"


def _gt_scene_haiper(image_name: str) -> str:
    stem = Path(image_name).stem
    return stem.split("_")[0]


def run_dataset(name: str, img_dir: Path, gt_fn) -> dict:
    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not images:
        print(f"  SKIP {name}: no images found in {img_dir}")
        return {}
    image_paths = {p.name: p for p in images}
    ground_truth = {p.name: gt_fn(p.name) for p in images}

    gt_scenes = sorted(set(ground_truth.values()))
    print(f"\n{'='*60}")
    print(f"Dataset: {name}  ({len(images)} images, GT scenes: {gt_scenes})")
    print(f"Threshold: total_lowe_matches >= {GOOD_MATCH_THRESHOLD}")

    result = evaluate_clustering(image_paths, ground_truth, verbose=True)

    print(f"\nResults:")
    print(f"  Same-scene recall    : {result['same_scene_recall']:.1%}  "
          f"({result['correct_same']}/{result['total_same_pairs']} pairs)")
    print(f"  Diff-scene precision : {result['diff_scene_precision']:.1%}  "
          f"({result['correct_diff']}/{result['total_diff_pairs']} pairs)")

    pred_scenes = sorted(set(result['scene_map'].values()))
    print(f"  Predicted scenes     : {pred_scenes}")

    # Show per-image predictions
    gt_vals = sorted(set(ground_truth.values()))
    for gt_scene in gt_vals:
        imgs_in_gt = [n for n, s in ground_truth.items() if s == gt_scene]
        pred_for_gt = [result['scene_map'][n] for n in imgs_in_gt]
        print(f"  GT '{gt_scene}' ({len(imgs_in_gt)} imgs) -> predicted: {sorted(set(pred_for_gt))}")

    return result


if __name__ == "__main__":
    train_root = REPO / "data" / "official" / "train"
    results = {}

    ets_dir = train_root / "ETs"
    if ets_dir.is_dir():
        results["ETs"] = run_dataset("ETs", ets_dir, _gt_scene_ETs)
    else:
        print(f"ETs training dir not found: {ets_dir}")

    haiper_dir = train_root / "imc2023_haiper"
    if haiper_dir.is_dir():
        results["haiper"] = run_dataset("imc2023_haiper", haiper_dir, _gt_scene_haiper)
    else:
        print(f"haiper training dir not found: {haiper_dir}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    for dname, r in results.items():
        if r:
            print(f"  {dname}: same-scene recall={r['same_scene_recall']:.1%}, "
                  f"diff-scene precision={r['diff_scene_precision']:.1%}")

    # Check at least one dataset passed
    all_good = all(
        r.get('same_scene_recall', 0) >= 0.8 and r.get('diff_scene_precision', 0) >= 0.8
        for r in results.values() if r
    )
    if all_good:
        print("\nSCENE CLUSTERING: PASS (>=80% recall and precision)")
    else:
        print("\nSCENE CLUSTERING: PARTIAL (some metrics below 80%)")

    sys.exit(0 if all_good else 1)
