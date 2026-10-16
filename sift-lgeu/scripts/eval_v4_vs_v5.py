"""Evaluate V4 vs V5 pipelines on locally available labeled training datasets.

Uses only camera registration count as the primary metric (no pose alignment needed).
Scene discovery accuracy is also measured against ground truth scene labels.

Datasets evaluated:
  - ETs (22 images, 3 scenes) — public test set; LGEU was trained on this
  - stairs (51 images, 2 scenes) — public test set; LGEU was trained on this
  - imc2023_haiper (54 images, 3 scenes) — private test; LGEU was trained on this

NOTE: all 3 datasets were used for LGEU training, so this is IN-SAMPLE evaluation.
The metrics show reconstruction quality, not generalization of LGEU to new domains.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imc2025_lgeu.lgeu import load_model
from imc2025_lgeu.scene_clustering import discover_scenes, discover_scenes_v5
from imc2025_lgeu.sfm import run_sfm_per_scene, run_sfm_per_scene_v5


def load_gt_labels(labels_path: Path) -> dict[str, dict[str, dict]]:
    """Returns {dataset: {image_name: {scene, rotation_matrix, translation_vector}}}."""
    gt: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            gt[row["dataset"]][row["image"]] = {
                "scene": row["scene"],
                "rotation_matrix": row["rotation_matrix"],
                "translation_vector": row["translation_vector"],
            }
    return dict(gt)


def scene_discovery_accuracy(
    scene_map: dict[str, str],
    gt_map: dict[str, str],
) -> dict:
    """Measure scene discovery quality against ground truth.

    Returns pairwise same-scene accuracy: fraction of image pairs that
    the predicted and ground truth agree on (same-scene / diff-scene).
    """
    names = sorted(set(scene_map) & set(gt_map))
    if len(names) < 2:
        return {"pairwise_accuracy": None, "n_pairs": 0}

    correct = 0
    total = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pred_same = scene_map[a] == scene_map[b]
            gt_same = gt_map[a] == gt_map[b]
            if pred_same == gt_same:
                correct += 1
            total += 1
    return {
        "pairwise_accuracy": correct / total if total > 0 else None,
        "n_pairs": total,
        "correct": correct,
    }


def evaluate_dataset(
    dataset: str,
    train_dir: Path,
    gt: dict[str, dict],
    lgeu_payload: dict | None,
    verbose: bool = True,
) -> dict:
    """Run V4 and V5 on a training dataset directory and compare results."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset} ({len(gt)} images with GT labels)")

    # Collect image paths
    image_paths: dict[str, Path] = {}
    for img_name in gt:
        p = train_dir / img_name
        if p.is_file():
            image_paths[img_name] = p
    print(f"  Images found locally: {len(image_paths)}/{len(gt)}")

    if len(image_paths) < 2:
        print("  SKIP: insufficient images")
        return {}

    gt_scene_map = {img: info["scene"] for img, info in gt.items() if img in image_paths}

    results = {}

    # ---- V4 ----
    print(f"\n  --- V4 ---")
    t0 = time.time()
    try:
        scene_map_v4, _edge_data_v4 = discover_scenes(image_paths, verbose=False)
        v4_clustering_time = time.time() - t0

        t1 = time.time()
        poses_v4, sfm_v4 = run_sfm_per_scene(image_paths, scene_map_v4, verbose=False)
        v4_sfm_time = time.time() - t1

        v4_non_id = sum(1 for img in image_paths if poses_v4.get(img, (None,))[0] is not None
                       and _is_non_identity(poses_v4[img]))
        v4_registered = sum(
            s.get("registered", 0) for s in sfm_v4.values()
        )
        v4_3d_pts = sum(s.get("num_points3D", 0) for s in sfm_v4.values())
        v4_scenes = sorted(set(scene_map_v4.values()))
        v4_accuracy = scene_discovery_accuracy(scene_map_v4, gt_scene_map)

        print(f"  Scenes discovered: {v4_scenes} (GT: {sorted(set(gt_scene_map.values()))})")
        print(f"  Scene pairwise accuracy: {v4_accuracy['pairwise_accuracy']:.3f}" if v4_accuracy['pairwise_accuracy'] else "  Scene accuracy: N/A")
        print(f"  Registered cameras: {v4_registered}/{len(image_paths)}")
        print(f"  3D points: {v4_3d_pts}")
        print(f"  Clustering time: {v4_clustering_time:.1f}s | SfM time: {v4_sfm_time:.1f}s | Total: {v4_clustering_time+v4_sfm_time:.1f}s")
        for scene, info in sorted(sfm_v4.items()):
            reg = info.get("registered", 0)
            tot = info.get("total", 0)
            pts = info.get("num_points3D", 0)
            print(f"    {scene}: {reg}/{tot} cams, {pts} pts")

        results["v4"] = {
            "registered": v4_registered,
            "total": len(image_paths),
            "num_points3D": v4_3d_pts,
            "clustering_time": v4_clustering_time,
            "sfm_time": v4_sfm_time,
            "pairwise_accuracy": v4_accuracy["pairwise_accuracy"],
            "scenes_predicted": v4_scenes,
            "sfm_info": sfm_v4,
        }
    except Exception as exc:
        print(f"  V4 ERROR: {exc}")
        results["v4"] = {"error": str(exc)}

    # ---- V5 ----
    print(f"\n  --- V5 ---")
    t0 = time.time()
    try:
        scene_map_v5, _edge_data_v5 = discover_scenes_v5(
            image_paths, lgeu_model_payload=lgeu_payload, verbose=False
        )
        v5_clustering_time = time.time() - t0

        t1 = time.time()
        poses_v5, sfm_v5 = run_sfm_per_scene_v5(image_paths, scene_map_v5, verbose=False)
        v5_sfm_time = time.time() - t1

        v5_registered = sum(s.get("registered", 0) for s in sfm_v5.values())
        v5_3d_pts = sum(s.get("num_points3D", 0) for s in sfm_v5.values())
        v5_scenes = sorted(set(scene_map_v5.values()))
        v5_accuracy = scene_discovery_accuracy(scene_map_v5, gt_scene_map)

        lgeu_used = lgeu_payload is not None and len(image_paths) > 40
        print(f"  LGEU pair filtering: {'YES (N={})'.format(len(image_paths)) if lgeu_used else 'NO (N<={} threshold)'.format(40)}")
        print(f"  Scenes discovered: {v5_scenes} (GT: {sorted(set(gt_scene_map.values()))})")
        print(f"  Scene pairwise accuracy: {v5_accuracy['pairwise_accuracy']:.3f}" if v5_accuracy['pairwise_accuracy'] else "  Scene accuracy: N/A")
        print(f"  Registered cameras: {v5_registered}/{len(image_paths)}")
        print(f"  3D points: {v5_3d_pts}")
        print(f"  Clustering time: {v5_clustering_time:.1f}s | SfM time: {v5_sfm_time:.1f}s | Total: {v5_clustering_time+v5_sfm_time:.1f}s")
        for scene, info in sorted(sfm_v5.items()):
            reg = info.get("registered", 0)
            tot = info.get("total", 0)
            pts = info.get("num_points3D", 0)
            vid = " [video]" if info.get("is_video") else ""
            print(f"    {scene}: {reg}/{tot} cams, {pts} pts{vid}")

        results["v5"] = {
            "registered": v5_registered,
            "total": len(image_paths),
            "num_points3D": v5_3d_pts,
            "clustering_time": v5_clustering_time,
            "sfm_time": v5_sfm_time,
            "pairwise_accuracy": v5_accuracy["pairwise_accuracy"],
            "scenes_predicted": v5_scenes,
            "sfm_info": sfm_v5,
            "lgeu_used": lgeu_used,
        }
    except Exception as exc:
        print(f"  V5 ERROR: {exc}")
        results["v5"] = {"error": str(exc)}

    # ---- Summary ----
    if "error" not in results.get("v4", {}) and "error" not in results.get("v5", {}):
        v4 = results["v4"]
        v5 = results["v5"]
        print(f"\n  --- Comparison ---")
        print(f"  Cameras registered:   V4={v4['registered']}/{v4['total']}, V5={v5['registered']}/{v5['total']}")
        print(f"  3D points:            V4={v4['num_points3D']}, V5={v5['num_points3D']}")
        print(f"  Scene accuracy:       V4={v4['pairwise_accuracy']:.3f}, V5={v5['pairwise_accuracy']:.3f}" if v4['pairwise_accuracy'] and v5['pairwise_accuracy'] else "")
        print(f"  Clustering time:      V4={v4['clustering_time']:.1f}s, V5={v5['clustering_time']:.1f}s ({v4['clustering_time']/max(v5['clustering_time'],0.01):.1f}x slower)")

    return results


def _is_non_identity(pose) -> bool:
    """Check if a pose tuple has a non-identity rotation."""
    if pose is None or len(pose) < 1 or pose[0] is None:
        return False
    rot = pose[0]
    try:
        flat = list(rot.flatten())
        eye = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        return any(abs(a - b) > 1e-6 for a, b in zip(flat, eye))
    except Exception:
        return False


def main():
    train_dir_base = REPO / "data" / "official" / "train"
    labels_path = REPO / "data" / "raw" / "train_labels.csv"
    output_dir = REPO / "outputs" / "eval_v4_v5"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    gt_all = load_gt_labels(labels_path)
    print(f"Loaded GT labels: {sum(len(v) for v in gt_all.values())} images across {len(gt_all)} datasets")

    # Load LGEU model
    model_path = REPO / "models" / "lgeu.joblib"
    lgeu_payload = None
    if model_path.is_file():
        try:
            lgeu_payload = load_model(model_path)
            print(f"LGEU model loaded: {lgeu_payload['metrics'].get('roc_auc', '?'):.3f} ROC AUC")
        except Exception as exc:
            print(f"WARNING: LGEU model load failed: {exc}")
    else:
        print("WARNING: lgeu.joblib not found — V5 will run without LGEU pair filtering")

    # Evaluate on available training datasets
    eval_datasets = ["ETs", "stairs", "imc2023_haiper"]
    all_results = {}

    for ds in eval_datasets:
        train_dir = train_dir_base / ds
        if not train_dir.is_dir():
            print(f"\nSKIP {ds}: training directory not found at {train_dir}")
            continue
        if ds not in gt_all:
            print(f"\nSKIP {ds}: no GT labels")
            continue

        gt_ds = gt_all[ds]
        results = evaluate_dataset(ds, train_dir, gt_ds, lgeu_payload, verbose=True)
        all_results[ds] = results

    # Write results
    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults written to {out_path}")

    # Print aggregate summary
    print(f"\n{'='*60}")
    print("AGGREGATE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<25} {'V4 reg':>8} {'V5 reg':>8} {'V4 pts':>8} {'V5 pts':>8} {'V4 accur':>10} {'V5 accur':>10} {'V4 t(s)':>8} {'V5 t(s)':>8}")
    print("-" * 100)
    for ds, res in all_results.items():
        v4 = res.get("v4", {})
        v5 = res.get("v5", {})
        v4_reg = f"{v4.get('registered','ERR')}/{v4.get('total','?')}"
        v5_reg = f"{v5.get('registered','ERR')}/{v5.get('total','?')}"
        v4_pts = v4.get("num_points3D", "ERR")
        v5_pts = v5.get("num_points3D", "ERR")
        v4_acc = f"{v4.get('pairwise_accuracy','N/A'):.3f}" if isinstance(v4.get("pairwise_accuracy"), float) else "N/A"
        v5_acc = f"{v5.get('pairwise_accuracy','N/A'):.3f}" if isinstance(v5.get("pairwise_accuracy"), float) else "N/A"
        v4_t = f"{v4.get('clustering_time',0)+v4.get('sfm_time',0):.1f}"
        v5_t = f"{v5.get('clustering_time',0)+v5.get('sfm_time',0):.1f}"
        print(f"{ds:<25} {v4_reg:>8} {v5_reg:>8} {v4_pts:>8} {v5_pts:>8} {v4_acc:>10} {v5_acc:>10} {v4_t:>8} {v5_t:>8}")


if __name__ == "__main__":
    main()
