"""Gate B smoke test: two-image matching and geometric verification.

Uses SIFT + FLANN + RANSAC fundamental matrix (OpenCV 5).
No torch/deep-learning backend available in current environment.
Documents what a pretrained deep-learning matcher (ALIKED+LightGlue) would need.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
ETS_DIR = REPO / "data" / "official" / "test" / "ETs"


def sift_match(path_a: Path, path_b: Path, ratio: float = 0.75) -> dict:
    img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return {"status": "missing_image", "matches": 0, "inliers": 0}

    sift = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.04)
    kpa, desa = sift.detectAndCompute(img_a, None)
    kpb, desb = sift.detectAndCompute(img_b, None)

    if desa is None or desb is None or len(kpa) < 8 or len(kpb) < 8:
        return {"status": "few_keypoints", "kp_a": len(kpa or []), "kp_b": len(kpb or []), "matches": 0, "inliers": 0}

    index_params = {"algorithm": 1, "trees": 5}  # FLANN_INDEX_KDTREE
    search_params = {"checks": 50}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    raw = flann.knnMatch(desa.astype(np.float32), desb.astype(np.float32), k=2)
    good = [m for m, n in raw if n is not None and m.distance < ratio * n.distance]

    if len(good) < 8:
        return {"status": "few_matches", "kp_a": len(kpa), "kp_b": len(kpb), "matches": len(good), "inliers": 0}

    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])

    F, mask = cv2.findFundamentalMat(pts_a, pts_b, cv2.USAC_MAGSAC, ransacReprojThreshold=1.0, confidence=0.999)
    if F is None or mask is None:
        return {"status": "fund_failed", "kp_a": len(kpa), "kp_b": len(kpb), "matches": len(good), "inliers": 0}

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / len(good)

    # Estimate relative pose (assumes equal focal ~ max(dim)/1.2)
    h_a, w_a = img_a.shape
    focal = max(h_a, w_a, img_b.shape[0], img_b.shape[1]) / 1.2
    pp_a = (w_a / 2.0, h_a / 2.0)
    K = np.array([[focal, 0, pp_a[0]], [0, focal, pp_a[1]], [0, 0, 1.0]])
    E = K.T @ F @ K

    # Recover pose
    inlier_pts_a = pts_a[mask.ravel() == 1]
    inlier_pts_b = pts_b[mask.ravel() == 1]
    n_cheiro, R, t, pose_mask = cv2.recoverPose(E, inlier_pts_a, inlier_pts_b, K)

    R_det = np.linalg.det(R)
    t_norm = float(np.linalg.norm(t))

    return {
        "status": "ok",
        "kp_a": len(kpa),
        "kp_b": len(kpb),
        "matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 3),
        "pose_inliers": int(n_cheiro),
        "R_det": round(float(R_det), 6),
        "t_norm": round(t_norm, 6),
        "R_finite": bool(np.all(np.isfinite(R))),
        "t_finite": bool(np.all(np.isfinite(t))),
        "R_is_rotation": bool(abs(R_det - 1.0) < 0.01 and np.allclose(R @ R.T, np.eye(3), atol=1e-4)),
    }


def run():
    images = sorted(ETS_DIR.glob("*.png"))
    if len(images) < 2:
        print("ERROR: need at least 2 images in", ETS_DIR)
        sys.exit(1)

    print(f"Gate B Smoke Test — SIFT + FLANN + RANSAC (Fundamental Matrix + recoverPose)")
    print(f"Using OpenCV {cv2.__version__}, {len(images)} images available in ETs/")
    print()

    # Test 1: Same-scene pair (should match well)
    # another_et images are from same scene
    same_scene_candidates = [p for p in images if "another_et" in p.name]
    if len(same_scene_candidates) >= 2:
        pa, pb = same_scene_candidates[0], same_scene_candidates[1]
    else:
        pa, pb = images[0], images[1]

    print(f"Test 1 (same-scene pair): {pa.name}  <->  {pb.name}")
    t0 = time.time()
    r1 = sift_match(pa, pb)
    elapsed1 = time.time() - t0
    print(f"  Result: {r1}")
    print(f"  Time: {elapsed1:.2f}s")
    pass1 = r1["status"] == "ok" and r1["R_is_rotation"] and r1["inliers"] >= 10
    print(f"  PASS: {pass1}")
    print()

    # Test 2: Cross-scene pair (ET vs another_ET — likely harder)
    et_imgs = [p for p in images if p.name.startswith("et_")]
    another_et_imgs = [p for p in images if "another_et" in p.name]
    if et_imgs and another_et_imgs:
        pa2, pb2 = et_imgs[0], another_et_imgs[0]
        label = "cross-scene pair (et vs another_et)"
    else:
        pa2, pb2 = images[0], images[-1]
        label = "cross-scene pair"

    print(f"Test 2 ({label}): {pa2.name}  <->  {pb2.name}")
    t0 = time.time()
    r2 = sift_match(pa2, pb2)
    elapsed2 = time.time() - t0
    print(f"  Result: {r2}")
    print(f"  Time: {elapsed2:.2f}s")
    pass2 = r2["status"] in ("ok", "few_matches", "few_keypoints")  # failure is expected for cross-scene
    print(f"  PASS (cross-scene expected low matches): {pass2}")
    print()

    # Test 3: outlier image vs same-scene image
    outlier_imgs = [p for p in images if "out_" in p.name]
    if outlier_imgs and same_scene_candidates:
        pa3, pb3 = outlier_imgs[0], same_scene_candidates[0]
        label3 = "outlier vs same-scene"
    else:
        pa3, pb3 = images[-2], images[-1]
        label3 = "two random images"

    print(f"Test 3 ({label3}): {pa3.name}  <->  {pb3.name}")
    t0 = time.time()
    r3 = sift_match(pa3, pb3)
    elapsed3 = time.time() - t0
    print(f"  Result: {r3}")
    print(f"  Time: {elapsed3:.2f}s")
    pass3 = r3["status"] in ("ok", "few_matches", "few_keypoints", "fund_failed")
    print(f"  PASS (any non-crash result): {pass3}")
    print()

    overall = pass1 and pass2 and pass3
    print("=" * 60)
    print(f"Gate B Smoke Test: {'PASS' if overall else 'PARTIAL'}")
    print()
    print("MATCHER ASSESSMENT:")
    print("  Used: SIFT + FLANN + RANSAC (cv2.USAC_MAGSAC) + recoverPose")
    print("  torch/kornia NOT available — ALIKED+LightGlue NOT runnable.")
    print("  To enable deep-learning matchers: install torch via uv in pyproject.toml")
    print()
    print("SAME-SCENE PAIR:")
    if r1["status"] == "ok":
        print(f"  matches={r1['matches']}, inliers={r1['inliers']}, ratio={r1['inlier_ratio']}")
        print(f"  R valid rotation: {r1['R_is_rotation']}, t_norm: {r1['t_norm']}")
    else:
        print(f"  FAILED: {r1['status']}")

    return overall, r1, r2, r3


if __name__ == "__main__":
    ok, r1, r2, r3 = run()
    sys.exit(0 if ok else 1)
