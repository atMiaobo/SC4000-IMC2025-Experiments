from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def identity_pose() -> tuple[np.ndarray, np.ndarray]:
    return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def estimate_relative_pose(path_a: Path, path_b: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    img_a = cv2.imread(str(path_a), cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_GRAYSCALE)
    if img_a is None or img_b is None:
        return *identity_pose(), {"status": "missing_image", "matches": 0, "inliers": 0}
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=7)
    kpa, desa = orb.detectAndCompute(img_a, None)
    kpb, desb = orb.detectAndCompute(img_b, None)
    if desa is None or desb is None or len(kpa) < 8 or len(kpb) < 8:
        return *identity_pose(), {"status": "few_keypoints", "matches": 0, "inliers": 0}
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(desa, desb, k=2)
    good = [m for m, n in raw if n is not None and m.distance < 0.78 * n.distance]
    if len(good) < 8:
        return *identity_pose(), {"status": "few_matches", "matches": len(good), "inliers": 0}
    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    focal = float(max(img_a.shape + img_b.shape))
    pp = (img_a.shape[1] / 2.0, img_a.shape[0] / 2.0)
    essential, mask = cv2.findEssentialMat(pts_a, pts_b, focal=focal, pp=pp, method=cv2.RANSAC, prob=0.999, threshold=1.5)
    if essential is None or mask is None:
        return *identity_pose(), {"status": "essential_failed", "matches": len(good), "inliers": 0}
    inliers, rotation, translation, pose_mask = cv2.recoverPose(essential, pts_a, pts_b, focal=focal, pp=pp, mask=mask)
    if inliers < 8:
        return *identity_pose(), {"status": "pose_failed", "matches": len(good), "inliers": int(inliers)}
    return rotation.astype(np.float64), translation.reshape(3).astype(np.float64), {
        "status": "ok",
        "matches": len(good),
        "inliers": int(inliers),
    }


def chain_scene_poses(image_paths: dict[str, Path], edges: pd.DataFrame) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict]]:
    poses = {image: identity_pose() for image in image_paths}
    adjacency: dict[str, list[tuple[str, np.ndarray, np.ndarray, dict]]] = defaultdict(list)
    diagnostics: list[dict] = []
    for row in edges.itertuples(index=False):
        if row.image_a not in image_paths or row.image_b not in image_paths:
            continue
        rotation, translation, info = estimate_relative_pose(image_paths[row.image_a], image_paths[row.image_b])
        rec = {"image_a": row.image_a, "image_b": row.image_b, **info}
        diagnostics.append(rec)
        if info["status"] != "ok":
            continue
        adjacency[row.image_a].append((row.image_b, rotation, translation, info))
        adjacency[row.image_b].append((row.image_a, rotation.T, -rotation.T @ translation, info))
    visited = set()
    for start in sorted(image_paths):
        if start in visited:
            continue
        poses[start] = identity_pose()
        visited.add(start)
        queue = deque([start])
        while queue:
            src = queue.popleft()
            r_src, t_src = poses[src]
            for dst, r_rel, t_rel, _ in adjacency[src]:
                if dst in visited:
                    continue
                poses[dst] = (r_rel @ r_src, r_rel @ t_src + t_rel)
                visited.add(dst)
                queue.append(dst)
    return poses, diagnostics
