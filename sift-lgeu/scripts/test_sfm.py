"""Test PyCOLMAP multiview reconstruction on a small official training scene.

Uses ETs/another_et (10 images) as the test scene — known same-scene from clustering test.
Reports: cameras registered, 3D points, and a sanity check that R is valid rotation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imc2025_lgeu.sfm import run_sfm


def main():
    ets_train = REPO / "data" / "official" / "train" / "ETs"
    if not ets_train.is_dir():
        print(f"ERROR: training ETs dir not found: {ets_train}")
        sys.exit(1)

    # Use "another_et" scene — 10 images with strong overlap
    scene_images = sorted(ets_train.glob("another_et_*.png"))
    if len(scene_images) < 3:
        print(f"ERROR: need at least 3 another_et images, found {len(scene_images)}")
        sys.exit(1)

    print(f"PyCOLMAP SfM test — another_et scene ({len(scene_images)} images)")
    print(f"Images: {[p.name for p in scene_images]}")
    print()

    image_paths = {p.name: p for p in scene_images}
    poses, info = run_sfm(image_paths, verbose=True)

    print(f"\nRegistration summary:")
    print(f"  Registered   : {info.get('registered', 0)} / {info.get('total', len(image_paths))}")
    print(f"  3D points    : {info.get('num_points3D', 'n/a')}")
    if "error" in info:
        print(f"  Error        : {info['error']}")

    n_valid_R = 0
    n_identity = 0
    for name, (R, t) in poses.items():
        is_identity = np.allclose(R, np.eye(3)) and np.allclose(t, 0)
        if is_identity:
            n_identity += 1
        else:
            det = np.linalg.det(R)
            valid = abs(det - 1.0) < 0.01 and np.allclose(R @ R.T, np.eye(3), atol=1e-4)
            if valid:
                n_valid_R += 1
            else:
                print(f"  WARNING: {name} has invalid rotation (det={det:.4f})")

    print(f"  Valid poses  : {n_valid_R} (non-identity, valid rotation)")
    print(f"  Identity     : {n_identity} (not registered / fallback)")

    # Show a few registered poses
    print()
    print("Sample poses (first 3 registered):")
    shown = 0
    for name, (R, t) in sorted(poses.items()):
        if not (np.allclose(R, np.eye(3)) and np.allclose(t, 0)):
            print(f"  {name}")
            print(f"    R = {R[0].tolist()}")
            print(f"    t = {t.tolist()}")
            shown += 1
            if shown >= 3:
                break

    n_registered = info.get("registered", 0)
    passed = n_registered >= 3
    print()
    print(f"SfM test: {'PASS' if passed else 'FAIL'} "
          f"({n_registered}/{len(image_paths)} cameras registered, need >=3)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
