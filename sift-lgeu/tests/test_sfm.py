"""Tests for PyCOLMAP SfM module.

Requires pycolmap to be installed and at least 3 another_et training images
present in data/official/train/ETs/.
"""
from pathlib import Path

import numpy as np
import pytest

TRAIN_ETS = Path("data/official/train/ETs")
ANOTHER_ET_IMAGES = sorted(TRAIN_ETS.glob("another_et_*.png")) if TRAIN_ETS.is_dir() else []


@pytest.mark.skipif(
    len(ANOTHER_ET_IMAGES) < 3,
    reason="need >=3 another_et training images in data/official/train/ETs/",
)
def test_sfm_registers_cameras():
    from imc2025_lgeu.sfm import run_sfm

    # Use just the first 5 images for speed
    images = ANOTHER_ET_IMAGES[:5]
    image_paths = {p.name: p for p in images}
    poses, info = run_sfm(image_paths, verbose=False)

    assert info.get("registered", 0) >= 3, (
        f"Expected >=3 cameras registered, got {info.get('registered', 0)}. Info: {info}"
    )
    assert info.get("num_points3D", 0) > 0, "Expected at least 1 3D point"

    for name, (R, t) in poses.items():
        assert R.shape == (3, 3), f"R shape mismatch for {name}"
        assert t.shape == (3,), f"t shape mismatch for {name}"
        assert np.all(np.isfinite(R)), f"R has non-finite values for {name}"
        assert np.all(np.isfinite(t)), f"t has non-finite values for {name}"
        det = np.linalg.det(R)
        assert abs(det - 1.0) < 0.05 or np.allclose(R, np.eye(3)), (
            f"R for {name} is not a rotation (det={det:.4f})"
        )
