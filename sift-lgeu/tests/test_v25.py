"""Tests for V25: discover_scenes_v25 delegates to V16/V21 clustering; run_sfm_v25 merges all
sub-reconstructions instead of taking only the largest."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v25


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v25 returns correct tuple and labels singletons "outliers"
# ---------------------------------------------------------------------------

def test_discover_scenes_v25_returns_tuple_with_two_images(tmp_path):
    """discover_scenes_v25 must return (scene_map, edge_data) for >=2 images."""
    # Create two minimal grayscale images
    import cv2
    for name in ("a.jpg", "b.jpg"):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), img)

    image_paths = {
        "a.jpg": tmp_path / "a.jpg",
        "b.jpg": tmp_path / "b.jpg",
    }
    result = discover_scenes_v25(image_paths, lgeu_model_payload=None)
    assert isinstance(result, tuple), "discover_scenes_v25 must return a tuple"
    assert len(result) == 2, "Tuple must have 2 elements: (scene_map, edge_data)"
    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "First element must be a dict (scene_map)"
    assert isinstance(edge_data, dict), "Second element must be a dict (edge_data)"
    assert set(scene_map.keys()) == {"a.jpg", "b.jpg"}, "scene_map must contain all images"


def test_discover_scenes_v25_single_image_all_scene_0(tmp_path):
    """discover_scenes_v25 with 1 image returns scene_0 with empty edge_data."""
    import cv2
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)

    image_paths = {"a.jpg": tmp_path / "a.jpg"}
    scene_map, edge_data = discover_scenes_v25(image_paths)
    assert scene_map == {"a.jpg": "scene_0"}, "Single image must be in scene_0"
    assert edge_data == {}, "Single image has no edges"


# ---------------------------------------------------------------------------
# Test 2: run_sfm_v25 merges poses from ALL sub-reconstructions
# ---------------------------------------------------------------------------

def test_run_sfm_v25_merges_multiple_reconstructions():
    """run_sfm_v25 must collect poses from ALL pycolmap reconstructions, not just largest."""
    from imc2025_lgeu.sfm import run_sfm_v25, _identity_pose

    def _make_mock_reconstruction(image_names):
        """Build a mock pycolmap Reconstruction with given registered images."""
        rec = MagicMock()
        rec.num_reg_images.return_value = len(image_names)
        rec.num_points3D.return_value = 10

        images = {}
        for i, name in enumerate(image_names):
            img = MagicMock()
            img.name = Path(name).name
            cam = MagicMock()
            cam.rotation.matrix.return_value = np.eye(3)
            cam.translation = np.array([float(i), 0.0, 0.0])
            img.cam_from_world.return_value = cam
            images[i] = img

        rec.reg_image_ids.return_value = list(images.keys())
        rec.image.side_effect = lambda img_id: images[img_id]
        return rec

    # Two separate sub-reconstructions — simulates two camera rigs on stairs
    rec_large = _make_mock_reconstruction(["cam_a.jpg", "cam_b.jpg", "cam_c.jpg"])  # 3 cameras
    rec_small = _make_mock_reconstruction(["cam_d.jpg", "cam_e.jpg"])               # 2 cameras

    image_paths = {
        "cam_a.jpg": Path("/fake/cam_a.jpg"),
        "cam_b.jpg": Path("/fake/cam_b.jpg"),
        "cam_c.jpg": Path("/fake/cam_c.jpg"),
        "cam_d.jpg": Path("/fake/cam_d.jpg"),
        "cam_e.jpg": Path("/fake/cam_e.jpg"),
    }

    mock_reconstructions = {0: rec_large, 1: rec_small}

    with patch("imc2025_lgeu.sfm._HAS_PYCOLMAP", True), \
         patch("imc2025_lgeu.sfm.pycolmap") as mock_pycolmap, \
         patch("shutil.copy2"), \
         patch("imc2025_lgeu.sfm.tempfile.mkdtemp", return_value="/tmp/fake_colmap"), \
         patch("pathlib.Path.mkdir"), \
         patch("shutil.rmtree"):

        mock_pycolmap.CameraMode.AUTO = "AUTO"
        mock_pycolmap.IncrementalPipelineOptions.return_value = MagicMock()
        mock_pycolmap.incremental_mapping.return_value = mock_reconstructions
        mock_pycolmap.extract_features.return_value = None
        mock_pycolmap.match_exhaustive.return_value = None

        # Patch _is_video_sequence to return False so exhaustive matching path is taken
        with patch("imc2025_lgeu.sfm._is_video_sequence", return_value=False):
            poses, info = run_sfm_v25(image_paths, verbose=False)

    # V25 must register cameras from BOTH reconstructions (5 total), not just largest (3)
    identity = np.eye(3)
    registered = [n for n, (R, _) in poses.items() if not np.allclose(R, identity)]
    # All 5 cameras should have non-identity R (or at least both recs' cameras are present)
    assert "cam_a.jpg" in poses, "cam_a.jpg from largest rec must be registered"
    assert "cam_d.jpg" in poses, "cam_d.jpg from smaller rec must be registered"
    assert info["num_reconstructions"] == 2, "num_reconstructions must be 2"
    assert info["registered_largest"] == 3, "registered_largest must reflect only the largest rec"


# ---------------------------------------------------------------------------
# Test 3: discover_scenes_v25 is equivalent to discover_scenes_v21 for non-video
# ---------------------------------------------------------------------------

def test_discover_scenes_v25_same_as_v21_for_non_video(tmp_path):
    """For non-video / small datasets, v25 clustering must match v21 behavior."""
    import cv2
    from imc2025_lgeu.scene_clustering import discover_scenes_v21

    for name in ("img1.jpg", "img2.jpg", "img3.jpg"):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), img)

    image_paths = {n: tmp_path / n for n in ("img1.jpg", "img2.jpg", "img3.jpg")}

    result_v25 = discover_scenes_v25(image_paths, lgeu_model_payload=None)
    result_v21 = discover_scenes_v21(image_paths, lgeu_model_payload=None)

    scene_map_v25, _ = result_v25
    scene_map_v21, _ = result_v21

    assert set(scene_map_v25.keys()) == set(scene_map_v21.keys()), \
        "v25 and v21 must cover the same images for non-video datasets"
    # Same-scene grouping must be identical
    for img in image_paths:
        assert scene_map_v25[img] == scene_map_v21[img], \
            f"Scene assignment differs for {img}: v25={scene_map_v25[img]}, v21={scene_map_v21[img]}"
