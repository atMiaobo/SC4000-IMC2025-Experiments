"""Tests for V26: discover_scenes_v26 delegates to V16/V21 clustering;
run_sfm_v26 combines init_min_num_inliers=3 with merge-all-sub-reconstructions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v26


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v26 returns correct (scene_map, edge_data) tuple
# ---------------------------------------------------------------------------

def test_discover_scenes_v26_returns_tuple(tmp_path):
    """discover_scenes_v26 must return (scene_map, edge_data) for >=2 images."""
    import cv2
    for name in ("a.jpg", "b.jpg"):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), img)

    image_paths = {
        "a.jpg": tmp_path / "a.jpg",
        "b.jpg": tmp_path / "b.jpg",
    }
    result = discover_scenes_v26(image_paths, lgeu_model_payload=None)
    assert isinstance(result, tuple), "discover_scenes_v26 must return a tuple"
    assert len(result) == 2, "Tuple must have 2 elements: (scene_map, edge_data)"
    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "First element must be a dict (scene_map)"
    assert isinstance(edge_data, dict), "Second element must be a dict (edge_data)"
    assert set(scene_map.keys()) == {"a.jpg", "b.jpg"}, "scene_map must contain all images"


def test_discover_scenes_v26_single_image(tmp_path):
    """discover_scenes_v26 with 1 image returns scene_0 with empty edge_data."""
    import cv2
    img = np.zeros((64, 64), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)

    image_paths = {"a.jpg": tmp_path / "a.jpg"}
    scene_map, edge_data = discover_scenes_v26(image_paths)
    assert scene_map == {"a.jpg": "scene_0"}, "Single image must be in scene_0"
    assert edge_data == {}, "Single image has no edges"


# ---------------------------------------------------------------------------
# Test 2: run_sfm_v26 uses init_min_num_inliers=3 for video scenes
# ---------------------------------------------------------------------------

def test_run_sfm_v26_sets_init_min_num_inliers_3_for_video():
    """run_sfm_v26 must set init_min_num_inliers=3 on mapper options for video scenes."""
    from imc2025_lgeu.sfm import run_sfm_v26

    # Use timestamp-based names so _is_video_sequence returns True
    image_paths = {
        "1234567890_cam.jpg": Path("/fake/1234567890_cam.jpg"),
        "1234567891_cam.jpg": Path("/fake/1234567891_cam.jpg"),
        "1234567892_cam.jpg": Path("/fake/1234567892_cam.jpg"),
    }

    captured_options = []

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        captured_options.append(options)
        return {}

    mock_pipeline_options = MagicMock()
    mock_pipeline_options.mapper = MagicMock()

    with patch("imc2025_lgeu.sfm._HAS_PYCOLMAP", True), \
         patch("imc2025_lgeu.sfm.pycolmap") as mock_pycolmap, \
         patch("shutil.copy2"), \
         patch("imc2025_lgeu.sfm.tempfile.mkdtemp", return_value="/tmp/fake_colmap_v26"), \
         patch("pathlib.Path.mkdir"), \
         patch("shutil.rmtree"):

        mock_pycolmap.CameraMode.AUTO = "AUTO"
        mock_pycolmap.IncrementalPipelineOptions.return_value = mock_pipeline_options
        mock_pycolmap.incremental_mapping.side_effect = fake_incremental_mapping
        mock_pycolmap.extract_features.return_value = None
        mock_pycolmap.match_exhaustive.return_value = None

        poses, info = run_sfm_v26(image_paths, verbose=False)

    # Verify init_min_num_inliers was set to 3 (not 5 as in V21/V25)
    assert mock_pipeline_options.mapper.init_min_num_inliers == 3, (
        "run_sfm_v26 must set init_min_num_inliers=3 for video scenes"
    )
    # Verify abs_pose_min_num_inliers is still 3 (V21 setting)
    assert mock_pipeline_options.mapper.abs_pose_min_num_inliers == 3, (
        "run_sfm_v26 must keep abs_pose_min_num_inliers=3 (V21 setting)"
    )
    # Verify init_min_tri_angle stays at 1.0 (not 0.5 which caused V22 regression)
    assert mock_pipeline_options.mapper.init_min_tri_angle == 1.0, (
        "run_sfm_v26 must keep init_min_tri_angle=1.0 (not 0.5 which caused V22 regression)"
    )
    # no_reconstruction means 0 cameras, no error raised
    assert info.get("error") == "no_reconstruction"


# ---------------------------------------------------------------------------
# Test 3: run_sfm_v26 merges poses from ALL sub-reconstructions (like V25)
# ---------------------------------------------------------------------------

def test_run_sfm_v26_merges_multiple_reconstructions():
    """run_sfm_v26 must collect poses from ALL pycolmap reconstructions, not just the largest."""
    from imc2025_lgeu.sfm import run_sfm_v26

    def _make_mock_reconstruction(image_names, n_pts=10):
        rec = MagicMock()
        rec.num_reg_images.return_value = len(image_names)
        rec.num_points3D.return_value = n_pts
        images = {}
        for i, name in enumerate(image_names):
            img = MagicMock()
            img.name = Path(name).name
            cam = MagicMock()
            cam.rotation.matrix.return_value = np.eye(3) * (i + 2)  # non-identity
            cam.translation = np.array([float(i), 0.0, 0.0])
            img.cam_from_world.return_value = cam
            images[i] = img
        rec.reg_image_ids.return_value = list(images.keys())
        rec.image.side_effect = lambda img_id: images[img_id]
        return rec

    rec_large = _make_mock_reconstruction(["cam_a.jpg", "cam_b.jpg", "cam_c.jpg"])
    rec_small = _make_mock_reconstruction(["cam_d.jpg", "cam_e.jpg"])

    image_paths = {
        "cam_a.jpg": Path("/fake/cam_a.jpg"),
        "cam_b.jpg": Path("/fake/cam_b.jpg"),
        "cam_c.jpg": Path("/fake/cam_c.jpg"),
        "cam_d.jpg": Path("/fake/cam_d.jpg"),
        "cam_e.jpg": Path("/fake/cam_e.jpg"),
    }

    with patch("imc2025_lgeu.sfm._HAS_PYCOLMAP", True), \
         patch("imc2025_lgeu.sfm.pycolmap") as mock_pycolmap, \
         patch("shutil.copy2"), \
         patch("imc2025_lgeu.sfm.tempfile.mkdtemp", return_value="/tmp/fake_colmap_v26b"), \
         patch("pathlib.Path.mkdir"), \
         patch("shutil.rmtree"):

        mock_pycolmap.CameraMode.AUTO = "AUTO"
        mock_pycolmap.IncrementalPipelineOptions.return_value = MagicMock()
        mock_pycolmap.incremental_mapping.return_value = {0: rec_large, 1: rec_small}
        mock_pycolmap.extract_features.return_value = None
        mock_pycolmap.match_exhaustive.return_value = None

        with patch("imc2025_lgeu.sfm._is_video_sequence", return_value=False):
            poses, info = run_sfm_v26(image_paths, verbose=False)

    assert "cam_a.jpg" in poses, "cam_a.jpg from largest rec must be registered"
    assert "cam_d.jpg" in poses, "cam_d.jpg from smaller rec must be present"
    assert info["num_reconstructions"] == 2, "num_reconstructions must be 2"
    assert info["registered_largest"] == 3, "registered_largest must reflect only the largest rec"


# ---------------------------------------------------------------------------
# Test 4: discover_scenes_v26 is equivalent to discover_scenes_v21 for non-video
# ---------------------------------------------------------------------------

def test_discover_scenes_v26_same_as_v21_for_non_video(tmp_path):
    """For non-video / small datasets, v26 clustering must match v21 behavior."""
    import cv2
    from imc2025_lgeu.scene_clustering import discover_scenes_v21

    for name in ("img1.jpg", "img2.jpg", "img3.jpg"):
        img = np.zeros((64, 64), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / name), img)

    image_paths = {n: tmp_path / n for n in ("img1.jpg", "img2.jpg", "img3.jpg")}

    result_v26 = discover_scenes_v26(image_paths, lgeu_model_payload=None)
    result_v21 = discover_scenes_v21(image_paths, lgeu_model_payload=None)

    scene_map_v26, _ = result_v26
    scene_map_v21, _ = result_v21

    assert set(scene_map_v26.keys()) == set(scene_map_v21.keys()), (
        "v26 and v21 must cover the same images for non-video datasets"
    )
    for img in image_paths:
        assert scene_map_v26[img] == scene_map_v21[img], (
            f"Scene assignment differs for {img}: v26={scene_map_v26[img]}, v21={scene_map_v21[img]}"
        )
