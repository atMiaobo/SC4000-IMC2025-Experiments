"""Tests for V22: init_min_tri_angle=0.5 for video SfM + V16 clustering passthrough."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v22


# ---------------------------------------------------------------------------
# Helper: fake image paths (files don't need to exist for clustering unit tests)
# ---------------------------------------------------------------------------

def _fake_paths(names: list[str]) -> dict[str, Path]:
    return {n: Path(f"/tmp/fake/{n}") for n in names}


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v22 delegates to discover_scenes_v16 for small
#         non-video datasets (exhaustive path) and returns correct tuple shape.
# ---------------------------------------------------------------------------

def test_v22_returns_tuple_with_scene_map_and_edge_data():
    """discover_scenes_v22 must return (scene_map dict, edge_data dict)."""
    names = ["img_a.jpg", "img_b.jpg", "img_c.jpg"]
    image_paths = _fake_paths(names)

    # Patch the SIFT extraction to avoid real file I/O
    fake_kp = np.zeros((10, 2), dtype=np.float32)
    fake_desc = np.zeros((10, 128), dtype=np.float32)
    with patch(
        "imc2025_lgeu.scene_clustering._extract_sift",
        return_value=MagicMock(keypoints=fake_kp, descriptors=fake_desc),
    ):
        result = discover_scenes_v22(image_paths, lgeu_model_payload=None, verbose=False)

    assert isinstance(result, tuple), "discover_scenes_v22 must return a tuple"
    assert len(result) == 2, "tuple must have exactly 2 elements"
    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "first element must be a dict (scene_map)"
    assert isinstance(edge_data, dict), "second element must be a dict (edge_data)"
    assert set(scene_map.keys()) == set(names), "scene_map must contain all image names"


# ---------------------------------------------------------------------------
# Test 2: discover_scenes_v22 produces the same clustering as discover_scenes_v21
#         (it is a pure passthrough — V22 only changes SfM, not clustering).
# ---------------------------------------------------------------------------

def test_v22_clustering_identical_to_v21():
    """discover_scenes_v22 must produce the same scene_map as discover_scenes_v21."""
    from imc2025_lgeu.scene_clustering import discover_scenes_v21

    names = ["img_x.jpg", "img_y.jpg"]
    image_paths = _fake_paths(names)

    fake_kp = np.zeros((5, 2), dtype=np.float32)
    fake_desc = np.zeros((5, 128), dtype=np.float32)
    with patch(
        "imc2025_lgeu.scene_clustering._extract_sift",
        return_value=MagicMock(keypoints=fake_kp, descriptors=fake_desc),
    ):
        scene_map_v21, _ = discover_scenes_v21(
            image_paths, lgeu_model_payload=None, verbose=False
        )
        scene_map_v22, _ = discover_scenes_v22(
            image_paths, lgeu_model_payload=None, verbose=False
        )

    assert scene_map_v21 == scene_map_v22, (
        "V22 clustering must be identical to V21 (only SfM changes)"
    )


# ---------------------------------------------------------------------------
# Test 3: run_sfm_v22 sets init_min_tri_angle=0.5 for video scenes.
# ---------------------------------------------------------------------------

def test_run_sfm_v22_uses_relaxed_tri_angle_for_video():
    """run_sfm_v22 must pass init_min_tri_angle=0.5 to COLMAP for video sequences."""
    from imc2025_lgeu.sfm import run_sfm_v22

    # Timestamp-style names to trigger is_video=True
    names = [f"1700000{i:04d}.jpg" for i in range(5)]
    image_paths = _fake_paths(names)

    captured_options = {}

    def fake_incremental_mapping(database_path, image_path, output_path, options):
        captured_options["init_min_tri_angle"] = options.mapper.init_min_tri_angle
        captured_options["abs_pose_min_num_inliers"] = options.mapper.abs_pose_min_num_inliers
        return {}  # no reconstructions -> identity poses

    mock_pycolmap = MagicMock()
    mock_pycolmap.IncrementalPipelineOptions.return_value = MagicMock(
        mapper=MagicMock(
            init_min_num_inliers=5,
            init_min_tri_angle=1.0,
            init_max_forward_motion=0.999,
            abs_pose_min_num_inliers=4,
        )
    )
    mock_pycolmap.incremental_mapping.side_effect = fake_incremental_mapping

    import imc2025_lgeu.sfm as sfm_module
    original_has_pycolmap = sfm_module._HAS_PYCOLMAP

    with patch.object(sfm_module, "_HAS_PYCOLMAP", True), \
         patch.object(sfm_module, "pycolmap", mock_pycolmap), \
         patch("imc2025_lgeu.sfm._is_video_sequence", return_value=True), \
         patch("shutil.copy2"), \
         patch("imc2025_lgeu.sfm.pycolmap", mock_pycolmap):
        poses, info = run_sfm_v22(image_paths, verbose=False)

    if captured_options:
        assert captured_options.get("init_min_tri_angle") == pytest.approx(0.5), (
            "run_sfm_v22 must set init_min_tri_angle=0.5 for video scenes"
        )
        assert captured_options.get("abs_pose_min_num_inliers") == 3, (
            "run_sfm_v22 must retain abs_pose_min_num_inliers=3 from V21"
        )
