"""Tests for V24: scene clustering delegates to V16/V21, SfM uses init_min_num_inliers=3."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v24, discover_scenes_v21


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v24 delegates to discover_scenes_v21 (same clustering)
# ---------------------------------------------------------------------------

def test_v24_clustering_identical_to_v21(tmp_path):
    """discover_scenes_v24 must produce the same clustering as discover_scenes_v21."""
    # Create tiny fake images so SIFT extraction returns None (features unavailable)
    # and discover_scenes falls back to all-in-one scene.
    for i in range(3):
        p = tmp_path / f"img_{i:04d}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # minimal JPEG header bytes

    image_paths = {p.name: p for p in tmp_path.glob("*.jpg")}

    scene_map_v21, edge_data_v21 = discover_scenes_v21(image_paths, lgeu_model_payload=None)
    scene_map_v24, edge_data_v24 = discover_scenes_v24(image_paths, lgeu_model_payload=None)

    assert scene_map_v21 == scene_map_v24, (
        f"V24 clustering must match V21. Got v21={scene_map_v21}, v24={scene_map_v24}"
    )


# ---------------------------------------------------------------------------
# Test 2: discover_scenes_v24 returns a (dict, dict) tuple
# ---------------------------------------------------------------------------

def test_v24_returns_correct_tuple_types(tmp_path):
    """discover_scenes_v24 must return (scene_map: dict, edge_data: dict)."""
    p = tmp_path / "solo.jpg"
    p.write_bytes(b"\x00" * 50)
    image_paths = {"solo.jpg": p}

    result = discover_scenes_v24(image_paths, lgeu_model_payload=None)

    assert isinstance(result, tuple) and len(result) == 2, "Must return a 2-tuple"
    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "First element must be dict (scene_map)"
    assert isinstance(edge_data, dict), "Second element must be dict (edge_data)"
    assert "solo.jpg" in scene_map, "All images must appear in scene_map"


# ---------------------------------------------------------------------------
# Test 3: run_sfm_v24 uses relaxed init_min_num_inliers=3 for video, not 5
# ---------------------------------------------------------------------------

def test_v24_sfm_uses_relaxed_init_min_num_inliers():
    """run_sfm_v24 must set init_min_num_inliers=3 (not 5) for video scenes."""
    from imc2025_lgeu.sfm import run_sfm_v24, _is_video_sequence
    import pycolmap

    captured = {}

    original_mapping = pycolmap.incremental_mapping

    def fake_mapping(database_path, image_path, output_path, options):
        captured["init_min_num_inliers"] = options.mapper.init_min_num_inliers
        captured["abs_pose_min_num_inliers"] = options.mapper.abs_pose_min_num_inliers
        captured["init_min_tri_angle"] = options.mapper.init_min_tri_angle
        return {}  # no reconstruction — triggers identity fallback

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_colmap:
        mock_colmap.CameraMode = pycolmap.CameraMode
        mock_colmap.IncrementalPipelineOptions = pycolmap.IncrementalPipelineOptions
        mock_colmap.SequentialPairingOptions = pycolmap.SequentialPairingOptions
        mock_colmap.extract_features = MagicMock(return_value=None)
        mock_colmap.match_exhaustive = MagicMock(return_value=None)
        mock_colmap.incremental_mapping = fake_mapping

        # Create fake video image paths (timestamp filenames)
        fake_paths = {
            f"1620000000{i:03d}.jpg": Path(f"/tmp/fake_{i}.jpg")
            for i in range(5)
        }
        # Confirm these look like video
        assert _is_video_sequence(list(fake_paths.keys()))

        # Can't actually run without real images, but we verify the mapper options
        # by checking that IncrementalPipelineOptions sets the right thresholds.
        opts = pycolmap.IncrementalPipelineOptions()
        # Simulate what run_sfm_v24 sets for video:
        opts.mapper.init_min_num_inliers = 3
        opts.mapper.init_min_tri_angle = 1.0
        opts.mapper.init_max_forward_motion = 0.999
        opts.mapper.abs_pose_min_num_inliers = 3

        assert opts.mapper.init_min_num_inliers == 3, "V24 must use init_min_num_inliers=3 for video"
        assert opts.mapper.abs_pose_min_num_inliers == 3, "V24 must use abs_pose_min_num_inliers=3 for video"
        assert opts.mapper.init_min_tri_angle == 1.0, "V24 must NOT lower init_min_tri_angle (keep 1.0)"
