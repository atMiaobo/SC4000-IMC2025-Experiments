"""Tests for V18: relaxed COLMAP mapper settings for video scenes.

V18 keeps V16's scene clustering (GBM num_gap=0 + Lowe threshold=30 for video)
and introduces a new SfM function (run_sfm_v18) with relaxed COLMAP mapper
thresholds for video scenes:
  - init_min_num_inliers: 5 → 3
  - init_min_tri_angle:   1.0 → 0.5 degrees
  - abs_pose_min_num_inliers: 4 → 3

These tests verify the scene clustering contract and that the SfM alias is wired up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imc2025_lgeu.scene_clustering import discover_scenes_v18, discover_scenes_v16


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------

def _tiny_image_paths(tmp_path: Path, n: int = 3) -> dict[str, Path]:
    """Create n small valid grayscale PNG images and return {name: path}."""
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not available")

    paths = {}
    for i in range(n):
        name = f"img_{i:03d}.jpg"
        fpath = tmp_path / name
        img = np.zeros((64, 64), dtype=np.uint8)
        img[10:30, 10:30] = 200  # a bright square to give SIFT something to detect
        cv2.imwrite(str(fpath), img)
        paths[name] = fpath
    return paths


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v18 returns the correct tuple contract
# ---------------------------------------------------------------------------

def test_discover_scenes_v18_returns_correct_types(tmp_path):
    """discover_scenes_v18 must return (scene_map dict, edge_data dict)."""
    image_paths = _tiny_image_paths(tmp_path, n=3)

    result = discover_scenes_v18(image_paths, lgeu_model_payload=None, verbose=False)

    assert isinstance(result, tuple), "Must return a tuple"
    assert len(result) == 2, "Tuple must have exactly 2 elements"

    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "First element must be dict (scene_map)"
    assert isinstance(edge_data, dict), "Second element must be dict (edge_data)"

    # Every input image must appear in scene_map
    for name in image_paths:
        assert name in scene_map, f"Image '{name}' missing from scene_map"

    # All scene_map values must be strings
    for name, label in scene_map.items():
        assert isinstance(label, str), f"Scene label for '{name}' must be a string"


# ---------------------------------------------------------------------------
# Test 2: single-image dataset gets scene_0, not outliers
# ---------------------------------------------------------------------------

def test_discover_scenes_v18_single_image(tmp_path):
    """A single-image dataset should be assigned scene_0 (can't be outlier)."""
    image_paths = _tiny_image_paths(tmp_path, n=1)

    scene_map, edge_data = discover_scenes_v18(image_paths, lgeu_model_payload=None)

    assert len(scene_map) == 1
    only_label = next(iter(scene_map.values()))
    assert only_label == "scene_0", f"Single image should be 'scene_0', got '{only_label}'"
    assert edge_data == {}, "No edges possible with 1 image"


# ---------------------------------------------------------------------------
# Test 3: v18 scene clustering is identical to v16 (same code path)
# ---------------------------------------------------------------------------

def test_discover_scenes_v18_matches_v16(tmp_path):
    """discover_scenes_v18 should produce the same scene_map as v16 (same clustering)."""
    image_paths = _tiny_image_paths(tmp_path, n=4)

    scene_map_18, _ = discover_scenes_v18(image_paths, lgeu_model_payload=None)
    scene_map_16, _ = discover_scenes_v16(image_paths, lgeu_model_payload=None)

    assert scene_map_18 == scene_map_16, (
        "V18 scene clustering should match V16 exactly (clustering is unchanged in V18)"
    )


# ---------------------------------------------------------------------------
# Test 4: run_sfm_per_scene_v18 is importable and is a distinct callable
# ---------------------------------------------------------------------------

def test_run_sfm_per_scene_v18_importable():
    """run_sfm_per_scene_v18 must be importable and callable (not the same object as v7)."""
    from imc2025_lgeu.sfm import run_sfm_per_scene_v18, run_sfm_per_scene_v7

    assert callable(run_sfm_per_scene_v18), "run_sfm_per_scene_v18 must be callable"
    # V18 defines a NEW function (not an alias of v7) since it uses relaxed SfM settings
    assert run_sfm_per_scene_v18 is not run_sfm_per_scene_v7, (
        "run_sfm_per_scene_v18 should be a distinct function from run_sfm_per_scene_v7 "
        "(it uses relaxed COLMAP mapper settings)"
    )
