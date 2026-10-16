"""Tests for V21: discover_scenes_v21 (V16 clustering) + run_sfm_per_scene_v21."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v21


def _make_dummy_images(names, tmpdir):
    """Write tiny 10x10 JPEG images and return {name: Path} dict."""
    try:
        import cv2
        paths = {}
        for name in names:
            p = tmpdir / name
            img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            cv2.imwrite(str(p), img)
            paths[name] = p
        return paths
    except ImportError:
        pytest.skip("cv2 not available")


def test_discover_scenes_v21_returns_correct_types():
    """discover_scenes_v21 must return (dict, dict) for a small non-video dataset."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        names = ["imgA.jpg", "imgB.jpg", "imgC.jpg"]
        image_paths = _make_dummy_images(names, tmpdir)

        scene_map, edge_data = discover_scenes_v21(
            image_paths,
            lgeu_model_payload=None,
            verbose=False,
        )

    assert isinstance(scene_map, dict), "scene_map must be a dict"
    assert isinstance(edge_data, dict), "edge_data must be a dict"
    assert set(scene_map.keys()) == set(names), "scene_map must contain all image names"
    for v in scene_map.values():
        assert isinstance(v, str), "scene labels must be strings"


def test_discover_scenes_v21_all_images_assigned():
    """Every image must receive a scene label (no image dropped)."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        names = [f"frame_{i:04d}.jpg" for i in range(5)]
        image_paths = _make_dummy_images(names, tmpdir)

        scene_map, _ = discover_scenes_v21(
            image_paths,
            lgeu_model_payload=None,
            verbose=False,
        )

    assert set(scene_map.keys()) == set(names)
    assert all(v.startswith("scene_") or v == "outliers" for v in scene_map.values()), (
        "All scene labels must be 'scene_N' or 'outliers'"
    )


def test_discover_scenes_v21_single_image():
    """Single-image datasets are handled gracefully (scene_0 assigned)."""
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        image_paths = _make_dummy_images(["only.jpg"], tmpdir)

        scene_map, edge_data = discover_scenes_v21(
            image_paths,
            lgeu_model_payload=None,
            verbose=False,
        )

    assert "only.jpg" in scene_map
    assert edge_data == {} or isinstance(edge_data, dict)


def test_discover_scenes_v21_delegates_to_v16():
    """V21 clustering must produce the same result as V16 (it delegates to it)."""
    from imc2025_lgeu.scene_clustering import discover_scenes_v16

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        names = ["a.jpg", "b.jpg", "c.jpg", "d.jpg"]
        image_paths = _make_dummy_images(names, tmpdir)

        scene_map_v21, _ = discover_scenes_v21(image_paths, lgeu_model_payload=None)
        scene_map_v16, _ = discover_scenes_v16(image_paths, lgeu_model_payload=None)

    assert scene_map_v21 == scene_map_v16, (
        "discover_scenes_v21 must produce identical results to discover_scenes_v16 "
        "when lgeu_model_payload is None (small non-video dataset path)"
    )
