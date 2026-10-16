"""Tests for V23: CameraMode.SINGLE for video scenes in SfM."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from imc2025_lgeu.scene_clustering import discover_scenes_v23


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"\d{10,13}")


def _video_names(n: int) -> dict[str, Path]:
    """Fake video-sequence image names (timestamp-based)."""
    return {f"1700000{i:04d}.jpg": Path(f"/fake/{i}.jpg") for i in range(n)}


def _non_video_names(n: int) -> dict[str, Path]:
    """Fake non-video image names (no timestamps)."""
    return {f"img_{i:04d}.jpg": Path(f"/fake/{i}.jpg") for i in range(n)}


# ---------------------------------------------------------------------------
# Test 1: discover_scenes_v23 returns a valid (scene_map, edge_data) tuple
# ---------------------------------------------------------------------------

def test_discover_scenes_v23_returns_tuple():
    """discover_scenes_v23 must return (dict, dict) for any input size."""
    image_paths = _non_video_names(3)
    # Patch the underlying v16/v6 functions so we don't need real images.
    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v16",
        return_value=(
            {name: "scene_0" for name in image_paths},
            {},
        ),
    ):
        result = discover_scenes_v23(image_paths, lgeu_model_payload=None)

    assert isinstance(result, tuple), "discover_scenes_v23 must return a tuple"
    assert len(result) == 2, "tuple must have exactly 2 elements"
    scene_map, edge_data = result
    assert isinstance(scene_map, dict), "first element must be a dict (scene_map)"
    assert isinstance(edge_data, dict), "second element must be a dict (edge_data)"


# ---------------------------------------------------------------------------
# Test 2: discover_scenes_v23 delegates to discover_scenes_v21 (V16 clustering)
# ---------------------------------------------------------------------------

def test_discover_scenes_v23_delegates_to_v21():
    """discover_scenes_v23 must call discover_scenes_v21 (which calls V16 clustering)."""
    image_paths = _non_video_names(4)
    expected_scene_map = {name: "scene_0" for name in image_paths}

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v21",
        return_value=(expected_scene_map, {}),
    ) as mock_v21:
        scene_map, edge_data = discover_scenes_v23(
            image_paths, lgeu_model_payload=None, verbose=False
        )

    mock_v21.assert_called_once_with(
        image_paths, lgeu_model_payload=None, verbose=False
    )
    assert scene_map == expected_scene_map


# ---------------------------------------------------------------------------
# Test 3: All images appear in scene_map output (no image dropped)
# ---------------------------------------------------------------------------

def test_discover_scenes_v23_covers_all_images():
    """Every image passed in must appear in the returned scene_map."""
    image_paths = _non_video_names(5)
    fake_scene_map = {name: "scene_0" for name in image_paths}

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v21",
        return_value=(fake_scene_map, {}),
    ):
        scene_map, _ = discover_scenes_v23(image_paths, lgeu_model_payload=None)

    for name in image_paths:
        assert name in scene_map, f"Image '{name}' missing from scene_map"


# ---------------------------------------------------------------------------
# Test 4: Single-image dataset returns scene_0 without crashing
# ---------------------------------------------------------------------------

def test_discover_scenes_v23_single_image():
    """discover_scenes_v23 must handle a single-image dataset gracefully."""
    image_paths = {"only.jpg": Path("/fake/only.jpg")}
    # discover_scenes_v6 (called via v21 chain) handles N<2 by returning scene_0.
    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v21",
        return_value=({"only.jpg": "scene_0"}, {}),
    ):
        scene_map, edge_data = discover_scenes_v23(image_paths)

    assert scene_map.get("only.jpg") == "scene_0"
