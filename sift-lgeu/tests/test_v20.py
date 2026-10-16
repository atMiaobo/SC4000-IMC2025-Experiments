"""Tests for discover_scenes_v20: V16 clustering + outlier reassignment."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import (
    MAX_EXHAUSTIVE_N,
    discover_scenes_v20,
)


def _make_video_names(n: int) -> list[str]:
    base = 1_700_000_000_000
    return [f"{base + i * 600}.jpg" for i in range(n)]


def _make_nonvideo_names(n: int) -> list[str]:
    return [f"img_{i:04d}.jpg" for i in range(n)]


def _make_image_paths(names: list[str], tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"\xff\xd8\xff")
        paths[name] = p
    return paths


def _make_lgeu_payload() -> dict:
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [np.zeros(len(X)), np.ones(len(X))]
    )
    return {"model": model, "feature_columns": []}


# ---------------------------------------------------------------------------
# Test 1: outlier images are reassigned to a non-outlier scene
# ---------------------------------------------------------------------------

def test_outliers_reassigned_to_scene(tmp_path):
    """Outlier images must be reassigned to an existing scene after V16 clustering."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)  # N=45, video path
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    # First 40 images get edges (30 Lowe matches + 3 MAGSAC inliers).
    # Last 5 images get 0 matches → become outliers from V16.
    call_count = [0]
    total_names = names

    def fake_match_pair(fa, fb, ratio=0.75):
        call_count[0] += 1
        return (30, 3)

    # V16 will form edges for all selected pairs (all return 30 matches + 3 inliers).
    # But the 5 last images are less similar → GBM mock picks them last → they may end
    # up with no edges if TOPK=20 doesn't include enough of their pairs.
    # To reliably produce outliers, we control V16's internal scene_map directly.

    # We patch discover_scenes_v16 to return a controlled scene_map with 3 outliers.
    outlier_indices = [0, 1, 2]
    non_outlier_scene = "scene_0"
    controlled_scene_map = {}
    for i, name in enumerate(names):
        if i in outlier_indices:
            controlled_scene_map[name] = "outliers"
        else:
            controlled_scene_map[name] = non_outlier_scene
    controlled_edge_data = {}

    # Each outlier gets a distinct descriptor so we can verify correct assignment.
    # Non-outliers all get descriptor [1, 0, ...] — outliers get [1, 0, ...] too → same scene.
    desc_nonoutlier = np.ones(133, dtype=np.float32)
    desc_nonoutlier /= np.linalg.norm(desc_nonoutlier)

    def fake_global_descriptor(path):
        name = path.name
        if name in [names[i] for i in outlier_indices]:
            # Slightly different but still closest to scene_0 images
            desc = np.ones(133, dtype=np.float32) * 0.9
        else:
            desc = np.ones(133, dtype=np.float32)
        return desc, {"width": 100, "height": 100, "aspect": 1.0}

    with (
        patch(
            "imc2025_lgeu.scene_clustering.discover_scenes_v16",
            return_value=(controlled_scene_map, controlled_edge_data),
        ),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            side_effect=fake_global_descriptor,
        ),
    ):
        scene_map, edge_data = discover_scenes_v20(
            image_paths, lgeu_model_payload=payload, verbose=False
        )

    # All outliers must have been reassigned — "outliers" label must not appear
    assert "outliers" not in set(scene_map.values()), (
        f"Expected no 'outliers' after reassignment, got: {set(scene_map.values())}"
    )
    # All images must be covered
    assert set(scene_map.keys()) == set(names)
    # Reassigned outliers must go to scene_0 (closest to non-outlier images)
    for i in outlier_indices:
        assert scene_map[names[i]] == "scene_0", (
            f"Outlier {names[i]} was assigned to '{scene_map[names[i]]}' instead of 'scene_0'"
        )


# ---------------------------------------------------------------------------
# Test 2: non-video datasets bypass the video path (standard V9 logic)
# ---------------------------------------------------------------------------

def test_nonvideo_uses_v6_not_v20_video_path(tmp_path):
    """Non-video datasets must not enter the video-specific outlier-reassignment path."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)  # N=45, but non-video
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    v16_called = [False]

    def fake_v16(*args, **kwargs):
        v16_called[0] = True
        return ({n: "scene_0" for n in names}, {})

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering.discover_scenes_v16", side_effect=fake_v16),
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v20(
            image_paths, lgeu_model_payload=payload, verbose=False
        )

    # Non-video path goes through discover_scenes_v6, NOT through discover_scenes_v16
    assert not v16_called[0], "Non-video path should NOT call discover_scenes_v16"
    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(names)


# ---------------------------------------------------------------------------
# Test 3: when V16 produces no outliers, scene_map is returned unchanged
# ---------------------------------------------------------------------------

def test_no_outliers_returns_v16_unchanged(tmp_path):
    """When V16 produces no outliers, discover_scenes_v20 must return the same scene_map."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    clean_scene_map = {n: "scene_0" for n in names}
    clean_edge_data = {(names[0], names[1]): (50, 10)}

    global_desc_called = [False]

    def fake_global_desc(path):
        global_desc_called[0] = True
        return np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}

    with (
        patch(
            "imc2025_lgeu.scene_clustering.discover_scenes_v16",
            return_value=(clean_scene_map, clean_edge_data),
        ),
        patch("imc2025_lgeu.scene_clustering.global_descriptor", side_effect=fake_global_desc),
    ):
        scene_map, edge_data = discover_scenes_v20(
            image_paths, lgeu_model_payload=payload, verbose=False
        )

    # No outliers → global_descriptor should NOT be called (no reassignment needed)
    assert not global_desc_called[0], (
        "global_descriptor should not be called when there are no outliers"
    )
    assert scene_map == clean_scene_map
    assert edge_data == clean_edge_data


# ---------------------------------------------------------------------------
# Test 4: return type is always (dict, dict)
# ---------------------------------------------------------------------------

def test_return_type_is_tuple_of_dicts(tmp_path):
    """discover_scenes_v20 must always return (dict, dict) regardless of input size."""
    # Small dataset — falls through to non-video/exhaustive path
    names = _make_video_names(3)
    image_paths = _make_image_paths(names, tmp_path)

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
    ):
        result = discover_scenes_v20(image_paths, lgeu_model_payload=None, verbose=False)

    assert isinstance(result, tuple) and len(result) == 2
    scene_map, edge_data = result
    assert isinstance(scene_map, dict)
    assert isinstance(edge_data, dict)
    assert set(scene_map.keys()) == set(names)
