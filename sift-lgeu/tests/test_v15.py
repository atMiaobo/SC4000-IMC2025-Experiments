"""Tests for discover_scenes_v15: lower Lowe threshold (30) for video sequences."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import (
    GOOD_MATCH_THRESHOLD,
    MAX_EXHAUSTIVE_N,
    discover_scenes_v15,
    discover_scenes_v6,
)


def _make_video_names(n: int) -> list[str]:
    """Generate n filenames with 13-digit ms timestamps."""
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
# Test 1: video sequence calls discover_scenes_v6 with good_match_threshold=30
# ---------------------------------------------------------------------------

def test_video_uses_lowered_threshold(tmp_path):
    """For video sequences, discover_scenes_v6 must be called with good_match_threshold=30."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)  # N=45 > 40, triggers video path
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v6",
        wraps=discover_scenes_v6,
    ) as mock_v6:
        with (
            patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
            patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
            patch(
                "imc2025_lgeu.scene_clustering.global_descriptor",
                return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
            ),
        ):
            discover_scenes_v15(image_paths, lgeu_model_payload=payload, verbose=False)

    mock_v6.assert_called_once()
    _, kwargs = mock_v6.call_args
    assert kwargs.get("good_match_threshold") == 30, (
        f"Expected good_match_threshold=30 for video sequences, got {kwargs.get('good_match_threshold')!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: non-video sequences use standard threshold (40) via v6
# ---------------------------------------------------------------------------

def test_nonvideo_uses_standard_threshold(tmp_path):
    """Non-video sequences must NOT use the lowered threshold."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)  # N=45 but no timestamps
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v6",
        wraps=discover_scenes_v6,
    ) as mock_v6:
        with (
            patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
            patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
            patch(
                "imc2025_lgeu.scene_clustering.global_descriptor",
                return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
            ),
        ):
            discover_scenes_v15(image_paths, lgeu_model_payload=payload, verbose=False)

    mock_v6.assert_called_once()
    _, kwargs = mock_v6.call_args
    # Non-video path should NOT pass good_match_threshold=30 explicitly
    # (delegates with default, which is GOOD_MATCH_THRESHOLD=40)
    assert kwargs.get("good_match_threshold", GOOD_MATCH_THRESHOLD) == GOOD_MATCH_THRESHOLD, (
        f"Non-video path should use default threshold {GOOD_MATCH_THRESHOLD}, "
        f"got {kwargs.get('good_match_threshold')!r}"
    )


# ---------------------------------------------------------------------------
# Test 3: small datasets (N <= MAX_EXHAUSTIVE_N) delegate to v6 without threshold=30
# ---------------------------------------------------------------------------

def test_small_dataset_uses_standard_threshold(tmp_path):
    """Small datasets (N <= 40) always use standard threshold even with timestamp filenames."""
    names = _make_video_names(MAX_EXHAUSTIVE_N - 1)  # N=39 <= 40
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v6",
        wraps=discover_scenes_v6,
    ) as mock_v6:
        with (
            patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
            patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        ):
            discover_scenes_v15(image_paths, lgeu_model_payload=payload, verbose=False)

    mock_v6.assert_called_once()
    _, kwargs = mock_v6.call_args
    assert kwargs.get("good_match_threshold", GOOD_MATCH_THRESHOLD) != 30, (
        "Small datasets must not use lowered threshold=30"
    )


# ---------------------------------------------------------------------------
# Test 4: edge admitted at 32 Lowe matches for video (would be rejected at threshold=40)
# ---------------------------------------------------------------------------

def test_video_weak_edge_admitted(tmp_path):
    """A pair with 32 Lowe matches and 4 MAGSAC inliers must form an edge for video sequences."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 2)  # N=42 > 40
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    # All pairs return 32 Lowe matches and 4 MAGSAC inliers — above threshold=30 but below 40
    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(32, 4)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v15(image_paths, lgeu_model_payload=payload, verbose=False)

    # With threshold=30 and MAGSAC inliers=4 >= MIN_MAGSAC_INLIERS=3, all selected pairs
    # should form edges, meaning images should NOT all be in outliers
    non_outlier = [v for v in scene_map.values() if v != "outliers"]
    assert len(non_outlier) > 0, (
        "With 32 Lowe matches and threshold=30, at least some images should form scenes (not outliers). "
        f"scene_map: {scene_map}"
    )


# ---------------------------------------------------------------------------
# Test 5: return type is always (dict, dict)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_video,n", [
    (True, MAX_EXHAUSTIVE_N + 3),   # video path (lowered threshold)
    (False, MAX_EXHAUSTIVE_N + 3),  # non-video path (standard threshold)
    (True, MAX_EXHAUSTIVE_N - 1),   # small dataset path
])
def test_return_type(tmp_path, use_video, n):
    """discover_scenes_v15 always returns (scene_map: dict, edge_data: dict)."""
    names = _make_video_names(n) if use_video else _make_nonvideo_names(n)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        result = discover_scenes_v15(image_paths, lgeu_model_payload=payload, verbose=False)

    assert isinstance(result, tuple) and len(result) == 2
    scene_map, edge_data = result
    assert isinstance(scene_map, dict)
    assert isinstance(edge_data, dict)
    assert set(scene_map.keys()) == set(names)
