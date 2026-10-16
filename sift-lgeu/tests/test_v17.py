"""Tests for discover_scenes_v17: nfeatures=8000 + num_gap=0 + Lowe threshold=30 for video."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import (
    MAX_EXHAUSTIVE_N,
    MIN_MAGSAC_INLIERS,
    discover_scenes_v17,
    discover_scenes_v6,
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
# Test 1: for video sequences, _extract_sift is called with nfeatures=8000
# ---------------------------------------------------------------------------

def test_video_uses_nfeatures_8000(tmp_path):
    """For video sequences, SIFT extraction must use nfeatures=8000 (not 4000)."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)  # N=45 > 40, video path
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
    captured_nfeatures = []

    def fake_extract_sift(path, nfeatures=4000):
        captured_nfeatures.append(nfeatures)
        return dummy_feat

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", side_effect=fake_extract_sift),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        discover_scenes_v17(image_paths, lgeu_model_payload=payload, verbose=False)

    assert captured_nfeatures, "_extract_sift was never called"
    assert all(n == 8000 for n in captured_nfeatures), (
        f"Expected all nfeatures=8000 for video path, got: {set(captured_nfeatures)}"
    )


# ---------------------------------------------------------------------------
# Test 2: for video sequences, num_gap is zeroed in the GBM feature vector
# ---------------------------------------------------------------------------

def test_video_num_gap_zeroed(tmp_path):
    """For video sequences, the GBM model must be called with num_gap=0 for all pairs."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)  # N=45 > 40, video path
    image_paths = _make_image_paths(names, tmp_path)

    model = MagicMock()
    captured_feats = []

    def fake_predict_proba(X):
        captured_feats.append(X.copy())
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    model.predict_proba.side_effect = fake_predict_proba
    payload = {"model": model, "feature_columns": []}

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        discover_scenes_v17(image_paths, lgeu_model_payload=payload, verbose=False)

    assert captured_feats, "GBM model.predict_proba was never called"
    feats = captured_feats[0]  # shape (n_pairs, 7): [cosine, mrm, mrx, aspect_diff, area_rat, ng, nj]
    num_gap_col = feats[:, 5]  # 6th feature (index 5) is num_gap
    assert np.all(num_gap_col == 0.0), (
        f"All num_gap values must be 0.0 for video sequences, got max={num_gap_col.max()}"
    )


# ---------------------------------------------------------------------------
# Test 3: non-video datasets bypass the video path (nfeatures unchanged)
# ---------------------------------------------------------------------------

def test_nonvideo_delegates_to_v6(tmp_path):
    """Non-video datasets (no timestamp filenames) must not enter the video path."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)  # N=45, but non-video
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    captured_nfeatures = []

    def fake_extract_sift(path, nfeatures=4000):
        captured_nfeatures.append(nfeatures)
        return ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", side_effect=fake_extract_sift),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v17(image_paths, lgeu_model_payload=payload, verbose=False)

    # Non-video path uses discover_scenes_v6, which calls _extract_sift with default nfeatures=4000
    assert captured_nfeatures, "_extract_sift was never called"
    assert all(n == 4000 for n in captured_nfeatures), (
        f"Non-video path must use nfeatures=4000, got: {set(captured_nfeatures)}"
    )
    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(names)


# ---------------------------------------------------------------------------
# Test 4: video path uses Lowe threshold=30 (not 40)
# ---------------------------------------------------------------------------

def test_video_lowe_threshold_30(tmp_path):
    """For video sequences, pairs with exactly 30 Lowe matches must form an edge."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    # Return 30 Lowe matches + MIN_MAGSAC_INLIERS inliers — just enough to form an edge
    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(30, MIN_MAGSAC_INLIERS)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v17(image_paths, lgeu_model_payload=payload, verbose=False)

    # With 30 matches + MIN_MAGSAC_INLIERS, all selected pairs form edges → no outliers
    assert "outliers" not in set(scene_map.values()), (
        "With 30 Lowe matches + MAGSAC inliers, no image should be labeled 'outliers' "
        f"(scene_map values: {set(scene_map.values())})"
    )
