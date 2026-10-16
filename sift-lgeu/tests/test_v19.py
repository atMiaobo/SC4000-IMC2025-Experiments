"""Tests for discover_scenes_v19: loosened FLANN Lowe ratio (0.75→0.80) for video sequences."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imc2025_lgeu.scene_clustering import (
    MAX_EXHAUSTIVE_N,
    MIN_MAGSAC_INLIERS,
    discover_scenes_v19,
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
# Test 1: video path calls _match_pair with ratio=0.80
# ---------------------------------------------------------------------------

def test_video_uses_ratio_080(tmp_path):
    """For video sequences, _match_pair must be called with ratio=0.80 (not 0.75)."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
    captured_ratios = []

    def fake_match_pair(fa, fb, ratio=0.75):
        captured_ratios.append(ratio)
        return (0, 0)

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", side_effect=fake_match_pair),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        discover_scenes_v19(image_paths, lgeu_model_payload=payload, verbose=False)

    assert captured_ratios, "_match_pair was never called for video sequence"
    assert all(r == 0.80 for r in captured_ratios), (
        f"Expected all ratio=0.80 for video path, got: {set(captured_ratios)}"
    )


# ---------------------------------------------------------------------------
# Test 2: video path still uses Lowe threshold=30 (V16 inheritance)
# ---------------------------------------------------------------------------

def test_video_lowe_threshold_30(tmp_path):
    """For video sequences, pairs with exactly 30 Lowe matches must form an edge."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(30, MIN_MAGSAC_INLIERS)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v19(image_paths, lgeu_model_payload=payload, verbose=False)

    assert "outliers" not in set(scene_map.values()), (
        "With 30 Lowe matches + MAGSAC inliers, no image should be 'outliers' "
        f"(scene_map values: {set(scene_map.values())})"
    )


# ---------------------------------------------------------------------------
# Test 3: video path uses num_gap=0 in GBM features (inherited from V16)
# ---------------------------------------------------------------------------

def test_video_num_gap_zeroed(tmp_path):
    """For video sequences, the GBM model must be called with num_gap=0 for all pairs."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)
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
        discover_scenes_v19(image_paths, lgeu_model_payload=payload, verbose=False)

    assert captured_feats, "GBM model.predict_proba was never called"
    feats = captured_feats[0]  # shape (n_pairs, 7): [cosine, mrm, mrx, aspect_diff, area_rat, ng, nj]
    num_gap_col = feats[:, 5]  # index 5 is num_gap
    assert np.all(num_gap_col == 0.0), (
        f"All num_gap values must be 0.0 for video sequences in V19, got max={num_gap_col.max()}"
    )


# ---------------------------------------------------------------------------
# Test 4: non-video large datasets use ratio=0.75 (standard path via v6)
# ---------------------------------------------------------------------------

def test_nonvideo_uses_standard_ratio(tmp_path):
    """Non-video large datasets must NOT enter the video path (ratio stays 0.75)."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    captured_ratios = []

    def fake_match_pair(fa, fb, ratio=0.75):
        captured_ratios.append(ratio)
        return (0, 0)

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", side_effect=fake_match_pair),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v19(image_paths, lgeu_model_payload=payload, verbose=False)

    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(names)
    # Non-video path delegates to v6 which uses default ratio=0.75
    if captured_ratios:
        assert all(r == 0.75 for r in captured_ratios), (
            f"Non-video path should use ratio=0.75, got: {set(captured_ratios)}"
        )


# ---------------------------------------------------------------------------
# Test 5: run_sfm_per_scene_v19 is aliased to run_sfm_per_scene_v18
# ---------------------------------------------------------------------------

def test_run_sfm_per_scene_v19_importable():
    """run_sfm_per_scene_v19 must be importable and point to v18 (relaxed mapper)."""
    from imc2025_lgeu.sfm import run_sfm_per_scene_v18, run_sfm_per_scene_v19

    assert callable(run_sfm_per_scene_v19), "run_sfm_per_scene_v19 must be callable"
    assert run_sfm_per_scene_v19 is run_sfm_per_scene_v18, (
        "run_sfm_per_scene_v19 should alias run_sfm_per_scene_v18 (relaxed COLMAP mapper)"
    )
