"""Tests for discover_scenes_v16: num_gap=0 GBM scoring + Lowe threshold=30 for video."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import (
    GOOD_MATCH_THRESHOLD,
    MAX_EXHAUSTIVE_N,
    MIN_MAGSAC_INLIERS,
    discover_scenes_v16,
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
# Test 1: for video sequences, num_gap is zeroed in the GBM feature vector
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
        discover_scenes_v16(image_paths, lgeu_model_payload=payload, verbose=False)

    assert captured_feats, "GBM model.predict_proba was never called"
    feats = captured_feats[0]  # shape (n_pairs, 7): [cosine, mrm, mrx, aspect_diff, area_rat, ng, nj]
    num_gap_col = feats[:, 5]  # 6th feature (index 5) is num_gap
    assert np.all(num_gap_col == 0.0), (
        f"All num_gap values must be 0.0 for video sequences, got max={num_gap_col.max()}"
    )


# ---------------------------------------------------------------------------
# Test 2: video sequence uses lowered Lowe threshold (30), not standard 40
# ---------------------------------------------------------------------------

def test_video_weak_edge_admitted(tmp_path):
    """A pair with 32 Lowe matches and 4 MAGSAC inliers must form an edge for video (threshold=30)."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 2)  # N=42 > 40
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(32, 4)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        scene_map, edge_data = discover_scenes_v16(image_paths, lgeu_model_payload=payload)

    # 32 >= 30 (video threshold) and MAGSAC inliers=4 >= MIN_MAGSAC_INLIERS=3
    # so edges should form and images should not all be outliers
    non_outlier = [v for v in scene_map.values() if v != "outliers"]
    assert len(non_outlier) > 0, (
        "32 Lowe matches with threshold=30 should form edges; images should not all be outliers. "
        f"scene_map: {scene_map}"
    )


# ---------------------------------------------------------------------------
# Test 3: same pairs with threshold=40 would all be outliers (verify contrast)
# ---------------------------------------------------------------------------

def test_video_weak_edge_rejected_at_standard_threshold(tmp_path):
    """32 Lowe matches are BELOW standard threshold=40 — v9 path would make all outliers."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 2)
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(32, 4)),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        # V9 path uses threshold=40; all pairs return 32 Lowe matches -> all outliers
        scene_map_v9, _ = discover_scenes_v6(
            image_paths,
            good_match_threshold=40,
            lgeu_model_payload=payload,
            lgeu_topk=20,
            verbose=False,
        )

    all_outliers = all(v == "outliers" for v in scene_map_v9.values())
    assert all_outliers, (
        "With threshold=40, 32 Lowe matches should NOT form edges (all outliers). "
        f"scene_map: {scene_map_v9}"
    )


# ---------------------------------------------------------------------------
# Test 4: non-video sequences delegate to discover_scenes_v6 (standard path)
# ---------------------------------------------------------------------------

def test_nonvideo_delegates_to_v6(tmp_path):
    """Non-video large datasets must use standard V6 path (no num_gap=0, no threshold=30)."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)
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
            discover_scenes_v16(image_paths, lgeu_model_payload=payload, verbose=False)

    mock_v6.assert_called_once()
    _, kwargs = mock_v6.call_args
    assert kwargs.get("good_match_threshold", GOOD_MATCH_THRESHOLD) == GOOD_MATCH_THRESHOLD, (
        f"Non-video path should use standard threshold {GOOD_MATCH_THRESHOLD}"
    )


# ---------------------------------------------------------------------------
# Test 5: return type is always (dict, dict)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_video,n", [
    (True, MAX_EXHAUSTIVE_N + 3),   # video path
    (False, MAX_EXHAUSTIVE_N + 3),  # non-video path
    (True, MAX_EXHAUSTIVE_N - 1),   # small dataset path
])
def test_return_type(tmp_path, use_video, n):
    """discover_scenes_v16 always returns (scene_map: dict, edge_data: dict)."""
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
        result = discover_scenes_v16(image_paths, lgeu_model_payload=payload, verbose=False)

    assert isinstance(result, tuple) and len(result) == 2
    scene_map, edge_data = result
    assert isinstance(scene_map, dict)
    assert isinstance(edge_data, dict)
    assert set(scene_map.keys()) == set(names)
