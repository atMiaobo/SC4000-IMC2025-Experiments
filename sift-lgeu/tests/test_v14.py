"""Tests for discover_scenes_v14: GBM LGEU with num_gap zeroed for video sequences."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from imc2025_lgeu.scene_clustering import (
    MAX_EXHAUSTIVE_N,
    discover_scenes_v14,
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
        p.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header stub
        paths[name] = p
    return paths


def _make_lgeu_payload(n_features: int = 7) -> dict:
    """Minimal mock LGEU payload whose model records feature arrays passed to it."""
    model = MagicMock()
    # predict_proba returns shape (N_pairs, 2) — column 1 is same-scene prob
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [np.zeros(len(X)), np.ones(len(X))]
    )
    return {"model": model, "feature_columns": []}


# ---------------------------------------------------------------------------
# Test 1: video sequence uses num_gap=0 in feature matrix
# ---------------------------------------------------------------------------

def test_video_num_gap_zeroed(tmp_path):
    """For video sequences, the num_gap feature (index 5) must be 0.0 for all pairs."""
    names = _make_video_names(MAX_EXHAUSTIVE_N + 5)  # N=45 > 40, triggers video path
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    # Patch _extract_sift and _match_pair to avoid real image processing
    dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
    dummy_match = (0, 0)

    with (
        patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
        patch("imc2025_lgeu.scene_clustering._match_pair", return_value=dummy_match),
        patch(
            "imc2025_lgeu.scene_clustering.global_descriptor",
            return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
        ),
    ):
        discover_scenes_v14(image_paths, lgeu_model_payload=payload, verbose=False)

    assert payload["model"].predict_proba.called, "model.predict_proba was never called"
    feat_array = payload["model"].predict_proba.call_args[0][0]  # shape (N_pairs, 7)
    num_gap_col = feat_array[:, 5]
    assert np.all(num_gap_col == 0.0), (
        f"num_gap column (index 5) must be all zeros for video sequences; "
        f"got non-zero values: {num_gap_col[num_gap_col != 0]}"
    )


# ---------------------------------------------------------------------------
# Test 2: non-video sequences delegate to discover_scenes_v6 unchanged
# ---------------------------------------------------------------------------

def test_nonvideo_delegates_to_v6(tmp_path):
    """Non-video sequences (no timestamp filenames) must use discover_scenes_v6."""
    names = _make_nonvideo_names(MAX_EXHAUSTIVE_N + 5)  # N=45 but no timestamps
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v6",
        wraps=discover_scenes_v6,
    ) as mock_v6:
        dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
        with (
            patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
            patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
            patch(
                "imc2025_lgeu.scene_clustering.global_descriptor",
                return_value=(np.ones(133, dtype=np.float32), {"width": 100, "height": 100, "aspect": 1.0}),
            ),
        ):
            discover_scenes_v14(image_paths, lgeu_model_payload=payload, verbose=False)

        mock_v6.assert_called_once()
    # model.predict_proba should NOT have been called (v6 calls _lgeu_select_pairs internally
    # with the original model, but we didn't patch that path here — just verify delegation)
    # The key is that discover_scenes_v6 was called, not the inline video path.


# ---------------------------------------------------------------------------
# Test 3: small datasets (N <= MAX_EXHAUSTIVE_N) delegate to v6 regardless
# ---------------------------------------------------------------------------

def test_small_dataset_delegates_to_v6(tmp_path):
    """Small datasets (N <= 40) always delegate to v6 even if filenames have timestamps."""
    names = _make_video_names(MAX_EXHAUSTIVE_N - 1)  # N=39 <= 40
    image_paths = _make_image_paths(names, tmp_path)
    payload = _make_lgeu_payload()

    with patch(
        "imc2025_lgeu.scene_clustering.discover_scenes_v6",
        wraps=discover_scenes_v6,
    ) as mock_v6:
        dummy_feat = ([MagicMock()], np.zeros((1, 128), dtype=np.float32))
        with (
            patch("imc2025_lgeu.scene_clustering._extract_sift", return_value=dummy_feat),
            patch("imc2025_lgeu.scene_clustering._match_pair", return_value=(0, 0)),
        ):
            discover_scenes_v14(image_paths, lgeu_model_payload=payload, verbose=False)

        mock_v6.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: return type is always (dict, dict) regardless of path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("use_video,n", [
    (True, MAX_EXHAUSTIVE_N + 3),   # video path
    (False, MAX_EXHAUSTIVE_N + 3),  # non-video path
    (True, MAX_EXHAUSTIVE_N - 1),   # small dataset path
])
def test_return_type(tmp_path, use_video, n):
    """discover_scenes_v14 always returns (scene_map: dict, edge_data: dict)."""
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
        result = discover_scenes_v14(image_paths, lgeu_model_payload=payload, verbose=False)

    assert isinstance(result, tuple) and len(result) == 2
    scene_map, edge_data = result
    assert isinstance(scene_map, dict)
    assert isinstance(edge_data, dict)
    assert set(scene_map.keys()) == set(names)
