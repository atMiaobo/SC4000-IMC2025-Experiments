"""Tests for V13 scene clustering (cosine-only pair ranking for video sequences)."""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from imc2025_lgeu.scene_clustering import discover_scenes_v13


def test_v13_returns_tuple_with_all_keys(tmp_path):
    """discover_scenes_v13 must return (scene_map, edge_data) with all input keys."""
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(4)}
    for p in imgs.values():
        p.touch()
    result = discover_scenes_v13(imgs)
    assert isinstance(result, tuple) and len(result) == 2
    scene_map, edge_data = result
    assert isinstance(scene_map, dict)
    assert isinstance(edge_data, dict)
    assert set(scene_map.keys()) == set(imgs.keys())


def test_v13_small_n_delegates_to_v6_with_topk20():
    """For N <= MAX_EXHAUSTIVE_N, v13 delegates to discover_scenes_v6 with lgeu_topk=20."""
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock_v6:
        mock_v6.return_value = ({}, {})
        small_imgs = {f"img_{i}.jpg": Path(f"img_{i}.jpg") for i in range(5)}
        discover_scenes_v13(small_imgs, lgeu_model_payload=None)
        mock_v6.assert_called_once()
        _, kwargs = mock_v6.call_args
        assert kwargs.get("lgeu_topk") == 20


def test_v13_non_video_large_n_delegates_to_v6():
    """For N > 40 but non-timestamp filenames, v13 delegates to discover_scenes_v6."""
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock_v6:
        mock_v6.return_value = ({}, {})
        dummy_model = {"model": object(), "feature_columns": []}
        large_non_video = {f"photo_{i:04d}.jpg": Path(f"photo_{i:04d}.jpg") for i in range(50)}
        discover_scenes_v13(large_non_video, lgeu_model_payload=dummy_model)
        mock_v6.assert_called_once()


def test_v13_video_large_n_uses_cosine_not_lgeu(tmp_path):
    """For video sequences (N > 40, timestamp filenames), v13 must NOT call _lgeu_select_pairs."""
    base_ts = 1700000000000
    imgs = {}
    for i in range(45):
        name = f"{base_ts + i * 600}.jpg"
        p = tmp_path / name
        p.touch()
        imgs[name] = p

    dummy_model = {"model": object(), "feature_columns": []}

    lgeu_called = []

    def fake_lgeu_select(image_paths, names, model, topk, verbose=False):
        lgeu_called.append(True)
        return []

    def fake_global_descriptor(path):
        rng = np.random.default_rng(hash(path.name) % (2**32))
        desc = rng.standard_normal(133).astype(np.float32)
        desc /= np.linalg.norm(desc) + 1e-8
        return desc, {"width": 100, "height": 100, "aspect": 1.0}

    def fake_extract_sift(path, nfeatures=4000):
        return ([], None)

    def fake_match_pair(fa, fb, ratio=0.75):
        return 0, 0

    with patch("imc2025_lgeu.scene_clustering._lgeu_select_pairs", side_effect=fake_lgeu_select), \
         patch("imc2025_lgeu.scene_clustering._extract_sift", side_effect=fake_extract_sift), \
         patch("imc2025_lgeu.scene_clustering._match_pair", side_effect=fake_match_pair), \
         patch("imc2025_lgeu.descriptors.global_descriptor", side_effect=fake_global_descriptor):
        scene_map, edge_data = discover_scenes_v13(imgs, lgeu_model_payload=dummy_model)

    assert not lgeu_called, "v13 must not call _lgeu_select_pairs for video sequences"
    assert set(scene_map.keys()) == set(imgs.keys())


def test_v13_video_selects_topk20_pairs_per_image(tmp_path):
    """For video sequences, v13 selects at most TOPK*N/2 unique pairs (cosine top-20)."""
    base_ts = 1700000000000
    N = 45
    imgs = {}
    for i in range(N):
        name = f"{base_ts + i * 600}.jpg"
        p = tmp_path / name
        p.touch()
        imgs[name] = p

    dummy_model = {"model": object(), "feature_columns": []}

    matched_pairs = []

    def fake_global_descriptor(path):
        rng = np.random.default_rng(hash(path.name) % (2**32))
        desc = rng.standard_normal(133).astype(np.float32)
        desc /= np.linalg.norm(desc) + 1e-8
        return desc, {"width": 100, "height": 100, "aspect": 1.0}

    def fake_extract_sift(path, nfeatures=4000):
        return ([], None)

    def fake_match_pair(fa, fb, ratio=0.75):
        matched_pairs.append(1)
        return 0, 0

    with patch("imc2025_lgeu.scene_clustering._lgeu_select_pairs", return_value=[]), \
         patch("imc2025_lgeu.scene_clustering._extract_sift", side_effect=fake_extract_sift), \
         patch("imc2025_lgeu.scene_clustering._match_pair", side_effect=fake_match_pair), \
         patch("imc2025_lgeu.descriptors.global_descriptor", side_effect=fake_global_descriptor):
        scene_map, edge_data = discover_scenes_v13(imgs, lgeu_model_payload=dummy_model)

    TOPK = 20
    max_possible_pairs = N * (N - 1) // 2
    # Cosine top-20 per image: at most N*TOPK/2 unique pairs
    assert len(edge_data) <= N * TOPK, f"Too many pairs matched: {len(edge_data)}"
    assert len(edge_data) > 0, "Should have matched some pairs for N=45"
    assert len(edge_data) < max_possible_pairs, "Should not match all pairs (not exhaustive)"
