"""Tests for V12 scene clustering (GBM LGEU + TOPK=20 + temporal pair augmentation)."""
from pathlib import Path
from unittest.mock import patch

from imc2025_lgeu.scene_clustering import discover_scenes_v12


def test_v12_small_n_returns_valid_scene_map(tmp_path):
    """Small N (<=40): should return a scene_map with all image keys."""
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(4)}
    for p in imgs.values():
        p.touch()
    scene_map, edge_data = discover_scenes_v12(imgs)
    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(imgs.keys())
    assert isinstance(edge_data, dict)


def test_v12_small_n_delegates_to_v6():
    """For N <= MAX_EXHAUSTIVE_N, v12 should delegate to discover_scenes_v6 with lgeu_topk=20."""
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock_v6:
        mock_v6.return_value = ({}, {})
        small_imgs = {f"img_{i}.jpg": Path(f"img_{i}.jpg") for i in range(5)}
        discover_scenes_v12(small_imgs, lgeu_model_payload=None)
        mock_v6.assert_called_once()
        _, kwargs = mock_v6.call_args
        assert kwargs.get("lgeu_topk") == 20


def test_v12_non_video_large_n_delegates_to_v6():
    """For N > 40 but non-video filenames, v12 should use standard LGEU logic (v6)."""
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock_v6:
        mock_v6.return_value = ({}, {})
        dummy_model = {"model": object(), "feature_columns": []}
        large_non_video = {f"scene_img_{i:04d}.jpg": Path(f"img_{i}.jpg") for i in range(50)}
        discover_scenes_v12(large_non_video, lgeu_model_payload=dummy_model)
        mock_v6.assert_called_once()


def test_v12_video_large_n_adds_temporal_pairs(tmp_path):
    """For N > 40 with timestamp filenames (video), v12 should augment pairs with temporal neighbors."""
    base_ts = 1700000000000
    imgs = {}
    for i in range(45):
        name = f"frame_{base_ts + i * 33}_cam0.jpg"
        p = tmp_path / name
        p.touch()
        imgs[name] = p

    temporal_pairs_added = []

    def fake_lgeu_select(image_paths, names, model, topk, verbose=False):
        return []

    def fake_match_pair(fa, fb, ratio=0.75):
        return 0, 0

    def fake_extract_sift(path, nfeatures=4000):
        return ([], None)

    with patch("imc2025_lgeu.scene_clustering._lgeu_select_pairs", side_effect=fake_lgeu_select), \
         patch("imc2025_lgeu.scene_clustering._match_pair", side_effect=fake_match_pair), \
         patch("imc2025_lgeu.scene_clustering._extract_sift", side_effect=fake_extract_sift):
        dummy_model = {"model": object(), "feature_columns": []}
        scene_map, edge_data = discover_scenes_v12(imgs, lgeu_model_payload=dummy_model)

    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(imgs.keys())
    n = len(imgs)
    assert len(edge_data) > 0, "temporal pairs should have been matched"
    max_temporal_pairs = n * 2
    assert len(edge_data) <= max_temporal_pairs + 10
