"""Tests for V11 scene clustering (GBM LGEU + TOPK=20 + lower Lowe threshold=30)."""
from pathlib import Path
import pytest
from unittest.mock import patch

from imc2025_lgeu.scene_clustering import discover_scenes_v11


def test_v11_small_n(tmp_path):
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(3)}
    for p in imgs.values():
        p.touch()
    scene_map, _ = discover_scenes_v11(imgs)
    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(imgs.keys())


def test_v11_passes_threshold30_for_small_n():
    """For N <= MAX_EXHAUSTIVE_N, v11 should use threshold=30."""
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock:
        mock.return_value = ({}, {})
        small_imgs = {f"img_{i}.jpg": Path(f"img_{i}.jpg") for i in range(5)}
        discover_scenes_v11(small_imgs)
        _, kwargs = mock.call_args
        assert kwargs.get("good_match_threshold") == 30


def test_v11_smoke_4_images(tmp_path):
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(4)}
    for p in imgs.values():
        p.touch()
    scene_map, _ = discover_scenes_v11(imgs)
    assert len(scene_map) == 4
