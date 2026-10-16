"""Tests for V10 scene clustering (GBM LGEU + TOPK=25)."""
from pathlib import Path
import pytest
from unittest.mock import patch

from imc2025_lgeu.scene_clustering import discover_scenes_v10


def test_v10_small_n(tmp_path):
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(3)}
    for p in imgs.values():
        p.touch()
    scene_map, _ = discover_scenes_v10(imgs)
    assert isinstance(scene_map, dict)
    assert set(scene_map.keys()) == set(imgs.keys())


def test_v10_uses_topk25():
    with patch("imc2025_lgeu.scene_clustering.discover_scenes_v6") as mock:
        mock.return_value = ({}, {})
        discover_scenes_v10({"a.jpg": Path("a.jpg")})
        _, kwargs = mock.call_args
        assert kwargs.get("lgeu_topk") == 25


def test_v10_smoke_4_images(tmp_path):
    imgs = {f"img_{i:03d}.jpg": tmp_path / f"img_{i:03d}.jpg" for i in range(4)}
    for p in imgs.values():
        p.touch()
    scene_map, _ = discover_scenes_v10(imgs)
    assert len(scene_map) == 4
