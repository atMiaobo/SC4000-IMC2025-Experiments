"""Tests for V7: exhaustive matching for small video sequences."""
from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2


def _make_images(tmp_path: Path, n: int = 5, size: int = 64) -> dict[str, Path]:
    """Create n images with timestamp filenames (detected as video sequence)."""
    paths = {}
    rng = np.random.default_rng(0)
    base_ts = 1697123456000  # 13-digit millisecond timestamp
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        img[i % size, :] = 200
        name = f"frame_{base_ts + i * 100}.jpg"
        p = tmp_path / name
        cv2.imwrite(str(p), img)
        paths[name] = p
    return paths


def _make_nonvideo_images(tmp_path: Path, n: int = 5, size: int = 64) -> dict[str, Path]:
    paths = {}
    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        name = f"img_{i:04d}.jpg"
        p = tmp_path / name
        cv2.imwrite(str(p), img)
        paths[name] = p
    return paths


# ---------------------------------------------------------------------------
# Test: v7 uses exhaustive for small video
# ---------------------------------------------------------------------------

def test_v7_video_small_uses_exhaustive(tmp_path):
    """Video with N <= 100 should use exhaustive COLMAP matching."""
    from imc2025_lgeu.sfm import run_sfm_v7, _is_video_sequence

    images = _make_images(tmp_path, n=4)
    names = sorted(images)
    assert _is_video_sequence(names), "frame_NNNNN pattern should be detected as video"

    exhaustive_called = []
    sequential_called = []

    def fake_extract(*a, **k): pass
    def fake_exhaustive(*a, **k): exhaustive_called.append(True)
    def fake_sequential(*a, **k): sequential_called.append(True)
    def fake_mapping(*a, **k): return {}

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_pyc:
        mock_pyc.CameraMode.AUTO = "AUTO"
        mock_pyc.extract_features = fake_extract
        mock_pyc.match_exhaustive = fake_exhaustive
        mock_pyc.match_sequential = fake_sequential
        mock_pyc.incremental_mapping = fake_mapping
        mock_pyc.IncrementalPipelineOptions = MagicMock(return_value=MagicMock(mapper=MagicMock()))

        import imc2025_lgeu.sfm as sfm_mod
        sfm_mod._HAS_PYCOLMAP = True
        run_sfm_v7(images, verbose=False)
        sfm_mod._HAS_PYCOLMAP = True  # reset

    assert exhaustive_called, "run_sfm_v7 should call match_exhaustive for video N=4 <= 100"
    assert not sequential_called, "run_sfm_v7 should NOT call match_sequential for video N=4"


def test_v7_video_large_uses_sequential(tmp_path):
    """Video with N > 100 should use sequential COLMAP matching."""
    from imc2025_lgeu.sfm import run_sfm_v7

    # Create 101 fake images with timestamp naming pattern (video detection)
    images = {}
    base_ts = 1697123456000
    for i in range(101):
        name = f"frame_{base_ts + i * 100}.jpg"
        p = tmp_path / name
        rng = np.random.default_rng(i)
        img = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(p), img)
        images[name] = p

    sequential_called = []

    def fake_extract(*a, **k): pass
    def fake_exhaustive(*a, **k): pass
    def fake_sequential(*a, **k): sequential_called.append(True)
    def fake_mapping(*a, **k): return {}

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_pyc:
        mock_pyc.CameraMode.AUTO = "AUTO"
        mock_pyc.extract_features = fake_extract
        mock_pyc.match_exhaustive = fake_exhaustive
        mock_pyc.match_sequential = fake_sequential
        mock_pyc.SequentialPairingOptions = MagicMock(return_value=MagicMock(overlap=10, loop_detection=False))
        mock_pyc.incremental_mapping = fake_mapping
        mock_pyc.IncrementalPipelineOptions = MagicMock(return_value=MagicMock(mapper=MagicMock()))

        import imc2025_lgeu.sfm as sfm_mod
        sfm_mod._HAS_PYCOLMAP = True
        run_sfm_v7(images, verbose=False)

    assert sequential_called, "run_sfm_v7 should use sequential for video N=101 > 100"


def test_v7_nonvideo_always_exhaustive(tmp_path):
    """Non-video images should always use exhaustive matching."""
    from imc2025_lgeu.sfm import run_sfm_v7

    images = _make_nonvideo_images(tmp_path, n=5)
    exhaustive_called = []

    def fake_extract(*a, **k): pass
    def fake_exhaustive(*a, **k): exhaustive_called.append(True)
    def fake_mapping(*a, **k): return {}

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_pyc:
        mock_pyc.CameraMode.AUTO = "AUTO"
        mock_pyc.extract_features = fake_extract
        mock_pyc.match_exhaustive = fake_exhaustive
        mock_pyc.incremental_mapping = fake_mapping
        mock_pyc.IncrementalPipelineOptions = MagicMock(return_value=MagicMock(mapper=MagicMock()))

        import imc2025_lgeu.sfm as sfm_mod
        sfm_mod._HAS_PYCOLMAP = True
        run_sfm_v7(images, verbose=False)

    assert exhaustive_called, "non-video images always get exhaustive matching"


def test_v7_match_mode_in_info(tmp_path):
    """run_sfm_v7 records match_mode in the info dict."""
    from imc2025_lgeu.sfm import run_sfm_v7

    images = _make_images(tmp_path, n=3)  # video, N=3 <= 100

    def fake_extract(*a, **k): pass
    def fake_exhaustive(*a, **k): pass
    def fake_mapping(*a, **k): return {}

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_pyc:
        mock_pyc.CameraMode.AUTO = "AUTO"
        mock_pyc.extract_features = fake_extract
        mock_pyc.match_exhaustive = fake_exhaustive
        mock_pyc.incremental_mapping = fake_mapping
        mock_pyc.IncrementalPipelineOptions = MagicMock(return_value=MagicMock(mapper=MagicMock()))

        import imc2025_lgeu.sfm as sfm_mod
        sfm_mod._HAS_PYCOLMAP = True
        _, info = run_sfm_v7(images, verbose=False)

    assert "match_mode" in info, "info should contain match_mode key"
    assert info["match_mode"] == "exhaustive", f"expected 'exhaustive', got {info['match_mode']!r}"


def test_v7_per_scene_isolates_exceptions(tmp_path):
    """run_sfm_per_scene_v7 catches exceptions per scene."""
    from imc2025_lgeu.sfm import run_sfm_per_scene_v7

    images = _make_images(tmp_path, n=4)
    scene_map = {name: "scene_0" for name in images}

    def boom(*a, **k):
        raise RuntimeError("COLMAP exploded")

    with patch("imc2025_lgeu.sfm.pycolmap") as mock_pyc:
        mock_pyc.CameraMode.AUTO = "AUTO"
        mock_pyc.extract_features = boom  # will raise
        mock_pyc.IncrementalPipelineOptions = MagicMock()

        import imc2025_lgeu.sfm as sfm_mod
        sfm_mod._HAS_PYCOLMAP = True
        poses, info = run_sfm_per_scene_v7(images, scene_map, verbose=False)

    # Must return identity poses, not raise
    assert len(poses) == len(images)
    for name in images:
        R, t = poses[name]
        assert np.allclose(R, np.eye(3)), "should be identity rotation"
        assert np.allclose(t, np.zeros(3)), "should be zero translation"
