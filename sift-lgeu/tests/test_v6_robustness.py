"""Robustness tests for V6 pipeline: crash isolation, fallback behaviour, edge cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_images(tmp_path: Path, n: int = 3, size: int = 64) -> dict[str, Path]:
    """Create n distinct synthetic JPEG images."""
    paths = {}
    rng = np.random.default_rng(42)
    for i in range(n):
        img = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        img[i % size, :] = 255  # ensure some variation per image
        name = f"img_{i:04d}.jpg"
        p = tmp_path / name
        cv2.imwrite(str(p), img)
        paths[name] = p
    return paths


def _sample_csv(tmp_path: Path, dataset: str = "ds", n: int = 3) -> Path:
    rows = [
        {
            "image_id": f"{dataset}/s/{i}.jpg",
            "dataset": dataset,
            "scene": "s",
            "image": f"{i}.jpg",
            "rotation_matrix": "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0",
            "translation_vector": "0.0;0.0;0.0",
        }
        for i in range(n)
    ]
    p = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# discover_scenes_v6: SIFT extraction guards
# ---------------------------------------------------------------------------

def test_discover_v6_missing_image_does_not_crash(tmp_path):
    from imc2025_lgeu.scene_clustering import discover_scenes_v6

    real = tmp_path / "real.jpg"
    cv2.imwrite(str(real), np.ones((64, 64, 3), dtype=np.uint8) * 128)

    paths = {
        "real.jpg": real,
        "missing.jpg": tmp_path / "missing.jpg",  # doesn't exist
    }
    scene_map, _ = discover_scenes_v6(paths, verbose=False)
    assert set(scene_map.keys()) == {"real.jpg", "missing.jpg"}


def test_discover_v6_all_missing_images_gives_scene_0(tmp_path):
    from imc2025_lgeu.scene_clustering import discover_scenes_v6

    paths = {
        "a.jpg": tmp_path / "a.jpg",
        "b.jpg": tmp_path / "b.jpg",
    }
    scene_map, _ = discover_scenes_v6(paths, verbose=False)
    assert len(scene_map) == 2


def test_discover_v6_single_image_returns_scene_0(tmp_path):
    from imc2025_lgeu.scene_clustering import discover_scenes_v6

    p = tmp_path / "img.jpg"
    cv2.imwrite(str(p), np.ones((64, 64, 3), dtype=np.uint8) * 200)
    scene_map, _ = discover_scenes_v6({"img.jpg": p}, verbose=False)
    assert scene_map == {"img.jpg": "scene_0"}


# ---------------------------------------------------------------------------
# discover_scenes_v6: LGEU fallback
# ---------------------------------------------------------------------------

def test_discover_v6_lgeu_failure_falls_back_to_exhaustive(tmp_path):
    from imc2025_lgeu.scene_clustering import discover_scenes_v6

    # 45 images to trigger LGEU path (> MAX_EXHAUSTIVE_N=40)
    paths = _make_images(tmp_path, n=45, size=32)

    broken_model = MagicMock()
    broken_model.predict_proba.side_effect = RuntimeError("sklearn version mismatch")
    lgeu_payload = {
        "model": broken_model,
        "feature_columns": [
            "cosine", "mutual_rank_mean", "mutual_rank_max",
            "aspect_absdiff", "area_ratio", "num_gap_min", "name_jaccard",
        ],
    }
    # Must not raise; should fall back to exhaustive pairs
    scene_map, _ = discover_scenes_v6(paths, lgeu_model_payload=lgeu_payload, verbose=True)
    assert len(scene_map) == 45


def test_discover_v6_lgeu_import_error_falls_back(tmp_path):
    from imc2025_lgeu.scene_clustering import discover_scenes_v6

    paths = _make_images(tmp_path, n=45, size=32)

    broken_model = MagicMock()
    broken_model.predict_proba.side_effect = ImportError("no module named descriptors")
    lgeu_payload = {
        "model": broken_model,
        "feature_columns": [
            "cosine", "mutual_rank_mean", "mutual_rank_max",
            "aspect_absdiff", "area_ratio", "num_gap_min", "name_jaccard",
        ],
    }
    scene_map, _ = discover_scenes_v6(paths, lgeu_model_payload=lgeu_payload, verbose=False)
    assert len(scene_map) == 45


# ---------------------------------------------------------------------------
# run_sfm_v6: pycolmap exception guards
# ---------------------------------------------------------------------------

def test_run_sfm_v6_feature_extraction_exception_returns_identity(tmp_path):
    from imc2025_lgeu.sfm import run_sfm_v6, _HAS_PYCOLMAP

    if not _HAS_PYCOLMAP:
        pytest.skip("pycolmap not available")

    paths = _make_images(tmp_path, n=3, size=64)

    with patch("imc2025_lgeu.sfm.pycolmap.extract_features",
               side_effect=RuntimeError("COLMAP internal crash")):
        poses, info = run_sfm_v6(paths, verbose=False)

    assert len(poses) == 3
    for R, t in poses.values():
        assert np.allclose(R, np.eye(3))
        assert np.allclose(t, np.zeros(3))
    assert "error" in info
    assert "feature_extraction_failed" in info["error"]


def test_run_sfm_v6_matching_exception_returns_identity(tmp_path):
    from imc2025_lgeu.sfm import run_sfm_v6, _HAS_PYCOLMAP

    if not _HAS_PYCOLMAP:
        pytest.skip("pycolmap not available")

    paths = _make_images(tmp_path, n=3, size=64)

    with patch("imc2025_lgeu.sfm.pycolmap.match_exhaustive",
               side_effect=RuntimeError("OOM in matching")):
        poses, info = run_sfm_v6(paths, verbose=False)

    assert len(poses) == 3
    assert "error" in info
    assert "matching_failed" in info["error"]


def test_run_sfm_v6_mapping_exception_returns_identity(tmp_path):
    from imc2025_lgeu.sfm import run_sfm_v6, _HAS_PYCOLMAP

    if not _HAS_PYCOLMAP:
        pytest.skip("pycolmap not available")

    paths = _make_images(tmp_path, n=3, size=64)

    with patch("imc2025_lgeu.sfm.pycolmap.incremental_mapping",
               side_effect=RuntimeError("COLMAP segfault")):
        poses, info = run_sfm_v6(paths, verbose=False)

    assert len(poses) == 3
    assert "error" in info
    assert "mapping_failed" in info["error"]


def test_run_sfm_v6_no_pycolmap_returns_identity(tmp_path):
    from imc2025_lgeu import sfm as sfm_module

    paths = _make_images(tmp_path, n=3, size=64)

    original = sfm_module._HAS_PYCOLMAP
    try:
        sfm_module._HAS_PYCOLMAP = False
        poses, info = sfm_module.run_sfm_v6(paths, verbose=False)
    finally:
        sfm_module._HAS_PYCOLMAP = original

    assert len(poses) == 3
    assert info.get("error") == "pycolmap_not_available"


def test_run_sfm_v6_single_image_returns_identity(tmp_path):
    from imc2025_lgeu.sfm import run_sfm_v6

    paths = _make_images(tmp_path, n=1, size=64)
    poses, info = run_sfm_v6(paths, verbose=False)
    assert len(poses) == 1
    assert info.get("registered", 0) == 0


# ---------------------------------------------------------------------------
# build_submission_v6: per-dataset isolation
# ---------------------------------------------------------------------------

def test_build_v6_no_images_gives_identity_submission(tmp_path):
    from imc2025_lgeu.submission import build_submission_v6
    from imc2025_lgeu.contract import SUBMISSION_COLUMNS

    sample_path = _sample_csv(tmp_path, dataset="ds1", n=3)
    output = tmp_path / "submission.csv"

    # No image files exist → identity poses for all rows
    result = build_submission_v6(
        sample_path=sample_path,
        image_root=tmp_path,
        output_path=output,
        verbose=False,
    )

    assert output.exists()
    assert list(result.columns) == SUBMISSION_COLUMNS
    assert len(result) == 3


def test_build_v6_clustering_exception_does_not_crash(tmp_path):
    from imc2025_lgeu.submission import build_submission_v6

    # Create real images so the pipeline enters the clustering branch
    test_dir = tmp_path / "test" / "ds"
    test_dir.mkdir(parents=True)
    for i in range(3):
        img = np.ones((64, 64, 3), dtype=np.uint8) * (i * 60 + 40)
        cv2.imwrite(str(test_dir / f"{i}.jpg"), img)

    sample_path = _sample_csv(tmp_path, dataset="ds", n=3)
    output = tmp_path / "submission.csv"

    with patch("imc2025_lgeu.submission.discover_scenes_v6",
               side_effect=RuntimeError("clustering crash")):
        result = build_submission_v6(
            sample_path=sample_path,
            image_root=tmp_path,
            output_path=output,
            verbose=True,
        )

    assert output.exists()
    assert len(result) == 3


def test_build_v6_sfm_exception_does_not_crash(tmp_path):
    from imc2025_lgeu.submission import build_submission_v6

    test_dir = tmp_path / "test" / "ds"
    test_dir.mkdir(parents=True)
    for i in range(3):
        img = np.ones((64, 64, 3), dtype=np.uint8) * (i * 60 + 40)
        cv2.imwrite(str(test_dir / f"{i}.jpg"), img)

    sample_path = _sample_csv(tmp_path, dataset="ds", n=3)
    output = tmp_path / "submission.csv"

    with patch("imc2025_lgeu.submission.run_sfm_per_scene_v6",
               side_effect=RuntimeError("SfM crash")):
        result = build_submission_v6(
            sample_path=sample_path,
            image_root=tmp_path,
            output_path=output,
            verbose=True,
        )

    assert output.exists()
    assert len(result) == 3


def test_build_v6_multiple_datasets_one_crashes(tmp_path):
    """A crash in dataset A does not prevent dataset B from being processed."""
    from imc2025_lgeu.submission import build_submission_v6
    from imc2025_lgeu.contract import SUBMISSION_COLUMNS

    # Two datasets in sample
    rows = []
    for ds in ("ds_a", "ds_b"):
        for i in range(2):
            rows.append({
                "image_id": f"{ds}/s/{i}.jpg",
                "dataset": ds,
                "scene": "s",
                "image": f"{i}.jpg",
                "rotation_matrix": "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0",
                "translation_vector": "0.0;0.0;0.0",
            })
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)

    # Create real images for ds_b only
    test_dir = tmp_path / "test" / "ds_b"
    test_dir.mkdir(parents=True)
    for i in range(2):
        img = np.ones((64, 64, 3), dtype=np.uint8) * (i * 100 + 50)
        cv2.imwrite(str(test_dir / f"{i}.jpg"), img)

    output = tmp_path / "submission.csv"

    # Patch so ds_a clustering raises, ds_b runs normally
    real_discover = __import__("imc2025_lgeu.scene_clustering", fromlist=["discover_scenes_v6"]).discover_scenes_v6

    call_count = {"n": 0}

    def _patched_discover(image_paths, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("first dataset crashes")
        return real_discover(image_paths, **kwargs)

    with patch("imc2025_lgeu.submission.discover_scenes_v6", side_effect=_patched_discover):
        result = build_submission_v6(
            sample_path=sample_path,
            image_root=tmp_path,
            output_path=output,
            verbose=True,
        )

    assert output.exists()
    assert len(result) == 4  # all 4 rows present
    assert list(result.columns) == SUBMISSION_COLUMNS


def test_build_v6_progressive_flush(tmp_path):
    """Partial CSV is written after the first dataset even if there's a second dataset to go."""
    from imc2025_lgeu.submission import build_submission_v6

    rows = []
    for ds in ("ds_a", "ds_b"):
        rows.append({
            "image_id": f"{ds}/s/0.jpg",
            "dataset": ds,
            "scene": "s",
            "image": "0.jpg",
            "rotation_matrix": "1.0;0.0;0.0;0.0;1.0;0.0;0.0;0.0;1.0",
            "translation_vector": "0.0;0.0;0.0",
        })
    sample_path = tmp_path / "sample.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)

    output = tmp_path / "submission.csv"
    wrote_after_first = {"v": False}
    original_write = __import__("imc2025_lgeu.contract", fromlist=["write_submission"]).write_submission

    call_count = {"n": 0}

    def _patched_write(out_rows, path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First progressive flush (after ds_a)
            wrote_after_first["v"] = True
        return original_write(out_rows, path)

    with patch("imc2025_lgeu.submission.write_submission", side_effect=_patched_write):
        build_submission_v6(
            sample_path=sample_path,
            image_root=tmp_path,
            output_path=output,
            verbose=False,
        )

    assert wrote_after_first["v"], "Expected a progressive flush after the first dataset"
