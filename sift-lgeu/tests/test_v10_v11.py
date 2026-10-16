"""Tests for V10 (TOPK=25) and V11 (lower Lowe threshold) scene clustering."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from unittest.mock import MagicMock, patch
import numpy as np
from imc2025_lgeu.scene_clustering import discover_scenes_v10, discover_scenes_v11


def _fake_model_payload():
    model = MagicMock()
    model.predict_proba.return_value = np.zeros((1, 2))
    return {"model": model, "metrics": {"roc_auc": 0.69}, "feature_columns": []}


# ---------------------------------------------------------------------------
# V10 tests: TOPK=25
# ---------------------------------------------------------------------------

def test_v10_large_n_calls_v6_with_topk25():
    """For N > MAX_EXHAUSTIVE_N, v10 calls discover_scenes_v6 with lgeu_topk=25."""
    from pathlib import Path
    names = [f"img_{i:04d}.jpg" for i in range(42)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}

    with patch('imc2025_lgeu.scene_clustering.discover_scenes_v6') as mock_v6:
        mock_v6.return_value = ({n: "scene_0" for n in names}, {})
        discover_scenes_v10(image_paths, lgeu_model_payload=_fake_model_payload(), verbose=False)

    call_kwargs = mock_v6.call_args.kwargs
    assert call_kwargs.get('lgeu_topk') == 25, f"Expected lgeu_topk=25, got {call_kwargs}"


def test_v10_smoke_small():
    """V10 produces a valid scene_map for small N with no model."""
    from pathlib import Path
    names = [f"a{i}.jpg" for i in range(4)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}
    with patch('imc2025_lgeu.scene_clustering._extract_sift', return_value=None):
        scene_map, _ = discover_scenes_v10(image_paths)
    assert len(scene_map) == 4
    assert all(isinstance(v, str) for v in scene_map.values())


# ---------------------------------------------------------------------------
# V11 tests: lower Lowe threshold for small N
# ---------------------------------------------------------------------------

def test_v11_small_n_uses_threshold_30():
    """For N <= MAX_EXHAUSTIVE_N, v11 calls discover_scenes_v6 with good_match_threshold=30."""
    from pathlib import Path
    # N=22 (ETs size) — exhaustive path
    names = [f"img_{i:02d}.jpg" for i in range(22)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}

    with patch('imc2025_lgeu.scene_clustering.discover_scenes_v6') as mock_v6:
        mock_v6.return_value = ({n: "scene_0" for n in names}, {})
        discover_scenes_v11(image_paths, lgeu_model_payload=None)

    call_kwargs = mock_v6.call_args.kwargs
    threshold = call_kwargs.get('good_match_threshold', None)
    assert threshold == 30, f"Expected good_match_threshold=30 for small N, got {threshold}"


def test_v11_large_n_uses_default_threshold():
    """For N > MAX_EXHAUSTIVE_N, v11 uses the caller-provided threshold (default 40)."""
    from pathlib import Path
    names = [f"img_{i:04d}.jpg" for i in range(42)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}

    with patch('imc2025_lgeu.scene_clustering.discover_scenes_v6') as mock_v6:
        mock_v6.return_value = ({n: "scene_0" for n in names}, {})
        discover_scenes_v11(image_paths, lgeu_model_payload=_fake_model_payload())

    call_kwargs = mock_v6.call_args.kwargs
    threshold = call_kwargs.get('good_match_threshold', 40)
    assert threshold == 40, f"Expected good_match_threshold=40 for large N, got {threshold}"


def test_v11_smoke_small():
    """V11 produces a valid scene_map for small N with no model."""
    from pathlib import Path
    names = [f"b{i}.jpg" for i in range(4)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}
    with patch('imc2025_lgeu.scene_clustering._extract_sift', return_value=None):
        scene_map, _ = discover_scenes_v11(image_paths)
    assert len(scene_map) == 4
    assert all(isinstance(v, str) for v in scene_map.values())
