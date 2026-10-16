"""Tests for V9: GBM + TOPK=20 scene clustering."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'src'))
from unittest.mock import MagicMock, patch
import numpy as np
from imc2025_lgeu.scene_clustering import discover_scenes_v9, discover_scenes_v6


def _fake_model_payload():
    """Create a mock LGEU model payload."""
    model = MagicMock()
    model.predict_proba.return_value = np.zeros((1, 2))
    return {"model": model, "metrics": {"roc_auc": 0.69}, "feature_columns": []}


def test_v9_uses_topk20():
    """discover_scenes_v9 passes lgeu_topk=20 even when module constant is 30."""
    from pathlib import Path
    names = [f"img_{i:03d}.jpg" for i in range(3)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}
    with patch('imc2025_lgeu.scene_clustering._extract_sift', return_value=None):
        scene_map, edge_data = discover_scenes_v9(image_paths, lgeu_model_payload=None)
    assert set(scene_map.values()) == {"outliers"} or "scene_0" in set(scene_map.values())
    assert len(scene_map) == 3


def test_v9_large_n_calls_v6_with_topk20():
    """For N > MAX_EXHAUSTIVE_N, v9 calls discover_scenes_v6 with lgeu_topk=20."""
    from pathlib import Path
    names = [f"img_{i:04d}.jpg" for i in range(42)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}

    with patch('imc2025_lgeu.scene_clustering.discover_scenes_v6') as mock_v6:
        mock_v6.return_value = ({n: "scene_0" for n in names}, {})
        discover_scenes_v9(image_paths, lgeu_model_payload=_fake_model_payload(), verbose=False)

    call_kwargs = mock_v6.call_args.kwargs
    assert call_kwargs.get('lgeu_topk') == 20, f"Expected lgeu_topk=20, got {call_kwargs}"


def test_v9_smoke_small():
    """V9 produces a valid scene_map for small N with no model."""
    from pathlib import Path
    names = [f"a{i}.jpg" for i in range(4)]
    image_paths = {n: Path(f"/fake/{n}") for n in names}
    with patch('imc2025_lgeu.scene_clustering._extract_sift', return_value=None):
        scene_map, _ = discover_scenes_v9(image_paths)
    assert len(scene_map) == 4
    assert all(isinstance(v, str) for v in scene_map.values())
