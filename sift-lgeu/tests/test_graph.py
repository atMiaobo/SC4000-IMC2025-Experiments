import pandas as pd

from imc2025_lgeu.graph import select_budgeted_graph, select_topk


def _pairs():
    return pd.DataFrame(
        [
            {"image_a": "a", "image_b": "b", "score": 0.9},
            {"image_a": "b", "image_b": "c", "score": 0.8},
            {"image_a": "c", "image_b": "d", "score": 0.7},
            {"image_a": "a", "image_b": "d", "score": 0.6},
            {"image_a": "a", "image_b": "c", "score": 0.5},
        ]
    )


def test_budgeted_graph_is_deterministic_and_respects_budget():
    p = _pairs()
    first = select_budgeted_graph(p, "score", budget=3)
    second = select_budgeted_graph(p, "score", budget=3)
    assert first[["image_a", "image_b"]].values.tolist() == second[["image_a", "image_b"]].values.tolist()
    assert len(first) == 3
    assert set(first["image_a"]) | set(first["image_b"]) == {"a", "b", "c", "d"}


def test_topk_selects_edges_per_image():
    selected = select_topk(_pairs(), "score", per_image_k=1)
    assert len(selected) >= 2
    assert ("a" in set(selected["image_a"])) or ("a" in set(selected["image_b"]))
