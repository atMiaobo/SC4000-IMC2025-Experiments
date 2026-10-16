import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from imc2025_lgeu.graph import select_budgeted_graph
from imc2025_lgeu.lgeu import score_pairs


def test_model_scores_change_graph_selection():
    pair_df = pd.DataFrame(
        [
            {"image_a": "a", "image_b": "b", "cosine": 0.99, "mutual_rank_mean": 1, "mutual_rank_max": 1, "aspect_absdiff": 0, "area_ratio": 1, "num_gap_min": 0.1, "name_jaccard": 0.1},
            {"image_a": "b", "image_b": "c", "cosine": 0.20, "mutual_rank_mean": 2, "mutual_rank_max": 2, "aspect_absdiff": 0, "area_ratio": 1, "num_gap_min": 0.0, "name_jaccard": 0.9},
            {"image_a": "a", "image_b": "c", "cosine": 0.10, "mutual_rank_mean": 3, "mutual_rank_max": 3, "aspect_absdiff": 0, "area_ratio": 1, "num_gap_min": 0.0, "name_jaccard": 0.9},
        ]
    )
    cols = ["cosine", "mutual_rank_mean", "mutual_rank_max", "aspect_absdiff", "area_ratio", "num_gap_min", "name_jaccard"]
    x = pair_df[cols].to_numpy(dtype=np.float32)
    y = np.array([0, 1, 1])
    model = LogisticRegression().fit(x, y)
    learned = pair_df.copy()
    learned["score"] = score_pairs(learned, {"model": model, "feature_columns": cols})
    selected = select_budgeted_graph(learned, "score", budget=2)
    assert ("b", "c") in set(zip(selected["image_a"], selected["image_b"]))
