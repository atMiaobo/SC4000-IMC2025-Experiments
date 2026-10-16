from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .pairs import PAIR_FEATURE_COLUMNS


def scene_disjoint_split(pair_df: pd.DataFrame, holdout_fraction: float = 0.35, seed: int = 17) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scenes = sorted({str(x) for x in pd.concat([pair_df["scene_a"], pair_df["scene_b"]]).dropna().unique()})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(scenes))
    n_holdout = max(1, int(round(len(shuffled) * holdout_fraction)))
    holdout = set(shuffled[:n_holdout])
    valid = pair_df["scene_a"].isin(holdout) & pair_df["scene_b"].isin(holdout)
    train = ~pair_df["scene_a"].isin(holdout) & ~pair_df["scene_b"].isin(holdout)
    return train.to_numpy(), valid.to_numpy(), sorted(holdout)


def train_lgeu(pair_df: pd.DataFrame, seed: int = 17, model_type: str = "gbm") -> tuple[Pipeline, dict]:
    """Train LGEU pair classifier.

    Args:
        pair_df: DataFrame with PAIR_FEATURE_COLUMNS and 'label' column.
        seed: random seed for reproducibility.
        model_type: 'gbm' (HistGradientBoosting, default, AUC≈0.69) or
                    'lr' (LogisticRegression, legacy, AUC≈0.61).
    """
    pair_df = pair_df.dropna(subset=["label"]).copy()
    train_mask, valid_mask, holdout = scene_disjoint_split(pair_df, seed=seed)
    if pair_df.loc[train_mask, "label"].nunique() < 2:
        raise ValueError("training split does not contain both classes")
    if pair_df.loc[valid_mask, "label"].nunique() < 2:
        raise ValueError("validation split does not contain both classes")
    x_train = pair_df.loc[train_mask, PAIR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = pair_df.loc[train_mask, "label"].to_numpy(dtype=np.int32)
    x_valid = pair_df.loc[valid_mask, PAIR_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_valid = pair_df.loc[valid_mask, "label"].to_numpy(dtype=np.int32)
    if model_type == "gbm":
        base = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=seed,
        )
    else:
        base = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
            ]
        )
    model = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_valid)[:, 1]
    pred = (prob >= 0.5).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(y_valid, pred, average="binary", zero_division=0)
    metrics = {
        "train_pairs": int(train_mask.sum()),
        "valid_pairs": int(valid_mask.sum()),
        "holdout_scenes": holdout,
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_valid": float(y_valid.mean()),
        "roc_auc": float(roc_auc_score(y_valid, prob)),
        "average_precision": float(average_precision_score(y_valid, prob)),
        "brier": float(brier_score_loss(y_valid, prob)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
    }
    return model, metrics


def save_model(model, path: Path, metrics: dict, feature_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns or PAIR_FEATURE_COLUMNS,
            "metrics": metrics,
            "competition_slug": "image-matching-challenge-2025",
        },
        path,
    )


def load_model(path: Path):
    payload = joblib.load(path)
    if payload.get("competition_slug") != "image-matching-challenge-2025":
        raise ValueError("checkpoint was not created for image-matching-challenge-2025")
    return payload


def score_pairs(pair_df: pd.DataFrame, model_payload: dict | None) -> np.ndarray:
    if model_payload is None:
        return pair_df["cosine"].to_numpy(dtype=np.float32)
    cols = model_payload["feature_columns"]
    return model_payload["model"].predict_proba(pair_df[cols].to_numpy(dtype=np.float32))[:, 1]
