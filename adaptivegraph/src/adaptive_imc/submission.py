from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .reconstruction import PosePrediction


# Unregistered / outlier images must still carry a FINITE pose: the official IMC scorer
# rejects non-finite values (NaN/Inf) -> "Submission Scoring Error". We use the standard
# placeholder (identity rotation + zero translation). Such poses are geometrically wrong, so
# they fall outside every mAA threshold (== not registered) and do not affect scene clustering,
# but they keep the submission finite and valid. (The vendored 7th-place metric tolerated NaN,
# which is why local scoring never caught this.)
def _matrix_to_string(R: np.ndarray | None) -> str:
    if R is None:
        R = np.eye(3, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(R)):
        R = np.eye(3, dtype=np.float64)
    return ";".join(f"{x:.12g}" for x in R.ravel())


def _vector_to_string(t: np.ndarray | None) -> str:
    if t is None:
        t = np.zeros(3, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(t)):
        t = np.zeros(3, dtype=np.float64)
    return ";".join(f"{x:.12g}" for x in t)


def rows_for_dataset(
    dataset: str,
    image_paths: list[Path],
    predictions: dict[int, PosePrediction],
) -> list[dict[str, str]]:
    rows = []
    for i, path in enumerate(image_paths):
        pred = predictions.get(i, PosePrediction("outliers", None, None))
        rows.append(
            {
                "dataset": dataset,
                "scene": pred.scene,
                "image": path.name,
                "rotation_matrix": _matrix_to_string(pred.rotation),
                "translation_vector": _vector_to_string(pred.translation),
            }
        )
    return rows


def write_submission(
    rows: list[dict[str, str]],
    output_path: str | Path,
    *,
    sample_submission: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred = pd.DataFrame(rows)

    if sample_submission is not None and Path(sample_submission).exists():
        sample = pd.read_csv(sample_submission)
        key_cols = [c for c in ("dataset", "image") if c in sample.columns]
        if len(key_cols) == 2:
            value_cols = ["scene", "rotation_matrix", "translation_vector"]
            # Handle zero/partial predictions robustly (e.g. initial pre-reconstruction
            # fallback submission, or a dataset that produced no rows): if pred lacks the
            # needed columns, use an empty slice so EVERY sample row falls back.
            need = ["dataset", "image"] + value_cols
            if all(c in pred.columns for c in need):
                pred_slim = pred[need]
            else:
                pred_slim = pd.DataFrame(columns=need)
            merged = sample.drop(columns=[c for c in value_cols if c in sample.columns]).merge(
                pred_slim,
                on=["dataset", "image"],
                how="left",
                validate="one_to_one",
            )
            # Fill any sample rows we produced no prediction for with the standard
            # outlier fallback (scene='outliers', NaN pose). This is required on Kaggle:
            # during a Save&Run commit only the visible test scenes exist, so most
            # sample rows are unseen; during the hidden rerun every image is processed.
            # Raising here would make the committed version unsubmittable.
            n_missing = int(merged["scene"].isna().sum())
            if n_missing:
                print(f"[write_submission] {n_missing}/{len(merged)} sample rows had no "
                      f"prediction (not in processed test dirs); filling outlier fallback.")
            fallback = {
                "scene": "outliers",
                "rotation_matrix": _matrix_to_string(None),
                "translation_vector": _vector_to_string(None),
            }
            for c in value_cols:
                merged[c] = merged[c].fillna(fallback[c])
            # Preserve the exact sample column order when possible.
            ordered = [c for c in sample.columns if c in merged.columns]
            extras = [c for c in merged.columns if c not in ordered]
            pred = merged[ordered + extras]

    pred.to_csv(output_path, index=False)
    return output_path
