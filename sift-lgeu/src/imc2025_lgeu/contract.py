from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

COMPETITION_SLUG = "image-matching-challenge-2025"
SUBMISSION_COLUMNS = [
    "image_id",
    "dataset",
    "scene",
    "image",
    "rotation_matrix",
    "translation_vector",
]


def assert_competition_slug(slug: str) -> None:
    if slug != COMPETITION_SLUG:
        raise ValueError(f"wrong competition slug {slug!r}; expected {COMPETITION_SLUG!r}")


def parse_vector_field(text: str, expected: int) -> np.ndarray:
    parts = str(text).split(";")
    if len(parts) != expected:
        raise ValueError(f"expected {expected} semicolon-delimited values, got {len(parts)}")
    arr = np.asarray([float(p) for p in parts], dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("pose field contains non-finite values")
    return arr


def parse_rotation(text: str) -> np.ndarray:
    return parse_vector_field(text, 9).reshape(3, 3)


def parse_translation(text: str) -> np.ndarray:
    return parse_vector_field(text, 3)


def format_rotation(rotation: np.ndarray) -> str:
    arr = np.asarray(rotation, dtype=np.float64).reshape(9)
    if not np.all(np.isfinite(arr)):
        raise ValueError("rotation contains non-finite values")
    return ";".join(f"{x:.9f}" for x in arr)


def format_translation(translation: np.ndarray) -> str:
    arr = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        raise ValueError("translation contains non-finite values")
    return ";".join(f"{x:.9f}" for x in arr)


def read_sample(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"wrong sample schema: {list(df.columns)!r}")
    if df.empty:
        raise ValueError("sample submission is empty")
    if df["image_id"].isna().any() or df["image_id"].duplicated().any():
        raise ValueError("sample image_id values must be present and unique")
    return df


def validate_submission(sample: pd.DataFrame, submission: pd.DataFrame) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"wrong submission schema: {list(submission.columns)!r}")
    sample_ids = list(sample["image_id"])
    sub_ids = list(submission["image_id"])
    if len(sub_ids) != len(set(sub_ids)):
        raise ValueError("submission contains duplicate image_id values")
    if set(sub_ids) != set(sample_ids):
        missing = sorted(set(sample_ids) - set(sub_ids))[:5]
        extra = sorted(set(sub_ids) - set(sample_ids))[:5]
        raise ValueError(f"submission ID mismatch; missing={missing}, extra={extra}")
    for _, row in submission.iterrows():
        if not row["dataset"] or not row["scene"] or not row["image"]:
            raise ValueError("dataset, scene, and image must be non-empty")
        rotation = parse_rotation(row["rotation_matrix"])
        parse_translation(row["translation_vector"])
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-2):
            raise ValueError(f"malformed rotation for image_id={row['image_id']}")


def validate_kernel_metadata(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sources = data.get("competition_sources")
    if sources != [COMPETITION_SLUG]:
        raise ValueError(f"competition_sources must be exactly [{COMPETITION_SLUG!r}], got {sources!r}")
    dataset_sources = data.get("dataset_sources")
    if not dataset_sources:
        raise ValueError(
            "dataset_sources must be non-empty — add the slug of your code/weights Kaggle Dataset "
            "(e.g. 'username/imc2025-lgeu-assets'); offline Kaggle runs cannot find src/ or model "
            "without an attached dataset."
        )
    return data


def write_submission(rows: Iterable[dict], path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=SUBMISSION_COLUMNS)
    df.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)
    return df
