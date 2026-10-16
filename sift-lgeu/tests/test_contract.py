from pathlib import Path

import pandas as pd
import pytest

from imc2025_lgeu.contract import (
    SUBMISSION_COLUMNS,
    assert_competition_slug,
    format_rotation,
    format_translation,
    read_sample,
    validate_kernel_metadata,
    validate_submission,
)


def test_slug_guard_rejects_wrong_competition():
    with pytest.raises(ValueError):
        assert_competition_slug("image-matching-challenge-2025-ongoing")


def test_sample_schema_matches_downloaded_contract():
    sample = read_sample(Path("contract/sample_submission.csv"))
    assert list(sample.columns) == SUBMISSION_COLUMNS
    assert sample["image_id"].is_unique


def test_submission_rejects_missing_extra_and_bad_rotation(tmp_path):
    sample = read_sample(Path("contract/sample_submission.csv")).head(2).copy()
    good_rows = []
    for row in sample.itertuples(index=False):
        good_rows.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "scene": row.scene,
                "image": row.image,
                "rotation_matrix": format_rotation([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                "translation_vector": format_translation([0, 0, 0]),
            }
        )
    validate_submission(sample, pd.DataFrame(good_rows, columns=SUBMISSION_COLUMNS))
    missing = pd.DataFrame(good_rows[:1], columns=SUBMISSION_COLUMNS)
    with pytest.raises(ValueError, match="ID mismatch"):
        validate_submission(sample, missing)
    bad = pd.DataFrame(good_rows, columns=SUBMISSION_COLUMNS)
    bad.loc[0, "rotation_matrix"] = "1;2;3"
    with pytest.raises(ValueError):
        validate_submission(sample, bad)
    bad = pd.DataFrame(good_rows, columns=SUBMISSION_COLUMNS)
    bad.loc[0, "rotation_matrix"] = "1;0;0;0;1;0;0;0;2"
    with pytest.raises(ValueError, match="malformed rotation"):
        validate_submission(sample, bad)


def test_kernel_metadata_guard(tmp_path):
    path = tmp_path / "kernel-metadata.json"
    # Valid: correct competition slug and non-empty dataset_sources
    path.write_text(
        '{"competition_sources":["image-matching-challenge-2025"],'
        '"dataset_sources":["mihao/imc2025-lgeu-assets"]}',
        encoding="utf-8",
    )
    validate_kernel_metadata(path)
    # Wrong competition slug
    path.write_text(
        '{"competition_sources":["image-matching-challenge-2025-ongoing"],'
        '"dataset_sources":["mihao/imc2025-lgeu-assets"]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        validate_kernel_metadata(path)
    # Empty dataset_sources — should fail (offline Kaggle won't find code/model)
    path.write_text(
        '{"competition_sources":["image-matching-challenge-2025"],'
        '"dataset_sources":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dataset_sources"):
        validate_kernel_metadata(path)
