#!/usr/bin/env python3
"""Fail-closed check of the OFFICIAL downloaded competition sample_submission.csv."""
import csv
from pathlib import Path
import sys

EXPECTED = [
    "image_id", "dataset", "scene", "image",
    "rotation_matrix", "translation_vector",
]
path = Path("contract/sample_submission.csv")
if not path.is_file():
    sys.exit(f"FAIL: official {path} missing; download it from the exact Kaggle competition")
with path.open(newline="", encoding="utf-8-sig") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames != EXPECTED:
        sys.exit(f"FAIL: wrong schema: {reader.fieldnames!r} != {EXPECTED!r}")
    rows = list(reader)
if not rows:
    sys.exit("FAIL: sample is empty")
ids = [row["image_id"] for row in rows]
if not all(ids) or len(ids) != len(set(ids)):
    sys.exit("FAIL: sample has missing or repeated image_id")
if any(not row["image"] or not row["dataset"] for row in rows):
    sys.exit("FAIL: sample has missing image or dataset")
print(f"PASS: official sample has exact expected schema and {len(rows)} unique image_ids")
print("Example rotation field:", repr(rows[0]["rotation_matrix"]))
print("Example translation field:", repr(rows[0]["translation_vector"]))
print("NOTE: The real hidden test may contain different rows. Do not hard-code count or IDs.")
