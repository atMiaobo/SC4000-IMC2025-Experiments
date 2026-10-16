from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .contract import parse_translation
from .descriptors import global_descriptor, numeric_tokens


PAIR_FEATURE_COLUMNS = [
    "cosine",
    "mutual_rank_mean",
    "mutual_rank_max",
    "aspect_absdiff",
    "area_ratio",
    "num_gap_min",
    "name_jaccard",
]


@dataclass(frozen=True)
class PairRecord:
    dataset: str
    image_a: str
    image_b: str
    scene_a: str | None
    scene_b: str | None
    features: np.ndarray
    label: int | None = None
    pose_distance: float | None = None


def image_path(root: Path, split: str, dataset: str, image: str) -> Path:
    return root / split / dataset / image


def compute_descriptor_table(labels: pd.DataFrame, image_root: Path, split: str = "train") -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    descs: dict[str, np.ndarray] = {}
    for row in labels.itertuples(index=False):
        path = image_path(image_root, split, row.dataset, row.image)
        if not path.is_file():
            continue
        desc, meta = global_descriptor(path)
        key = f"{row.dataset}/{row.image}"
        descs[key] = desc
        rows.append(
            {
                "dataset": row.dataset,
                "scene": getattr(row, "scene", None),
                "image": row.image,
                "key": key,
                **meta,
            }
        )
    return pd.DataFrame(rows), descs


def _name_jaccard(a: str, b: str) -> float:
    toks_a = set(a.lower().replace(".", "_").split("_"))
    toks_b = set(b.lower().replace(".", "_").split("_"))
    if not toks_a and not toks_b:
        return 1.0
    return len(toks_a & toks_b) / max(len(toks_a | toks_b), 1)


def _num_gap(a: str, b: str) -> float:
    ta = numeric_tokens(a)
    tb = numeric_tokens(b)
    if not ta or not tb:
        return 1.0
    return min(abs(x - y) for x in ta for y in tb) / 1000.0


def build_pair_records(
    labels_or_meta: pd.DataFrame,
    descs: dict[str, np.ndarray],
    max_pairs_per_dataset: int | None = None,
    seed: int = 17,
    label_from_pose: bool = True,
) -> list[PairRecord]:
    rng = np.random.default_rng(seed)
    records: list[PairRecord] = []
    for dataset, group in labels_or_meta.groupby("dataset", sort=True):
        group = group.reset_index(drop=True)
        keys = [f"{r.dataset}/{r.image}" for r in group.itertuples(index=False)]
        present = [i for i, k in enumerate(keys) if k in descs]
        if len(present) < 2:
            continue
        matrix = np.stack([descs[keys[i]] for i in present])
        sim = matrix @ matrix.T
        ranks = np.argsort(np.argsort(-sim, axis=1), axis=1) + 1
        pairs = list(combinations(range(len(present)), 2))
        if max_pairs_per_dataset and len(pairs) > max_pairs_per_dataset:
            pairs = list(rng.choice(np.array(pairs, dtype=object), size=max_pairs_per_dataset, replace=False))
        for local_i, local_j in pairs:
            i, j = present[int(local_i)], present[int(local_j)]
            a = group.iloc[i]
            b = group.iloc[j]
            key_a, key_b = keys[i], keys[j]
            cosine = float(np.dot(descs[key_a], descs[key_b]))
            area_a = float(a.width * a.height)
            area_b = float(b.width * b.height)
            features = np.array(
                [
                    cosine,
                    float((ranks[present.index(i), present.index(j)] + ranks[present.index(j), present.index(i)]) / 2.0),
                    float(max(ranks[present.index(i), present.index(j)], ranks[present.index(j), present.index(i)])),
                    float(abs(a.aspect - b.aspect)),
                    float(min(area_a, area_b) / max(area_a, area_b, 1.0)),
                    float(_num_gap(a.image, b.image)),
                    float(_name_jaccard(a.image, b.image)),
                ],
                dtype=np.float32,
            )
            label = None
            pose_distance = None
            if label_from_pose and {"scene", "translation_vector"}.issubset(labels_or_meta.columns):
                same_scene = str(a.scene) == str(b.scene)
                try:
                    ta = parse_translation(a.translation_vector)
                    tb = parse_translation(b.translation_vector)
                    pose_distance = float(np.linalg.norm(ta - tb))
                    finite_pose = True
                except ValueError:
                    pose_distance = None
                    finite_pose = False
                label = int(
                    finite_pose
                    and same_scene
                    and 0.05 <= float(pose_distance) <= 5.0
                    and str(a.scene).lower() != "outliers"
                )
            records.append(
                PairRecord(
                    dataset=str(dataset),
                    image_a=str(a.image),
                    image_b=str(b.image),
                    scene_a=str(a.scene) if "scene" in group.columns else None,
                    scene_b=str(b.scene) if "scene" in group.columns else None,
                    features=features,
                    label=label,
                    pose_distance=pose_distance,
                )
            )
    return records


def records_to_frame(records: list[PairRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {
            "dataset": rec.dataset,
            "image_a": rec.image_a,
            "image_b": rec.image_b,
            "scene_a": rec.scene_a,
            "scene_b": rec.scene_b,
            "label": rec.label,
            "pose_distance": rec.pose_distance,
        }
        row.update({name: float(value) for name, value in zip(PAIR_FEATURE_COLUMNS, rec.features)})
        rows.append(row)
    return pd.DataFrame(rows)
