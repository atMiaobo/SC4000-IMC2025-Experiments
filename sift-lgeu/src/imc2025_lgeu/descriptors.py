from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np


def image_id_key(dataset: str, image: str) -> str:
    return f"{dataset}/{image}"


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def global_descriptor(path: Path) -> tuple[np.ndarray, dict]:
    image = load_image(path)
    h, w = image.shape[:2]
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).reshape(-1)
    hist = hist / max(float(hist.sum()), 1.0)
    gray_f = gray.astype(np.float32) / 255.0
    moments = np.array(
        [gray_f.mean(), gray_f.std(), np.percentile(gray_f, 10), np.percentile(gray_f, 90)],
        dtype=np.float32,
    )
    edges = cv2.Canny(gray, 80, 160)
    edge_stats = np.array([edges.mean() / 255.0], dtype=np.float32)
    desc = np.concatenate([hist.astype(np.float32), moments, edge_stats])
    norm = float(np.linalg.norm(desc))
    if norm > 0:
        desc = desc / norm
    return desc.astype(np.float32), {"width": w, "height": h, "aspect": w / max(h, 1)}


def numeric_tokens(name: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", name)]


def stable_unit_hash(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)
