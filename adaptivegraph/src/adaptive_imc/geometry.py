from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math
import cv2
import numpy as np


@dataclass
class PairGeometry:
    pts0: np.ndarray
    pts1: np.ndarray
    raw_count: int
    inlier_count: int
    inlier_ratio: float
    coverage: float
    median_sampson: float
    F: Optional[np.ndarray]

    @property
    def is_valid(self) -> bool:
        return self.inlier_count >= 8


def _dedupe_correspondences(pts0: np.ndarray, pts1: np.ndarray, quant: float = 0.5):
    if len(pts0) == 0:
        return pts0.astype(np.float32), pts1.astype(np.float32)
    packed = np.concatenate([pts0, pts1], axis=1)
    keys = np.round(packed / quant).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    keep = np.sort(keep)
    return pts0[keep].astype(np.float32), pts1[keep].astype(np.float32)


def _grid_coverage(pts: np.ndarray, width: int, height: int, grid: int) -> float:
    if len(pts) == 0 or width <= 0 or height <= 0:
        return 0.0
    x = np.clip((pts[:, 0] / max(width, 1) * grid).astype(int), 0, grid - 1)
    y = np.clip((pts[:, 1] / max(height, 1) * grid).astype(int), 0, grid - 1)
    occupied = np.unique(y * grid + x).size
    return float(occupied / (grid * grid))


def spatial_coverage(
    pts0: np.ndarray,
    pts1: np.ndarray,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    grid: int = 4,
) -> float:
    h0, w0 = shape0
    h1, w1 = shape1
    return 0.5 * (
        _grid_coverage(pts0, w0, h0, grid) + _grid_coverage(pts1, w1, h1, grid)
    )


def sampson_errors(F: np.ndarray, pts0: np.ndarray, pts1: np.ndarray) -> np.ndarray:
    if F is None or F.shape != (3, 3) or len(pts0) == 0:
        return np.full(len(pts0), np.inf, dtype=np.float64)
    x0 = np.c_[pts0.astype(np.float64), np.ones(len(pts0))]
    x1 = np.c_[pts1.astype(np.float64), np.ones(len(pts1))]
    Fx0 = (F @ x0.T).T
    Ftx1 = (F.T @ x1.T).T
    numer = np.sum(x1 * Fx0, axis=1) ** 2
    denom = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return numer / np.maximum(denom, 1e-12)


def verify_correspondences(
    pts0: np.ndarray,
    pts1: np.ndarray,
    shape0: tuple[int, int],
    shape1: tuple[int, int],
    *,
    reproj_threshold: float = 4.0,
    confidence: float = 0.9999,
    max_iters: int = 20000,
    coverage_grid: int = 4,
) -> PairGeometry:
    pts0 = np.asarray(pts0, dtype=np.float32).reshape(-1, 2)
    pts1 = np.asarray(pts1, dtype=np.float32).reshape(-1, 2)
    pts0, pts1 = _dedupe_correspondences(pts0, pts1)
    raw_count = len(pts0)

    if raw_count < 8:
        return PairGeometry(
            pts0=np.empty((0, 2), dtype=np.float32),
            pts1=np.empty((0, 2), dtype=np.float32),
            raw_count=raw_count,
            inlier_count=0,
            inlier_ratio=0.0,
            coverage=0.0,
            median_sampson=float("inf"),
            F=None,
        )

    try:
        F, mask = cv2.findFundamentalMat(
            pts0,
            pts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=float(reproj_threshold),
            confidence=float(confidence),
            maxIters=int(max_iters),
        )
    except cv2.error:
        F, mask = None, None

    if F is None or mask is None:
        return PairGeometry(
            pts0=np.empty((0, 2), dtype=np.float32),
            pts1=np.empty((0, 2), dtype=np.float32),
            raw_count=raw_count,
            inlier_count=0,
            inlier_ratio=0.0,
            coverage=0.0,
            median_sampson=float("inf"),
            F=None,
        )

    if F.ndim == 2 and F.shape[0] > 3:
        F = F[:3, :3]
    mask = mask.ravel().astype(bool)
    if mask.size != raw_count:
        mask = mask[:raw_count]

    in0, in1 = pts0[mask], pts1[mask]
    n = len(in0)
    ratio = float(n / max(raw_count, 1))
    coverage = spatial_coverage(in0, in1, shape0, shape1, coverage_grid)
    errors = sampson_errors(F, in0, in1)
    median_sampson = float(np.median(errors)) if len(errors) else float("inf")

    return PairGeometry(
        pts0=in0,
        pts1=in1,
        raw_count=raw_count,
        inlier_count=n,
        inlier_ratio=ratio,
        coverage=coverage,
        median_sampson=median_sampson,
        F=F,
    )


def pair_quality_score(geom: PairGeometry, similarity: float) -> float:
    """A smooth score used only for ranking rescue candidates / graph edges."""
    e = geom.median_sampson
    err_term = 0.0 if not math.isfinite(e) else -0.10 * math.log1p(e)
    return (
        math.log1p(max(geom.inlier_count, 0))
        + 1.5 * geom.inlier_ratio
        + 2.0 * geom.coverage
        + 0.6 * float(similarity)
        + err_term
    )
