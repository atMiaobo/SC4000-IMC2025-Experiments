from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm


TensorDict = dict[str, object]


def _map_tensors(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _map_tensors(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_map_tensors(v, fn) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    return obj


def _to_cpu(obj):
    return _map_tensors(obj, lambda x: x.detach().cpu())


def _to_device(obj, device: torch.device):
    return _map_tensors(obj, lambda x: x.to(device, non_blocking=True))


def _unbatch(d: TensorDict) -> TensorDict:
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.ndim > 0 and v.shape[0] == 1:
            out[k] = v[0]
        else:
            out[k] = v
    return out


@dataclass
class FeatureView:
    feats: TensorDict
    to_original: Callable[[np.ndarray], np.ndarray]
    tag: str


class AdaptiveLightGlueMatcher:
    """ALIKED + LightGlue with cached base, tiled and rotation views.

    The original contribution is not a new descriptor; it is the adaptive routing:
    cheap full-image matching first, then expensive views only for ambiguous or
    topologically important pairs.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        max_keypoints: int = 4096,
        detection_threshold: float = 0.01,
        filter_threshold: float = 0.15,
        width_confidence: float = 0.99,
        depth_confidence: float = -1.0,
        base_resize: int = 1280,
        tile_resize: int = 1024,
        use_amp: bool = True,
    ):
        from lightglue import ALIKED, LightGlue
        from lightglue.utils import load_image

        self.load_image = load_image
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.base_resize = int(base_resize)
        self.tile_resize = int(tile_resize)

        self.extractor = ALIKED(
            max_num_keypoints=max_keypoints,
            detection_threshold=detection_threshold,
        ).eval().to(self.device)
        self.matcher = LightGlue(
            features="aliked",
            filter_threshold=filter_threshold,
            width_confidence=width_confidence,
            depth_confidence=depth_confidence,
        ).eval().to(self.device)

        self._base_cache: dict[str, FeatureView] = {}
        self._tile_cache: dict[str, list[FeatureView]] = {}
        self._rotation_cache: dict[tuple[str, int], FeatureView] = {}
        self._shape_cache: dict[str, tuple[int, int]] = {}

    def _read_tensor(self, path: Path) -> torch.Tensor:
        return self.load_image(path)

    def image_shape(self, path: Path) -> tuple[int, int]:
        key = str(path)
        if key not in self._shape_cache:
            image = self._read_tensor(path)
            self._shape_cache[key] = (int(image.shape[-2]), int(image.shape[-1]))
        return self._shape_cache[key]

    @torch.inference_mode()
    def _extract(self, image: torch.Tensor, resize: int | None) -> TensorDict:
        image = image.to(self.device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            feats = self.extractor.extract(image, resize=resize)
        return _to_cpu(feats)

    def base_view(self, path: Path) -> FeatureView:
        key = str(path)
        if key not in self._base_cache:
            image = self._read_tensor(path)
            self._shape_cache[key] = (int(image.shape[-2]), int(image.shape[-1]))
            feats = self._extract(image, self.base_resize)
            self._base_cache[key] = FeatureView(
                feats=feats,
                to_original=lambda p: np.asarray(p, dtype=np.float32),
                tag="base",
            )
        return self._base_cache[key]

    def tile_views(self, path: Path) -> list[FeatureView]:
        key = str(path)
        if key in self._tile_cache:
            return self._tile_cache[key]

        image = self._read_tensor(path)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        self._shape_cache[key] = (h, w)
        ymid, xmid = h // 2, w // 2
        boxes = [
            (0, 0, xmid, ymid, "q00"),
            (xmid, 0, w, ymid, "q10"),
            (0, ymid, xmid, h, "q01"),
            (xmid, ymid, w, h, "q11"),
        ]
        views: list[FeatureView] = []
        for x0, y0, x1, y1, tag in boxes:
            crop = image[:, y0:y1, x0:x1]
            feats = self._extract(crop, self.tile_resize)

            def mapper(p, x0=x0, y0=y0):
                q = np.asarray(p, dtype=np.float32).copy()
                q[:, 0] += x0
                q[:, 1] += y0
                return q

            views.append(FeatureView(feats=feats, to_original=mapper, tag=tag))
        self._tile_cache[key] = views
        return views

    def rotation_view(self, path: Path, k_ccw: int) -> FeatureView:
        k_ccw %= 4
        if k_ccw == 0:
            return self.base_view(path)
        key = (str(path), k_ccw)
        if key in self._rotation_cache:
            return self._rotation_cache[key]

        image = self._read_tensor(path)
        h, w = int(image.shape[-2]), int(image.shape[-1])
        self._shape_cache[str(path)] = (h, w)
        rot = torch.rot90(image, k_ccw, dims=(-2, -1))
        feats = self._extract(rot, self.base_resize)

        def inverse_map(points: np.ndarray) -> np.ndarray:
            p = np.asarray(points, dtype=np.float32)
            x, y = p[:, 0], p[:, 1]
            if k_ccw == 1:
                # original -> CCW: (x',y')=(y, W-1-x)
                xo = (w - 1) - y
                yo = x
            elif k_ccw == 2:
                xo = (w - 1) - x
                yo = (h - 1) - y
            else:  # k=3, original -> CW: (x',y')=(H-1-y, x)
                xo = y
                yo = (h - 1) - x
            return np.stack([xo, yo], axis=1).astype(np.float32)

        view = FeatureView(feats=feats, to_original=inverse_map, tag=f"rot{k_ccw * 90}")
        self._rotation_cache[key] = view
        return view

    @torch.inference_mode()
    def match_views(self, view0: FeatureView, view1: FeatureView) -> tuple[np.ndarray, np.ndarray]:
        f0 = _to_device(view0.feats, self.device)
        f1 = _to_device(view1.feats, self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            pred = self.matcher({"image0": f0, "image1": f1})
        pred = _unbatch(_to_cpu(pred))
        u0 = _unbatch(view0.feats)
        u1 = _unbatch(view1.feats)
        matches = pred.get("matches")
        if matches is None:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        if isinstance(matches, (list, tuple)):
            if len(matches) == 0:
                return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
            matches = matches[0]
        if matches.numel() == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        matches = matches.detach().cpu().numpy().astype(np.int64)
        k0 = u0["keypoints"].detach().cpu().numpy()[matches[:, 0]]
        k1 = u1["keypoints"].detach().cpu().numpy()[matches[:, 1]]
        return view0.to_original(k0), view1.to_original(k1)

    def match_base(self, path0: Path, path1: Path) -> tuple[np.ndarray, np.ndarray]:
        return self.match_views(self.base_view(path0), self.base_view(path1))

    def match_tiled(
        self,
        path0: Path,
        path1: Path,
        *,
        include_cross_base: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        views0 = [self.base_view(path0)] + self.tile_views(path0)
        views1 = [self.base_view(path1)] + self.tile_views(path1)
        pts0_all, pts1_all = [], []
        for i, v0 in enumerate(views0):
            for j, v1 in enumerate(views1):
                if i == 0 and j == 0:
                    continue  # already computed by base pass
                if not include_cross_base and (i == 0 or j == 0):
                    continue
                p0, p1 = self.match_views(v0, v1)
                if len(p0):
                    pts0_all.append(p0)
                    pts1_all.append(p1)
        if not pts0_all:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
        return np.concatenate(pts0_all), np.concatenate(pts1_all)

    def match_rotations(
        self,
        path0: Path,
        path1: Path,
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        v0 = self.base_view(path0)
        out = []
        for k in (1, 2, 3):
            p0, p1 = self.match_views(v0, self.rotation_view(path1, k))
            out.append((k, p0, p1))
        return out

    def clear_dataset_cache(self) -> None:
        self._base_cache.clear()
        self._tile_cache.clear()
        self._rotation_cache.clear()
        self._shape_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
