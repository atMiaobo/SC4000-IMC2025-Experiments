"""MASt3R pairwise matcher producing pixel correspondences in ORIGINAL image coordinates.

This is standard MASt3R API usage (NAVER MASt3R, CC-BY-NC-SA-4.0): encode each image once,
decode per pair, extract reciprocal-NN correspondences on the 24-D descriptors, map back to
original resolution. The coordinate-rescale math follows the public MASt3R demo / the
teammate reference `mast3r_colmap/match.py` (standard usage, attributed in THIRD_PARTY_NOTICES).

Interface mirrors AdaptiveGraph's matcher: `match_pair(path0, path1) -> (pts0, pts1)` as
float32 [N,2] arrays in original-image pixel coordinates, so the rest of the pipeline
(MAGSAC verification, graph, COLMAP) is reused unchanged.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def _add_mast3r_paths(third_party_root: str | Path) -> None:
    root = Path(third_party_root)
    for p in (root / "mast3r", root / "mast3r" / "dust3r", root / "mast3r" / "dust3r" / "croco"):
        sp = str(p)
        if p.exists() and sp not in sys.path:
            sys.path.insert(0, sp)


def _image_transform(path: Path, size: int = 512) -> dict:
    """Replicate dust3r load_images(size) resize+centre-crop so match coords can be
    mapped back to the original resolution. Mirrors mast3r_colmap/match.py."""
    with Image.open(path) as im:
        width, height = im.size
    rw, rh = (int(round(x * size / max(width, height))) for x in (width, height))
    cx, cy = rw // 2, rh // 2
    halfw = ((2 * cx) // 16) * 8
    halfh = ((2 * cy) // 16) * 8
    if rw == rh:
        halfh = int(3 * halfw / 4)
    return {
        "original_wh": [int(width), int(height)],
        "resized_wh": [int(rw), int(rh)],
        "crop_xy": [int(round(cx - halfw)), int(round(cy - halfh))],
        "processed_wh": [int(round(2 * halfw)), int(round(2 * halfh))],
    }


def _original_xy(xy: np.ndarray, t: dict) -> np.ndarray:
    scale = np.asarray(t["original_wh"], dtype=np.float64) / np.asarray(t["resized_wh"], dtype=np.float64)
    return ((np.asarray(xy, dtype=np.float64) + np.asarray(t["crop_xy"]) + 0.5) * scale - 0.5).astype(np.float32)


class MASt3RMatcher:
    def __init__(
        self,
        model_dir: str | Path,
        third_party_root: str | Path,
        *,
        device: str = "cuda",
        size: int = 512,
        subsample: int = 8,
        max_matches: int = 4096,
        conf_threshold: float = 1.001,  # desc_conf uses exp(.) >= 1; >1 keeps confident matches
        block_size: int = 2 ** 13,
        use_amp: bool = True,
    ):
        _add_mast3r_paths(third_party_root)
        from mast3r.model import AsymmetricMASt3R  # noqa
        from dust3r.utils.image import load_images  # noqa
        from mast3r.fast_nn import fast_reciprocal_NNs  # noqa

        self._load_images = load_images
        self._fast_nn = fast_reciprocal_NNs
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.size = int(size)
        self.subsample = int(subsample)
        self.max_matches = int(max_matches)
        self.conf_threshold = float(conf_threshold)
        self.block_size = int(block_size)

        self.model = AsymmetricMASt3R.from_pretrained(str(model_dir)).eval().to(self.device)

        self._enc_cache: dict[str, tuple] = {}   # path -> (feat_cpu, pos_cpu, shape)
        self._tf_cache: dict[str, dict] = {}     # path -> transform

    # ---- per-image encode (cached on CPU) ----
    @torch.inference_mode()
    def encode(self, path: Path) -> tuple:
        key = str(path)
        if key in self._enc_cache:
            return self._enc_cache[key]
        view = self._load_images([str(path)], size=self.size, verbose=False)[0]
        shape = torch.as_tensor(view["true_shape"], dtype=torch.int32)
        img = view["img"].to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            feat, pos, _ = self.model._encode_image(img, shape)
        self._enc_cache[key] = (feat.cpu(), pos.cpu(), shape)
        self._tf_cache[key] = _image_transform(path, self.size)
        return self._enc_cache[key]

    def transform(self, path: Path) -> dict:
        key = str(path)
        if key not in self._tf_cache:
            self.encode(path)
        return self._tf_cache[key]

    # ---- pairwise match -> original-image pixel correspondences ----
    @torch.inference_mode()
    def match_pair(self, path0: Path, path1: Path) -> tuple[np.ndarray, np.ndarray]:
        f0 = self.encode(path0)
        f1 = self.encode(path1)
        feat1, pos1, shape1 = f0[0].to(self.device), f0[1].to(self.device), f0[2]
        feat2, pos2, shape2 = f1[0].to(self.device), f1[1].to(self.device), f1[2]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            dec1, dec2 = self.model._decoder(feat1, pos1, feat2, pos2)
            pred1 = self.model._downstream_head(1, [x.float() for x in dec1], shape1)
            pred2 = self.model._downstream_head(2, [x.float() for x in dec2], shape2)

        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()
        xy0, xy1 = self._fast_nn(
            desc1, desc2,
            subsample_or_initxy1=self.subsample,
            device=self.device,
            dist="dot",
            block_size=self.block_size,
        )
        empty = (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
        if len(xy0) == 0:
            return empty

        t0, t1 = self.transform(path0), self.transform(path1)
        w0, h0 = t0["processed_wh"]
        w1, h1 = t1["processed_wh"]
        xy0 = np.asarray(xy0); xy1 = np.asarray(xy1)
        m = 3
        valid = (
            (xy0[:, 0] >= m) & (xy0[:, 0] < w0 - m) & (xy0[:, 1] >= m) & (xy0[:, 1] < h0 - m)
            & (xy1[:, 0] >= m) & (xy1[:, 0] < w1 - m) & (xy1[:, 1] >= m) & (xy1[:, 1] < h1 - m)
        )
        xy0, xy1 = xy0[valid], xy1[valid]
        if len(xy0) == 0:
            return empty

        # descriptor confidence (min over the two views)
        c0 = pred1["desc_conf"].squeeze(0).detach().cpu().numpy()[xy0[:, 1], xy0[:, 0]]
        c1 = pred2["desc_conf"].squeeze(0).detach().cpu().numpy()[xy1[:, 1], xy1[:, 0]]
        quality = np.minimum(c0, c1)
        keep = np.flatnonzero(np.isfinite(quality) & (quality >= self.conf_threshold))
        if len(keep) == 0:
            keep = np.flatnonzero(np.isfinite(quality))  # fallback: keep all finite
        if len(keep) > self.max_matches:
            keep = keep[np.argsort(-quality[keep], kind="stable")[: self.max_matches]]
        xy0, xy1 = xy0[keep], xy1[keep]

        return _original_xy(xy0, t0), _original_xy(xy1, t1)

    def clear_cache(self) -> None:
        self._enc_cache.clear()
        self._tf_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
