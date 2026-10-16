from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm


@dataclass(frozen=True)
class CandidatePair:
    i: int
    j: int
    similarity: float


class DINOv2Retriever:
    """DINOv2 global retrieval.

    Local/Nix path: timm pretrained model.
    Kaggle/offline path: set IMC_DINO_LOCAL_PATH to a local HuggingFace
    DINOv2 model directory (e.g. /kaggle/input/dinov2/pytorch/base/1).
    """
    def __init__(self, model_name: str, device: str = "cuda", use_amp: bool = True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        local_path = os.environ.get("IMC_DINO_LOCAL_PATH")

        if local_path:
            from transformers import AutoImageProcessor, AutoModel
            self.backend = "transformers"
            self.processor = AutoImageProcessor.from_pretrained(local_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(local_path, local_files_only=True).eval().to(self.device)
            self.transform = None
        else:
            import timm
            from timm.data import resolve_model_data_config, create_transform
            self.backend = "timm"
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model = self.model.eval().to(self.device)
            data_cfg = resolve_model_data_config(self.model)
            self.transform = create_transform(**data_cfg, is_training=False)
            self.processor = None

    @torch.inference_mode()
    def encode(self, image_paths: Sequence[Path], batch_size: int = 8) -> np.ndarray:
        outputs: list[torch.Tensor] = []
        for start in tqdm(range(0, len(image_paths), batch_size), desc="DINOv2 retrieval"):
            batch_paths = image_paths[start : start + batch_size]
            if self.backend == "timm":
                batch = []
                for p in batch_paths:
                    with Image.open(p) as im:
                        batch.append(self.transform(im.convert("RGB")))
                x = torch.stack(batch).to(self.device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    feats = self.model(x)
            else:
                images = []
                for p in batch_paths:
                    with Image.open(p) as im:
                        images.append(im.convert("RGB").copy())
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                    out = self.model(**inputs)
                    feats = out.last_hidden_state[:, 0]

            feats = F.normalize(feats.float(), p=2, dim=1)
            outputs.append(feats.cpu())
        return torch.cat(outputs, dim=0).numpy().astype(np.float32)


def make_candidate_pairs(
    embeddings: np.ndarray,
    min_neighbors: int,
    max_neighbors: int,
    similarity_threshold: float,
) -> tuple[list[CandidatePair], np.ndarray]:
    if len(embeddings) < 2:
        return [], np.eye(len(embeddings), dtype=np.float32)

    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    pair_map: dict[tuple[int, int], float] = {}
    n = len(embeddings)
    kmin = min(max(1, min_neighbors), n - 1)
    kmax = min(max(kmin, max_neighbors), n - 1)

    for i in range(n):
        order = np.argsort(-sim[i])
        selected: list[int] = []
        for rank, j in enumerate(order[:kmax]):
            if rank < kmin or float(sim[i, j]) >= similarity_threshold:
                selected.append(int(j))
        for j in selected:
            a, b = sorted((i, j))
            pair_map[(a, b)] = max(pair_map.get((a, b), -1.0), float(sim[a, b]))

    pairs = [CandidatePair(i=a, j=b, similarity=s) for (a, b), s in pair_map.items()]
    pairs.sort(key=lambda p: p.similarity, reverse=True)
    return pairs, sim
