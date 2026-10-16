#!/usr/bin/env python3
"""Download/cache DINOv2, ALIKED and LightGlue weights before a long run."""
from pathlib import Path
import os
import sys

import torch

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(root / "models" / "torch"))
os.environ.setdefault("HF_HOME", str(root / "models" / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(root / "models" / "huggingface" / "hub"))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Run inside `nix develop`, then source .venv/bin/activate.")

print("GPU:", torch.cuda.get_device_name(0))

print("Caching DINOv2...")
import timm
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
del model

print("Caching ALIKED + LightGlue...")
from lightglue import ALIKED, LightGlue
_ = ALIKED(max_num_keypoints=4096)
_ = LightGlue(features="aliked")

print("Model cache ready under:")
print("  TORCH_HOME:", os.environ["TORCH_HOME"])
print("  HF_HOME:", os.environ["HF_HOME"])
