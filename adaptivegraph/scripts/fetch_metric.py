#!/usr/bin/env python3
from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution/main/imc25metric.py"
root = Path(__file__).resolve().parents[1]
out = root / "third_party" / "imc25metric.py"
out.parent.mkdir(parents=True, exist_ok=True)
if out.exists() and out.stat().st_size > 10000:
    print(f"Metric already present: {out}")
else:
    print(f"Fetching public IMC25 metric from {URL}")
    with urlopen(URL, timeout=60) as r:
        data = r.read()
    if len(data) < 10000:
        raise RuntimeError("Downloaded metric looks unexpectedly small")
    out.write_bytes(data)
    print(f"Saved {len(data):,} bytes -> {out}")
