#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imc2025_lgeu.submission import build_submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, default=Path("contract/sample_submission.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data/official"))
    parser.add_argument("--model", type=Path, default=Path("models/lgeu.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/submission.csv"))
    parser.add_argument("--diagnostics", type=Path, default=Path("outputs/submission_diagnostics.json"))
    args = parser.parse_args()
    df = build_submission(args.sample, args.image_root, args.output, args.model, diagnostics_path=args.diagnostics)
    print(f"wrote {args.output} with {len(df)} rows")
    print(f"wrote {args.diagnostics}")


if __name__ == "__main__":
    main()
