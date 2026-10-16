#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--sample", type=Path, default=None)
    a=p.parse_args()
    df=pd.read_csv(a.csv)
    required=["dataset","scene","image","rotation_matrix","translation_vector"]
    missing=[c for c in required if c not in df.columns]
    if missing: raise SystemExit(f"Missing columns: {missing}")
    if df.duplicated(["dataset","image"]).any():
        raise SystemExit("Duplicate dataset/image rows detected")
    badR=[]; badt=[]
    for k,row in df.iterrows():
        try:
            R=np.array([float(x) for x in str(row.rotation_matrix).split(";")]).reshape(3,3)
        except Exception: badR.append(k)
        try:
            t=np.array([float(x) for x in str(row.translation_vector).split(";")]).reshape(3)
        except Exception: badt.append(k)
    if badR or badt: raise SystemExit(f"Malformed matrices: R rows {badR[:10]}, t rows {badt[:10]}")
    if a.sample and a.sample.exists():
        s=pd.read_csv(a.sample)
        expected=set(zip(s.dataset,s.image)); actual=set(zip(df.dataset,df.image))
        if expected != actual:
            raise SystemExit(f"Row identity mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    print(f"OK: {len(df)} rows, format valid")

if __name__ == "__main__": main()
