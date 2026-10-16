#!/usr/bin/env python3
"""Reproduce the hidden 'Submission Scoring Error' by running OUR submission writer output
through the official metric on synthetic hidden-like GT covering edge cases."""
import importlib.util, sys, tempfile, csv
from pathlib import Path
import numpy as np

AG = Path("/home/mihao/Projects/SC4000-IMC2025-2/IMC2025_AdaptiveGraph")
sys.path.insert(0, str(AG / "src"))
from adaptive_imc.submission import rows_for_dataset, write_submission
from adaptive_imc.reconstruction import PosePrediction

spec = importlib.util.spec_from_file_location("m", AG / "third_party" / "imc25metric.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

rng = np.random.default_rng(0)
def randR():
    A = rng.standard_normal((3, 3)); Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0: Q[:, 0] = -Q[:, 0]
    return Q

# ---- synthetic GT: 3 datasets ----
# A: 2 scenes x 3 imgs (normal). B: 1 scene x 4 imgs. C: 1 scene x 1 img (singleton gt).
gt_spec = {
    "dsA": {"sceneA0": ["a0.png","a1.png","a2.png"], "sceneA1": ["a3.png","a4.png","a5.png"]},
    "dsB": {"sceneB0": ["b0.png","b1.png","b2.png","b3.png"]},
    "dsC": {"sceneC0": ["c0.png"]},
}
def write_gt(path):
    with open(path,"w",newline="\n") as f:
        w=csv.writer(f); w.writerow(["dataset","scene","image","rotation_matrix","translation_vector"])
        for ds,sc in gt_spec.items():
            for s,imgs in sc.items():
                for im in imgs:
                    R=randR().ravel(); t=rng.standard_normal(3)
                    w.writerow([ds,s,im,";".join(f"{x:.9f}" for x in R),";".join(f"{x:.9f}" for x in t)])
def write_th(path):
    with open(path,"w",newline="\n") as f:
        w=csv.writer(f); w.writerow(["dataset","scene","thresholds"])
        for ds,sc in gt_spec.items():
            for s in sc: w.writerow([ds,s,";".join(str(x) for x in [0.1,0.25,0.5,1,2,5])])

# ---- user predictions via OUR writer, with edge cases ----
# dsA: registered into ONE merged cluster (over-merge) -> all cluster0
# dsB: ALL outliers (0 registered, nan poses)  <-- key edge case
# dsC: singleton outlier
def user_rows():
    rows=[]
    # dsA all images, all 'cluster0', real poses
    pathsA=[Path(x) for x in gt_spec["dsA"]["sceneA0"]+gt_spec["dsA"]["sceneA1"]]
    predsA={i:PosePrediction("cluster0", randR(), rng.standard_normal(3)) for i in range(len(pathsA))}
    rows+=rows_for_dataset("dsA", pathsA, predsA)
    # dsB all outliers nan
    pathsB=[Path(x) for x in gt_spec["dsB"]["sceneB0"]]
    predsB={i:PosePrediction("outliers", None, None) for i in range(len(pathsB))}
    rows+=rows_for_dataset("dsB", pathsB, predsB)
    # dsC singleton outlier
    pathsC=[Path("c0.png")]
    predsC={0:PosePrediction("outliers", None, None)}
    rows+=rows_for_dataset("dsC", pathsC, predsC)
    return rows

with tempfile.TemporaryDirectory() as td:
    td=Path(td)
    write_gt(td/"gt.csv"); write_th(td/"th.csv")
    # sample = exactly the gt images (dataset,image), scene placeholder
    import pandas as pd
    gt=pd.read_csv(td/"gt.csv"); sample=gt[["dataset","image"]].copy()
    sample.insert(0,"image_id",[f"{r.dataset}_{r.image}_public" for r in gt.itertuples()])
    sample["scene"]="cluster0"; sample["rotation_matrix"]="1;0;0;0;1;0;0;0;1"; sample["translation_vector"]="0;0;0"
    sample=sample[["image_id","dataset","scene","image","rotation_matrix","translation_vector"]]
    sample.to_csv(td/"sample.csv",index=False)
    write_submission(user_rows(), td/"sub.csv", sample_submission=td/"sample.csv")
    print("submission written. Running official metric...")
    try:
        res=M.score(gt_csv=str(td/"gt.csv"),user_csv=str(td/"sub.csv"),thresholds_csv=str(td/"th.csv"),
                    mask_csv=None,combo_mode="harmonic",inl_cf=0,strict_cf=-1,verbose=False)
        print("SCORE OK:", {k:round(float(np.mean(list(v.values()))),2) for k,v in res.items()})
    except Exception as e:
        import traceback; traceback.print_exc()
        print("\n*** SCORER CRASHED:", type(e).__name__, e)
