#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imc2025_lgeu.graph import select_budgeted_graph, select_topk
from imc2025_lgeu.lgeu import load_model, score_pairs


def summarize_selection(name: str, selected: pd.DataFrame, all_pairs: pd.DataFrame, elapsed: float) -> dict:
    y = selected["label"].astype(int)
    return {
        "method": name,
        "selected_pairs": int(len(selected)),
        "positive_rate_selected": float(y.mean()) if len(y) else 0.0,
        "precision_selected": float(precision_score(y, [1] * len(y), zero_division=0)) if len(y) else 0.0,
        "recall_of_all_positives": float(selected["label"].sum() / max(all_pairs["label"].sum(), 1)),
        "wall_time_sec": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, default=Path("experiments/pair_training_table.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/lgeu.joblib"))
    parser.add_argument("--out-csv", type=Path, default=Path("experiments/ablation_results.csv"))
    parser.add_argument("--out-json", type=Path, default=Path("experiments/ablation_receipt.json"))
    parser.add_argument("--out-fig", type=Path, default=Path("experiments/ablation_positive_rate.png"))
    parser.add_argument("--budget", type=int, default=220)
    args = parser.parse_args()

    pair_df = pd.read_csv(args.pairs).dropna(subset=["label"]).copy()
    pair_df["label"] = pair_df["label"].astype(int)
    payload = load_model(args.model)
    pair_df["lgeu_score"] = score_pairs(pair_df, payload)
    pair_df["retrieval_score"] = pair_df["cosine"]
    rows = []

    start = time.perf_counter()
    rows.append(summarize_selection("A_fixed_topk_retrieval", select_topk(pair_df, "retrieval_score", per_image_k=3), pair_df, time.perf_counter() - start))

    start = time.perf_counter()
    rows.append(summarize_selection("B_graph_retrieval_only", select_budgeted_graph(pair_df, "retrieval_score", budget=args.budget), pair_df, time.perf_counter() - start))

    start = time.perf_counter()
    rows.append(summarize_selection("C_lgeu_naive_topk", select_topk(pair_df, "lgeu_score", per_image_k=3), pair_df, time.perf_counter() - start))

    start = time.perf_counter()
    rows.append(summarize_selection("D_full_lgeu_budget_graph", select_budgeted_graph(pair_df, "lgeu_score", budget=args.budget), pair_df, time.perf_counter() - start))

    result = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_csv, index=False)
    receipt = {
        "pair_count": int(len(pair_df)),
        "positive_pairs": int(pair_df["label"].sum()),
        "lgeu_pair_auc": float(roc_auc_score(pair_df["label"], pair_df["lgeu_score"])),
        "lgeu_pair_average_precision": float(average_precision_score(pair_df["label"], pair_df["lgeu_score"])),
        "methods": rows,
    }
    args.out_json.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    plt.figure(figsize=(7, 4))
    plt.bar(result["method"], result["positive_rate_selected"])
    plt.ylabel("Selected positive rate")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=160)
    print(result.to_string(index=False))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
