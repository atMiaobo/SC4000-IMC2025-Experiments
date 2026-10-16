from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class UnionFind:
    parent: dict[str, str]

    @classmethod
    def from_nodes(cls, nodes: list[str]) -> "UnionFind":
        return cls({node: node for node in nodes})

    def find(self, node: str) -> str:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def select_topk(pair_df: pd.DataFrame, score_col: str, per_image_k: int) -> pd.DataFrame:
    selected = set()
    for image_col, other_col in [("image_a", "image_b"), ("image_b", "image_a")]:
        for image, group in pair_df.groupby(image_col, sort=True):
            top = group.sort_values([score_col, other_col], ascending=[False, True]).head(per_image_k)
            for idx in top.index:
                selected.add(idx)
    return pair_df.loc[sorted(selected)].copy()


def select_budgeted_graph(pair_df: pd.DataFrame, score_col: str, budget: int, min_degree: int = 2) -> pd.DataFrame:
    if pair_df.empty:
        return pair_df.copy()
    nodes = sorted(set(pair_df["image_a"]) | set(pair_df["image_b"]))
    budget = max(0, min(int(budget), len(pair_df)))
    ranked = pair_df.sort_values([score_col, "image_a", "image_b"], ascending=[False, True, True])
    selected: list[int] = []
    selected_set: set[int] = set()
    uf = UnionFind.from_nodes(nodes)
    for idx, row in ranked.iterrows():
        if len(selected) >= budget:
            break
        if uf.union(row.image_a, row.image_b):
            selected.append(idx)
            selected_set.add(idx)
    degree = {node: 0 for node in nodes}
    for idx in selected:
        row = pair_df.loc[idx]
        degree[row.image_a] += 1
        degree[row.image_b] += 1
    for idx, row in ranked.iterrows():
        if len(selected) >= budget:
            break
        if idx in selected_set:
            continue
        if degree[row.image_a] < min_degree or degree[row.image_b] < min_degree:
            selected.append(idx)
            selected_set.add(idx)
            degree[row.image_a] += 1
            degree[row.image_b] += 1
    for idx, _ in ranked.iterrows():
        if len(selected) >= budget:
            break
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
    return pair_df.loc[selected].copy()
