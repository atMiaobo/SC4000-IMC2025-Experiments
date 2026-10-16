from __future__ import annotations

from dataclasses import dataclass
import networkx as nx


@dataclass
class EdgeStats:
    i: int
    j: int
    inliers: int
    ratio: float
    coverage: float
    similarity: float
    quality: float
    route: str


def build_graph(num_images: int, edges: list[EdgeStats]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_images))
    for e in edges:
        g.add_edge(
            e.i,
            e.j,
            inliers=e.inliers,
            ratio=e.ratio,
            coverage=e.coverage,
            similarity=e.similarity,
            quality=e.quality,
            route=e.route,
        )
    return g


def triangle_support(g: nx.Graph, i: int, j: int) -> int:
    if i not in g or j not in g:
        return 0
    return len(set(g.neighbors(i)).intersection(g.neighbors(j)))


def low_degree_nodes(g: nx.Graph, target_degree: int = 2) -> list[int]:
    return [n for n in g.nodes if g.degree[n] < target_degree]


def components(g: nx.Graph, min_size: int = 2) -> list[set[int]]:
    comps = [set(c) for c in nx.connected_components(g) if len(c) >= min_size]
    comps.sort(key=len, reverse=True)
    return comps
