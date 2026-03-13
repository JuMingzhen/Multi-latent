#!/usr/bin/env python3
"""
ProsQA 路径可视化工具：给定一条数据，绘制 DAG 图，
标出所有节点、边，并用不同颜色区分从节点 0 到目标 concept 的正确路径。
# 从 JSONL 中画第 0 条，只弹窗显示
python visualize_prosqa.py prosqa_sample.jsonl --index 0

# 从 JSONL 中画第 2 条并保存为 PNG
python visualize_prosqa.py prosqa_sample.jsonl -i 2 -o graph.png

# 从单条 JSON 文件画图
python visualize_prosqa.py single_record.json -o out.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx


def find_all_paths_bfs(
    start: int,
    end: int,
    edges: list[tuple[int, int]],
    max_paths: int = 100,
) -> list[list[int]]:
    """从 start 到 end 的所有路径（用于无 paths 字段的旧数据）。"""
    graph: dict[int, list[int]] = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)

    paths: list[list[int]] = []
    queue: deque[tuple[int, list[int]]] = deque([(start, [start])])
    while queue and len(paths) < max_paths:
        node, path = queue.popleft()
        if node == end:
            paths.append(path)
            continue
        for neighbor in graph.get(node, []):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return paths


def edges_to_tuples(edges: list) -> list[tuple[int, int]]:
    """将 [[u,v],...] 或 [(u,v),...] 统一为 list[tuple[int,int]]。"""
    out = []
    for e in edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((int(e[0]), int(e[1])))
    return out


def get_nodes_from_edges(edges: list[tuple[int, int]]) -> set[int]:
    nodes = set()
    for u, v in edges:
        nodes.add(u)
        nodes.add(v)
    return nodes


def dag_layer_layout(edges: list[tuple[int, int]], nodes: set[int]) -> dict[int, tuple[float, float]]:
    """
    按 DAG 层级布局：从入度为 0 的节点开始分层，同一层节点垂直排列。
    返回 node -> (x, y)，y 为层（0 为根），同一层内 x 均匀分布。
    """
    pred: dict[int, list[int]] = {n: [] for n in nodes}
    children: dict[int, list[int]] = {n: [] for n in nodes}
    in_degree = {n: 0 for n in nodes}
    for u, v in edges:
        if v in pred:
            pred[v].append(u)
        if u in children:
            children[u].append(v)
        in_degree[v] += 1

    roots = [n for n in nodes if in_degree[n] == 0]
    # 拓扑序 + 层（到根的最长距离）
    order: list[int] = []
    layer: dict[int, int] = {}
    q = deque(roots)
    for n in roots:
        layer[n] = 0
    while q:
        u = q.popleft()
        order.append(u)
        for v in children.get(u, []):
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
                layer[v] = layer[u] + 1

    # 同一层的节点按 order 中的顺序排，x 均匀
    by_layer: dict[int, list[int]] = {}
    for n in order:
        L = layer.get(n, 0)
        if L not in by_layer:
            by_layer[L] = []
        by_layer[L].append(n)

    pos = {}
    for L, nlist in sorted(by_layer.items()):
        width = max(1, len(nlist))
        for i, n in enumerate(nlist):
            x = (i - (width - 1) / 2) * 1.2
            y = -L * 1.0
            pos[n] = (x, y)
    return pos


def path_edges(paths: list[list[int]]) -> set[tuple[int, int]]:
    """从路径列表中收集所有 (u,v) 边（用于高亮）。"""
    out: set[tuple[int, int]] = set()
    for path in paths:
        for i in range(len(path) - 1):
            out.add((path[i], path[i + 1]))
    return out


def prepare_record(record: dict) -> dict:
    """
    统一数据格式：确保有 edges(list of tuple), nodes(set), paths(list of list),
    names(dict node->str), target_node(int)。
    若 record 无 paths，则根据 answer 与 concept_a_node/concept_b_node 计算。
    """
    edges = edges_to_tuples(record["edges"])
    nodes = get_nodes_from_edges(edges)
    names = record.get("names")
    if names is not None:
        names = {int(k): v for k, v in names.items()}
    else:
        names = {n: str(n) for n in nodes}

    paths = record.get("paths")
    target_node = record.get("concept_a_node") or record.get("concept_b_node")
    if paths is None and target_node is not None:
        answer = record.get("answer")
        concept_a = record.get("concept_a")
        concept_b = record.get("concept_b")
        concept_a_node = record.get("concept_a_node")
        concept_b_node = record.get("concept_b_node")
        if answer is not None and concept_a is not None and concept_b is not None:
            target_node = concept_a_node if answer == concept_a else concept_b_node
        paths = find_all_paths_bfs(0, target_node, edges)
    if paths is None:
        paths = []

    return {
        "edges": edges,
        "nodes": nodes,
        "paths": paths,
        "names": names,
        "target_node": target_node,
        "question": record.get("question", ""),
    }


def visualize(
    record: dict,
    out_path: str | Path | None = None,
    path_colors: list[str] | None = None,
    edge_non_path_color: str = "#c0c0c0",
    node_color: str = "#e8e8e8",
    node_path_color: str = "#b3d9ff",
    target_color: str = "#ffcccb",
    root_color: str = "#98fb98",
    figsize: tuple[float, float] = (12, 10),
    dpi: int = 150,
) -> None:
    """
    绘制 DAG：所有节点与边，正确路径上的边用不同颜色区分。
    """
    data = prepare_record(record)
    edges = data["edges"]
    nodes = data["nodes"]
    paths = data["paths"]
    names = data["names"]
    target_node = data["target_node"]
    question = data["question"]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = dag_layer_layout(edges, nodes)
    correct_edges = path_edges(paths)

    if path_colors is None:
        path_colors = list(mcolors.TABLEAU_COLORS) + ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"]

    # 为每条路径分配颜色，并记录每条边属于哪条路径（用于着色）
    edge_to_path_indices: dict[tuple[int, int], list[int]] = {}
    for idx, path in enumerate(paths):
        color = path_colors[idx % len(path_colors)]
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            if e not in edge_to_path_indices:
                edge_to_path_indices[e] = []
            edge_to_path_indices[e].append(idx)

    # 先画非路径边（灰色）
    non_path_edges = [e for e in edges if e not in correct_edges]
    nx.draw_networkx_edges(
        G, pos,
        edgelist=non_path_edges,
        edge_color=edge_non_path_color,
        arrows=True, arrowsize=14,
        connectionstyle="arc3,rad=0.05",
        width=1.2,
    )

    # 再画路径边：同一条边可能属于多条路径，取第一种路径颜色
    for e in correct_edges:
        if e not in G.edges:
            continue
        indices = edge_to_path_indices.get(e, [0])
        color = path_colors[indices[0] % len(path_colors)]
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[e],
            edge_color=color,
            arrows=True, arrowsize=14,
            connectionstyle="arc3,rad=0.05",
            width=2.5,
        )

    # 节点颜色：根 0、目标节点、路径上的其他节点、其余节点
    root_node = 0
    path_nodes = set()
    for path in paths:
        path_nodes.update(path)

    node_colors_list = []
    for n in G.nodes():
        if n == root_node:
            node_colors_list.append(root_color)
        elif n == target_node:
            node_colors_list.append(target_color)
        elif n in path_nodes:
            node_colors_list.append(node_path_color)
        else:
            node_colors_list.append(node_color)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors_list, node_size=800, edgecolors="black", linewidths=1.2)
    labels = {n: names.get(n, str(n)) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    plt.title(question or "ProsQA DAG", fontsize=11, wrap=True)
    plt.axis("off")
    plt.tight_layout()

    # 图例：路径条数
    if paths:
        legend_handles = []
        for idx, path in enumerate(paths[: min(len(paths), len(path_colors))]):
            from matplotlib.lines import Line2D
            legend_handles.append(
                Line2D([0], [0], color=path_colors[idx], linewidth=3, label=f"Path {idx + 1} (len={len(path)})")
            )
        if len(paths) > len(path_colors):
            from matplotlib.lines import Line2D
            legend_handles.append(
                Line2D([0], [0], color="gray", linewidth=2, label=f"... +{len(paths) - len(path_colors)} more")
            )
        plt.legend(handles=legend_handles, loc="upper left", fontsize=8)

    if out_path is not None:
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="ProsQA 路径可视化：给定一条数据，绘制 DAG 并高亮正确路径。")
    parser.add_argument("input", type=str, help="JSONL 文件路径，或单条 JSON 文件路径。")
    parser.add_argument("--index", "-i", type=int, default=0, help="JSONL 中条目的索引（从 0 开始）。")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出图片路径；不指定则只弹窗显示。")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI。")
    args = parser.parse_args()

    path = Path(args.input)
    if not path.is_file():
        raise FileNotFoundError(f"Not found: {path}")

    if path.suffix.lower() == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if args.index >= len(lines):
            raise IndexError(f"Index {args.index} out of range (file has {len(lines)} lines).")
        record = json.loads(lines[args.index])
    else:
        with open(path, "r", encoding="utf-8") as f:
            record = json.load(f)

    out_path = Path(args.output) if args.output else None
    visualize(record, out_path=out_path, dpi=args.dpi)


if __name__ == "__main__":
    main()
