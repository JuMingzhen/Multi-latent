#!/usr/bin/env python3
"""
ProsQA dataset generation: DAG-based logical reasoning binary questions.
Algorithm 1 from the paper: Graph Construction for ProsQA.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Algorithm 1: DAG construction
# ---------------------------------------------------------------------------

def depth_to_root(c: int, edges: list[tuple[int, int]], nodes: set[int]) -> float:
    """
    Longest path length from node 0 or node 1 to node c.
    Used to assign sampling weights so deeper nodes are preferred (longer reasoning chains).
    """
    pred: dict[int, list[int]] = {n: [] for n in nodes}
    children: dict[int, list[int]] = {n: [] for n in nodes}
    for u, v in edges:
        if v in pred:
            pred[v].append(u)
        if u in children:
            children[u].append(v)
    dist = {n: 0 if n in (0, 1) else -1 for n in nodes}
    in_degree = {n: 0 for n in nodes}
    for u, v in edges:
        in_degree[v] += 1
    q = deque([n for n in nodes if in_degree[n] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in children[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                q.append(v)
    for v in order:
        if v in (0, 1):
            continue
        for u in pred[v]:
            if dist[u] >= 0 and dist[v] < dist[u] + 1:
                dist[v] = dist[u] + 1
    return max(0.0, float(dist.get(c, 0)))


def build_dag(N: int, rng: np.random.Generator) -> tuple[list[tuple[int, int]], set[int], dict[int, int], dict[int, set[int]]]:
    """
    Algorithm 1: Graph Construction for ProsQA.
    Returns: edges, nodes, labels, groups.
    Labels: 1 = descendant of node 0, 2 = descendant of node 1, 3 = both, 0 = neither.
    """
    edges: list[tuple[int, int]] = []
    nodes: set[int] = {0, 1}
    labels: dict[int, int] = {0: 1, 1: 2}
    # groups[l] = set of node ids with label l
    groups: dict[int, set[int]] = {0: set(), 1: {0}, 2: {1}, 3: set()}
    idx = 2

    while idx < N:
        n_in_nodes = rng.poisson(1.5)
        rand = rng.random()

        if rand < 0.35:
            candidates = groups[0] | groups[1]
        elif rand <= 0.7:
            candidates = groups[0] | groups[2]
        else:
            candidates = set(nodes)

        n_in_nodes = min(len(candidates), max(0, n_in_nodes))
        if n_in_nodes <= 0:
            in_nodes = []
        else:
            cand_list = list(candidates)
            weights = np.array([depth_to_root(c, edges, nodes) * 1.5 + 1.0 for c in cand_list])
            total = weights.sum()
            if total <= 0:
                probs = np.ones(len(cand_list)) / len(cand_list)
            else:
                probs = weights / total
            in_nodes = list(rng.choice(len(cand_list), size=min(n_in_nodes, len(cand_list)), replace=False, p=probs))
            in_nodes = [cand_list[i] for i in in_nodes]

        cur_label = 0
        for in_idx in in_nodes:
            cur_label |= labels[in_idx]
            edges.append((in_idx, idx))

        groups[cur_label].add(idx)
        labels[idx] = cur_label
        nodes.add(idx)
        idx += 1

    return edges, nodes, labels, groups


def get_leaf_nodes(edges: list[tuple[int, int]], nodes: set[int]) -> set[int]:
    """Nodes with no outgoing edges."""
    has_out = set()
    for u, v in edges:
        has_out.add(u)
    return nodes - has_out


def find_concept_leaves(
    edges: list[tuple[int, int]],
    nodes: set[int],
    labels: dict[int, int],
) -> tuple[int | None, int | None]:
    """
    Find one leaf with label 1 (Concept A) and one leaf with label 2 (Concept B).
    Node 0 is [Entity], so it must not be chosen as Concept A or B.
    Returns (node_id_for_A, node_id_for_B) or (None, None) if not both exist.
    """
    leaves = get_leaf_nodes(edges, nodes)
    leaf_1 = [n for n in leaves if labels.get(n) == 1 and n != 0]
    leaf_2 = [n for n in leaves if labels.get(n) == 2 and n != 0]
    a = random.choice(leaf_1) if leaf_1 else None
    b = random.choice(leaf_2) if leaf_2 else None
    return (a, b)


# ---------------------------------------------------------------------------
# Name assignment and question generation
# ---------------------------------------------------------------------------

def load_names(config_path: Path) -> tuple[list[str], list[str]]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["entity_names"], data["concept_names"]


def assign_node_names(
    nodes: set[int],
    edges: list[tuple[int, int]],
    entity_names: list[str],
    concept_names: list[str],
    rng: np.random.Generator,
) -> dict[int, str]:
    """
    Nodes without parents get entity names; others get concept names.
    """
    parents = set()
    for u, v in edges:
        parents.add(v)
    root_nodes = nodes - parents  # 0 and possibly 1 have no parents

    name_pool_entity = list(entity_names)
    name_pool_concept = list(concept_names)
    rng.shuffle(name_pool_entity)
    rng.shuffle(name_pool_concept)

    names: dict[int, str] = {}
    entity_idx = 0
    concept_idx = 0
    names[0] = name_pool_entity[0 % len(name_pool_entity)]
    entity_idx += 1
    for n in sorted(nodes)[1:]:
        if n in root_nodes:
            names[n] = name_pool_entity[entity_idx % len(name_pool_entity)]
            entity_idx += 1
        else:
            names[n] = name_pool_concept[concept_idx % len(name_pool_concept)]
            concept_idx += 1
    return names

def find_all_paths_dfs(
    start: int, 
    end: int, 
    edges: list[tuple[int, int]], 
    max_paths: int = 10
) -> list[list[int]]:
    """
    Find all paths from start to end.
    Return list of paths, each path is a list of node IDs.
    """
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
    
    paths = []
    def dfs(node, target, path, visited):
        if node == target:
            paths.append(path[:])
            return
        if len(paths) >= max_paths:
            return
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, target, path, visited)
                path.pop()
                visited.remove(neighbor)
    
    dfs(start, end, [start], {start})
    return paths

def find_all_paths_bfs(
    start: int, 
    end: int, 
    edges: list[tuple[int, int]], 
    max_paths: int = 10
) -> list[list[int]]:
    """
    Find all paths from start to end.
    Return list of paths, each path is a list of node IDs.
    """
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
    
    paths = []
    queue = deque([(start, [start])])
    while queue and len(paths) < max_paths:
        node, path = queue.popleft()
        if node == end:
            paths.append(path)
            continue
        for neighbor in graph.get(node, []):
            if neighbor not in path:
                queue.append((neighbor, path + [neighbor]))
    return paths

def generate_one_problem(
    N: int,
    entity_names: list[str],
    concept_names: list[str],
    rng: np.random.Generator,
    max_retries: int = 50,
) -> dict | None:
    """
    Build one DAG, assign names, pick Concept A (leaf label 1) and Concept B (leaf label 2),
    and form the question with random permutation of A and B.
    """
    for _ in range(max_retries):
        edges, nodes, labels, groups = build_dag(N, rng)
        concept_a_node, concept_b_node = find_concept_leaves(edges, nodes, labels)
        if concept_a_node is None or concept_b_node is None:
            continue

        names = assign_node_names(nodes, edges, entity_names, concept_names, rng)

        entity_name = names[0]
        concept_a_name = names[concept_a_node]
        concept_b_name = names[concept_b_node]
    
        # Random permutation of [Concept A] and [Concept B]
        if rng.random() < 0.5:
            opt1, opt2 = concept_a_name, concept_b_name
            answer = concept_a_name
        else:
            opt1, opt2 = concept_b_name, concept_a_name
            answer = concept_a_name  # still the correct answer
        paths = find_all_paths_dfs(0, concept_a_node, edges)
        question = f"Is {entity_name} a {opt1} or {opt2}?"
        return {
            "question": question, # [Entity] is [Concept A] or [Concept B]?
            "answer": answer, # [Concept A]
            "concept_a_node": concept_a_node,
            "concept_b_node": concept_b_node, # node id of [Concept B]
            "n_nodes": len(nodes), # number of nodes in the DAG
            "edges": edges, # edges of the DAG
            "names": names, # names of the nodes
            "labels": labels, # labels of the nodes
            "paths": paths, # all paths from [Entity] to [Concept A] and [Concept B]
            "number_of_paths": len(paths), # number of paths from [Entity] to [Concept A] and [Concept B]
            "mean_path_length": np.mean([len(path) for path in paths]), # mean length of the paths
        }
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate ProsQA logical reasoning dataset.")
    parser.add_argument("--output", "-o", type=str, default="prosqa.jsonl", help="Output JSONL file.")
    parser.add_argument("--num", "-n", type=int, default=1000, help="Number of problems to generate.")
    parser.add_argument("--nodes", type=int, default=20, help="Number of nodes N per DAG (Algorithm 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config", type=str, default="config/prosqa_names.json", help="Path to prosqa_names.json (default: config/prosqa_names.json).")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config) if args.config else script_dir / "config" / "prosqa_names.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entity_names, concept_names = load_names(config_path)
    if len(entity_names) < 1 or len(concept_names) < 2:
        raise ValueError("Config must have at least 1 entity name and 2 concept names.")

    failed = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(args.num):
            record = generate_one_problem(
                args.nodes,
                entity_names,
                concept_names,
                rng,
            )
            if record is None:
                failed += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1} problems...")

    if failed:
        print(f"Warning: {failed} problems skipped (no valid leaf pair).")
    print(f"Done. Wrote {args.num - failed} problems to {out_path}")


if __name__ == "__main__":
    main()
