from collections import deque

import numpy as np

from . import BaseGraph, to_unweighted, to_undirected, to_adjacency_matrix


def build_residual_graph(graph):
    # only edges matter
    # assume no multiedges
    residual_graph = to_undirected(to_unweighted(graph))
    assert residual_graph.order() > 0
    capacity = to_adjacency_matrix(graph)
    return residual_graph, capacity


def ford_fulkerson(graph: BaseGraph, s=0, t=None):
    """Basic Ford-Fulkerson algorithm with DFS for augmented paths. Tries to find path from s to t in residual
    network (augmented path) and updates flow function values in found path. Finds max_flow in graph with integer
    capacities in O(Ef), f = maximum flow"""
    n = graph.order()
    if t is None:
        t = n - 1

    graph, C = build_residual_graph(graph)
    max_flow = 0
    F = np.zeros((n, n))
    pred = np.full(n, -1)

    def dfs(v, flow=np.inf):
        # augmented path in residual network found
        if v == t:
            return flow

        for u in graph.successors(v):
            if pred[u] == -1 and F[v][u] < C[v][u]:
                pred[u] = v
                pushed_flow = dfs(u, min(flow, C[v][u] - F[v][u]))
                if pushed_flow:
                    # update flow in augmented path
                    F[v][u] += pushed_flow
                    F[u][v] -= pushed_flow
                    return pushed_flow
        return 0

    while True:
        pred.fill(-1)
        pred[s] = s
        added_flow = dfs(s)
        if added_flow == 0:
            break
        max_flow += added_flow

    return max_flow, F


def edmonds_karp(graph: BaseGraph, s=0, t=None):
    """Edmonds-Karp algorithm for maximum flow. It is variant of Furd-Fulkerson method with BFS for augmented paths.R
    Finds max_flow in O(V*E^2)
    """
    n = graph.order()
    if t is None:
        t = n - 1

    graph, C = build_residual_graph(graph)
    max_flow = 0
    F = np.zeros((n, n))
    pred = np.full(n, -1)

    def bfs():
        queue = deque()
        queue.append((s, np.inf))
        while queue:
            v, f = queue.popleft()
            for u in graph.successors(v):
                if pred[u] == -1 and F[v][u] < C[v][u]:
                    pred[u] = v
                    pushed = min(f, C[v][u] - F[v][u])
                    if u == t:
                        return pushed
                    queue.append((u, pushed))
        return 0

    while True:
        pred.fill(-1)
        pred[s] = s
        added_flow = bfs()
        if added_flow == 0:
            break
        max_flow += added_flow
        v = t
        while v != s:
            p = pred[v]
            F[p][v] += added_flow
            F[v][p] -= added_flow
            v = p
    return max_flow, F
