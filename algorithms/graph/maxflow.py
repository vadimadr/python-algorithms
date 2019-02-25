import numpy as np

from algorithms.graph import BaseGraph


def ford_fulkerson(graph: BaseGraph, s=0, t=None):
    """Basic Ford-Fulkerson algorithm with DFS for augmented paths. Tries to find path from s to t in residual
    network (augmented path) and updates flow function values in found path"""
    n = graph.order()
    if t is None:
        t = n - 1

    maxflow = 0
    F = np.zeros((n, n))
    pred = np.full(n, -1)

    def dfs(v, flow=np.inf):
        # augmented path in residual network found
        if v == t:
            return flow

        for u, d in graph.successors(v):
            if pred[u] == -1 and F[v][u] < d:
                pred[u] = v
                pushed_flow = dfs(u, min(flow, d - F[v][u]))
                if pushed_flow:
                    # update flow in augmented path
                    F[v][u] += pushed_flow
                    F[u][v] -= pushed_flow
                    return pushed_flow
        return 0

    while True:
        pred[s] = s
        pred.fill(-1)
        added_flow = dfs(s)
        if added_flow == 0:
            break
        maxflow += added_flow

    return maxflow, F
