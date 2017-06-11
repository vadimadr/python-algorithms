from collections import deque

from algorithms.structures.heap import BinaryHeap
from . import BaseGraph

inf = float('inf')


def dfs(g: BaseGraph, u):
    """
    Depth First Search.

    Parameters
    ----------
    g : Graph
        Graph to search in.
    u
        Starting vertex

    Returns
    -------
    time_in : list
        time_in[v] - time when node v first discovered

    time_out : list
        time_out[v] - time when search finishes discovering v's adjacency
        list.

    Notes
    --------

    Colors are used to find cycles in graph. If one of v's successors is gray
    then loop found. Timer is used to determine order of search.

    If we sort time_out in descending order then we will have nodes in
    topological order (topological sort)

    Every new iteration of outer loop we start in a new connected component.

    """

    V = g.order()

    # 0 - edge never visited (white)
    # 1 - edge visited but not finished yet (gray)
    # 2 - edge is finished (black)

    colors = [0] * V
    time_in = [0] * V
    time_out = [0] * V
    timestamp = 0

    def dfs_visit(v):
        nonlocal timestamp
        colors[v] = 1
        time_in[v] = timestamp
        timestamp += 1

        for w in g.successors(v):
            if colors[w] == 0:
                dfs_visit(w)
        colors[v] = 2
        time_out[v] = timestamp
        timestamp += 1

    for v in g:
        if colors[v] == 0:
            dfs_visit(v)
    return time_in, time_out


def dfs_iter(g: BaseGraph, v, used=None, preorder=True):
    if used is None:
        used = set()

    if preorder:
        yield v

    used.add(v)  # black
    for u in g.successors(v):
        if u not in used:
            yield from dfs_iter(g, u, used, preorder)

    if not preorder:  # post order
        yield v


def bfs(g: BaseGraph, v):
    """
    Breadth-first search all nodes within one connectivity component

    Parameters
    ----------
    g : Graph
    v
        Starting vertex

    Returns
    -------
    d : list
        d[u] is the number of hops from v to u
    p : list
        predecessors

    Notes
    ------
    BFS (with some modifications) is used for: shortest path in unweighted
    graph, find connectivity components, solve a game with minimal steps,
    find shortest cycle, find all edges or vertices of path, shortest even
    path [1]_

    Time, Memory: O(V + E)

    .. [1] http://e-maxx.ru/algo/bfs

    """
    n = g.order()  # |V|
    p, d = [-1] * n, [0] * n
    used = [False] * n
    q = deque((v,))
    used[v] = True

    while q:
        u = q.popleft()
        for u_ in g.successors(u):
            if not used[u_]:
                used[u_] = True
                q.append(u_)
                d[u_] = d[u] + 1
                p[u_] = u

    return d, p


def bfs_iter(g, v):
    n = g.order()  # |V|
    used = [False] * n
    q = deque((v,))
    used[v] = True

    while q:
        u = q.popleft()
        for u_ in g.successors(u):
            if not used[u_]:
                used[u_] = True
                q.append(u_)
        yield u


def restore_path(p: list, v):
    """
    Convert predecessors list to path list

    Parameters
    ----------
    p : list
        predecessors
    v
        searched node

    Returns
    -------
    path : list
        path[i] is the node visited in the ith step

    """
    path = [v]
    u = p[v]
    while u != -1:
        path.append(u)
        u = p[u]
    return list(reversed(path))


def dijkstra_search(g: BaseGraph, from_):
    """
    Find all shortest paths from node u to all others in O(N^2 + E)

    assume no negative weights

    Returns
    -------
    d : list
        d[u] is the number of hops from v to u
    p : list
        predecessors
    """
    n = g.order()  # |V|
    d = [inf] * n
    p = [-1] * n

    used = [False] * n
    pq = BinaryHeap()  # use priority queue for finding minimum.
    pq.push((0, from_))

    d[from_] = 0
    while pq.data:
        # unused node with minimal distance (d[v])
        dist, v = pq.pop()
        # heaps do not support deleting random element. So check if dist is
        # valid
        if used[v] or dist != d[v]:
            continue

        # unreachable from u, end of connected component
        if d[v] == inf:
            break

        used[v] = True
        # relaxation
        for u_, dist in g.successors(v):
            relaxed = d[v] + dist
            if relaxed < d[u_]:
                p[u_] = v
                d[u_] = relaxed
                pq.push((relaxed, u_))

    return d, p


def bellman_ford_search(g: BaseGraph, from_):
    """Shortest path problem using Bellman-Ford algorithm. Slower than
    Dijkstra's but allows edges with negative weights, ut without negative
    cycles. Complexity: O(n*m)
    """
    n = g.order()  # |V|
    d = [inf] * n
    p = [-1] * n

    # graph must be represented as edges list.
    edges = list(g.edges())

    d[from_] = 0
    for i in range(n - 1):
        relaxed = False  # early stopping
        for u_, v_, dist in edges:
            # first check is needed when negative edges to prevent inf - k
            if d[u_] < inf and d[v_] > d[u_] + dist:
                d[v_] = d[u_] + dist
                p[v_] = u_
                relaxed = True
        if not relaxed:
            break

    # additionally detect negative cycles here with additional iteration.

    return d, p
