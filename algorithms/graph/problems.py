from collections import deque

from algorithms.graph import Graph
from algorithms.graph.searching import dfs_iter, restore_path


def find_cycle(g: Graph, v, p=None, c=None):
    """Find first cycle"""

    if p is None:
        p = [-1] * g.order()
        c = [0] * g.order()

    c[v] = 1
    for u in g.successors(v, distances=False):
        if c[u] == 0:
            p[u] = v
            cycle = find_cycle(g, u, p, c)
            if cycle:
                return cycle
        elif c[u] == 1 and p[v] != u:  # cycle found
            cycle = restore_path(p, v)
            cycle.append(cycle[0])
            return cycle
    c[v] = 2
    return False


def is_connected(g: Graph, start=0):
    q = deque()

    q.append(start)
    used = [False] * g.order()
    while q:
        v = q.popleft()
        used[v] = True
        for u in g.successors(v, distances=False):
            if not used[u]:
                q.append(u)
    return False not in used


def topological_sort(g: Graph):
    """Reorder nodes of acyclic oriented graph, in the way that
    for any edge (u, w): u < w

    Topological order is not unique

    Returns
    --------
    t
       t[i] is the ith node in topological order
    """
    t = []
    n = g.order()
    c = [0] * n

    def dfs(u):
        c[u] = 1
        for v in g.successors(u):
            if c[v] == 0:
                if not dfs(v):
                    return False
            elif c[v] == 1:
                return False
        c[u] = 2
        t.append(u)
        return True

    for i in range(n):
        if c[i] == 0:
            if not dfs(i):
                return False
    t.reverse()
    return t


def euler_graph_test(g: Graph):
    """Test whether g is euler graph"""

    # Number of odd vertices is 0 or 2 (start and end of euler path)
    p = None  # start of path
    q = None  # end of path
    for v in g:
        if g.degree(v) % 2 == 1:
            if g.directed:
                if g.in_degree(v) == g.out_degree(v) - 1:
                    if p is not None:  # there is only one starting node
                        return False
                    p = v
                else:
                    if q is not None:
                        return False
                    q = v
            # for undirected graphs only parity of node is important
            else:
                if p is None:
                    p = v
                elif q is None:
                    q = v
                else:
                    return False

    # Only one connected component
    visited = [False] * g.order()
    dfs_done = False
    for v in g:
        if g.degree(v) > 0:
            if dfs_done and not visited[v]:
                return False
            for u in dfs_iter(g, v):
                visited[u] = True
            dfs_done = True

    return True


def euler_path(g: Graph):
    """Find euler path or cycle

    Correctness:
    1. every edge is visited once (because edges are deleted and algorithm
    won't stop while there is edges left
    2. p forms up a correct path

    so p is the euler path
    """
    if not euler_graph_test(g):
        return False

    # if there is an odd vertex, then start from it
    v = None
    for u in g:
        if g.degree(u) % 2 == 1:
            if not g.directed or g.in_degree(u) == g.out_degree(u) - 1:
                v = u
                break
        if v is None and g.degree(u) > 0:
            v = u

    p = []
    s = [v]  # stack can be replaced with simple recursive solution
    while s:
        w = s[-1]
        for u in g.successors(w):
            s.append(u)
            g.remove_edge(w, u)
            break
        # all incident edges are processed
        if w == s[-1]:
            p.append(s.pop())
    p.reverse()
    return p
