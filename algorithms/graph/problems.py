from algorithms.graph import Graph
from algorithms.graph.searching import restore_path, dfs_iter


def find_cycle(g: Graph, v, p=None, c=None):
    """Find first cycle"""

    if p is None:
        p = [-1] * g.order()
        c = [0] * g.order()

    c[v] = 1
    for u in g.neighbours(v):
        if c[u] == 0:
            p[u] = v
            cycle = find_cycle(g, u, p, c)
            if cycle:
                return cycle
        elif c[u] == 1 and p[v] != u:  # cycle found
            # cycle = [u_]
            # while v_ != u_:
            #     cycle.append(v_)
            #     v_ = p[v_]
            # cycle.append(u_)  # start of cycle
            # cycle.reverse()
            cycle = restore_path(p, v)
            cycle.append(cycle[0])
            return cycle
    c[v] = 2
    return False


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
    n_odd = 0
    for v in g:
        if g.degree(v) % 2 == 1:
            n_odd += 1
    if n_odd > 2:
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
