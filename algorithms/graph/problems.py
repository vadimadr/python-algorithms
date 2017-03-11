from algorithms.graph import Graph
from algorithms.graph.searching import restore_path


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
