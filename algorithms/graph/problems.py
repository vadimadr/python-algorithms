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
        elif p[v] != u:  # cycle found
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
