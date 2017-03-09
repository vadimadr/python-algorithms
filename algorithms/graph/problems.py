from algorithms.graph import Graph
from algorithms.graph.searching import restore_path
from algorithms.graph.utils import nodes_ids_to_labels


def find_cycle(g: Graph, v, p=None, c=None):
    """Find first cycle"""

    if p is None:
        p = [-1] * g.order()
        c = [0] * g.order()

    v_ = g.id[v]
    c[v_] = 1
    for u in g.neighbours(v):
        u_ = g.id[u]
        if c[u_] == 0:
            p[u_] = v_
            cycle = find_cycle(g, u, p, c)
            if cycle:
                return cycle
        elif p[v_] != u_:  # cycle found
            # cycle = [u_]
            # while v_ != u_:
            #     cycle.append(v_)
            #     v_ = p[v_]
            # cycle.append(u_)  # start of cycle
            # cycle.reverse()
            cycle = restore_path(p, v_)
            cycle.append(cycle[0])
            return nodes_ids_to_labels(g, cycle)
    c[v_] = 2
    return False
