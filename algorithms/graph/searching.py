from . import BaseGraph


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

    # 0 - edge never visited
    # 1 - edge visited but not finished yet
    # 2 - edge is finished

    colors = [0] * V
    time_in = [0] * V
    time_out = [0] * V
    timestamp = 0

    # name -> id
    names = dict(zip(g, range(V)))

    def dfs_visit(v):
        nonlocal timestamp
        colors[v] = 1
        time_in[v] = timestamp
        timestamp += 1

        for w in g.successors(v):
            w = names[w]
            if colors[w] == 0:
                dfs_visit(w)
        colors[v] = 2
        time_out[v] = timestamp
        timestamp += 1

    for v in g:
        w = names[v]
        if colors[w] == 0:
            dfs_visit(w)
    return time_in, time_out


def dfs_iter(g: BaseGraph, v, used=None):
    if used is None:
        used = set()

    used.add(v)
    for u in g.successors(v):
        if u not in used:
            yield from dfs_iter(g, u, used)
    yield u


def bfs(g: BaseGraph, u):
    pass


def dijkstra_search(g: BaseGraph, u, v):
    pass
