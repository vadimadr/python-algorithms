from itertools import chain

import numpy as np

from algorithms.graph import BaseGraph


def to_adjacency_matrix(g: BaseGraph):
    n = g.order()

    mx = np.zeros((n, n))
    names = {k: v for k, v in zip(g, range(n))}

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                ui, vi, w = names[u], names[v[0]], v[1]
            else:
                ui, vi, w = names[u], names[v], 1

            mx[ui, vi] = w

    return mx


def to_adjacency_list(g: BaseGraph):
    n = g.order()

    adj_list = [[] for _ in range(n)]
    names = {k: v for k, v in zip(g, range(n))}

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                ui, vi = names[u], names[v[0]]
                adj_list[ui].append((vi, v[1]))
            else:
                ui, vi = names[u], names[v]
                adj_list[ui].append(vi)

    return adj_list


def to_edge_list(g: BaseGraph):
    n = g.order()

    edge_list = []
    names = {k: v for k, v in zip(g, range(n))}

    if not g.directed:
        edge_set = set()

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                edge = names[u], names[v[0]], v[1]
                edge_rev = names[v[0]], names[u], v[1]
            else:
                edge = (names[u], names[v])
                edge_rev = names[v], names[u]
            if not g.directed and edge not in edge_set:
                edge_set.add(edge_rev)
                edge_set.add(edge)
                edge_list.append(edge)
            elif g.directed:
                edge_list.append(edge)

    return sorted(edge_list)


def subgraph(g: BaseGraph, nodes):
    nodes = frozenset(nodes)

    weighted = g.weighted
    directed = g.directed
    g_new = g.__class__(weighted=weighted, directed=directed)

    for u in g:
        if u in nodes:
            for v in g.successors(u):
                if g.weighted and v[0] in nodes:
                    g_new.add_edge(u, v[0], v[1])

                elif v in nodes:
                    g_new.add_edge(u, v)
    return g_new


def graph_union(g1: BaseGraph, g2: BaseGraph):
    weighted = g1.weighted
    directed = g1.directed
    g_new = g1.__class__(weighted, directed)

    for u in chain(g1, g2):
        for v in g1.successors(u):
            if g1.weighted:
                g_new.add_edge(u, v[0], v[1])
            else:
                g_new.add_edge(u, v)
    return g_new


def graph_join(g1, g2, weight=1):
    g1_nodes = list(g1)
    g2_nodes = list(g2)

    inter_edges = []

    for v in g1_nodes:
        for u in g2_nodes:
            inter_edges.append((u, v, weight))
            if g1.directed:
                inter_edges.append((v, u, weight))

    g_new = g1.__class__(directed=g1.directed, weighted=g1.weighted)

    for u in chain(g1, g2):
        for v in g1.successors(u):
            if g1.weighted:
                g_new.add_edge(u, v[0], v[1])
            else:
                g_new.add_edge(u, v)

    g_new.add_edges_from(inter_edges)


def graph_copy(g: BaseGraph):
    g_new = g.__class__(directed=g.directed, weighted=g.weighted)

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                g_new.add_edge(u, v[0], v[1])
            else:
                g_new.add_edge(u, v)
    return g_new


def graph_intersect(g1: BaseGraph, g2: BaseGraph):
    g1_edges = set(to_edge_list(g1))
    g2_edges = set(to_edge_list(g2))

    intersection = g1_edges.intersection(g2_edges)
    return g1.__class__.from_edge_list(intersection, directed=g1.directed,
                                       weighted=g1.weighted)


def to_undirected(g: BaseGraph, min_weight=True):
    g_new = g.__class__(weighted=g.weighted, directed=False)

    if g.weighted:
        edges = {}

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                # chose max weight between (x,y) and (y,x)
                e = (max(u, v[0]), min(u, v[0]))
                if e in edges:
                    if min_weight:
                        d = min(v[1], edges[e])
                        g_new.remove_edge(u, v[0])
                        g_new.add_edge(u, v[0], d)
                    else:
                        d = max(v[1], edges[e])
                        g_new.remove_edge(u, v[0])
                        g_new.add_edge(u, v[0], d)
                else:
                    g_new.add_edge(u, v[0], v[1])
            else:
                g_new.add_edge(u, v)
    return g_new


def to_unweighted(g: BaseGraph):
    g_new = g.__class__(directed=g.directed, weighted=False)

    for u in g:
        for v in g.successors(u):
            if g.weighted:
                g_new.add_edge(u, v[0])
            else:
                g_new.add_edge(u, v)
    return g_new


def is_complete_graph(g: BaseGraph):
    n = g.order()
    for u in g:
        if g.degree(u) != n - 1:
            return False
    return True
