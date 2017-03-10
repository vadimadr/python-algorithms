from itertools import product
from operator import eq

import numpy as np
import pytest
from scipy.sparse import csgraph as scipy_graph

from algorithms.graph import (AdjMxGraph, AdjSetGraph, EdgeListGraph,
                              is_complete_graph, subgraph, to_adjacency_list,
                              to_adjacency_matrix, to_edge_list, to_undirected)
from algorithms.graph.problems import find_cycle
from algorithms.graph.searching import (bfs, bfs_iter, dfs_iter,
                                        dijkstra_search, restore_path)
from algorithms.graph.utils import normalize_edge_list, \
    normalize_adjacency_dict

k5 = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4),
      (3, 4)]
k7 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (1, 2), (1, 3), (1, 4),
      (1, 5), (1, 6), (2, 3), (2, 4), (2, 5), (2, 6), (3, 4), (3, 5), (3, 6),
      (4, 5), (4, 6), (5, 6)]

# complete bipariate graph
k3_3 = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]

k4_4m = [[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1],
         [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]

# star graph
s6 = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]

# cycle graph
c5 = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]


def labeled_path(path, labels):
    return [labels[i] for i in path]


@pytest.fixture(scope="class", params=[AdjMxGraph, AdjSetGraph, EdgeListGraph],
                ids=["Matrix", "Set", "Edge List"])
def graph_cls(request):
    request.cls.graph = request.param


@pytest.mark.usefixtures("graph_cls")
class TestGraphBasics:
    def testk5(self):
        g = self.graph.from_edge_list(k5)
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert g.has_edge(i, j)
            assert g.degree(i) == 4
            assert g.in_degree(i) == 4
            assert g.out_degree(i) == 4

    def testk7(self):
        g = self.graph.from_edge_list(k7)
        for i in range(7):
            for j in range(7):
                if i != j:
                    assert g.has_edge(i, j)

    def testk3_3(self):
        g = self.graph.from_edge_list(k3_3)
        for i in range(6):
            for j in range(6):
                if i == j:
                    assert not g.has_edge(i, j)
                if (i > 2) != (j > 2):
                    assert g.has_edge(i, j)
                else:
                    assert not g.has_edge(i, j)

    def test_star(self):
        g = self.graph.from_edge_list(s6)

        for i in range(6):
            if i == 0:
                assert g.degree(i) == 5
            else:
                assert g.degree(i) == 1

    def test_star_directed(self):
        g = self.graph.from_edge_list(s6, directed=True)

        for i in range(6):
            if i == 0:
                assert g.out_degree(i) == 5
                assert g.in_degree(i) == 0
            else:
                assert g.out_degree(i) == 0
                assert g.in_degree(i) == 1

    def test_cycle(self):
        g = self.graph.from_edge_list(c5)

        for i in range(5):
            assert g.degree(i) == 2

        g = self.graph.from_edge_list(c5, directed=True)

        for i in range(5):
            assert g.degree(i) == 2

    def test_from_adjmx(self):
        g = self.graph.from_adjacency_matrix(k4_4m)

        for i in range(8):
            for j in range(8):
                if i == j:
                    assert not g.has_edge(i, j)
                if (i > 3) != (j > 3):
                    assert g.has_edge(i, j)
                else:
                    assert not g.has_edge(i, j)

    def test_remove_edge(self):
        g = self.graph.from_edge_list(k7)

        g.remove_edge(2, 3)

        in_degs = []
        out_degs = []
        degs = []

        for i in range(7):
            in_degs.append(g.in_degree(i))
            out_degs.append(g.out_degree(i))
            degs.append(g.degree(i))

        assert sorted(in_degs) == [5, 5, 6, 6, 6, 6, 6]
        assert sorted(out_degs) == [5, 5, 6, 6, 6, 6, 6]
        assert sorted(degs) == [5, 5, 6, 6, 6, 6, 6]

    def test_remove_edge_directed(self):
        g = self.graph.from_edge_list(c5, directed=True)

        g.remove_edge(2, 3)

        in_degs = []
        out_degs = []
        degs = []

        for i in range(5):
            in_degs.append(g.in_degree(i))
            out_degs.append(g.out_degree(i))
            degs.append(g.degree(i))

        assert sorted(in_degs) == [0, 1, 1, 1, 1]
        assert sorted(out_degs) == [0, 1, 1, 1, 1]
        assert sorted(degs) == [1, 1, 2, 2, 2]

    def test_order(self):
        g = self.graph.from_edge_list(s6)

        assert g.order() == 6

        g = self.graph.from_edge_list(k7)

        assert g.order() == 7

    def test_iter(self):

        g = self.graph.from_edge_list(k5)

        for u in g:
            assert g.degree(u) == 4


def test_normalize_edge_list():
    e1 = [('a', 'b'), ('b', 'c'), ('c', 'a')]
    e2 = [(1, 2, 0.3), (2, 5, 0.6), (1, 3, 2.5), (3, 5, 0.1)]

    e1_, _ = normalize_edge_list(e1)
    e2_, m = normalize_edge_list(e2)

    assert e1_ == [(0, 1), (1, 2), (2, 0)]
    assert e2_ == [(0, 1, 0.3), (1, 3, 0.6), (0, 2, 2.5), (2, 3, 0.1)]

    for i, e in enumerate(e2):
        u, v, _ = e
        assert u == m.label[e2_[i][0]]
        assert v == m.label[e2_[i][1]]


def test_normalize_adj_dict():
    d1 = {'a': ['b', 'c', 'd'], 'b': ['a', 'c'], 'd': ['a']}

    d1_, _ = normalize_adjacency_dict(d1)

    assert d1_ == [[1, 2, 3], [0, 2], [], [0]]


@pytest.mark.usefixtures("graph_cls")
class TestGraphUtils:
    def test_to_adjmx(self):
        g = self.graph.from_edge_list(k3_3)
        m = np.array(
            [[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1],
             [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0]],
            dtype=float)

        assert (m == to_adjacency_matrix(g)).all()

    def test_to_adjlst(self):
        g = self.graph.from_edge_list(k3_3)

        l = [[3, 4, 5], [3, 4, 5], [3, 4, 5], [0, 1, 2], [0, 1, 2], [0, 1, 2]]

        assert l == to_adjacency_list(g)

    def test_to_edgelist(self):
        g = self.graph.from_edge_list(k3_3)

        l = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
             (2, 5)]

        assert sorted(l) == sorted(to_edge_list(g))

    def test_subgraph(self):
        g = self.graph.from_edge_list(k5)

        g2 = subgraph(g, [0, 1, 2])

        l = [(0, 1), (0, 2), (1, 2)]

        assert to_edge_list(g2) == l

    def test_is_complete(self):
        g = self.graph.from_edge_list(k5)

        assert is_complete_graph(g)

        g2 = subgraph(g, [0, 1, 2])

        assert is_complete_graph(g2)

    def test_to_undirected(self):
        g = self.graph.from_edge_list(k3_3, directed=True)
        g2 = to_undirected(g)

        l = [(0, 3), (0, 4), (0, 5), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4),
             (2, 5)]

        assert l == to_edge_list(g2)


@pytest.mark.usefixtures("graph_cls")
class TestSearch:
    def test_dfs_iter(self):
        path1 = [0, 1, 2, 3, 4]  # expected dfs order
        path2 = [0, 1, 4, 3, 2]

        graph = np.array([[0, 1, 2, 0, 0],
                          [1, 0, 0, 0, 3],
                          [2, 0, 0, 7, 0],
                          [0, 0, 7, 0, 1],
                          [0, 3, 0, 1, 0]])

        for directed, preorder in product([True, False], [True, False]):
            g1 = self.graph.from_edge_list(c5, directed=directed)
            g2 = self.graph.from_adjacency_matrix(graph, directed=directed)

            dfs1 = list(dfs_iter(g1, 0, preorder=preorder))
            dfs2 = list(dfs_iter(g2, 0, preorder=preorder))
            path1_ = path1 if preorder else reversed(path1)
            path2_ = path2 if preorder else reversed(path2)

            assert all(map(eq, dfs1, path1_))
            assert all(map(eq, dfs2, path2_))

    def test_bfs_iter(self):
        graph = np.array([[0, 1, 2, 0, 0],
                          [1, 0, 0, 0, 3],
                          [2, 0, 0, 7, 0],
                          [0, 0, 7, 0, 1],
                          [0, 3, 0, 1, 0]])

        path1 = [0, 1, 4, 2, 3]  # c5, undir
        path2 = [0, 1, 2, 3, 4]  # c5, dir
        path3 = [0, 1, 2, 4, 3]  # g2, dir / undir

        for directed in (True, False):
            g1 = self.graph.from_edge_list(c5, directed=directed)
            g2 = self.graph.from_adjacency_matrix(graph, directed=directed)

            bfs1 = list(bfs_iter(g1, 0))
            bfs2 = list(bfs_iter(g2, 0))

            assert all(map(eq, bfs2, path3))
            if not directed:
                assert all(map(eq, bfs1, path1))
            else:
                assert all(map(eq, bfs1, path2))

    def test_bfs_order(self):
        for directed in (True, False):
            g = self.graph.from_edge_list(c5, directed=directed)

            d, p = bfs(g, 0)

            if directed:
                assert p == [-1, 0, 1, 2, 3]
                assert d == [0, 1, 2, 3, 4]
            else:
                assert p == [-1, 0, 1, 4, 0]
                assert d == [0, 1, 2, 2, 1]

    def test_restore_path(self):
        assert restore_path([-1, 0, 1, 2, 3], 4) == [0, 1, 2, 3, 4]
        assert restore_path([-1, 0, 1, 4, 0], 3) == [0, 4, 3]
        assert restore_path([-1, 0, 1, 2, 3], 0) == [0]

    def test_dijkstra_directed(self):
        directed_G = np.array([[0, 3, 3, 0, 0],
                               [0, 0, 0, 2, 4],
                               [0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [2, 0, 0, 2, 0]], dtype=float)

        g = self.graph.from_adjacency_matrix(directed_G, directed=True,
                                             weighted=True)

        P = []
        D = []
        for i in g:
            d, p = dijkstra_search(g, i)
            P.append(p)
            D.append(d)

        D_, P_ = scipy_graph.dijkstra(directed_G, directed=True,
                                      return_predecessors=True)
        P_[P_ == -9999] = -1
        assert (np.array(P, dtype=int) - P_ == 0).all()
        assert (np.array(P, dtype=float) - P_ < 1e-6).all()

    def test_dijkstra_undirected(self):
        undirected_G = np.array([[0, 3, 3, 1, 2],
                                 [3, 0, 0, 2, 4],
                                 [3, 0, 0, 0, 0],
                                 [1, 2, 0, 0, 2],
                                 [2, 4, 0, 2, 0]], dtype=float)

        g = self.graph.from_adjacency_matrix(undirected_G, directed=False,
                                             weighted=True)

        P = []
        D = []
        for i in g:
            d, p = dijkstra_search(g, i)
            P.append(p)
            D.append(d)

        D_, P_ = scipy_graph.dijkstra(undirected_G, return_predecessors=True)
        P_[P_ == -9999] = -1
        assert (np.array(P, dtype=int) - P_ == 0).all()
        assert (np.array(P, dtype=float) - P_ < 1e-6).all()

    def test_find_cycle(self):
        test_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

        g = self.graph.from_edge_list(test_edges)
        cycle = find_cycle(g, 1)
        assert cycle is not False
