import numpy as np
import pytest

from algorithms.graph import (AdjMxGraph, AdjSetGraph, EdgeListGraph,
                              is_complete_graph, subgraph, to_adjacency_list,
                              to_adjacency_matrix, to_edge_list, to_undirected)
from algorithms.graph.searching import dfs_iter

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

    def test_named(self):
        s5_named = [('a', 'b'), ('a', 'c'), ('a', 'd'), ('a', 'e')]
        g = self.graph.from_edge_list(s5_named)

        for u in g:
            if u == 'a':
                assert g.degree(u) == 4
            else:
                assert g.degree(u) == 1


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
    @pytest.mark.skip
    def test_dfs_iter(self):
        # TODO: fix DFS
        g = self.graph.from_edge_list(c5)

        l = [4, 3, 2, 1, 0]

        assert list(dfs_iter(g, 0)) == l
