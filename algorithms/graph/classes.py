"""
Common operations:
1. Add edge
2. Remove edge
3. Search (check if edge (x,y) exists)
4. Get all neighbours for given node
5*. Add node (less priority)

V - number of vertices (nodes)
E - number of edges
K - number of adjacent nodes

Common graph representations are:
* Adjacency matrix (or distance matrix)
* Collection of edges (list or set)
* Collection of adjacent nodes (Adjacency list or Hash List)

Complexity:
Operation    | Adj. matrix | SortedEdgeList | AdjList
-------------+-------------+----------------+----------
Memory       | O(V^2)      | O(E)           | O(V + E)
add edge     | O(1)        | O(E*log E)     | O(1)
remove edge  | O(E)        | O(E*log E)     | O(K)
search       | O(1)        | O(log E)       | O(K)
enumerate    | O(V)        | O(log E + K)   | O(K)

In assumptions that: edge list is sorted

References:
    * https://arxiv.org/pdf/0908.3089.pdf
    * http://codeforces.com/blog/entry/12217?locale=ru&mobile=false
"""
from contextlib import suppress

import numpy as np
from abc import ABC, ABCMeta
from bisect import insort, bisect_left


class BaseGraph(ABC, metaclass=ABCMeta):
    # Each node has internal id from 0 to |V| - 1

    # To implement graph structure methods
    # init(), add_node(), remove_node(), distance() should be overriden
    # additionally enumerating methods (successors(), predcessors(), ...)
    # may be overridden for efficiency. It is obligatory if graph is multigraph

    def order(self):
        return self._n_nodes

    def size(self):
        # Number of edges (|E|)
        return self._n_edges

    def __iter__(self):
        # enumerate all nodes
        yield from self._nodes

    def __getitem__(self, item):
        yield from self.neighbours(item)

    def __contains__(self, item):
        return item in self._nodes

    def __init__(self, weighted=False, directed=False, *args, **kwargs):
        self._weighted = weighted
        self._directed = directed
        self._n_nodes = 0  # |V|
        self._n_edges = 0  # |E|

        # Mapping from node name to node id
        # e.g. {'Ivan': 0, 'Bob': 1, ...}
        self._nodes = {}

    @classmethod
    def from_edge_list(cls, edge_list, *args, **kwargs):
        graph = cls(*args, **kwargs)
        for edge in edge_list:
            graph.add_edge(*edge)
        return graph

    @classmethod
    def from_adjacency_matrix(cls, A, names=None, *args, **kwargs):
        graph = cls(*args, **kwargs)

        n = len(A)
        if names is None:
            names = list(range(n))

        for v in names:
            graph.add_node(v)

        for u, u_successors in enumerate(A):
            for v, d in enumerate(u_successors):
                if d > 0:
                    graph.add_edge(names[u], names[v], d)

        return graph

    @classmethod
    def from_adjacency_list(cls, adj_list, names=None, *args, **kwargs):
        graph = cls(*args, **kwargs)

        n = len(adj_list)
        if names is None:
            names = list(range(n))

        for v in names:
            graph.add_node(v)

        if not graph._weighted:
            for u, u_adj in enumerate(adj_list):
                for v in u_adj:
                    graph.add_edge(u, v)
        else:
            for u, u_adj in enumerate(adj_list):
                for v, w in u_adj:
                    graph.add_edge(u, v, w)

    def add_node(self, u):
        if u in self._nodes:
            raise Exception('Node already exists')

        self._nodes[u] = self._n_nodes
        self._n_nodes += 1

    def remove_node(self, u):
        # isolates node

        if self._directed:
            for v in self.predecessors(u):
                self.remove_edge(v, u)
        for v in self.successors(u):
            self.remove_edge(u,v)


    def has_node(self, u):
        return u in self._nodes

    def has_edge(self, u, v):
        return self.distance(u, v) > 0

    def distance(self, u, v):
        if u not in self._nodes:
            return 0
        if v not in self._nodes:
            return 0

    def add_edge(self, u, v, weight=None):
        if self._weighted and weight is None:
            raise Exception('Graph is weighted but weight is None')

        if not self.has_edge(u, v):
            self._n_edges += 1

        if u not in self._nodes:
            self.add_node(u)
        if v not in self._nodes:
            self.add_node(v)

    def add_edges_from(self, edges):
        for edge in edges:
            self.add_edge(*edge)

    def remove_edge(self, u, v):
        self._n_edges -= 1

    def successors(self, v):
        # if an arrow (x,y) exists,
        # then y is said to be a direct successor of x

        for name, node in self._nodes.items():
            d = self.distance(v, node)
            if d > 0:
                if self._weighted:
                    yield v, name, d
                else:
                    yield v, name

    def predecessors(self, v):
        # if an arrow (x,y) exists,
        # then x is said to be a direct predecessor of y

        for name, node in self._nodes.items():
            d = self.distance(node, v)
            if d > 0:
                if self._weighted:
                    yield v, name, d
                else:
                    yield v, name

    def neighbours(self, v):
        # Enumerate all adjacent nodes

        if not self._directed:
            yield from self.successors(v)
        else:
            yield from self.successors(v)
            yield from self.predecessors(v)

    def in_degree(self, v):
        # deg-(v)
        # the number of head ends adjacent to a vertex
        return sum(1 for _ in self.predecessors(v))

    def out_degree(self, v):
        # deg+(v)
        # the number of tail ends adjacent to a vertex
        # return sum(1 for _ in self.successors(v))

        k = 0
        t = []
        for i in self.successors(v):
            k += 1
            t.append(i)
        return k

    def degree(self, v):
        if self._directed:
            return self.in_degree(v) + self.out_degree(v)
        else:
            return self.out_degree(v)

    def _get_id(self, *nodes):
        if len(nodes) is 1:
            return self._nodes.get(nodes[0])
        return tuple(self._nodes.get(v) for v in nodes)

    def _get_names(self, ids):
        # for given node id yield node name

        id_set = frozenset(ids)
        for node_name, node_id in self._nodes.items():
            if node_id in id_set:
                yield node_name

    def _get_edges(self, u, edges):
        """
        Iterate all edges in the form (u, v, weight)

        :param u: heading node name (i.e. "A")
        :param edges: list of tailing nodes or pairs (node, weight)
        :return: all edges from node u given in a list
        """

        if self._weighted:
            id_set = dict(edges)
        else:
            id_set = frozenset(edges)

        if self._weighted:
            for node_name, node_id in self._nodes.items():
                if node_id in id_set:
                    yield node_name, id_set[node_id]
        else:
            for node_name, node_id in self._nodes.items():
                if node_id in id_set:
                    yield node_name

    @property
    def weighted(self):
        return self._weighted

    @property
    def directed(self):
        return self._directed


class AdjMxGraph(BaseGraph):
    # Adjacency matrix graph class.
    # _mx[u,v] is a distance from u to v
    # can not be multigraph and digraph simultaneously

    # equivalent to int g[V][V]
    # or vector<int>[V] or vector<vector<int>>

    def __init__(self, n_nodes=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # if self._multigraph:
        #     # TODO: implement multigraph
        #     raise Exception('Adjacency matrix can not be multigraph')

        if n_nodes is None:
            self._mx = np.zeros((16, 16))
        else:
            self._mx = np.zeros((n_nodes, n_nodes))

    def add_node(self, u):
        super().add_node(u)

        if self._n_nodes > self._mx.shape[0]:
            # add 16 to each dimension, fill with 0
            self._mx = np.pad(self._mx, ((0, 16), (0, 16)), mode='constant')

    def add_edge(self, u, v, weight=None):
        super().add_edge(u, v, weight)

        if not self._weighted:
            weight = 1

        u, v = self._get_id(u, v)

        self._mx[u, v] = weight

        if not self._directed:
            self._mx[v, u] = weight

    def remove_edge(self, u, v):
        super().remove_edge(u, v)
        u, v = self._get_id(u, v)
        if u is None or v is None:
            return

        if not self._directed:
            self._mx[v, u] = 0

        self._mx[u, v] = 0

    def predecessors(self, v):
        v_id = self._get_id(v)
        ids = np.where(self._mx[..., v_id] > 0)[0]
        if self._weighted:
            weights = self._mx[v_id, ids]
            yield from self._get_edges(v, zip(ids, weights))
        else:
            yield from self._get_edges(v, ids)

    def successors(self, v):
        v_id = self._get_id(v)
        ids = np.where(self._mx[v_id] > 0)[0]
        if self._weighted:
            weights = self._mx[v_id, ids]
            yield from self._get_edges(v, zip(ids, weights))
        else:
            yield from self._get_edges(v, ids)

    def distance(self, u, v):
        if super().distance(u, v) is 0:
            return 0
        u, v = self._get_id(u, v)

        return self._mx[u, v]


class AdjSetGraph(BaseGraph):
    # List of adjacency sets:
    # [{1,2,3}, {2}, {1,3}, ...]
    # _graph[v] is set of nodes adjacent to v

    # Can be implemented as vector<int>[V]
    # or as 3 arrays (headers, next, successors)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # _graph[v] is set of nodes adjacent to v
        self._graph = []

    def add_node(self, u):
        super().add_node(u)
        node = {} if self._weighted else set()
        self._graph.append(node)

    def add_edge(self, u, v, weight=None):
        super().add_edge(u, v, weight)
        if not self._weighted:
            weight = 1
        u, v = self._get_id(u, v)
        if self._weighted:
            self._graph[u][v] = weight
        else:
            self._graph[u].add(v)
            if not self._directed:
                self._graph[v].add(u)

    def distance(self, u, v):
        if super().distance(u, v) is 0:
            return 0
        u, v = self._get_id(u, v)
        if self._weighted:
            return self._graph[u][v]
        else:
            return 1 if v in self._graph[u] else 0

    def successors(self, v):
        v_id = self._get_id(v)
        yield from self._get_edges(v, self._graph[v_id])

    def predecessors(self, v):
        v_id = self._get_id(v)

        ids = []
        if not self._weighted:
            for u, adj_set in enumerate(self._graph):
                if v_id in adj_set:
                    ids.append(u)
            yield from self._get_edges(v, ids)
        else:
            weights = []
            for u, adj_set in enumerate(self._graph):
                if v_id in adj_set:
                    ids.append(u)
                    weights.append(adj_set[v_id])
            yield from self._get_edges(v, zip(ids, weights))

    def remove_edge(self, u, v):
        super().remove_edge(u, v)
        u, v = self._get_id(u, v)
        if u is None or v is None:
            return

        if self._weighted:
            self._graph[u].pop(v, None)
            if not self._directed:
                self._graph[v].pop(u, None)
        else:
            with suppress(KeyError):
                self._graph[u].remove(v)
                if not self._directed:
                    self._graph[v].remove(u)


class EdgeListGraph(BaseGraph):
    # List of edges graph:
    # [(0,1), (0,2), (2,3), ...]

    # Unsorted vs Sorted vs Set problem:
    # Unsorted is worst for searching (O(E))
    # Sorted is worse for adding and removing edges (O(E*log E))
    # Edge set is very bad for enumerating. (O(E))
    # Weights can be implemented as dict instead of set.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._edges = []

    def add_edge(self, u, v, weight=None):
        super().add_edge(u, v, weight)

        u, v = self._get_id(u, v)

        if self._weighted:
            node = (u, v, weight)
            node_reversed = (v, u, weight)
        else:
            node = (u, v)
            node_reversed = (v, u)

        if self._directed:
            insort(self._edges, node)
        else:
            insort(self._edges, node)
            insort(self._edges, node_reversed)

    def remove_edge(self, u, v):
        super().remove_edge(u, v)
        u, v = self._get_id(u, v)
        if u is None or v is None:
            return

        # binary search for edge
        i = bisect_left(self._edges, (u, v))
        # edge found
        if i != len(self._edges):

            node = self._edges[i]
            if node[0] == u and node[1] == v:
                self._edges.pop(i)

        if not self._directed:
            i_reversed = bisect_left(self._edges, (v, u))
            if i_reversed < len(self._edges):
                node = self._edges[i_reversed]
                if node[1] == u and node[0] == v:
                    self._edges.pop(i_reversed)

    def distance(self, u, v):
        if super().distance(u, v) == 0:
            return 0

        u, v = self._get_id(u, v)

        # binary search for edge
        i = bisect_left(self._edges, (u, v))
        # edge found
        if i != len(self._edges):

            if self._weighted:
                uu, vv, ww = self._edges[i]
            else:
                uu, vv = self._edges[i]
                ww = 1

            if u == uu and v == vv:
                return ww
        return 0

    def successors(self, v):
        v_id = self._get_id(v)

        i = bisect_left(self._edges, (v_id, 0))

        ids = []
        weights = []

        if i != len(self._edges):
            node = self._edges[i]

            while node[0] == v_id:
                ids.append(node[1])

                if self._weighted:
                    weights.append(node[2])

                i += 1
                if i >= len(self._edges):
                    break

                node = self._edges[i]
        if self._weighted:
            yield from self._get_edges(v, zip(ids, weights))
        else:
            yield from self._get_edges(v, ids)


class HashListGraph(BaseGraph):
    # Combination of hash map of edges and adjacency list

    # adding edge: O(1)
    # searching edge O(1)
    # enumerating O(K)

    # TODO: resizing, initializing, removing edge


    def distance(self, u, v):
        if super().distance(u, v) == 0:
            return 0

        u, v = self._get_id(u, v)

        h = hash((u, v))

        while self._used[h]:
            if self._edges[h] == (u, v):
                if self._weighted:
                    return self._weights[h]
                else:
                    return 1
            else:
                # linear probing
                h = (h + 1) % self._size
        return 0

    def add_edge(self, u, v, weight):
        super().add_edge(u, v, weight)

        u, v = self._get_id(u, v)

        h = hash((u, v))

        while self._used[h]:
            if self._edges[h] == (u, v):
                return
            else:
                # linear probing
                h = (h + 1) % self._size

        self._edges[h] = (u, v)
        self._used[h] = True
        self._next[h] = self._heads[u]
        self._heads[u] = h
        if self._weighted:
            self._weights[h] = weight

            # TODO: if size > F(loadfactor) then resize

    def successors(self, v):
        v_id = self._get_id(v)

        h = self._heads[v_id]

        ids = []
        weights = []
        while h != -1:
            uu, vv = self._edges[h]
            ids.append(vv)
            if self._weighted:
                weights.append(self._weights[h])
        if self._weighted:
            yield from self._get_edges(v, zip(ids, weights))
        else:
            yield from self._get_edges(v, ids)

    def remove_edge(self, u, v):
        super().remove_edge(u, v)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._heads = []  # _heads[x] is a pointer to first edge (x, y)
        self._next = []  # next pointer in a list
        self._edges = []  # which node edge goes to?
        self._used = []  # does this edge exist?

        self._size = 0  # size of hash map

        if self._weighted:
            self._weights = []

    def add_node(self, u):
        return super().add_node(u)