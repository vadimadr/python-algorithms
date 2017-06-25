"""Disjoint set data structure (union-find data structure) groups elements
into disjoint sets. Allows to find a set that contains x and union any two
sets in near linear time. Data structure is represented using disjoint-set
forest.

Any additional information about each set can be stored. Distance to the
leader can be stored (store tuples in the parent array). Explicit lists of
elements in each set can be stored (union would be O(M + N * log N)

Possible applications: Check if two vertices of graph are in the same
connected component, Kruskal's algorithm, check bipartite in online, RMQ in
offline.
"""


class DisjointSet:

    def __init__(self, data):
        sz = max(data)
        self.parent = [-1] * sz  # compressed parent array.
        self.rank = [0] * sz  # rank (height) of each set. Alternatively can be
        # implemented using size of each set.
        for el in data:
            self.add(el)

    def add(self, item):
        """Adds a new set with the single element item"""
        if len(self.parent) <= item:
            size_ = (item - len(self.parent) + 1)
            self.rank.extend([0] * size_)
            self.parent.extend([-1] * size_)

        self.rank[item] = 0
        self.parent[item] = item

    def find_set(self, item):
        if self.parent[item] == item:
            return item
        # path compression heuristic. Remembers the leader of each
        # element in the path.
        self.parent[item] = self.find_set(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        a = self.find_set(a)
        b = self.find_set(b)
        if a != b:
            # rank union heuristic. Append tree with lower size to the tree
            # with higher.
            if self.rank[a] < self.rank[b]:
                a, b = b, a
            self.parent[b] = a
            if self.rank[a] == self.rank[b]:
                self.rank[a] += 1
