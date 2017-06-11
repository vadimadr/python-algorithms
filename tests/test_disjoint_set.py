from algorithms.structures.disjoint_set import DisjointSet


def test_disjoint_set():
    a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ds = DisjointSet(a)
    assert ds.find_set(1) != ds.find_set(2)
    ds.union(1, 2)
    assert ds.find_set(1) == ds.find_set(2)
    assert ds.find_set(1) != ds.find_set(3)
    ds.union(2, 3)
    assert ds.find_set(1) == ds.find_set(2)
    assert ds.find_set(2) == ds.find_set(3)
