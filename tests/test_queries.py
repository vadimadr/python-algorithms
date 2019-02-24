from hypothesis import assume, given
from hypothesis.strategies import composite, lists, integers, tuples, permutations

from algorithms.structures.range_queries.fenwick_tree import FenwickTree
from algorithms.structures.range_queries.sqrt_decomposition import SQRTDecomposition


@composite
def shuffle(draw, data):
    p = draw(permutations(data))
    return [data[i] for i in p]


@composite
def intervals(draw, arr_size):
    queries = draw(lists(integers(1, arr_size)))
    return [('q', r) for r in queries]


@composite
def ranges(draw, arr_size):
    idxs = integers(1, arr_size)
    data = tuples(idxs, idxs)
    queries = draw(lists(data))
    return [('q', min(l, r), max(l, r)) for l, r in queries]


@composite
def updates(draw, arr_size, data_range):
    idxs = integers(0, arr_size - 1)
    updates = draw(lists(tuples(idxs, data_range)))
    return [('u', i, x) for i, x in updates]


@composite
def range_queries(draw, queries=ranges, updates=updates):
    data_range = integers(-1000, 1000)
    arr = draw(lists(data_range))
    assume(len(arr) > 0)

    queries = draw(queries(len(arr)))
    if updates:
        queries += draw(updates(len(arr), data_range))

    queries = draw(permutations(queries))
    return arr, queries


@given(range_queries(queries=intervals, updates=updates))
def test_fenwick(data):
    arr, queries = data
    tree = FenwickTree(arr)

    for q in queries:
        if q[0] == 'q':
            assert sum(arr[:q[1]]) == tree.query(q[1])
        if q[0] == 'u':
            arr[q[1]] += q[2]
            tree.update(q[1], q[2])


@given(range_queries(queries=ranges, updates=updates))
def test_sqrt_decompoisition(data):
    arr, queries = data
    struct = SQRTDecomposition(arr)
    arr = arr.copy()

    for q in queries:
        if q[0] == 'q':
            _, l, r = q
            assert sum(arr[l:r]) == struct.query(l, r)
        if q[0] == 'u':
            _, i, x = q
            arr[i] += x
            struct.update(i, x)
