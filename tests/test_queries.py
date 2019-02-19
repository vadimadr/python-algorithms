from hypothesis import assume, given
from hypothesis.strategies import composite, lists, integers, tuples, permutations

from algorithms.structures.range_queries.fenwick_tree import FenwickTree


@composite
def shuffle(draw, data):
    p = draw(permutations(data))
    return [data[i] for i in p]


@composite
def interval_queries(draw):
    data_range = integers(-1000, 1000)
    arr = draw(lists(data_range))
    assume(len(arr) > 0)

    upd_indexes = integers(0, len(arr) - 1)
    q_indexes = integers(1, len(arr))
    updates = draw(lists(tuples(upd_indexes, data_range)))
    queries = draw(lists(q_indexes))

    queries = [('q', x) for x in queries] + [('u', i, x) for i, x in updates]
    queries = draw(permutations(queries))
    return arr, queries


@given(interval_queries())
def test_fenwick(data):
    arr, queries = data
    tree = FenwickTree(arr)

    for q in queries:
        if q[0] == 'q':
            assert sum(arr[:q[1]]) == tree.query(q[1])
        if q[0] == 'u':
            arr[q[1]] += q[2]
            tree.update(q[1], q[2])
