from math import inf

from algorithms.structures.tree import BinaryTree


def bst_invariant(t, min_key=-inf, max_key=inf):
    if t is None:
        return True
    if max_key <= t.key or t.key <= min_key:
        return False
    return bst_invariant(t.left, min_key, t.key) and \
           bst_invariant(t.right, t.key, max_key)


def test_bst():
    t = [10, 20, 5, 1, 0, 2]
    t2 = [10, 20, 5, 1, 0, 3]
    bst_t = BinaryTree.from_list(t)

    assert bst_invariant(bst_t)
    assert [n.key for n in bst_t.breadth_first()] == [10, 5, 20, 1, 0, 2]
    assert 10 in bst_t
    assert 13 not in bst_t

    assert bst_t == BinaryTree.from_list(t)
    assert bst_t != BinaryTree.from_list(t2)


def test_bst_sort():
    t = [20, 15, 25, 5, 17, 23, 30, 3, 10, 16, 19, 27, 31, 1, 4, 7, 12]
    sorted1 = sorted(t)
    bst_t = BinaryTree.from_list(t)

    u = bst_t.search(sorted1[0])
    for i in sorted1:
        assert u.key == i
        u = u.successor()
    assert bst_invariant(bst_t)


def test_bst_delete():
    t = [20, 15, 25, 5, 17, 23, 30, 3, 10, 16, 19, 27, 31, 1, 4, 7, 12]
    bst_t = BinaryTree.from_list(t)

    for i in [5, 20, 31, 1, 17, 30]:
        assert i in bst_t
        assert bst_invariant(bst_t)
        bst_t.delete(i)
        assert i not in bst_t
    bst_t.delete(44)
    assert bst_invariant(bst_t)
