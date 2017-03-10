from algorithms.structures.tree import BinaryTree


def test_bst():
    t = [10, 20, 5, 1, 0, 2]
    t2 = [10, 20, 5, 1, 0, 3]
    bst_t = BinaryTree.from_list(t)

    assert [n.key for n in bst_t.breadth_first()] == [10, 5, 20, 1, 0, 2]
    assert 10 in bst_t
    assert 13 not in bst_t

    assert bst_t == BinaryTree.from_list(t)
    assert bst_t != BinaryTree.from_list(t2)
