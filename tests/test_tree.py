from math import inf

from hypothesis import given
from hypothesis.strategies import integers, lists

from algorithms.structures.tree.binary_search_tree import BinarySearchTree
from algorithms.structures.tree.red_black_tree import RedBlackTree, BLACK, RED


def bst_invariant(t, min_key=-inf, max_key=inf, parent=None):
    if t is None:
        return True
    if t.parent is not parent:
        return False

    # for some types of BST left child may be equal to the right child
    # e.g. RedBlackTree.from_list([1,0,0]) --> (0, (0, 1))
    if max_key < t.key or t.key < min_key:
        return False

    left_inv = bst_invariant(t.left, min_key, t.key, t)
    right_inv = bst_invariant(t.right, t.key, max_key, t)
    return left_inv and right_inv


def check_rb_tree_properties(tree, root=True):
    if not tree:
        # leaf is black
        return 1
    if root:
        assert tree.color is BLACK
    if tree.color is RED:
        if tree.left:
            assert tree.left.color is BLACK
        if tree.right:
            assert tree.right.color is BLACK
    num_blacks = check_rb_tree_properties(tree.right, root=False)
    assert check_rb_tree_properties(tree.left, root=False) == num_blacks
    if tree.color is BLACK:
        return num_blacks + 1
    return num_blacks


def is_left_child(parent, child):
    if child.parent is not parent:
        return False
    if parent.left is not child:
        return False
    return True


def is_right_child(parent, child):
    if child.parent is not parent:
        return False
    if parent.right is not child:
        return False
    return True


def test_bst():
    t = [10, 20, 5, 1, 0, 2]
    t2 = [10, 20, 5, 1, 0, 3]
    bst_t = BinarySearchTree.from_list(t)

    assert bst_invariant(bst_t)
    assert [n.key for n in bst_t.breadth_first()] == [10, 5, 20, 1, 0, 2]
    assert 10 in bst_t
    assert 13 not in bst_t

    assert bst_t == BinarySearchTree.from_list(t)
    assert bst_t != BinarySearchTree.from_list(t2)


def test_bst_sort():
    t = [20, 15, 25, 5, 17, 23, 30, 3, 10, 16, 19, 27, 31, 1, 4, 7, 12]
    sorted1 = sorted(t)
    bst_t = BinarySearchTree.from_list(t)

    u = bst_t.search(sorted1[0])
    for i in sorted1:
        assert u.key == i
        u = u.successor()
    assert bst_invariant(bst_t)


def test_bst_delete():
    t = [20, 15, 25, 5, 17, 23, 30, 3, 10, 16, 19, 27, 31, 1, 4, 7, 12]
    bst_t = BinarySearchTree.from_list(t)

    for i in [5, 20, 31, 1, 17, 30]:
        assert i in bst_t
        assert bst_invariant(bst_t)
        bst_t.delete(i)
        assert i not in bst_t
    bst_t.delete(44)
    assert bst_invariant(bst_t)


@given(lists(integers()))
def test_bst_invariant(t):
    bst = BinarySearchTree.from_list(t)
    assert bst_invariant(bst)


@given(lists(integers(), min_size=2, max_size=20, unique=True), integers(0, 1))
def test_bst_invariant_rotations(t, left):
    bst = BinarySearchTree.from_list(t)

    # try rotate every node
    for node in bst.depth_first():
        # check that rotation is possible for the current node
        if left and node.right is None or not left and node.left is None:
            continue

        before_rotation = [n.key for n in node.breadth_first()]
        parent = node.parent
        node.rotate(bool(left))
        assert bst_invariant(node, parent=parent), "Tree rotation must not break BST invariant"

        node.rotate(not bool(left))
        assert before_rotation == [n.key for n in node.breadth_first()], \
            "Symmetric rotations should result in the same tree"


def test_left_rotate():
    z = BinarySearchTree(0)
    x = z.add(5)
    y = x.add(7)
    a = x.add(3)
    b = y.add(6)
    c = y.add(8)

    assert x.rotate(left=True) is y
    assert z.key == 0
    assert is_right_child(z, x) and x.key == 7
    assert is_left_child(x, y) and y.key == 5
    assert is_left_child(y, a) and a.key == 3
    assert is_right_child(y, b) and b.key == 6
    assert is_right_child(x, c) and c.key == 8


def test_right_rotate():
    z = BinarySearchTree(0)
    x = z.add(5)
    y = x.add(3)
    a = x.add(8)
    b = y.add(2)
    c = y.add(4)

    assert x.rotate(left=False) is y
    assert z.key == 0
    assert is_right_child(z, x) and x.key == 3
    assert is_right_child(x, y) and y.key == 5
    assert is_left_child(y, c) and c.key == 4
    assert is_right_child(y, a) and a.key == 8
    assert is_left_child(x, b) and b.key == 2


def test_rb_tree():
    tree = RedBlackTree.from_list([1, 1, 1, 0, 0, 0, 0, 0])
    assert bst_invariant(tree)
    check_rb_tree_properties(tree)


@given(lists(integers()))
def test_rb_tree_invariant(t):
    tree = RedBlackTree.from_list(t)
    assert bst_invariant(tree)
    check_rb_tree_properties(tree)
