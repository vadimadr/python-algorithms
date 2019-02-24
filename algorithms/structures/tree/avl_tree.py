from .binary_search_tree import BinarySearchTree


def height(node):
    if node is None:
        return 0
    return node.height


class AVLTree(BinarySearchTree):
    """AVL tree. Balanced by tree height (height of the left and right subtrees differ no more than by two)

    height of the tree is h = O(log n)
    let m_h = minimal number of nodes in tree with height h, then
    m_h = m_h+1 + m_h+2 + 1 -> m_h = Fib_h+2 - 1
    m_1 = Fib_3 - 1 = 2 - 1
    m_h+1 = m_h + m_h-1 + 1 = Fib_h+2 + Fib_h+1 - 2 + 1 = Fib_h+3 - 1

    Fib_h ~ phi^n -> m_h >= phi^h -> log(m) >= h
    """

    def __init__(self, key, *args, **kwargs):
        super().__init__(key, *args, **kwargs)
        self.height = 1

    def big_rotate(self, left=True):
        if left:
            self.right.rotate(not left)
        else:
            self.left.rotate(left)
        self.rotate(left)

    def add(self, key):
        new_node = super().add(key)
        new_node._backtrack_and_rebalance()
        return new_node

    def delete(self, key):
        # standard tree deletion procedure
        v, y, x = super().delete(key)
        y._backtrack_and_rebalance()

    def rotate(self, left=True):
        super().rotate(left)

        # change height after rotation
        child = self.left if left else self.right
        child.height = max(height(child.left), height(child.right)) + 1
        self.height = max(height(self.left), height(self.right)) + 1

    def _backtrack_and_rebalance(self):
        """Traverse from the current subtree to the root and restore height and balance"""
        x = self
        while x is not None:
            x.height = max(height(x.left), height(x.right)) + 1
            x._rebalance()
            x = x.parent

    def _rebalance(self):
        left_height = height(self.left)
        right_height = height(self.right)
        if abs(left_height - right_height) < 2:
            return

        b = self.right if left_height < right_height else self.left
        p, q = b.left, b.right
        if left_height < right_height:
            p, q = q, p

        # big rotate
        if height(p) < height(q):
            b.rotate(not left_height < right_height)

        self.rotate(left_height < right_height)
