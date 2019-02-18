from algorithms.structures.tree.base import BaseTree


class BinarySearchTree(BaseTree):
    """Binary Search Tree"""

    def __init__(self, key, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.key = key

    def add(self, key):
        if self.key is None:
            self.key = key
            return self

        # search for the leaf
        parent, next = self, self
        while next is not None:
            parent = next
            if key < next.key:
                next = next.left
            else:
                next = next.right

        node = self.__class__(key, parent)
        if key < parent.key:
            parent.left = node
        else:
            parent.right = node
        return node

    def search(self, key):
        node = self
        while not node.leaf and node.key != key:
            if key < node.key:
                if node.left is None:
                    return node
                node = node.left
            else:
                if node.right is None:
                    return node
                node = node.right
        return node

    def delete(self, key):
        if isinstance(key, self.__class__):
            v = key
        else:
            v = self.search(key)
            if v.key != key:
                return
        p = v.parent
        if p is not None:
            i = 0 if p.left is v else 1

        # deleted node is leaf
        if v.leaf:
            if p is not None:
                p.children[i] = None
            else:
                v.key = None
            return

        # deleted node has one child
        if v.left is None or v.right is None:
            u = v.left or v.right
            u.parent = p
            if p is not None:
                p.children[i] = u
            else:
                v.key = u.key
                v.children = u.children
            return

        # deleted node has two children
        # swap v and its predecessor
        u = v.successor()
        v.key = u.key
        if u.parent.left is u:
            u.parent.left = u.right
        else:
            u.parent.right = u.right
        if u.right is not None:
            u.right.parent = u.parent

    def successor(self):
        """return node with key, that is successor of current node key"""

        # all elements in right subtree > key
        if self.right is not None:
            u = self.right
            # find minimum in right subtree (all left nodes)
            while u.left is not None:
                u = u.left
            return u
        # all parents of right subtree < key, thus move upwards until we are
        # in left subtree
        u = self
        p = u.parent
        while p is not None and u == p.right:
            u = p
            p = u.parent
        return p

    def predecessor(self):
        """return node with key, that is predecessor of current node key"""
        if self.left is not None:
            u = self.left
            while u.right is not None:
                u = u.right
            return u
        u = self
        p = u.parent
        while p is not None and u == p.left:
            u = p
            p = u.parent
        return p

    def rotate(self, left=True):
        """
        Performs sub-tree rotation. Does not violates RB-tree properties
        left-rot -> <- right-rot
            X             X
          c   Y         Y   c
             a b       a b

        Returns node that was in the root position before rotation.

        We do not store pointer to the tree root explicitly, thus we don't move root node and use swap instead
        """
        if left:
            new_root = self._left_rotate()
        else:
            new_root = self._right_rotate()
        return new_root

    def _left_rotate(self):
        x, y = self, self.right
        assert y is not None, "Right parent is assumed to exist before left rotation"
        c, a, b = x.left, y.left, y.right
        x._swap_nodes(y)
        x.left, x.right = y, b
        if b:
            b.parent = x
        y.left, y.right = c, a
        if c:
            c.parent = y
        return y

    def _right_rotate(self):
        x, y = self, self.left
        assert y is not None, "Left parent is assumed to exist before right rotation"
        c, a, b = x.right, y.left, y.right
        x._swap_nodes(y)
        x.left, x.right = a, y
        if a:
            a.parent = x
        y.left, y.right = b, c
        if c:
            c.parent = y
        return y

    def _swap_nodes(self, other):
        self.key, other.key = other.key, self.key

    def _get_child(self, i):
        if len(self.children) != 0:
            return self.children[i]
        return None

    def _set_child(self, i, val):
        if len(self.children) == 0:
            self.children.extend((None, None))
        self.children[i] = val

    @property
    def left(self):
        return self._get_child(0)

    @left.setter
    def left(self, value):
        self._set_child(0, value)

    @property
    def right(self):
        return self._get_child(1)

    @right.setter
    def right(self, value):
        self._set_child(1, value)

    def __eq__(self, other):
        if self.key != other.key:
            return False

        if self.left != other.left:
            return False

        if self.right != other.right:
            return False

        return True

    def __contains__(self, item):
        node = self.search(item)
        return node.key == item

    def __repr__(self):
        return "BTree({})".format(self.key)

    @classmethod
    def from_list(cls, lst):
        if not lst:
            return None

        tree = cls(lst[0])

        for n in lst[1:]:
            tree.add(n)

        return tree
