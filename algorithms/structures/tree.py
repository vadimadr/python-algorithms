from collections import deque
from enum import Enum


class BaseTree:

    def __init__(self, parent=None, *args, **kwargs):
        self.parent = parent
        self.children = []

    @property
    def leaf(self):
        return len(self.children) == 0

    @property
    def root(self):
        return self.parent is None

    def depth_first(self):
        yield self
        for c in self.children:
            if c is not None:
                yield from c.depth_first()

    def breadth_first(self):
        q = deque()
        q.append(self)

        while q:
            node = q.popleft()
            yield node
            for c in node.children:
                if c is not None:
                    q.append(c)


class BinaryTree(BaseTree):
    """Binary Search Tree"""

    def __init__(self, key, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.key = key

    def add(self, key):
        if self.key is None:
            self.key = key
            return self

        parent = self.search(key)

        if key == parent.key:
            return parent

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

        Returns new sub-tree root after rotation
        """
        if left:
            return self._left_rotate()
        else:
            return self._right_rotate()

    def _left_rotate(self):
        x, y = self, self.right
        assert y is not None, "Right parent is assumed to exist before left rotation"
        y.parent = x.parent
        if y.parent and y.parent.left is x:
            y.parent.left = y
        elif y.parent and y.parent.right is x:
            y.parent.right = y
        x.right = y.left
        if x.right:
            x.right.parent = x
        y.left = x
        x.parent = y
        return y

    def _right_rotate(self):
        x, y = self, self.left
        assert y is not None, "Left parent is assumed to exist before right rotation"
        y.parent = x.parent
        if y.parent and y.parent.right is x:
            y.parent.right = y
        elif y.parent and y.parent.left is x:
            y.parent.left = y
        x.left = y.right
        if x.left:
            x.left.parent = x
        y.right = x
        x.parent = y
        return y

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


class _RBColor(Enum):
    RED = 0
    BLACK = 1


RED = _RBColor.RED
BLACK = _RBColor.BLACK


class RedBlackTree(BinaryTree):
    """
    properties:
    1. Every node is either RED or BLACK
    2. The root is always black
    3. Every leaf (virtual) leaf is black
    4. If node is RED then both children are BLACK
    5. For each node x , all paths to a leve contains same number of black nodes = bh(x)
    """

    def __init__(self, key, *args, **kwargs):
        super().__init__(key, *args, **kwargs)
        # init NIL-like node
        self.color = BLACK
        self.parent = self  # make it parent to itself

    def add(self, key):
        new_node = super().add(key)
        new_node.color = RED
        # fix RBT properties after insertion
        z = new_node
        while z.parent.color is RED:
            # at most 1 property may be violated (2 or 4) at any iterationR
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color is RED:
                    # resolve case 1
                    z.parent.color = BLACK
                    y.color = BLACK
                    z.parent.parent.color = RED
                    z = z.parent.parent

        self.root.color = BLACK
