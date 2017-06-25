from collections import deque


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
            return

        parent = self.search(key)

        if key == parent.key:
            return

        node = BinaryTree(key, parent)
        if key < parent.key:
            parent.left = node
        else:
            parent.right = node

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
