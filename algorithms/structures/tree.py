from collections import deque


class BaseTree:
    def __init__(self, parent=None, *args, **kwargs):
        self.parent = parent
        self.children = []

    @property
    def leaf(self):
        return len(self.children) == 0

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
        pass

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
