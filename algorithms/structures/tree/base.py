from collections import deque


class BaseTree:
    def __init__(self, parent=None, root=None, *args, **kwargs):
        self.parent = parent
        self.children = []
        self._root = root if root else self

    @property
    def leaf(self):
        return len(self.children) == 0 or not any(self.children)

    @property
    def is_root(self):
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

    @property
    def root(self):
        root = self
        while root.parent:
            root = root.parent
        return root
