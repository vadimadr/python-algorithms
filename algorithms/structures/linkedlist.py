class LinkedList:
    """Double linked list implementation"""
    def __init__(self, key=None, next_=None, prev=None):
        self.key = key
        self.next = next_
        self.prev = prev

    def prepend(self, key=None):
        prev_ = self.prev
        new_node = LinkedList(key, self, prev_)
        self.prev = new_node
        if prev_:
            prev_.next = new_node
        return new_node

    def append(self, key=None):
        next_ = self.next
        new_node = LinkedList(key, next_, self)
        self.next = new_node
        if next_:
            next_.prev = new_node
        return new_node

    def search(self, key):
        node = self
        while node:
            if node.key == key:
                return node
            node = node.next

    def delete(self):
        prev_ = self.prev
        next_ = self.next
        if next_:
            next_.prev = prev_
        if prev_:
            prev_.next = next_
        return prev_ if prev_ else next_

    def swap(self, other):
        other_prev = other.prev
        other_next = other.next

        if self.prev:
            self.prev.next = other
        if self.next:
            self.next.prev = other
        if other_next:
            other_next.prev = self
        if other_prev:
            other_prev.next = self

        self.prev, other.prev = other.prev, self.prev
        self.next, other.next = other.next, self.next

    def reverse(self):
        head = self
        node = self.next
        head.next = None
        while node:
            next_ = node.next
            node.next = head
            head.prev = node
            head = node
            node = next_
        head.prev = None
        return head

    @property
    def last(self):
        prev, next_ = None, self
        while next_:
            prev, next_ = next_, next_.next
        return prev

    @property
    def head(self):
        prev, next_ = self, None
        while prev:
            prev, next_ = prev.prev, prev
        return next_

    @classmethod
    def from_pylist(cls, pylist):
        if not pylist:
            return None
        head = cls(pylist[0])
        tail = head
        for k in pylist[1:]:
            tail = tail.append(k)
        return head

    def __iter__(self):
        node = self
        while node:
            yield node.key
            node = node.next
