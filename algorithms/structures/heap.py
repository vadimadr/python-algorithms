"""Binary heap data structure. Heap is the binary tree with the following
properties:
1. parent is not greater (min-heap) or not less (max-heap) than it's children.
2. all levels (except of deepest) are fully filled
3. leaves are filled left to right.

Example of min-heap:

                                   0

                  1                                 2

          3               4                5               6

      7       8       9       10      11      12      13      14

    15 16   17 18   19 20   21 22   23 24   25 26   27 28   29 30

Heap is stored in array where a[0] is a root, and a[i*2 + 1] and a[i*2 + 2]
are left and right children respectively. Binary Heaps are used to represent
priority queues or to sort without additional memory.
"""


class BinaryHeap:
    def __init__(self, data=None, maxheap=False):
        self.maxheap = maxheap
        if data:
            self.data = data.copy()
            self.n = len(data)
            self.heapify()
        else:
            self.data = []
            self.n = 0

    def siftdown(self, i):
        """move element with index i to the down to restore heap properties

        All children of i already heaps. Place i at leaf and bubble it up
        until heap property is restored. It is possible to implement this
        procedure by swapping i with its child until heap property is restored
        but comparisons are potentially expensive and siftdown is usually
        called after heappop (i.e. for large element) so it make sense to
        start comparisons from the bottom.

        Complexity: O(log N)
        """

        a = self.data
        n = self.n
        item = a[i]
        root = i
        child = 2 * i + 1
        while child < n:
            # select smallest child
            maxheap = self.maxheap
            if child + 1 < n and (not maxheap and a[child + 1] < a[child] or
                    maxheap and a[child] < a[child + 1]):
                child += 1
            a[i] = a[child]
            i = child
            child = 2 * i + 1
        a[i] = item
        self.siftup(i, root)

    def siftup(self, i, root=0):
        """move element with index i to the root to restore heap properties

        Bubble up element until heap property is restored.

        Complexity: O(log N)
        """
        a = self.data
        item = a[i]
        while i > root:
            parent = (i - 1) // 2
            if not self.maxheap and item < a[parent] \
                    or self.maxheap and a[parent] < item:
                a[i] = a[parent]
                i = parent
            else:
                break
        a[i] = item

    def heapify(self):
        """Transform list to heap in O(N).

        Leaves are already heaps so start restoring heap property from its
        parents.
        """
        for i in reversed(range(self.n // 2)):
            self.siftdown(i)

    def push(self, item):
        """Add new item in O(log N)"""
        self.data.append(item)
        self.n += 1
        self.siftup(self.n - 1)

    def pop(self):
        """Pops minimum (maximum) element and restores heap property"""
        item = self.data[0]
        self.data[0] = self.data[-1]
        self.data.pop()
        self.n -= 1
        if self.n:
            self.siftdown(0)
        return item

    def replace(self, i, newval):
        """Replaces element with index i and restores heap property"""
        oldval = self.data[i]
        self.data[i] = newval
        if not self.maxheap and newval < oldval \
                or self.maxheap and oldval < newval:
            self.siftup(i)
        else:
            self.siftdown(i)

    def merge(self, second):
        """Merge heaps in O(N + M)"""
        self.data += second.data
        self.n += second.n
        self.heapify()

    def kth_element(self, k):
        """Returns kth smallest element in O(k*log k). When k log K >= n it
        is better to use quick select algorithm"""
        n = self.n
        temp = BinaryHeap(maxheap=self.maxheap)
        temp.push((self.data[0], 0))
        for i in range(k):
            val, idx = temp.data[0]
            left_child = 2 * idx + 1
            if left_child < n:
                temp.push((self.data[left_child], left_child))
            if left_child + 1 < n:
                temp.push((self.data[left_child + 1], left_child + 1))
            temp.pop()
        return temp.data[0][0]
