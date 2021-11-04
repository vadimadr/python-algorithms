from operator import add

import numpy as np


class SQRTDecomposition:
    """Divide input array into sub-arrays of size B = sqrt(n), pre-compute cumulative frequency in every block:
    [0, B), [B, 2*B), ... Then part of the query can be answered by taking the value from pre-computed block.
    Then complexity of the query will be: T(n) = 2*B + (n / B) = O(sqrt(n))
    """

    def __init__(self, arr, op=add, initial=0):
        self.initial = initial
        self.function = op
        self.block_size = int(np.sqrt(len(arr))) + 1
        self.blocks = np.full(self.block_size, initial)
        self.arr = arr

        for i in range(len(arr)):
            self.blocks[i // self.block_size] = op(
                self.blocks[i // self.block_size], arr[i]
            )

    def query(self, l, r):
        ans = self.initial

        # while l < r:
        #     if l % self.block_size == 0 and l + self.block_size - 1 < r:
        #         ans = self.function(ans, self.blocks[l // self.block_size])
        #         l += self.block_size
        #     else:
        #         ans = self.function(ans, self.arr[l])
        #         l += 1
        # return ans

        start_block = l // self.block_size
        end_block = (r - 1) // self.block_size
        if start_block >= end_block:
            for i in range(l, r):
                ans = self.function(ans, self.arr[i])
            return ans
        for i in range(l, self.block_size * (start_block + 1)):
            ans = self.function(ans, self.arr[i])
        for i in range(start_block + 1, end_block):
            ans = self.function(ans, self.blocks[i])
        for i in range(self.block_size * end_block, r):
            ans = self.function(ans, self.arr[i])
        return ans

    def update(self, i, x):
        self.arr[i] = self.function(self.arr[i], x)
        self.blocks[i // self.block_size] = self.function(
            self.blocks[i // self.block_size], x
        )
