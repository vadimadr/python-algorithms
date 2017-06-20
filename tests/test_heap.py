from hypothesis import given
from hypothesis.strategies import integers, lists

from algorithms.structures.heap import BinaryHeap


def is_heap(heap):
    n = heap.n
    a = heap.data
    maxheap = heap.maxheap

    for i in range(n):
        left_child = i * 2 + 1
        if left_child < n and (not maxheap and a[left_child] < a[i] or
                               maxheap and a[i] < a[left_child]):
            return False
        if left_child + 1 < n and (not maxheap and (a[left_child + 1] < a[i])
                                   or maxheap and a[i] < a[left_child + 1]):
            return False
    return True


@given(lists(integers()))
def test_heapify_min(data):
    heap = BinaryHeap(data)
    assert is_heap(heap)


@given(lists(integers()))
def test_heapify_max(data):
    heap = BinaryHeap(data, maxheap=True)
    assert is_heap(heap)


@given(lists(integers(), 1))
def test_heap_pop(data):
    heap = BinaryHeap(data)
    assert is_heap(heap)
    data_min = min(data)
    heap_pop = heap.pop()
    assert is_heap(heap)
    assert heap_pop == data_min


@given(lists(integers(), 1))
def test_heap_pop_max(data):
    heap = BinaryHeap(data, maxheap=True)
    data_max = max(data)
    heap_pop = heap.pop()
    assert is_heap(heap)
    assert heap_pop == data_max


@given(lists(integers(), 1), integers(0), integers())
def test_heap_replace(data, i, newval):
    heap = BinaryHeap(data)
    i %= len(data)
    heap.replace(i, newval)
    assert is_heap(heap)


@given(lists(integers(), 1), lists(integers(), 1))
def test_merge_heaps(data1, data2):
    heap1 = BinaryHeap(data1)
    heap2 = BinaryHeap(data2)
    heap1.merge(heap2)
    assert is_heap(heap1)


@given(lists(integers(), 1), integers(0))
def test_kth_element(data, i):
    heap = BinaryHeap(data)
    i %= len(data)
    kth = sorted(data)[i]
    assert heap.kth_element(i) == kth
