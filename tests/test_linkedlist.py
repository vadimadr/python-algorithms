import pytest
from hypothesis import given, assume
from hypothesis.strategies import lists, integers

from algorithms.structures.linkedlist import LinkedList


@given(lists(integers()))
def test_from_list(lst):
    assume(len(lst) > 0)
    head = LinkedList.from_pylist(lst)
    assert list(head) == lst


def test_append():
    head = LinkedList(5)
    tail = head
    tail = tail.append(4)
    tail = tail.append(3)
    tail = tail.append(2)
    tail = tail.append(1)
    assert list(head) == [5, 4, 3, 2, 1]


def test_prepend():
    head = LinkedList(1)
    head = head.prepend(2)
    head = head.prepend(3)
    head = head.prepend(4)
    head = head.prepend(5)
    assert list(head) == [5, 4, 3, 2, 1]


def test_delete():
    lst = [1, 2, 3]
    l1 = LinkedList.from_pylist(lst)
    assert list(l1.delete()) == [2, 3]
    l2 = LinkedList.from_pylist(lst)
    ret = l2.next.delete()
    assert ret == l2
    assert list(l2) == [1, 3]
    l3 = LinkedList.from_pylist(lst)
    l3.next.next.delete()
    assert l3.next.next is None
    assert list(l3) == [1, 2]


def test_head():
    head = LinkedList.from_pylist([1, 2, 3, 4, 5])
    item = head.next.next
    assert item.head == head


def test_last():
    head = LinkedList.from_pylist([1, 2, 3, 4, 5])
    assert head.last.key == 5


def test_search():
    head = LinkedList.from_pylist([1, 2, 3, 4, 5])
    item = head.search(3)
    assert item.key == 3
    item = head.search(10)
    assert not item


def reverse_list(head, limit=100):
    node = head.last
    lst = []
    i = 0
    while node:
        lst.append(node.key)
        node = node.prev
        i += 1
        if i > limit:
            raise Exception("Limit exceeded. List is most likely looped")
    return lst


def forward_list(head, limit=100):
    node = head
    lst = []
    i = 0
    while node:
        lst.append(node.key)
        node = node.next
        i += 1
        if i > limit:
            raise Exception("Limit exceeded. List is most likely looped")
    return lst


@pytest.mark.parametrize('lst,a,b', [
    ([1, 2, 3, 4, 5, 6, 7], 2, 6),
    ([1, 2, 3], 1, 2),
    ([1, 2, 3], 2, 3),
    ([1, 2, 3, 4], 2, 3),
    ([1, 2, 3, 4], 3, 2),
    ([1, 2, 3], 2, 2),
    ([1, 2, 3], 1, 1),
    ([1, 2, 3], 3, 3),
    ([1], 1, 1)
])
def test_swap(lst, a, b):
    head = LinkedList.from_pylist(lst)
    item1 = head.search(a)
    item2 = head.search(b)
    item1.swap(item2)
    head = head.head

    i, j = lst.index(a), lst.index(b)
    lst[i], lst[j] = lst[j], lst[i]
    assert forward_list(head) == lst
    assert reverse_list(head) == list(reversed(lst))


@pytest.mark.parametrize('lst', [
    [1, 2, 3, 4],
    [1, 2, 3],
    [1, 2],
    [1]
])
def test_reverse(lst):
    head = LinkedList.from_pylist(lst)
    head = head.reverse()
    assert forward_list(head) == list(reversed(lst))
    assert reverse_list(head) == lst
