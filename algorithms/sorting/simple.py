from algorithms.sorting.utils import imin, swap

from ._sorting import fast_sort_wrap


def selection_sort(a, l, r):
    for i in range(l, r + 1):
        m = imin(a, i)
        swap(a, i, m)


def bubble_sort(a, l, r):
    for i in range(l, r + 1):
        for j in range(1, r + 1):
            if a[j] < a[j - 1]:
                swap(a, j - 1, j)


def insertion_sort(a, l, r):
    for i in range(1, r + 1):
        j = i
        while j > 0 and a[j] < a[j - 1]:
            swap(a, j, j - 1)
            j -= 1


def quick_sort(a, l, r):
    """
    http://ru.wikipedia.org/wiki/Быстрая_сортировка
    http://en.wikipedia.org/wiki/Quicksort

    Реализация скопирована
    ru.wikibooks.org/wiki/Реализации_алгоритмов/Сортировка/Быстрая (Java/C#)

    в качестве опорного элемента выбирается середина
    """

    def partition(seq, start, end):
        cursor = start  # позиция опорного элемента
        swap(seq, (start + end) // 2,
             end - 1)  # Переместить опорный элемент в конец
        for i in range(start, end):
            if seq[i] <= seq[end - 1]:
                swap(seq, cursor, i)
                cursor += 1
        return cursor - 1  # на последней итерации будет свапнут seq[end],
        # поэтому нужно уменьшить указатель на 1

    def sort(seq, start, end):
        if start < end:
            p = partition(seq, start,
                          end)  # переместить элементы, меньшие или равные
            # опорному в начало списка
            sort(seq, start, p)
            sort(seq, p + 1, end)

    sort(a, l, r + 1)


def merge(seq, start, p, end):
    for i in range(start, end):
        if seq[i] > seq[p]:
            swap(seq, i, p)
            p = min(p + 1, end - 1)
    return seq
