def swap(arr, i, j):
    arr[i], arr[j] = arr[j], arr[i]


def selection_sort(a, lo, hi):
    for i in range(lo, hi):
        m = i
        for j in range(i, hi):
            if a[j] < a[m]:
                m = j
        swap(a, i, m)


def bubble_sort(a, lo, hi):
    for i in range(lo, hi):
        for j in range(1, hi - i):
            if a[j] < a[j - 1]:
                swap(a, j - 1, j)


def insertion_sort(a, lo, hi):
    for i in range(1, hi):
        j = i
        while j > 0 and a[j] < a[j - 1]:
            swap(a, j, j - 1)
            j -= 1


def quick_sort(a, lo, hi):
    """quick sort with median of left, middle, right as pivot element"""

    def pivot_median(seq, lo, hi):
        m = lo + (hi - lo) // 2  # middle element
        if seq[lo] < seq[m]:
            if seq[m] < seq[hi - 1]:
                return m
            elif seq[hi - 1] < seq[lo]:
                return lo
        else:
            if seq[hi - 1] < seq[m]:
                return m
            elif seq[lo] < seq[hi - 1]:
                return lo
        return hi - 1

    def partition(seq, start, end):
        p = start  # index of first elem not less than pivot
        pivot = pivot_median(seq, start, end)
        swap(seq, pivot, end - 1)  # move pivot to the end

        for i in range(start, end):
            if seq[i] < seq[end - 1]:
                swap(seq, p, i)
                p += 1
        swap(seq, p, end - 1)
        return p

    def sort(seq, start, end):
        if start < end:
            p = partition(seq, start, end)
            sort(seq, start, p)
            sort(seq, p + 1, end)

    sort(a, lo, hi)
