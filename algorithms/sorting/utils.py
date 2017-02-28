def simple_sort(fn):
    def wrap(a, *args, **kwargs):
        fn(a, 0, len(a) - 1, *args, **kwargs)
        return list(a)

    return wrap


def swap(seq, a, b):
    seq[a], seq[b] = seq[b], seq[a]


def imax(seq, start=0, end=None):
    """
    Находит максимальный элемент из интервала [start, end) (по умолчанию вся
    последовательность),
    и возвращает его индекс
    """
    end = end or len(seq)
    return max(range(start, end), key=seq.__getitem__)


def imin(seq, start=0, end=None):
    """
    Находит минималный элемент из интервала [start, end) (по умолчанию вся
    последовательность),
    и возвращает его индекс
    """
    end = end or len(seq)
    return min(range(start, end), key=seq.__getitem__)


def is_sorted(seq, start=0, end=None):
    end = end or len(seq) - 1
    for i in range(start + 1, end + 1):
        if seq[i - 1] > seq[i]:
            return False
    return True
