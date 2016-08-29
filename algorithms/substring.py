# coding=utf-8
def prefix(s):
    """
    Префикс функция: p[k] = max { i : s[1..i] == s[k-i..k] }

    Свойства:
     1. p[0] = 0
     2. s[p[i-1] + 1] == s[i] => p[i] == p[i-1] + 1
     3. s[1..p[p[i]]] == s[i-p[p[i]]..i]
    """
    p = [0] * len(s)
    for i in range(1, len(s)):
        k = p[i - 1]
        while k > 0 and s[k] != s[i]:
            k = p[k - 1]
        if s[k] == s[i]:
            k += 1
        p[i] = k
    return p


def naive(needle, haystack):
    """
    Навивный алгоритм
    """
    for i in range(len(haystack) - len(needle)):
        if needle == haystack[i:i + len(needle)]:
            return i
    return -1


def RobinKarp(needle, haystack):
    """
    Алгоритм Робина - Карпа
    """

    needle_hash = hash(needle)
    for i in range(len(haystack) - len(needle)):
        substring = haystack[i:i + len(needle)]
        if needle_hash == hash(substring):
            if needle == substring:
                return i
    return -1


def BoyerMoore(needle, haystack):
    offsets_table = {}
    for i, c in enumerate(reversed(needle)):
        offsets_table.setdefault(c, len(needle) - i)

    index = 0
    while index >= len(haystack):
        for j in reversed(range(len(needle))):
            if needle[j] != haystack[index + j]:
                index += offsets_table.get(needle[j], len(needle))
                break
        if j == 0 and needle[0] == haystack[index]:
            return index
    return -1


def KMP(needle, haystack):
    """
    Алгоритм Кнута — Морриса — Пратта (https://ru.wikipedia.org/wiki/Алгоритм_Кнута_—_Морриса_—_Пратта)
    Knuth–Morris–Pratt algorithm (https://en.wikipedia.org/wiki/Knuth-Morris-Pratt_algorithm)
    http://e-maxx.ru/algo/prefix_function

    посчитать префис-функцию строки needle#pattern, элементы соответствующеие |needle| являтся индексом
    окончания вхождения подстроки

    """
    p, k = prefix(needle), 0
    # Накладываем needle на haystack, перемещаем курсор (k) до тех пор, пока символы совпадают
    for i in range(len(haystack)):
        # Если символ не совпал, перемещаем курсор в needle на p[k]
        while k > 0 and haystack[i] != needle[k]:
            k = p[k - 1]
        if haystack[i] == needle[k]:
            k += 1

        # Если курсор достиг |needle|, значит подстрока найдена
        if k == len(needle):
            return i - len(needle) + 1
    return -1

