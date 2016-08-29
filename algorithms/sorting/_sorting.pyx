"""
Simple example of using cython for wrapping c++ code
"""

cimport _sorting

def fast_sort_wrap(xs):
    cdef vec xs_v

    # convert list to c++ vector
    xs_v = xs

    # call c++ function
    fast_sort(xs_v)

    # convert back to list
    return list(xs_v)
