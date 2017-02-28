from libcpp.vector cimport vector

cdef extern from "src/sorting.cc":
    ctypedef long long ll
    ctypedef vector[ll] vec

    void fast_sort(vec &)
