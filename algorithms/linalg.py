def dot_product(lhs, rhs):
    # lhs = M x N
    # rhs = K x Q

    M, N = len(lhs), len(lhs[0])
    K, Q = len(rhs), len(rhs[0])

    # implemet later
    assert N == K
    assert isinstance(M, int)
    assert isinstance(Q, int)
