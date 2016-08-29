def dot_product(lhs, rhs):
    # lhs = M x N
    # rhs = K x Q

    M, N = len(lhs), len(lhs[0])
    K, Q = len(rhs), len(rhs[0])
