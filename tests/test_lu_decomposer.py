from lu_decomposer import LUDecomposer
from numpy.testing import assert_allclose
from numpy import matmul


def test_lu_decomposer():
    A = [
        [ 2,  4,  3,  5],
        [-4, -7, -5, -8],
        [ 6,  8,  2,  9],
        [ 4,  9, -2, 14]
    ]
    decomposer = LUDecomposer()
    decomposer.set(A)

    L, U = decomposer.solve()
    
    assert_allclose(L, [
        [ 1,  0,  0,  0],
        [-2,  1,  0,  0],
        [ 3, -4,  1,  0],
        [ 2,  1,  3,  1],
    ], rtol=1e-6)

    assert_allclose(U, [
        [ 2,  4,  3,  5],
        [ 0,  1,  1,  2],
        [ 0,  0, -3,  2],
        [ 0,  0,  0, -4]
    ], rtol=1e-6)

    assert_allclose(A, matmul(L, U), rtol=1e-6)
