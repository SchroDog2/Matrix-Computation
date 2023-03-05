import pytest
from numpy.testing import assert_allclose
from gaussian_eliminator import GaussianEliminator


@pytest.fixture
def eliminator() -> GaussianEliminator:
    A = [
        [2, -6, -1, -38],
        [-3, -1, 7, -34],
        [-8, 1, -2, -20]
    ]
    eliminator = GaussianEliminator()
    eliminator.set(A)
    return eliminator


def test_solve(eliminator):
    result = eliminator.solve()
    assert_allclose(result, [ 4, 8., -2.], rtol=1e-6)


def test_swap_row(eliminator):
    eliminator.swap_row(1, 2)
    assert_allclose(eliminator.A[2], [-3, -1, 7, -34], rtol=1e-6)
    assert_allclose(eliminator.A[1], [-8, 1, -2, -20], rtol=1e-6)

    eliminator.swap_row(0, 2)
    assert_allclose(eliminator.A[0], [-3, -1, 7, -34], rtol=1e-6)
    assert_allclose(eliminator.A[2], [2, -6, -1, -38], rtol=1e-6)
    

def test_add_row(eliminator):
    eliminator.add_row(1, 2)
    assert_allclose(eliminator.A[2], [-11, 0, 5, -54], rtol=1e-6)
    assert_allclose(eliminator.A[1], [-3, -1, 7, -34], rtol=1e-6)
    eliminator.add_row(1, 0, 0.5)
    assert_allclose(eliminator.A[0], [0.5, -6.5, 2.5, -55], rtol=1e-6)


def test_partial_pivot(eliminator):
    eliminator.partial_pivot_and_swap(0)
    assert_allclose(eliminator.A[0], [-8, 1, -2, -20], rtol=1e-6)
    eliminator.partial_pivot_and_swap(1)
    assert_allclose(eliminator.A[1], [2, -6, -1, -38], rtol=1e-6)


def test_eliminate(eliminator):
    eliminator.eliminate(0)
    assert_allclose(eliminator.A, [
        [   2. ,   -6. ,   -1. ,  -38. ],
        [   0. ,  -10. ,    5.5,  -91. ],
        [   0. ,  -23. ,   -6. , -172. ]
    ], rtol=1e-6)

    eliminator.eliminate(1)
    assert_allclose(eliminator.A, [
        [  2.  ,  -6.  ,  -1.  , -38.  ],
        [  0.  , -10.  ,   5.5 , -91.  ],
        [  0.  ,   0.  , -18.65,  37.3 ]
    ], rtol=1e-6)
