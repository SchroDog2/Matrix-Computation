import numpy as np

from numpy.testing import assert_allclose
from lu_decomposition_solver import LUDecompositionSolver, matrix_inv_lu


def test_solve():
    A = [[7, 2, -3, 1], [2, 5, -3, 0], [1, -1, -6, 0]]
    solver = LUDecompositionSolver()
    solver.set(A)
    x = solver.solve()
    assert_allclose(np.matmul(np.array(A, dtype=float)[:, :3], x), [1, 0, 0], atol=1e-6)


def test_matrix_inv_lu():
    A = [[7, 2, -3], [2, 5, -3], [1, -1, -6]]
    inv = matrix_inv_lu(A)
    assert_allclose(np.matmul(np.array(A, dtype=float), inv), np.identity(3), atol=1e-6)
