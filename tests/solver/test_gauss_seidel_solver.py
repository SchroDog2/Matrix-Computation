import numpy as np
from numpy.testing import assert_allclose

from core.solver.gauss_seidel_solver import GuassSeidelSolver


def test_solve_when_converge():
    A = [[-3, 1, 15, 44], [6, -2, 1, 5], [5, 10, 1, 28]]
    solver = GuassSeidelSolver()
    # no relaxation
    solver.set(A, tolerance=0.05)
    solution = solver.solve()
    assert_allclose(
        np.matmul(
            np.array(A, dtype=float)[:, :3],
            np.array(solution, dtype=float).reshape(3, 1),
        ).reshape(3),
        np.array([44, 5, 28], dtype=float),
        rtol=0.05,
    )

    # relaxation 0.95
    solver.set(A, relaxation=0.95, tolerance=0.05)
    solution = solver.solve()
    assert_allclose(
        np.matmul(
            np.array(A, dtype=float)[:, :3],
            np.array(solution, dtype=float).reshape(3, 1),
        ).reshape(3),
        np.array([44, 5, 28], dtype=float),
        rtol=0.05,
    )
