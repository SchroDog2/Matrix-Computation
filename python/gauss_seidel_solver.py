from typing import Iterable

import numpy as np

from solver import Solver


# todo: some methods are the same as GaussianEliminationSolver. refactor
class GuassSeidelSolver(Solver):
    """class that implements Gauss-Seidel method for solving linear system"""

    def set(self, A: Iterable[Iterable], tolerance=0.05, relaxation=1) -> None:
        """tolerance specifies the stopping criterion for approximation error
        relaxation specifies x_new = relaxation * x_new + (1-relaxation) * x_old
        """
        super().set(A)
        self.tolerance = tolerance
        self.relaxation = relaxation

    def solve(self):
        # swap rows so that the values on diagonal are relatively large
        for i in range(self.N):
            self.partial_pivot_and_swap(i)

        # initial guess with zeroes
        curr = np.zeros(self.N)
        approx_err = 1
        while approx_err >= self.tolerance:
            prev = curr
            curr = self._solve_next_iter(prev)
            curr = self.relaxation * curr + (1 - self.relaxation) * prev
            approx_err = np.max(np.abs(1 - prev / curr))
            self.print_vector_if_verbose(prev, title=f"approx err {approx_err}")
            self.print_vector_if_verbose(curr, title=f"approx err {approx_err}")
        if any(np.isnan(curr)) or any(np.isinf(curr)):
            raise ValueError("Gauss-Seidel method doesn't converge.")
        return curr

    def partial_pivot_and_swap(self, i):
        """find largest element below element (i, i). then swap the rows
        so that the largest element is on the pivot of row i
        """
        row_to_swap = abs(self.A[i:, i]).argmax() + i
        if row_to_swap != i:
            self.swap_row(i, row_to_swap)
        self.print_matrix_if_verbose(self.A, title="Pivoting and Swap")

    def swap_row(self, i, j):
        """swaps row i and row j in matrix A. i and j are zero index based"""
        temp = self.A[i].copy()
        self.A[i] = self.A[j]
        self.A[j] = temp

    def _solve_next_iter(self, prev):
        """solve next iteration of x"""
        # todo: below commented the array operation yields wrong result. debug.
        # return (
        #     self.A[:, self.N] - sum(self.A[:, :self.N] * prev) + np.diag(self.A) * prev
        # ) / np.diag(self.A)
        x = np.zeros(self.N, dtype=float)
        for i in range(self.N):
            x[i] = (
                self.A[i, self.N]
                - sum(self.A[i, :i] * prev[:1])
                - sum(self.A[i, i + 1 : self.N] * prev[i + 1 :])
            ) / self.A[i, i]
        return x
