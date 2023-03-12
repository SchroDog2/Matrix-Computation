from abc import ABC, abstractmethod
import numpy as np

from typing import Iterable


class Solver(ABC):
    """base solver class for solving system of linear equations"""
    def __init__(self, verbose=True) -> None:
        """instantiate a solver object"""
        self.A = None  # extended matrix with dimension N * (N + 1)
        self.N = None  # number of rows
        self.verbose = verbose

    @abstractmethod
    def solve(self):
        pass

    def set(self, A: Iterable[Iterable]) -> None:
        """set up the linear equations to be sovled.
        A is the extended coefficient matrix of the system that includes right hand side coefficients.
        example:
            A = [
                [2, -6, -1, -38],
                [-3, -1, 7, -34],
                [-8, 1, -2, -20]
            ]
        """
        # note dtype of A must be float
        self.A = np.array(A, dtype=float)
        nrow, ncol = self.A.shape
        if not nrow + 1 == ncol:
            raise ValueError("Expecting square matrix for coefficient A")
        self.N = nrow
        self.print_matrix_if_verbose(self.A, title="Linear System Set Up")


class GaussianSolver(Solver):
    """base Gaussian solver class for solving system of linear equations.
    this base class implements the forward and backward substition methods
    """
    def __init__(self, verbose=True) -> None:
        """instantiate a solver object"""
        super().__init__(verbose)

    def backward_substitute(self):
        """solve upper triangular matrix using backward substitution"""
        # initiate solution vector
        x = np.ndarray(self.N, float)
        for i in range(self.N)[::-1]:
            x[i] = (
                self.A[i, self.N] - sum(self.A[i, (i + 1) : self.N] * x[i + 1 :])
            ) / self.A[i, i]
        self.print_vector_if_verbose(x, title="Solved Solution")
        return x

    def forward_substitute(self):
        """solve lower triangular matrix using forward substitution"""
        x = np.ndarray(self.N, float)
        for i in range(self.N):
            x[i] = (self.A[i, self.N] - sum(self.A[i, :i] * x[:i])) / self.A[i, i]
        return x

    def print_matrix_if_verbose(self, A, title=None):
        """print the given matrix if verbose"""
        if self.verbose:
            print(f"\n========== {title} ============")
            print(A)

    def print_vector_if_verbose(self, x, title=None):
        """print the given vector if verbose"""
        if self.verbose:
            print(f"\n========== {title} ============")
            print(x)
