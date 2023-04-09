import numpy as np
import pytest

from numpy.testing import assert_almost_equal
from core.factorization.qr import classical_gram_schmidt, modified_gram_schmidt, house_qr


@pytest.fixture
def A():
    """a normal well conditioned matrix.
    note that A is not a square matrix. The QR factorization of A = QR
    doesn't imply Q * Q.T == I. Only Q.T * Q == I is true.
    Q * Q.T == Q.T * Q == I is true only when A is a square matrix
    """
    return np.matrix([
        [1, -1, 4],
        [1, 4, -2],
        [1, 4,  2],
        [1, -1, 0]
    ], dtype=float
)

def test_classical_gram_schmidt(A):
    Q, R = classical_gram_schmidt(A)
    assert_almost_equal(Q.T * Q, np.identity(3))
    assert_almost_equal(Q * R, np.matrix(A))


def test_modified_gram_schmidt(A):
    Q, R = modified_gram_schmidt(A)
    assert_almost_equal(Q.T * Q, np.identity(3))
    assert_almost_equal(Q * R, np.matrix(A))
    print(Q)
    print(R)


def test_householder_reflection(A):
    Q, R = house_qr(A)
    assert_almost_equal(Q.T * Q, np.identity(3))
    assert_almost_equal(Q * R, np.matrix(A))
