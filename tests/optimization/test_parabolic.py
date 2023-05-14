from math import sin

from core.optimization.parabolic import parabolic


def f(x):
    return 2 * sin(x) - x ** 2 / 10


def test_parabolic():
    assert abs(parabolic(f, 0, 1, 4) - 1.4275523) < 1e-6

