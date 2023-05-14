from core.root.newton import newton


def f(x):
    return (x - 1)**2

def g(x):
    return 2 * x - 2

def test_newton():
    assert abs(newton(f, g, 0) - 1) < 1e-6
    
