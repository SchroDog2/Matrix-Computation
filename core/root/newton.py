import logging


def newton(f: callable, g: callable, x0: float, tol: float = 1e-6):
    """use Newton's method to locate the root of a 1-d function
    this finds the root iteratively by evaluating:
    x(i+1) = x(i) - f(x(i)) / f'(x(i))
    
    f: the original function. needs to be a callable function f(x)
    g: the first derivative function to f. ie: g = f'(x)
    x0: initial guess
    tol: the stopping criteria when relative error is smaller than tolerance
    """
    old = x0
    ea = 1

    while abs(ea) >= tol:
        new = old - f(old) / g(old)
        ea = (new - old) / new
        old = new
        logging.debug(f"new={new:}; old={old}; approx error={ea}")
        # print(f"new value={new};  old={old}; approx error={ea}")

    return new
