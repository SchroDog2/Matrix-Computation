import logging


def parabolic(f: callable, x0: float, x1: float, x2: float, tol: float = 1e-6):
    """parabolic interpolation method for solving maximum value
    of a function f.
    f: callable function f to be solved for the maxima
    x0: initial guess point one
    x1: initial guess point two
    x2: initial guess point three
    tol: stopping criteria for relative approximation error
    """
    def g1(x, a, b):
        """helper function for evaluating the numerators"""
        return f(x) * (a ** 2 - b ** 2)
    
    def g2(x, a, b):
        """helper function for evaluting denominators"""
        return 2 * f(x) * (a - b)
    
    ea = 1
    it = 0
    while abs(ea) >= tol:
        it += 1
        numerator = g1(x0, x1, x2) + g1(x1, x2, x0) + g1(x2, x0, x1)
        denominator = g2(x0, x1, x2) + g2(x1, x2, x0) + g2(x2, x0, x1)
        
        x3 = numerator / denominator

        ea = (x3 - x1) / x3

        # update x0, x1, x2
        # below is the simple update. 
        # todo: use a more effective update logic
        x0, x1, x2 = x1, x2, x3
        logging.info(f"iter {it}: x0={x0:.6f}; x1={x1:.6f}; x2={x2:.6f}; x3={x3:.6f}; ea={ea:.6f}")

    return x3
