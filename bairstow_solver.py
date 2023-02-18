import numpy as np

from typing import Iterable


class BairstowSolver:
    """class that implements Bairstow's Method to solve real and imaginary roots
    for real polynomials.
    https://archive.nptel.ac.in/content/storage2/courses/122104019/numerical-analysis/Rathish-kumar/ratish-1/f3node9.html
    """
    def __init__(self, er: float, es: float):
        """er and es are stopping criterion"""
        self.er_threshold = er
        self.es_threshold = es
        self.roots = []

    def solve(self, a: Iterable, r: float, s: float):
        """solves roots for real polynomials. a is an array of coef of the polynomial to be solved. 
        a[n] is the coef of degree n. ex: [4, -10, 10, -5, 1] stands for x^4 -5x^3 +10x^2 -10x + 4 
        r and s are the two initial guesses of the algorithm.
        The algorithms solves two roots at a time and reduces the degree of order by 2. 
        """
        # clear pervious results
        self.roots = []
        # use the algorithm when degree > 3
        while len(a) > 3:
            print(f"Solving polynomial a={a}")
            print("=" * 50)
            roots, a = self._solve_bairstow(a, r, s)
            print(f"Solved roots: {roots}")
            self.roots.extend(roots)
            print()

        if len(a) == 3:
            print(f"Solving polynomial a={a}")
            print("=" * 50)
            roots = self._solve_quadratic(a[2], a[1], a[0])
            print(f"Solved roots: {roots}")
            self.roots.extend(roots)
            print()

        elif len(a) == 2:
            print(f"Solving polynomial a={a}")
            print("=" * 50)
            roots = self._solve_quadratic(0, a[1], a[0])
            print(f"Solved roots: {roots}")
            self.roots.extend(roots)
            print()
        
        print("Final solved roots")
        print("=" * 50)
        self.show()

    def show(self):
        """display solved roots in a nice foramt"""
        roots = []
        for i, r in enumerate(self.roots):
            if r.imag == 0:
                roots.append(f"x{i+1} = {r.real:.4f}")
            else:
                roots.append(f"x{i+1} = {r.real:.4f} + {r.imag:.4f}i")
        print('\n'.join(roots))

    def _solve_bairstow(self, a: Iterable, r: float, s: float) -> Iterable:
        """returns tuple of (roots, coef of polynomial of order n-2)"""
        # initial setup for approximation error er and es
        er, es = 1, 1

        # start iteration
        iter = 0
        while er > self.er_threshold or es > self.es_threshold: 
            b = self._solve_coef_b(a, r, s)
            c = self._solve_coef_c(b, r, s)
            dr, ds = self._solve_increment(b, c)
            r, s = r + dr, s + ds
            er, es = self._compute_error(r, s, dr, ds)
            iter += 1
            # print(f"iter={iter}; b={b}; c={c}; dr={dr:.3f}; ds={ds:.3f}; r={r:.3f}; s={s:.3f}; er={er:.3f}; es={es:.3f}")
            print(f"iter={iter}; dr={dr:.3f}; ds={ds:.3f}; r={r:.3f}; s={s:.3f}; er={er:.3f}; es={es:.3f}")
        
        # compute the two roots
        roots = self._solve_quadratic(1, -r, -s)
        print(f"Criterion met after {iter} iterations.")
        print(f"New coef to solve b={b[2:]}")
        return roots, b[2:]

    def _solve_coef_b(self, a: Iterable, r: float, s: float):
        """b is an array of coef of polynomial of order n-2, with b1 and bo coef of the remainders"""
        # if len(a) <= 3, we have closed form solution
        l = len(a)
        assert l > 3
        
        # base 0 array
        n = l - 1
        
        # initialize array for b with na values
        b = np.full(l, np.nan)
        
        # b[n], b[n-1]
        b[n] = a[n]
        b[n-1] = a[n-1] + r * b[n]
        
        # b[n-2]
        for i in range(n-2, -1, -1):
            b[i] = a[i]  + r*b[i+1] + s*b[i+2]
        
        return b
    
    def _solve_coef_c(self, b: Iterable, r: float, s: float):
        """c is an array of coef that iteratively replace b"""
        l = len(b)
        assert l > 3
        
        # base 0 array
        n = l - 1
        
        # initialize array for c with na values
        c = np.full(l, np.nan)
        
        # c[n], c[n-1]
        c[n] = b[n]
        c[n-1] = b[n-1] + r * c[n]
        
        # b[n-2]
        for i in range(n-2, 0, -1):
            c[i] = b[i]  + r*c[i+1] + s*c[i+2]
        
        return c

    def _solve_increment(self, b: Iterable, c: Iterable):
        """returns dr and ds for updating r and s"""
        # b, c are base 0 arrays
        left = [ [c[2], c[3]], [c[1], c[2]] ]
        right = [-b[1], -b[0]]
        return np.linalg.solve(left, right)

    def _compute_error(self, r, s, dr, ds):
        """compute ralative approximation error where r and s are already updated in current iteration"""
        return abs(dr / r), abs(ds / s)
  
    def _solve_quadratic(self, a: float, b: float, c: float):
        """solves the roots for f(x) = a*x^2 + b*x + c = 0"""
        if a != 0:
            # convert to complex since root could be imaginary
            a, b, c = complex(a), complex(b), complex(c)
            sqrt_term = np.sqrt(b ** 2 - 4 * a * c)
            return [(-b + sqrt_term) / (2 * a), (-b - sqrt_term) / (2 * a)]
        elif b != 0:
            return [-c / b]


if __name__ == "__main__":
    solver = BairstowSolver(0.01, 0.01)
    solver.solve([4, -10, 10, -5, 1], r=0.5, s=-0.5)
    