# Random-Stuff

## Bairstow's Method for Solving Real and Imaginary Roots for Real Polynomials

This is an implementation following this link:
https://archive.nptel.ac.in/content/storage2/courses/122104019/numerical-analysis/Rathish-kumar/ratish-1/f3node9.html


```
from bairstow_solver import BairstowSolver

# define tolerance for r and s
solver = BairstowSolver(0.01, 0.01)
solver.solve([4, -10, 10, -5, 1], r=0.5, s=-0.5)
```

```
Solving polynomial a=[4, -10, 10, -5, 1]
==================================================
iter=1; dr=1.118; ds=0.296; r=1.618; s=-0.204; er=0.691; es=1.456
iter=2; dr=2.280; ds=0.325; r=3.898; s=0.121; er=0.585; es=2.678
iter=3; dr=-0.962; ds=1.297; r=2.936; s=1.418; er=0.328; es=0.914
iter=4; dr=-0.149; ds=-1.623; r=2.787; s=-0.204; er=0.053; es=7.938
iter=5; dr=-0.004; ds=-0.951; r=2.784; s=-1.156; er=0.001; es=0.823
iter=6; dr=0.100; ds=-0.556; r=2.883; s=-1.711; er=0.035; es=0.325
iter=7; dr=0.101; ds=-0.253; r=2.984; s=-1.965; er=0.034; es=0.129
iter=8; dr=0.016; ds=-0.035; r=3.000; s=-2.000; er=0.005; es=0.017
iter=9; dr=0.000; ds=-0.000; r=3.000; s=-2.000; er=0.000; es=0.000
Criterion met after 9 iterations.
New coef to solve b=[ 2.00027489 -2.00019138  1.        ]
Solved roots: [(2.000000005827486+0j), (0.9999999563878375-0j)]

Solving polynomial a=[ 2.00027489 -2.00019138  1.        ]
==================================================
Solved roots: [(1.000095690806682+1.0000417510853692j), (1.000095690806682-1.0000417510853692j)]

Final solved roots
==================================================
x1 = 2.0000
x2 = 1.0000
x3 = 1.0001 + 1.0000i
x4 = 1.0001 + -1.0000i
```