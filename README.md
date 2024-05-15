# OrthogPolyKANs
Kolmogorov-Arnold Networks (KAN) using orthogonal polynomials instead of B-splines.
- ! Based heavily on ChebyKAN implementation by SynodicMonth !
  - https://github.com/SynodicMonth/ChebyKAN
- There are many orthogonal polynomials: https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials
  - https://mathworld.wolfram.com/OrthogonalPolynomials.html
- Polynomials: Legendre, generalized Laguerre, Chebyshev 2nd kind, Gegenbauer, Hermite, Fibonacci, Bessel, Lucas, and Jacobi
- Working on:
  - More polynomials: Romanovski, Bernstein, Newton, Bernoulli, Euler, Zernike, Kravchuk, and Lucas Polynomials
  - Alternative to tanh for normalizing to [-1, 1] using MinMax...
  - Optimize by implementing polynomials with explicit formulas instead of recursive definitions...
- NB: This is a very rough implementation, and there is a lot to improve. 
