problems:
http://cs229.stanford.edu/materials/ps0.pdf
answer:
1.

(a) ∇f(x) = x^T A+ b^T

(b) ∇f(x) = ∇g(h(x)) * ∇h(x)

(c) ∇f(x) = A

(d) ∇f(x) = ∇g(a^T x) * a^T

∇^2 f(x) = ∇^2 g(a^T x) * a^T * a^T

2.

(a)A = z*z^T = ((z*z^T)^T)^T = (z*z^T)^T ==> A = A^T

x^T A x = x^T A^T x
= x^T (z z^T)^T x = x^T z^T z x = (zx)^T (zx)  = yT*y (for y=zx) = yT*I*y ==> A is positive semidefinte.

(b)because A is non-zero and the consequence of (a), the null-space of A is only 0.
the rank of A is 1.

(c) BAB^T is positively definte.
x^T BAB^T x = (B^Tx)^T A (B^Tx) = y^T A y for(y=B^T x)
Because A is postively semidefinte 
==> y^T A y>=0 
==> x^T BAB^T x
==> BAB^T is postively semidefinte.
