problems:
http://cs229.stanford.edu/materials/ps0.pdf  
answer:  
1.  
(a) ![](http://latex.codecogs.com/gif.latex?\\triangledown%20f(x)=x^{T}A+b^{T})  
(b) ![](http://latex.codecogs.com/gif.latex?\\triangledown%20f(x)=\\triangeldown%20g(f(x))\\triangeldown%20h(x))  
(c) ![](http://latex.codecogs.com/gif.latex?\\triangledown%20f(x)=A)  
(d) ![](http://latex.codecogs.com/gif.latex?\\triangledown%20f(x)=\\triangledown%20g(a^{T}x)\\cdot%20a^{T})  
![](http://latex.codecogs.com/gif.latex?\\triangledown^{2}f(x)=\\triangledown^{2}g(a^{T}x)a^{T}x^{T})  
2.  
(a) ![](http://latex.codecogs.com/gif.latex?A=z\\cdot%20z^{T}=((z\\cdot%20z^{T})^{T})^{T}=(z\\cdot%20z^{T})^{T})  
![](http://latex.codecogs.com/gif.latex?\\therefore%20A=A^{T})  
![](http://latex.codecogs.com/gif.latex?x^{T}Ax=x^{T}A^{T}x)  
![](http://latex.codecogs.com/gif.latex?x^{T}z\\cdot%20z^{T}x=x^{T}(z\\cdot%20z^{T})^{T}x)  
![](http://latex.codecogs.com/gif.latex?=(zx)^{T}zx=y^{T}y)=y^{T}Iy)(y=zx)  
![](http://latex.codecogs.com/gif.latex?\\therefore)A is positive semidefinte.  
(b)Because A is non-zero and ![](http://latex.codecogs.com/gif.latex?\\triangledown%20f(x)=x^{T}A+b^{T}), the null-space of A is 0.  
the rank of A is 1.  
(c) ![](http://latex.codecogs.com/gif.latex?BAB^{T})
is positively definte.  
![](http://latex.codecogs.com/gif.latex?x^{T}BAB^{T}x=(B^{T}x)^{T}A(B^{T}x)=y^{T}Ay(for%20y=B^{T}x))  
![](http://latex.codecogs.com/gif.latex?\\because) A is postively semidefinte  
![](http://latex.codecogs.com/gif.latex?\\therefore%20y^{T}Ay>=0)  
![](http://latex.codecogs.com/gif.latex?\\therefore%20x^{T}BAB^{T}x>=0)  
![](http://latex.codecogs.com/gif.latex?\\therefore%20BAB^{T})is postively semidefinte.


This problem sheet is designed for basic linear algebra.  
I wrote it to get familiar with embed latex code format, so far so good.
