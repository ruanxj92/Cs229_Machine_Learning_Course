problems:
http://cs229.stanford.edu/materials/ps0.pdf 
answer: 
1. **梯度与海森**
(a) $ \triangledown f(x)=x^{T}A+b$
(b) $ \triangledown f(x)= g'(h(x)) \triangledown h(x)$ 
(c) $ \triangledown^2 f(x)=A$ 
(d) $ \triangledown f(x)= \triangledown g(a^{T}x) \cdot a$ 
 $ \triangledown^{2}f(x)= \triangledown^{2}g(a^{T}x)ax^{T}$ 
2. 正定矩阵
(a) $A=z \cdot z^{T}=((z \cdot z^{T})^{T})^{T}=(z \cdot z^{T})^{T}$
$ \therefore A=A^{T}$
$x^{T}Ax=x^{T}A^{T}x$
$x^{T}z \cdot z^{T}x=x^{T}(z \cdot z^{T})^{T}x$
$=(zx)^{T}zx=y^{T}y=y^{T}Iy$其中$(y=zx)$
$ \therefore$A 是半正定的
(b)由A非零及 $ \triangledown f(x)=x^{T}A+b^{T}$，A的零空间是0，A的秩是1。（待考虑）
(c) $BAB^{T}$是正定的。
$x^{T}BAB^{T}x=(B^{T}x)^{T}A(B^{T}x)=y^{T}Ay$(其中 $y=B^{T}x)$
由于A正定，
$ \therefore y^{T}Ay>=0$
$ \therefore x^{T}BAB^{T}x>=0$
$ \therefore BAB^{T}$半正定
3. 特征值，特征向量和谱定理

This problem sheet is designed for basic linear algebra. 
I wrote it to get familiar with embed latex code format, so far so good.
