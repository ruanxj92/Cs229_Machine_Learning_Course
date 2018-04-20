写在开篇的话：  
这一系列的笔记是斯坦福大学cs229 machine learning 机器学习的课程笔记。初步的规划是将[课程网站](http://cs229.stanford.edu/syllabus.html)上的资料翻译成中文，同时也作为学习的一个记录。同时由于水平有限，可能有一些不能很好翻译的地方会用自己的话表达。同时课程的视频资源详见[bilibili](https://www.bilibili.com/video/av9912938/)。一共有20节课，一般平均每节课大约4-8小节不等。
本篇原文见[cs229-notes1.pdf](http://cs229.stanford.edu/notes/cs229-notes1.pdf)。
[TOC]

#CS229 课程笔记
吴恩达
##监督学习
让我们用几个简单的例子来开始讨论监督学习问题。假设我们有一个数据集包括了俄勒冈的波特兰市47所住房居住面积和价格：
居住面积(平方英尺)|价格(千美元)
---------------|----------
2104|400
1600|330
2400|369
1416|232
3000|540




这些数据画图是这样的：
![波特兰房价面积图](https://img-blog.csdn.net/20180413120426996?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbW9ucnVhbjky/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)










给出这样的数据，我们能怎样用一个居住面积的函数来预测住房价格呢？
为了作一些标记供今后使用，我们用$x^{(i)}$ 表示“输入”变量(在这个例子中是居住面积)，也叫做输入**特征**， 同时 $y^{(i)}$ 表示要预测的“输出”或**,目标变量**。一对（$x^{(i)},y^{(i)}$）叫做一个**训练样本**，并且把要用来学习的数据集-一列m个训练样本{$x^{(i)},y^{(i)}$:i=1,...,m}叫做**训练集**。注意记号的上角标“(i)”只是用来表示训练集中的序号，没什么需要解释的。我们也用 $ \cal X $ 表示输入变量的空间，用 $ \cal Y $ 表示输出变量的空间。在该例子中 $ \cal X=Y=\Bbb R $.
为了稍微正式一点来表示监督学习问题，我们的目标是对于给定的训练集，能学习到一个函数*h*(x):$ \cal X \mapsto Y$,使得*h*(x)是一个能很好预测对应y值的函数。由于历史原因，这个函数*h* 称为**假设**。 如图，因此过程像这样：
![监督学习函数](https://img-blog.csdn.net/20180413155118838?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbW9ucnVhbjky/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)















当需要预测的目标变量连续时，比如说我们房价的例子，我们把这样的学习问题叫做**回归**问题。当y只能取若干较小数量的离散值时（比如说给定居住面积，我们想预测某住所是独栋房还是公寓）我们称之为**分类**问题。
### 第一部分 线性回归
为了使得房价预测的例子更加有趣，让我们考虑一个更加丰富的数据集，其中包括了每栋房子的卧室数量：
居住面积(平方英尺)|卧室数|价格(千美元)
---------------|------|---------
2104|3|400
1600|3|330
2400|3|369
1416|2|232
3000|4|540
...|...|...





其中，x是 $\mathbb R^{2}$ 的二维向量。例如，$x_{1}^{(i)}$ 是训练集中第i所房子的居住面积，$ x_{2}^{(i)} $ 是它的卧室数量。（一般来说，当设计一个学习问题，它会取决于要选取那些特征，所以如果你在收集波特兰的房屋数据，你也许要考虑包含一些其他特征比如说房子里有没有壁炉，卫生间的数量等等。再之后的章节我们会对特征选取讨论更多，但现在暂时只考虑给定的特征。）
为了实行监督学习，我们必须要决定如何在计算机中表示函数或者叫假设。如一开始所决定的，我们选用近似$y作为x$的线性函数：  
$$
h_{\theta}(x)=\theta_{0}+\theta_{1}x_{1}+\theta_{2}x_{2} 
$$
上式中、$ \theta_{i} $ 是**参数**（也叫做**权重**）用来将$ \cal X 映射到 Y $ 的线性函数的空间参数化。当不引起歧义时，我们把 $ h_{\theta}(x) 的\theta$ 放到下标，更进一步简化成 $h(x)$。 为了简化，引入 $x_{0}=1$（这是**截距项**）故

$$ 
h(x)= \sum_{i=0}^{n}\theta_{i}x_{i}=\theta^{T}x
$$

上式右侧我们把$\theta, x$ 都看作向量，这里 $n$ 是输入变量的数目（不计$x_{0}$）。
现在，给定一个训练集，我们如何挑选或学习参数$\theta$？一个合理的方法看上去是让$h(x)靠近y$，至少在训练集上要接近。为了正式表达，我们定义一个函数来测量对于每一个$\theta$ 对应的$h(x^{(i)})$ 有多接近对应的$y$。定义**代价函数**（成本函数）：
$$
J(\theta)=\frac{1}{2}\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-y^{(i)})^{2}
$$
如果你已经见过线形回归，你会注意到这和普通的最小二乘法的代价函数很相似。无论之前是否见过，我们最终会知道这是更广泛算法中的一个特例。
####1 LMS算法
我们想通过选择$\theta$ 来使 $J(\theta)$最小化。为了找到这样的$\theta$, 我们用一些“猜测”的初值来开始搜索算法，然后不断改变$\theta$使得$J(\theta)$ 更小，直到收敛到一个$\theta$ 的值使得$J(\theta)$ 最小化。特别的，考虑梯度下降算法，该算法从一些初值$\theta$开始，重复以下操作更新 [^footnote]：
$$
\theta_{j}:=\theta_{j}-\alpha\frac{\partial}{\partial\theta_{j}}J(\theta).
$$
(该更新同时对所有$j=0,...,n.$进行操作)式中$\alpha$ 称作**学习率**。该算法自然地重复向$J$ 下降最陡峭的方向前进一步。
为了实现该算法，我们要推导出右边的偏导项。先从只有一个训练样本（$x,y$）开始,忽略$J$ 定义的求和项。我们有：
$$
\begin{align}
\frac{\partial}{\partial\theta_{j}}J(\theta)  
=&\frac{\partial}{\partial\theta_{j}}\frac{1}{2}(h_\theta(x)-y)^2\\
=&2\cdot\frac12(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_{j}}(h_\theta(x)-y)\\
=&(h_\theta(x)-y)\cdot\frac{\partial}{\partial\theta_{j}}(\sum_{i=0}^{n}\theta_{i}x_{i}-y)\\
=&(h_\theta(x)-y)x_{j}
\end{align}
$$
对于单个训练样本给出更新法则：
$$
\theta_{j}:=\theta_{j}-\alpha(y^i-h_\theta(x^{(i)}))x^{(i)}_j
$$
这个规则称为**LMS**法则（LMS代表最小均方），也叫做Widrow-Hoff学习规则。该规则有许多看上去自然又符合直觉的特性。比如更新的大小与**误差**项($y^{(i)}-h_{\theta}( {x^{(i)}}) $)成比例；因此，比如遇到一个训练样本预测值很接近$y^{(i)}$的真实值，然后发现不太需要调整参数；相反的，一个对于有大误差的预测值$h_{\theta}{x^{(i)}}$，参数需要有大的调整（比如距离$y^{(i)}$非常远）。

我们已经推导出一个样本的LMS法则。有两种方法来修改这个法则，让它能适用多个样本的训练集。第一个方法是改成如下的算法：
$$
\begin{align}
&重复直到收敛\{\\
&\theta_{j}:=\theta_{j}+\alpha\sum^m_{i=1}(h_\theta(x^{(i)})-y^i)x^{(i)}_j(对于每个j)\\
&\}
\end{align}
$$
读者可以很简单地验证以上更新规则的求和就是$\frac{\partial J}{\partial\theta_{j}}$(J是原始定义)。所以这就是简单对原先代价函数$J$的梯度下降。这个方法看起来在整个训练集的每一个样本的每一步中，称之为**批梯度下降**。注意，一般来说，梯度下降法受到局部极小值的影响。我们提出来是因为线性回归只有一个全局最优点，没有其他局部最优点。因此梯度下降法总是收敛（假设学习率$\alpha$不过大）到最小值。事实上，$J$是凸二次函数。这个是最小化二次函数的梯度下降的例子。
[^footnote]:我们用“$a := b$” 来表示（计算机程序的）赋值操作，设定变量a的数值等于变量b的数值。换句话说，该操作用b的值覆盖a的值。相对的，我们用 “$a = b$"断言事实的状态，a的值等于b的值。\


![二次函数等高线](https://img-blog.csdn.net/20180413203557724?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbW9ucnVhbjky/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)





上图的椭圆展示了二次函数的等高线。也展示了从（48，30）开始的梯度下降的轨迹。图中的$x$(和直线相接)标记出了梯度下降经过的若干连续的$\theta$值.
当我们在之前的数据集上跑批梯度下降来拟合$\theta$时，我们得到$\theta_{0}=71.27,\theta_{1}=0.1345$,如果我们画出$h_{\theta}(x)$作为$x$（面积）的函数图像，以及训练数据点时，我们得到以下图像：
![数据拟合](https://img-blog.csdn.net/20180413210046629?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3NpbW9ucnVhbjky/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



















如果卧室数量也包括在一个输入特征的话，我们得到$\theta_0=89.60,\theta_1=0.1392,\theta_2=-8.738$。
以上结果由批梯度下降得到。批梯度下降以外，另一个算法效果也非常好。考虑如下算法：
$$
\begin{align}
Loop\{\\
	&for\ i=1 to m,\{\\
		    &\theta_j:=\theta_j+\alpha(y^{(i)-h_\theta(x^{(i)})})x_j^{(i)} (对于每个j)\\
	&\}\\
\}\\&
\end{align}
$$
在这个算法中，我们重复遍历训练集，每次遇到一个训练样本，我们根据单个训练样本的误差的梯度更新参数。这个算法称为**随机梯度下降**(也称**增量梯度下降**)。 当批梯度下降法执行一步更新操作要遍历整个训练集-当m很大时，这个操作很费时-随机梯度下降可以直接开始，然后每过一个训练赝本都可以前进一步。经常是随机梯度下降靠近最小值比批梯度下降快得多。（注意有时候也许永远不能收敛到最小值，参数$\theta$会在$J(\theta)$的最小值附近振荡；但是实际上大多数足够靠近最小值的近似都是真正最小值的一个不错的近似值[^footnote2]）。考虑到以上原因，实际上当训练集足够大，随机梯度下降经常都是比批梯度下降更受欢迎的。

####2 正规方程
梯度下降给出了一种最小化$J$ 的方法。让我们讨论第二种方法，这次明确地做最小化而不用迭代算法。在这个方法中，我们明确的求对$\theta_j$的导数来并令之为零来I求$J$的最小值。为了不用写大量的代数和整页整页的矩阵的导数，让我们引入一些记号来做矩阵的计算。
[^footnote2]: 正如我们之前所描述的那样，通常随机梯度下降用固定的学习率$\alpha$，但缓慢下降学习率$\alpha$到0，可以让参数收敛到全局最小而不是仅仅在最小值附近震荡。
#####2.1矩阵导数
对于函数$f:\mathbb R ^{m\times n}\mapsto\mathbb R$从$m\times n$的矩阵映射到实数，我们将$f$对$A$的导数定义为：
$$
\triangledown_{A}f(A)=
\left[
\begin{matrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1n}}\\
\vdots & \ddots & \vdots\\
\frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}}\\
\end{matrix}
\right]
$$
因此，梯度$\triangledown_Af(A)$自己是$m\times n$的矩阵，其中第$(i,j)$个元素是$\frac{\partial f}{\partial A_{ij}}$。例如假设
$$
A=\begin{bmatrix} 
A_{11}&A_{12} \\
A_{21}&A_{22}
\end{bmatrix}
$$
是一个$2\times 2$的矩阵，函数$f:\mathbb R^{2\times2}\mapsto R$由下式给出：
$$
f(A)=\frac{3}{2}A_{11}+5A_{12}^2+A_{21}A_{22}
$$
其中$A_{ij}$表示矩阵A的第$(i,j)$个元素，则有：
$$
\triangledown_A f(A)=
\begin{bmatrix}
\frac32 &10A_{12}\\
A_{22} & A_{21}
\end{bmatrix}
$$
我们也引入**迹**算子，写作"tr."对于$2\times2$的方阵A，他的迹定义为对角线上所有元素的和：
$$
trA=\sum_{i=1}^nA_{ii}
$$
若a是实数（例如1$\times$1矩阵），那么$tr a=a$。（如果你还没有见过“算子记号”，可以把A的迹想象成tr(A)，把迹作为矩阵A的函数。通常不写括号。）
迹算子有以下性质：若两个矩阵A，B都是方阵的，他们的乘积AB也是方阵，则有$trAB=trBA$（自己验证）。作为该性质的推广我们不难得出：
$$
trABC=trCAB=trBCA,\\
trABCD=trDABC=trCDAB=trBCDA
$$
以下迹的性质也很容易验证。式中A和B都是方阵，a是实数：
$$
\begin{align}
trA&=trA^T\\
tr(A+B)&=trA+trB\\
traA&=atrA
\end{align}
$$
我们现不加证明地给出以下关于矩阵导数的结论（其中有一些我们在本章之后的地方才会用到）。公式（4）仅仅对非奇异的矩阵A适用，其中|A|是A的行列式。我们有：
$$
\begin{align}
\triangledown_AtrAB&=B^T\tag 1\\
\triangledown_{A^T}f(A)&=(\triangledown_Af(A))^T\tag 2\\
\triangledown_AtrABA^TC&=CAB+C^TAB^T\tag 3\\
\triangledown_A|A|&=|A|(A^{-1})^T\tag 4
\end{align}
$$
为了让矩阵记号更加具体，让我们具体解释一下第一个方程。假设我们有一常数矩阵$B\in\mathbb R^{n\times m}$。我们根据$f(A)=trAB$定义函数$f:\mathbb R^{m\times n}\mapsto R$。注意该定义是合理的，因$A\in\mathbb R^{m\times n}$,则AB是方阵，能对其应用迹算子；因此f事实上从$\mathbb R^{m\times n}$映射到$\mathbb R$。我们能用矩阵导数的定义得到$\triangledown_Af(A)$，这是$m\times n$矩阵。上式(1)表明该矩阵第$(i,j)$项由$B^T$的$(i,j)$项给出，即$B_{ji}$。
上式(1-3)的证明简单，留给读者作练习用。上式(4)可用矩阵的逆的伴随矩阵推导出[^footnote3].
[^footnote3]:如果我们把A'定义成这样一个矩阵，其中第（i，j）个元素是$(-1)^{i+j}$乘以A删去i行j列后的方阵的行列式，那么可以证明$A^{-1}=A'^T/|A|$(你可以用标准的方法算e二阶$A^{-1}$来验证一致性。如果要得出更一般的结论，请参考中高级的线性代数书比如*Charles Curtis,1991, Linear Algebra, Springer*)。$A'=|A|(A^{-1})^T$得证。而且矩阵的行列式可以写成$|A|=\sum_jA_{ij}A'_{ij}$。由于$(A')_{ij}$与$(A)_{ij}$无关(由定义)，这就说明$\frac{\partial}{\partial A_{ij}}|A|=A_{ij}'$。把所有的放到一起就是结果。

#####2.2  再议最小二乘
有了矩阵导数的帮助，让我们寻找使得$J(\theta)$最小化的$\theta$值的封闭形式。我们从用矩阵-向量记号重写$J$开始。
给定训练集，定义**设计矩阵**X为$m\times n$矩阵(实际上是$m\times n+1$，如果包含截距项)的行包含了训练集的输入值：
$$
X=
\begin{bmatrix}
-(x^{(1)})^T-\\
-(x^{(2)})^T-\\
\vdots\\
-(x^{(m)})^T-
\end{bmatrix}
$$
让$\overrightarrow{y}$为包含训练集所有目标值的m维向量：
$$
\overrightarrow{y}=
\begin{bmatrix}
y^{(1)}\\y^{(2)}\\
\vdots\\y^{(m)}\\
\end{bmatrix}
$$
由于$h_\theta(x^{(i)})=(x^{(i)})^T\theta$我们能很简单地证明：

$$
\begin{align}
X\theta-\overrightarrow{y} &=
\begin{bmatrix}
(x^{(1)})^T\theta\\
\vdots\\
(x^{(m)})^T\theta\\
\end{bmatrix}-\begin{bmatrix}
y^{(1)}\\
\vdots\\y^{(m)}\\
\end{bmatrix}
 &=\begin{bmatrix}
h_\theta(x^{(1)})-y^{(1)}\\
\vdots\\
h_\theta(x^{(m)})-y^{(m)}\\
\end{bmatrix}
\end{align}
$$

因此用向量z的一个性质我们有$z^Tz=\sum_iz^2_i$:

$$
\begin{align}
\frac12(X\theta-\overrightarrow y)^T(X\theta-\overrightarrow y)&=\frac12\sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)})^2\\
&=J(\theta)
\end{align}
$$
最后，最小化J，让我们求关于$\theta$的导数。联立方程(2)(3)得到：
$$
\triangledown_{A^T}trABA^TC=B^TA^TB^T+BA^TC\tag 5
$$
有：
$$
\begin{align}
\triangledown_{\theta}J(\theta)&=\triangledown_\theta\frac12(X\theta-\overrightarrow y)^T(X\theta-\overrightarrow y)\\
&=\frac12\triangledown_\theta(\theta^TX^TX\theta-\theta^TX^T\overrightarrow y-\overrightarrow y^TX\theta+\overrightarrow y^T\overrightarrow y)\\
&=\frac12\triangledown_\theta tr(\theta^TX^TX\theta-\theta^TX^T\overrightarrow y-\overrightarrow y^TX\theta+\overrightarrow y^T\overrightarrow y)\\
&=\frac12\triangledown_\theta(tr\theta^TX^TX\theta-2tr\overrightarrow y^TX\theta)\\
&=\frac12(X^TX\theta+X^TX\theta-2X^T\overrightarrow y)\\
&=X^TX\theta-X^T\overrightarrow y
\end{align}
$$
在第三步，利用了迹只是一个实数的这条性质；第四步利用了性质$trA=trA^T$;第五步利用了式(5),其中$A^T=\theta,B=B^T=X^TX及C=I$以及式(1)。为了求J最小值，令导数为零，得到正规方程：
$$
X^TX\theta=X^T\overrightarrow y
$$
那么$J(\theta)$取最小值时$\theta$的取值由下式这样的封闭形式得到：
$$
\theta=(X^TX)^{-1}X^T\overrightarrow y
$$
####3 概率解释
当我们面对回归问题的时候，为什么用线性回归，特别是为什么用最小二乘代价函数J是合理的选择？在这一节，我们会给出一系列概率假设，在这些假设下，最小二乘回归是一种非常自然的算法。
让让我们假设目标值和输入值满足以下方程：
$$
y^{(i)}=\theta^Tx^{i}+\epsilon^{(i)},
$$
其中$\epsilon^{(i)}$是误差项，可能原因是模型以外的因素（比如某特征与房价高度相关，但是没有被纳入回归）或者随机噪声。进一步假设$\epsilon^{(i)}$是与有均值零及一定的方差$\sigma$的高斯分布（或正态分布）独立同分布的。我们把这个假设写成"$\epsilon^{(i)}$～$\cal N(0,\sigma^2)$"。换句话说,$\epsilon^{(i)}$的概率密度函数满足：
$$
p(\epsilon^{(i)})=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(\epsilon^{(i)})^2}{2\sigma^2})
$$
这表明：
$$
p(y^{(i)}|x^{(i)};\theta)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
$$
记号$p(y^{(i)}|x^{(i)};\theta)$表示$y^{(i)}$在给定$x^{(i)}$下由$\theta$参数决定的概率密度函数。注意到不应该把$\theta$作为条件($p(y^{(i)}|x^{(i)},\theta)$),因为$\theta$不是随机变量。可以把$y^{(i)}$写成$y^{(i)}|x^{(i)};\theta\sim\cal N (\theta^Tx^{(i)},\sigma^2)$。
对于给定X（即设计矩阵，其包含所有$x^{(i)}$）及$\theta$，对应的$y^{(i)}$是什么样的分布？数据的概率由$p(\overrightarrow y|X;\theta)$给出。对于某个特定的$\theta$的值，这个概率通常被看做是$\overrightarrow y$(或者X)的函数。当我们把它看做一个$\theta$的具体的函数时，我们会把它叫做**似然**函数：
$$
L(\theta)=L(\theta;X,\overrightarrow y)=p(\overrightarrow y|X;\theta)
$$
注意到$\epsilon^{(i)}$的独立性假设（也包括$y^{(i)}和x^{(i)}$）,上式也可写成：
$$
\begin{align}
L(\theta)&=\prod^{m}_{i=1}p(y^{(i)}|x^{(i)},\theta)\\
&=\prod^{m}_{i=1}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})
\end{align}
$$
现在给定关于$y^{(i)}和x^{(i)}$的概率模型，那么对参数$\theta$的合理的最佳估计是什么？**极大似然**估计方法是指我们应该估计$\theta$使得数据的概率尽可能大。也就是说我，我们求使得$L(\theta)$最大化的$\theta$。
如果不求$L(\theta)$的最大值，我们也可以求任意严格递增的$L(\theta)$函数的最大值。特别是，求**对数似然函数**$\cal l\theta$的最大值的时候，导数更容易求:
$$
\begin{align}
\cal{l}(\theta)&=logL(\theta)\\
&=log\prod^{m}_{i=1}\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})\\
&=\sum^m_{i=1}log\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(y^{(i)}-\theta^Tx^{(i)})^2}{2\sigma^2})\\
&=mlog\frac{1}{\sqrt{2\pi}\sigma}-\frac{1}{\sigma^2}\cdot\frac{1}{2}\sum^m_{i=1}(y^{(i)}-\theta^Tx^{(i)})^2
\end{align}
$$
那么求$\cal{l}(\theta)$的最大值等价于求解下士的最小值：
$$
\frac{1}{2}\sum^m_{i=1}(y^{(i)}-\theta^Tx^{(i)})^2
$$
我们发现这就是原来的最小二乘代价函数。
总结：给予之前对于数据的概率假设，最小二乘回归恰好找到了$\theta$的最大似然估计。这就是在此假设下最小二乘回归是一个非常自然的方法，因为它就是做极大似然估计。（注意让最小二乘成为非常好或者很合理的方法而言，概率假设并不一定是*必要的*。并且有其他自然的假设用来证明其合理性）。
还注意到在我们之前的讨论中，最终$\theta$的取值与$\sigma$无关，实际上就算我们不知道$\sigma^2$也能求得同样的结果。我们也会在讨论指数和一般线性模型的时候再一次用上这个结论。
####4 局部加权线性回归
考虑到从$x\in\mathbb R$预测$y$的问题。下图最左侧的图展示了数据集上拟合$y=\theta_0+\theta_1x$的结果。我们看到数据点并没由完全落到直线上，所以拟合得并不好。
![三种拟合](http://img.blog.csdn.net/20180415160204209?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2ltb25ydWFuOTI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
然而如果我们在加一个特征$x^2$拟合$y=\theta_0+\theta_1x+\theta_2x^2$那么我们拟合数据稍好一点（见中间这幅图）。那么是不是特征越多，拟合数据就越好呢？然而，加太多特征也是有危险的：右图是五阶多项式的拟合结果$y=\sum^5){j=0}\theta_jx^j$。我们可以看到虽然拟合曲线完美地经过了每一个点，但是这并不像是一个好的房价(y)和居住面积(x)关系的拟合。不加正式定义的，我们把左图展示的这种情况称作**欠拟合**，其中数据的结构没有很好地由模型表达出来。右图的情况称作**过拟合**。(之后在这个课程里，当我们讨论学习理论时，我们会给出记号的正式定义，也会小心地给出假设好坏的定义)。
正如以上所述，在例子中已经展示了选择特征对于学习理论效果好坏的重要性。（当我们讨论模型选择时，会看到算法是如何自动选择一系列好的特征的）。在这一节，让我们简单地讨论以下局部加权线性回归(LWR)，LWR就是假设在有充足的训练数据的情况下，使得特征的选择略微不那么关键一点。这种处理是简明的，因为你会有机会自己在作业中自己探索一下LWR的一些性质。
在原本的线性回归算法中，当遇到一个有疑问的点x(来估计h(x))我们会：
1. 求$\sum_i(y^{(i)}-\theta^Tx^{(i)})^2$最小值来拟合
2. 输出$\theta^Tx$  
相对的是，局部加权线性回归算法会这样做：
1. 求$\sum_iw^{(i)}(y^{(i)}-\theta^Tx^{(i)})^2$最小值来拟合
2. 输出$\theta^Tx$  
式中$w^{(i)}$是非负**权重值**。直觉上，如果$w^{(i)}$对于一个特定i值很大，在挑选$\theta$时，我们努力让$(y^{(i)}-\theta^Tx^{(i)})^2$变小。如果$w^{(i)}$很小，那么误差项$(y^{(i)}-\theta^Tx^{(i)})^2$就在拟合的时候很容易被忽略掉。
决定权重的一种合适方法是[^footnote4]：
$$
w^{(i)}=exp(-\frac{(x^{(i)-x})^w}{2\tau^2})
$$
注意到权重取决于特定的要求值的x。更进一步说，如果$|x^{(i)}-x|$很小那么$w^{(i)}$接近1；如果$|x^{(i)}-x|$很大那么$w^{(i)}$很小。因此，$\theta$选择给予训练样本(的误差)上靠近所求点x更大的权重。也注意到当权重公式的形式上很类似高斯分布的概率密度分布，$w^{(i)}$并没有和高斯分布有什么直接关系特别是$w^{(i)}$不是随机变量，正太分布或者什么类似的东西。参数$\tau$控制着训练样本的权重对于待求点x随着距离$x^{(i)}$下降的快慢。$\tau$叫做**带宽**参数，也会在作业里遇到。
局部加权线性回归是我们看到的第一个**非参数**算法。之前见过的（不加权的）线性回归也称作**含参数**的学习算法，因为有固定的，有限个数的参数($\theta$),用来拟合数据。一旦完成了拟合，保留下参数$\theta$我们就不在需要原始数据来预测未来的情况。相对的，在用局部加权线性回归的时候我们需要保留整个训练集来做未来的预测。“非参数”指的是需要保留一堆东西来代表假设$h$随着训练集的尺度线性增长。
[^footnote4]:若是向量值，一般化的表达是$w^{(i)}=exp(-(x^{(i)}-x)^T\frac{x^{(i)}-x}{2\tau^2})$，或者选择合适$\tau$或者$\Sigma$下$w^{(i)}=exp(-(x^{(i)}-x)^T\sum^{-1}\frac{x^{(i)}-x}{2})$。

### 第二部分 分类与logistic回归
现在让我们来讨论分类问题。这就像是回归问题，对于值y，我们想预测他属于少数几个值。现在我们关注**二元分类**问题，其中y只能取两种值，0或者1.（我们说的大多数问题都可以推广到多分类问题）。比如我们要做一个垃圾邮件分类器，那$x^{(i)}$是一封邮件的若干特征，当邮件是垃圾邮件时y是1，否则是0.0也成为**负类**，1称为**正类**，有时也用“-”，“+”来表示。对于给定的$x^{(i)}$对应的$y^{(i)}$也叫做训练样本的**标签**。
#### 5 logistic回归
我们可以忽略y是离散值的事实来处理分类问题，就用老的线性回归来尝试x对应的y。然而，很容易就能构造一个让这个办法效果很差的样例。直观上来说，当我们已经知道$y\in\{0,1\}$,$h_\theta(x)$取值大于1或小于0都是没什么意义的。
为了修正这一点，让我们改变假设$h_\theta(x)$的形式。我们选择
$$
h_\theta(x)=g(\theta^Tx)=\frac{1}{1+\theta^{-\theta^Tx}},
$$
其中
$$
g(z)=\frac{1}{1+e^{-z}}
$$
叫做**logistic**函数或者**sigmoid**函数，下图是g(z)的函数图像：
![logistic function](http://img.blog.csdn.net/20180415193408134?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2ltb25ydWFuOTI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
注意当$z\rightarrow\infty$时，g(z)趋向于1，当$z\rightarrow-\infty$时，g(z)趋向于0。这样g(z)及h(x)就限制在0,1之间。和之前一样我们让$x_0=1$故有$\theta^Tx=\theta_0+\sum^n_{j=1}\theta_jx_j$.
现在让我们在给定的g中选择。其他光滑的从0增长到1的函数也可用，但是之后我们会解释理由（当我们讨论GLM时即讨论生成学习算法的时候），对于logistic函数的选择是非常自然的。在继续之前，以下是sigmoid函数导数的几个有用的性质,简记为g'：
$$
\begin{align}
g^{'}(z)&=\frac{d}{dz}\frac{1}{1+e^{-z}}\\
&=\frac{1}{(1+e^{-z})^2}(e^{-z})\\
&=\frac{1}{1+e^{-z}}(1-\frac{1}{1+e^{-z}})\\
&=g(z)(1-g(z))
\end{align}
$$
那么对于给定的logistic回归模型，我们怎么来拟合$\theta$呢？类似于之前最小二乘拟合可以由极大似然估计在一系列的概率假设下导出，然后通过极大似然法拟合参数。
让我们假设：
$$
\begin{align}
P(y=1|x;\theta)$=\\
P(y=0|x;\theta)$=1-h_{\theta}(x)\\
\end{align}
$$
注意到可以写得更简洁：
$$
p(y|x;\theta)=(h_{\theta}(x))^y(1-h_{\theta}(x))^{1-y}
$$
假设m个训练样本都是独立生成的，我们能写出参数的似然函数：
$$
\begin{align}
L(\theta) &=p(\overrightarrow y|X;\theta)\\
&=\prod^m_{i=1}p(y^{(i)}|x^{(i)};\theta)\\
&=\prod^m_{i=1}(h_{\theta}(x^{(i)}))^{y^{(i)}}(1-h_{\theta}(x^{(i)}))^{1-y^{(i)}}\\
\end{align}
$$
和之前一样，对数似然函数更容易求解：
$$
\begin{align}
l(\theta)&=logL(\theta)\\
&=\sum^m_{i=1}y^{(i)}logh(x^{(i)})+(1-y^{(i)})log(1-h(x^{(i)}))
\end{align}
$$
那怎么求似然函数最大值呢？类似线性回归的例子，我们可以用梯度上升。用向量记号写，我们的更新因此可以这样写$\theta:=\theta+\alpha\triangledown_{\theta}l(\theta)$(注意公式里是正号不是负号，因为这个函数现在是求最大值，不是最小值)。让我们从单个训练样本（x，y）开始用随机梯度上升规则：
$$
\begin{align}
\frac{\partial}{\partial\theta_j}l(\theta)&=(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})\frac{\partial}{\partial\theta_j}g(\theta^Tx)\\
&=(y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)})g(\theta^Tx)(1-g(\theta^Tx))\frac{\partial}{\partial\theta_j}\theta^Tx\\
&=(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx))x_j\\
&=(y-h_{\theta}(x))x_j
\end{align}
$$
以上我们利用了$g^{'}(z)=g(z)(1-g(z))$因此，给出了梯度上升的公式：
$$
\theta_j:=\theta_j+\alpha(y{(i)}-h_{\theta}(x^{(i)}))x_j
$$
若我们将之与最小二乘LMS的更新公式相比较，会发现它们看上去一样的。但是这*不*是同一种算法，因为现在$h_{\theta}(x^{(i)})$定义为$\theta^Tx^{(i)}$的非线性函数。不管怎么说，不同的算法，不同的学习问题最终结果一样还是有点惊讶的，或许背后有着更深层次的原因？当我们学习GLM的时候会得到答案。(详见题集1的Q3附加题)。

#### 6 题外话：感知机学习算法
我们说点题外话，现在简单讨论一个在历史上很有趣的算法，之后我们在讨论学习理论的时候又会回来讨论他。考虑修改logistics函数“强迫”其只能输出0或者1。
为了实现这样，把g的定义改成阈值函数看上去很自然：
$$
g(z)=
\begin{cases}
&1&if\ z\ \ge0\\
&0&if\ z\ <0\\
\end{cases}
$$
若和之前一样使$h_{\theta}(x)=g(\theta^Tx)$该是用修改后的g的定义，如果我们用如下更新规则：
$$
\theta_j:=\theta_j+\alpha(y^{(i)}-h_{\theta}(x^{(i)}))x_j^{(i)}
$$
那么我们就有了**感知机学习算法**。
在20世纪60年代，提出了“感知机”作为大脑中单个神经元的工作原理模型。由于这个算法如此简单，这给算法给我们在今后的课程上讨论学习算法提供了一个很好的起点。注意感知机在形式上并不类似于我们讨论的其他算法。实际上这是一种与logistics回归或者线性回归非常不同类型的算法。特别是很难以给他一个有含义的概率解释或者像极大似然算法那样的推导。

#### 7 另一种求$l(\theta)$最大值的算法
回到用sigmoid函数作为g(z)的logstics回归，让我们讨论另外一种求$l(\theta)$最大值的算法。
让我们考虑用牛顿法求一个函数的零点。具体来说我们有某函数$f:\mathbb R\mapsto\mathbb R$，我们希望找到一个$\theta$的值使得$f(\theta)=0$。其中$\theta\in\mathbb R$是实数。牛顿法为：
$$
\theta:=\theta-\frac{f(\theta)}{f^{'}(\theta)}
$$
这个算法有着非常自然的解释，就是我们用一个在当前$\theta$与函数f相切的线性函数来逼近函数f，解线性方程等于0的位置，并且用其来作下一个迭代值。
下图是牛顿法的过程的图像：
![牛顿法](http://img.blog.csdn.net/20180415193442587?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2ltb25ydWFuOTI=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
左图可以看到函数f图像及y=0的水平直线。我们尝试求$\theta$使得$f(\theta)=0$，由图得，$\theta$的值约1.3。假设算法初始值$\theta=4.5$牛顿法拟合$\theta=4.5$f的切线求0点(中图)。这给了我们下一个$\theta$的猜测值，大约2.8。右图是再下一次迭代的结果，$\theta$大约为1.8。在几次迭代后，我们迅速地得到了$\theta=1.3$。
就蹲发是一种求$f(\theta)=0$零点的方法。如果我们要求l最大值该怎么办？l的极大值点应该有一阶导数$l^{'}(\theta)=0$。所以令$f(\theta)=l^{'}(\theta)$我们可以用同样的方法来求l的最大值，我们得到了更新规则：
$$
\theta:=\theta-\frac{l^{'}(\theta)}{l^{''}(\theta)}
$$
(想一想，如果我们要用牛顿法求一个函数的最小值而不是最大值应该怎么做？)
最后，在我们的logistic回归中，$\theta$是向量，所以我们要把牛顿法推广到这上来。牛顿法的推广需要多维度设定（也叫做牛顿-拉夫逊法）由下式给出：
$$
\theta:=\theta-H^{-1}\triangledown_{\theta}l(\theta)
$$
其中$\triangledown_{\theta}l(\theta)$一般是$l(\theta)$关于$\theta_{i}$的向量偏导数；H是n×n矩阵（实际上(n+1)×(N+1)，假设假设我们忽略截距项），叫做海森矩阵：
$$
H_{ij}=\frac{\partial^2l(\theta)}{\partial\theta_i\partial\theta_j}
$$
典型上说，牛顿法收敛比（批）梯度下降快，需要更少的迭代就能求得最小值。虽说一次牛顿法的迭代比梯度梯度下降更加“昂贵”，因为需要求n×n海森矩阵，但是只要n不太大一般总是比较快的。当牛顿法应用于logistic对数极大似然函数$l(\theta)$时，结果的方法叫做**Fisher scoring**。

### 第三部分 广义线性模型[^footnote5]
目前为止，我们已经看过了一个回归模型，一个分类模型。在回归模型中，我们有$y|x;\theta\sim\cal{N}(\mu,\sigma^2)$，在分类模型中，我们有$y|x;\theta\sim Bernoulli(\phi)$，其中$\mu、\phi$是$x、\theta$有适当的定义的函数。在本章节，我们会说明以上两种方法是一更广泛模型类型的两个特例，称为（广义线性模型GLM）。我们也会说明GLM的其他模型类型能够导出并应用于其他分类和回归问题。

####8 指数族
为了适应GLM，我们先定义指数族分布。当满足下式时，我们说一类分布是指数族分布：
$$
p(y;\eta)=b(y)exp(\eta^TT(y)-a(\eta))\tag 6
$$
其中$\eta$称为分布的**自然参数**（也叫作**正则参数**）T(y)是充分统计量（对于我们考虑的分布，经常有T(y)=y），$a(\eta)$是**对数划分函数**. $e^{-a(\eta)}$的数量在归一化常数中起到非常重要的作用,他能保证$p(y;\eta)$的分布的和除以y得1.
T, a, b的选择定义了由$\eta$参数化的一族分布,当我们改变$\eta$的值,我们能获得这个分布族内的不同分布.
现在我们将说明伯努利分布和高斯分布是指数分布族内的两个例子. 均值为$\phi$的伯努利分布计记做$Bernoulli(\phi)$,其定义了$y \in \{0,1\}$的分布故$p(y=1;\phi)=\phi;p(y=0;\phi)=1-\phi$.当改变$\phi$的值,我们得到不同均值的伯努利分布. 现在说明的这类伯努利分布是改变$\phi$得到的一类指数族. 例如对于特定的$T,a,b$,那方程(6)就正好变成伯努利分布类型.
[^footnote5]:本章节的展示材料灵感来源于 Michael I.Jordan, *Learning in graphical models*(未出版书稿),以及McCullagh and Nelder, *Generalized Linear Models*(第二版)

我们把伯努利分布写成:
$$
\begin{align}
p(y;\phi)&=\phi^y(1-\phi)^{1-y}\\
&=exp(ylog\phi+(1-y)log(1-\phi))\\
exp&=((log(\frac{\phi}{1-\phi}))y+log(1-\phi))\\
\end{align}
$$
因此，自然参数由$\eta=log(\phi/(1-\phi))$给出。有趣的是，如果我们用$\eta$解出$\phi$的表达式，会得到$\phi\frac{1}{1+e^{-\eta}}$。这就是我们熟悉的sigmoid函数表达式！这就又回到了推导出logistic函数是一种GLM。为了完成伯努利分布的指数族分布公式，我们有:
$$
\begin{align}
T(y)&=y\\
a(\eta)&=-log(1-\phi)
&=log(1+^{\eta})
b(y)&=1
\end{align}
$$
这说明了伯努利分布分公式可以写成公式(6)的形式,只要选定合适的T,a,b。
现在让我们考虑高斯分布。回到我们推导线性回归的时候，$\sigma^2$的值对于最后$\theta 和h_{\theta}(x)$的取值没有影响。为了简化推导过程，令$\sigma^2=1$[^footnote6]。我们有：
$$
\begin{align}
p(y;\mu)&=\frac{1}{\sqrt{2\pi}}(-\frac{1}{2}(y-\mu)^2)\\
&=\frac{1}{\sqrt{2\pi}}(-y^2)\cdot exp(\mu y-\frac{1}{2}\mu^2)
\end{align}
$$
[^footnote6]: 若我们令 $\sigma^2$为变量，高斯分布也能以指数族的方式来表示其中$\eta\in\mathbb R^{2}$现在是基于$\mu 和\sigma$的二维向量为了GLM，虽说$\sigma^w$参数能被当做是更广义的指数族的定义：$p(y;\eta,\tau)=b(a,\tau)exp((\eta^TT(y)-a(\eta))/c(\tau))$.其中$\tau$称作**分散度参数**，对于高斯分布$c(\tau)=\sigma^2$;但是在我们的简化下，这里例子不需要考虑更一般的定义。

因此，我们可以看出高斯分布是指数族，有：
$$
\begin{align}
\eta&=\mu\\
T(y)&=y\\
a(\eta)&=\mu^2/2\\
&=\eta^2/2\\
b(y)&=(1/\sqrt{2\pi})exp(-y^2/2)
\end{align}
$$
有很多其他分布也是指数族：多项分布（之后会看到），泊松分布（用于计数数据建模），伽马分布和指数分布（用与连续非负随机变量建模，比如时间间隔）；beta分布和狄利克雷分布（用于概率分布），还有很多。在下一节中，我们会描述一个通用的办法来对任意分布的y（给定x和$\theta$）建模.
####9 构建GLM
假设你要基于一些特征建立一个模型来估计任意时间到达你的商店的数目y，比如商店优惠，最近广告，天气，星期几等等。我们知道泊松分布可以很好地预测来访者的数目。既然知道了，那对于我们的问题应该怎样来建一个模型呢？幸好泊松分布是指数族分布，所以我们可以用GLM。在这一节，我们会讨论一个办法来建立GLM模型来解决类似这样的问题。
更一般的来说，考虑分类或回归问题，其中要预测x的函数随机变量y。为了要推导出问题对应GLM，我们对我们的模型做以下三个条件关于给定x的y的分布假设：
1. $y|x;\theta$~$ExponentialFamily(\eta)$例如给定x和$\theta$，y的分布满足某个指数族分布，参数是$\eta$  
2. 给定x，我们的目标是预测给定x的对应T(y)的值。在我们大多数的例子中，我们有T(y)=y，故这意味着我们要通过我们学习到的假设h的输出值h(x)满足h(x)=E[y|x].(注意到这个假设$h_{\theta}(x)$对于logistic回归和线性回归都适用。比如说logistic回归有$h_{\theta}(x)=p(y=1|x;\theta)=0\cdot p(y=0|x;\theta)+1\cdot p(y=1|x;\theta)=E[y|x;\theta]$)
3. 自然参数$\eta$和输入x是线性相关的：$\eta=\theta^Tx$（或者如果$\eta$是向量的话那么$\eta_i=\theta^T_ix$）.
#####9.1 普通最小二乘
为了说明普通最小二乘是GLM族的模型，考虑令y为连续目标变量（也叫GLM的**响应变量**），我们建立模型给定x的条件分布y是满足高斯分布$\cal N(\mu,\sigma^2)$的(这里$\mu 可以依赖于x$)。所以我们让上面的$ExponentialFamily(\eta)$分布为高斯分布。正如我们之前所看到的，作为指数族分布的高斯分布的公式中标$\mu=\eta$所以我们有：
$$
\begin{align}
h_{\theta}(x)&=E[y|x;\theta]\\
&=\mu\\
&=\eta\\
&=\theta^Tx
\end{align}
$$
第一个公式遵循假设2；第二个公式遵循$y|x;\theta\cal N(\mu,\sigma^2)$。故其期望值为$\mu$第三个等式遵循假设1（我们之前的推导说明在作为指数分布的高斯分布公式中，$\mu=\eta$）;最后一个公式遵循假设3.
#####9.2 logistic 回归
我们现在考虑logistic回归。在此，我们对二元分来感兴趣,所以$y\in\{0,1\}$。给定y是个二元值，所以用伯努利族的分布来给对给定x的y建模是非常自然的事。在我们的作为指数族分布的伯努利分布公式中。我们有$\phi=1/(1+e^{-\eta})$。更进一步说注意到$y|x;\theta$~$Bernoulli(\phi)$，然后$E[y|x;\theta]=\phi$。所以由类似的推导有普通的最小二乘，有：
$$
\begin{align}
h_{\theta}(x)&=E[y|x;\theta]\\
&=\phi\\
&=1/(1+e^{-\eta})\\
&=1/(1+e^{-\theta^T}x)
\end{align}
$$
所以，这给了我们这样形式的假设公式$h_{\theta}=1/(1+e^{\theta^T}x)$，如果你之前想过logistic的公式$1/(1+e^{z})$是怎么推导的，这就能说明GLM和指数族分布的定义是怎么来的。
再介绍一点术语，g函数以自然参数$(g(\eta)=E[T(y);\eta])$的方式给出了分布的均值叫做**正则响应函数**。反函数$g^{-1}$叫做**正则链接函数**。那么，高斯族的正则响应函数就是恒等函数；伯努利分布的正则响应函数是logistic函数[^footnote7]。
#####9.3 softmax 回归
让我们看一个GLM的例子。考虑一个分类问题，其中响应变量y可以去k个值中的一个，所以$y\in\{1,2,...,k\}$。例如相比较于二元的垃圾，非垃圾的邮件分类，我们会想把邮件分成三类比如垃圾邮件，个人邮件，工作相关邮件。响应变量依然是离散的，但现在可以选多于两个值。我们会因此根据多项来把他建立成分布的模型。
[^footnote7]:许多书用g来表示链接函数，用来$g^{-1}$表示响应函数法；但是注意到我们在这里用的记号是从早期的机器学习书籍继承而来，这样会与我们在剩下的课程中使用的记号更相一致。

让我们对这种多项来推导一个GLM模型。为了这样，我们开始用指数族分布的形式来表达多项分布。为了要把k个输出的取值的多项分布参数化，可以用k个参数$\phi_{1},...,\phi_{k}$来表达每一个可能的输出。但是这些参数是有冗余的，或者说他们不是独立的（已经独立知道k-1个的$\phi_i$值可以推出最后一个，因为要满足$\sum^k_{i=1}\phi_i=1$）。所以我们只把多项用k-1个参数来表示，$\phi_1,...,\phi_{k-1},$其中$\phi_i=p(y=i;\phi),p(y=k;\phi)=1-\sum^{k-1}_{i=1}\phi_i$。为了简洁，令$\phi_k=1-\sum^{k-1}_{i=1}\phi_i$。但是要记得这不知一个参数，它由$\phi_1,...,\phi_{k-1}$所确定。
为了把多项分布用指数族分布来表示，我们会定义如下$T(y)\in\mathbb R^{k-1}$：
$$
T(1)=\begin{bmatrix}&1\\&0\\&0\\&\vdots\\&0\\\end{bmatrix},
T(2)=\begin{bmatrix}&0\\&1\\&0\\&\vdots\\&0\\\end{bmatrix},
T(3)=\begin{bmatrix}&0\\&0\\&1\\&\vdots\\&0\\\end{bmatrix},
\dots,
T(k-1)=\begin{bmatrix}&0\\&0\\&0\\&\vdots\\&1\\\end{bmatrix},
T(k)=\begin{bmatrix}&0\\&0\\&0\\&\vdots\\&0\\\end{bmatrix},
$$
不想我们之前的例子，这里没有T(y)=y;这里T(y)是k-1维的向量而不是一个实数。我们用$(T(y))_i$来表示向量T(y)的第i个元素。
我们现在引入一个非常有用的记号。指示函数1{$\cdot$}当变量为真时为1，否则为0（1{True}=1,1{False}=0）。比如1{2=3}=0,1{3=5-2}=1。那T(y)和y的关系可以写作$(T(y))_i=1{y=i}$(在你继续读下去之前，请确保你理解为什么这是真！)。进一步说$E[(T(y))_i]=P(y=i)=\phi_i$。
我们现在准备好证明多项分布是指数族的一类，我们有：
$$
\begin{align}
p(y;\phi)&=\phi^{1\{y=1\}}_{1}\phi^{1\{y=2\}}_{2}\cdots\phi^{1\{y=k\}}_{k}\\
&=\phi^{1\{y=1\}}_{1}\phi^{1\{y=2\}}_{2}\cdots\phi^{1-\sum^{k-1}_{i=1}1\{y=i\}}_{k}\\
&=\phi^{(T(y))_1}_{1}\phi^{(T(y))_2}_{2}\cdots\phi^{1-\sum^{k-1}_{i=1}(T(y))_i}_{k}\\
&=exp((T(y))_1log(\phi_1)+(T(y))_2log(\phi_2)+\cdots+(1-\sum^{k-1}_{i=1}(T(y))_i)log(\phi_k))\\
&=exp((T(y))_1log(\phi_1/\phi_k)+(T(y))_2log(\phi_2/\phi_k)+\cdots+((T(y))_{k-1}log(\phi_{k-1}/\phi_k)+log(\phi_k))\\
&=b(y)exp(\eta^TT(y)-a(\eta))
\end{align}
$$
其中
$$
\begin{align}
\eta&=\begin{bmatrix}
log(\phi_1/\phi_k)\\
log(\phi_2/\phi_k)\\
\vdots\\
log(\phi_{k-1}\phi_k)
\end{bmatrix},\\
a(\eta)&=-log(\phi_k)\\
b(y)&=1.
\end{align}
$$
这就完成了我们指数族形式的多项分布的公式。
链接公式(i=1,...,k)由下式给出：
$$
\eta_i=log\frac{\phi_i}{\phi_k}
$$
为了方便，我们也定义$\eta_k=log(\phi_k/\phi_k)=0$。翻转链接函数，导出响应函数，因此有：
$$
\begin{align}
e^{\eta_i}&=\frac{\phi_i}{\phi_k}\\
\phi_ke^{\eta_i}&=\phi_i \tag{7}\\
\phi_k\sum^k_{i=1}e^{\eta_i}&=\sum^k_{i=1}\phi_i=1
\end{align}
$$
这就说明$\phi_k=1/\sum^{k}_{i=1}e^{\eta_i}$，这带回公式(7)就给出响应方程：
$$
\phi_i=\frac{e^{\eta}_i}{\sum^k_{j=1}e^{\eta_j}}
$$
这个从$\eta$到$\phi$的映射称作**softmax**函数。
为了完成我们的模型，我们用假设3，之前给出的$\eta_i$与x的线性相关。所以有$\eta_i=\theta^T_ix$(i=1,...,k-1)，其中$\theta_1,...,\theta_{k-1}\in\mathbb R^{n+1}$是我们模型的参数。为了标记方便，我们定义$\theta_k=0$所以像之前给出的那样，$\eta_k=\theta^T_kx=0$。所以我们的模型假设对于给定x的y的条件分布由下式给出：
$$
\begin{align}
p(y=i|x;\theta)&=\phi_i\\
&=\frac{e^{\eta_i}}{\sum^{k}{j=1}e^{\eta_j}}\\
&=\frac{e^{\theta_i^Tx}}{\sum^{k}{j=1}e^{\theta_j^Tx}}\tag{8}\\
\end{align}
$$
该模型应用于分类问题，其中$y\in\{1,...,k\}$，这成为softmax回归。这是推广了的logistic回归。
我们的假设输出：
$$
\begin{array}\\
h_{\theta}(x)&=E[T(y)|x;\theta]\\
&=E\begin{bmatrix}\begin{matrix}
1\{y=1\}\\ 1\{y=2\}\\ \vdots\\ 1\{y=k-1\}\\  
\end{matrix}|x;\theta\end{bmatrix}\\
&=\begin{bmatrix}
\phi_1\\
\phi_2\\
\vdots\\
\phi_{k-1}\\
\end{bmatrix}\\
&=\begin{bmatrix}
\frac{exp(\theta^T_1x)}{\sum^k_{j=1}exp(\theta^T_jx)}\\\frac{exp(\theta^T_2x)}{\sum^k_{j=1}exp(\theta^T_jx)}\\
\vdots\\
\frac{exp(\theta^T_{k-1}x)}{\sum^k_{j=1}exp(\theta^T_jx)}\\
\end{bmatrix}\\
\end{array}
$$
换句话说，我们的假设会输出对于每个i=1,...,k$p(y=i|x;\theta)$的估计概率(即使说$h_{\theta}(x)$仅仅定义了k-1维，很清楚的是$p(y=k|x;\theta)能由1-\sum^{k-1}_{i=1}\phi_i$得到)。
最后，让我们讨论一下参数拟合。类似于我们原先的最小二乘和logistic回归的推导，如果我们有m个样本的训练集$\{(x^{(i),y^{(i)}};i=1,...,m\}$,希望学习该模型的参数$\theta_i$，我们可以开始写对数似然函数：
$$
\begin{align}
\cal l(\theta)&=\sum^{m}_{i=1}logp(y^{(i)}|x^{(i)};\theta)\\
&=\sum^{m}_{i=1}log\prod^{k}_{l=1}(\frac{e^{^T_lx^{(i)}}}{\sum^{k}_{j=1}x^{(i)}})^{1\{y^{(i)}=l\}}
\end{align}
$$
为了获得上面的第二行，我们用方程(8)的$p(y|x;\theta)$定义。我们现在通过以$\theta$的形式求$l(\theta)$的最大值来获得参数的最大似然估计，可以用诸如梯度上升或者牛顿法之类的方法。

