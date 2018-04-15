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
### 第一部分
####线性回归
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
#####1 LMS算法
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

#####2 正规方程
梯度下降给出了一种最小化$J$ 的方法。让我们讨论第二种方法，这次明确地做最小化而不用迭代算法。在这个方法中，我们明确的求对$\theta_j$的导数来并令之为零来I求$J$的最小值。为了不用写大量的代数和整页整页的矩阵的导数，让我们引入一些记号来做矩阵的计算。
[^footnote2]: 正如我们之前所描述的那样，通常随机梯度下降用固定的学习率$\alpha$，但缓慢下降学习率$\alpha$到0，可以让参数收敛到全局最小而不是仅仅在最小值附近震荡。
######2.1矩阵导数
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

######2.2  再议最小二乘
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
其中$\epsilon^{(i)}$是误差项，可能原因是模型以外的因素（比如某特征与房价高度相关，但是没有被纳入回归）或者随机噪声。进一步假设$\epsilon^{(i)}$是与有均值零及一定的方差$\sigma$的高斯分布（或正态分布）独立同分布的。我们把这个假设写成"$\epsilon^{(i)}～\cal N(0,\sigma^2)$"。换句话说,$\epsilon^{(i)}$的概率密度函数满足：
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
决定权重的一种合适方法是[^footnote 4]：
$$
w^{(i)}=exp(-\frac{(x^{(i)-x})^w}{2\tau^2})
$$
注意到权重取决于特定的要求值的x。更进一步说，如果$|x^{(i)}-x|$很小那么$w^{(i)}$接近1；如果$|x^{(i)}-x|$很大那么$w^{(i)}$很小。因此，$\theta$选择给予训练样本(的误差)上靠近所求点x更大的权重。也注意到当权重公式的形式上很类似高斯分布的概率密度分布，$w^{(i)}$并没有和高斯分布有什么直接关系特别是$w^{(i)}$不是随机变量，正太分布或者什么类似的东西。参数$\tau$控制着训练样本的权重对于待求点x随着距离$x^{(i)}$下降的快慢。$\tau$叫做**带宽**参数，也会在作业里遇到。
局部加权线性回归是我们看到的第一个**非参数**算法。之前见过的（不加权的）线性回归也称作**含参数**的学习算法，因为有固定的，有限个数的参数($\theta$),用来拟合数据。一旦完成了拟合，保留下参数$\theta$我们就不在需要原始数据来预测未来的情况。相对的，在用局部加权线性回归的时候我们需要保留整个训练集来做未来的预测。“非参数”指的是需要保留一堆东西来代表假设$h$随着训练集的尺度线性增长。
[^footnote]:若是向量值，一般化的表达是$w^{(i)}=exp(-(x^{(i)}-x)^T\frac{x^{(i)}-x}{2\tau^2})$，或者选择合适$\tau$或者$\Sigma$下$w^{(i)}=exp(-(x^{(i)}-x)^T\sum^{-1}\frac{x^{(i)}-x}{2})$。

###第二 部分 
