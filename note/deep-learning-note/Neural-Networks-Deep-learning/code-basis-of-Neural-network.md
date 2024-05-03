# code-basis-of-Neural-network

## 1. binary classification —— 二分类

<div align=center>
<img src="img/屏幕截图%202024-04-20%20193223.png" width=45%>
<img src="img/屏幕截图%202024-04-20%20193557.png" width=45%>
</div>

* 对于图像处理问题，每个图片由很多像素点组成。所以图片的特征向量是3通道的RGB矩阵，我们将其展平作为一个特征输入向量x xx。
* 约定一些符号：
  * $x$：表示一个$n_x$维数据，为输入数据，维度为$(n_x,1)$；
  * $y$：表示输出结果，取值为$(0,1)$；
  * $(x^i,y^i)$：表示第$i$组数据，可能是训练数据，也可能是测试数据，此处默认为训练数据；
  * $X=\begin{bmatrix}x^{(1)} x^{(2)} \dots x^{(m)} \\ \end{bmatrix}$:表示所有的训练数据集的输入值，放在一个$n_x×m$的矩阵中，其中$m$表示样本数目;
  * $Y=\begin{bmatrix}y^{(1)} y^{(2)} \dots y^{(m)} \\ \end{bmatrix}$:对应表示所有训练数据集的输出值，维度为$1×m$。

## 2. Logistic Regression —— 逻辑回归

$\quad$对于二元分类问题来讲，给定一个输入特征向量$X$，它可能对应一张图片，你想识别图片内容是否为一只猫，你需要算法能够输出预测$\hat{y}$，$\hat{y}$表示$y$等于$1$的一种可能性或者是概率。

$\quad$前提条件给定了输入特征$X$，$X$是一个$n_x$维的向量（相当于有$n_x$个特征的特征向量）。我们用$w$来表示逻辑回归的参数，这也是一个$n_x$维向量（因为$w$是特征权重，维度与特征向量相同），参数里面还有$b$，这是一个表示偏差的实数。

$\quad$所以给出输入$x$以及参数$w$和$b$之后，我们怎样产生输出预测值$\hat{y}={{w}^{T}}x+b$这样么? 答案是否定的。

<div align=center>
<img src="img/屏幕截图%202024-04-20%20195833.png" width=80%>
</div>

* 我们需要借助sigmoid函数(其中$z={{w}^{T}}x+b$)，令

$$
\hat{y}=\sigma \left( z \right)=\frac{1}{1+{{e}^{-z}}}
$$

* 我认为此定义原因如下:
  * 为了便于解释$\hat{y}$
  * 为了便于接下来代价函数的提出和理解。

## 3. Logistic Regression cost function —— 逻辑回归的代价函数

$\quad$为了衡量一个算法在模型上的表现并以此作为优化的依据，我们需要一个代价函数。在逻辑回归中，我们需要通过训练代价函数来得到优化后的参数$w$和参数$b$。

$Loss function:L\left( \hat{y},y \right)=-(y\log(\hat{y})+(1-y)\log (1-\hat{y}))$，交叉熵损失函数，常用于二分类问题。

$\quad$所有的样本的损失函数的平均值:

$$
J(w, b)=\frac{1}{m} \sum_{i=1}^{m} L\left(\hat{y}^{(i)}, y^{(i)}\right)=-\frac{1}{m} \sum_
{i=1}^{m}\left[y^{(i)} \log \left(\hat{y}^{(i)}\right)+\left(1-y^{(i)}\right) \log \left(1-\hat{y}^{(i)}\right)\right]
$$

目标就是找到合适的参数，使得代价函数最小化

<div align=center>
<img src="img/屏幕截图%202024-04-20%20200629.png" width=80%>
</div>

## 4. Gradient Descent —— 梯度下降法

$\quad$如何寻找合适的$w,b$使得代价函数最小呢?
$\quad$迭代的过程中，不断的在各参数的偏导数方向上更新参数值，$\alpha$是学习率

$$
w\coloneqq w-\alpha \frac{\partial J(w,b)}{\partial w} \quad b\coloneqq b-\alpha \frac{\partial J(w,b)}{\partial b}
$$

<div align=center>
<img src="img/屏幕截图%202024-04-20%20201519.png" width=46%>
<img src="img/屏幕截图%202024-04-20%20201553.png" width=44%>
</div>

## 5. 导数

$\quad$导数定义：函数在某一点的斜率，在不同的点，斜率可能是不同的。

## 6. Computer Graph —— 计算图

* 偏导的链式法则

$$
J(w,b)=\frac{1}{m}\sum_{i=1}^{m}L(a^{(i)},y^{(i)}) \quad a^{(i)}=\hat{y}^{(i)}=\sigma(z^{(i)}) \quad z^{(i)}=w^{T}x+b
$$

$$
\frac{\partial L(a^{(i)},y^{(i)})}{\partial z^{(i)}}=\frac{\partial L(a^{(i)},y^{(i)})}{\partial a^{(i)}}\frac{\partial
a^{(i)}}{\partial z^{(i)}}=\left(-\frac{y^{(i)}}{a^{(i)}}+\frac{1-y^{(i)}}{1-a^{(i)}}\right)\left(\frac{e^{-z}}{(
1+{{e}^{-z}})^2}\right)=\left(-\frac{y^{(i)}}{a^{(i)}}+\frac{1-y^{(i)}}{1-a^{(i)}}\right)a^{(i)}(1-a^{(i)})=a^{(i)}-y^{(
i)}
$$

$$
\frac{\partial J(w,b)}{\partial w_{j}}=\frac{1}{m}\sum_{i=1}^{m} \frac{\partial}{\partial w^{(i)}_{j}}L(a^{(i)},y^{(i)})
=\frac{1}{m}\sum_{i=1}^{m}x^{(i)}_{j}(a^{(i)}-y^{(i)})
$$

$$
\frac{\partial J(w,b)}{\partial b}=\frac{1}{m}\sum_{i=1}^{m} \frac{\partial}{\partial b}L(a^{(i)},y^{(i)})
=\frac{1}{m}\sum_{i=1}^{m}(a^{(i)}-y^{(i)})
$$

## 7. Vectorization —— 向量化

* 使用向量化可以充分使用并行化运算，可以大大增强运行速度

<div align=center>
<img src="img/屏幕截图%202024-04-20%20210150.png" width=80%>
</div>

```shell
# 测试结果
[1 2 3 4]
249880.65187010984
Vectorized version:0.0ms
249880.6518701139
For loop:584.7179889678955ms
```

## 8. 向量化的更多例子

* 使用向量的时候

<div align=center>
<img src="img/屏幕截图%202024-04-20%20210617.png" width=46%>
<img src="img/屏幕截图%202024-04-20%20210658.png" width=44%>
</div>

<div align=center>
<img src="img/屏幕截图%202024-04-20%20213929.png" width=80%>
</div>

* 使用矩阵

<div align=center>
<img src="img/屏幕截图%202024-04-20%20215024.png" width=45%>
<img src="img/屏幕截图%202024-04-20%20215047.png" width=45%>
</div>

<div align=center>
<img src="img/屏幕截图%202024-04-20%20215143.png" width=80%>
</div>

## 9. Broadcasting in Python —— 广播机制

* 测试文件在test.ipynb中！！！！！
* 广播机制：广播会在缺失和(或)长度为1的维度上进行，广播时会将这一部分进行copy

<div align=center>
<img src="img/屏幕截图%202024-04-20%20221702.png" width=45%>
<img src="img/屏幕截图%202024-04-20%20221721.png" width=45%>
</div>

## 10. 关于python中的numpy库的一些说明

* 具体见numpy_use.ipynb文件

## 11. logistic 损失函数的解释

<div align=center>
<img src="img/屏幕截图%202024-04-20%20224738.png" width=45%>
<img src="img/屏幕截图%202024-04-20%20224802.png" width=45%>
</div>


