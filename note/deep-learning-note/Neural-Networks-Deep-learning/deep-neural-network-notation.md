# deep-neural-network-notation

## 1. Deep L-layer Neural network

$$
L:\#(layers)
$$

$$
n^{[l]}:\#(units \ in \ layer \ l)
$$

$$
a^{[l]}=g^{[l]}(z^{[l]}):activations \ in \ layer \ l
$$

<div align=center>
<img src="img/屏幕截图%202024-04-24%20163752.png" width=80%>
</div>

## 2. Forward and backward propagation

* $m$:表示的是示例的数量
* 这里的$+ \bm{b^{[l]}}$:对其讨论的时候使用了广播机制！！！！

$$
\bm{X}=\begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & \dots & x_1^{(m)} \\
x_2^{(1)} & x_2^{(2)} & \dots & x_2^{(m)} \\
\vdots & \vdots & \vdots & \vdots \\
x_{n^{[0]}}^{(1)} & x_{n^{[0]}}^{(2)} & \dots & x_{n^{[0]}}^{(m)} \\
\end{bmatrix}

\quad

\bm{A^{[l]}}=\begin{bmatrix}
a_1^{(1)} & a_1^{(2)} & \dots & a_1^{(m)} \\
a_2^{(1)} & a_2^{(2)} & \dots & a_2^{(m)} \\
\vdots & \vdots & \vdots & \vdots \\
a_{n^{[l]}}^{(1)} & a_{n^{[l]}}^{(2)} & \dots & a_{n^{[l]}}^{(m)} \\
\end{bmatrix}

\quad

\bm{W^{[l]}}=\begin{bmatrix}
w_1^{(1)} & w_2^{(1)} & \dots & w_{n^{[l-1]}}^{(1)} \\
w_1^{(2)} & w_2^{(2)} & \dots & w_{n^{[l-1]}}^{(2)} \\
\vdots & \vdots & \vdots & \vdots \\
w_1^{(n^{[l]})} & w_2^{(n^{[l]})} & \dots & w_{n^{[l-1]}}^{(n^{[l]})} \\
\end{bmatrix}

\quad

\bm{b^{[l]}}=\begin{bmatrix}
b^{(1)} \\
b^{(2)} \\
\vdots \\
b^{(n^{[l]})} \\
\end{bmatrix}
$$

$$
\bm{Z^{[l]}}=\begin{bmatrix}
z_1^{(1)} & z_1^{(2)} & \dots & z_1^{(m)} \\
z_2^{(1)} & z_2^{(2)} & \dots & z_2^{(m)} \\
\vdots & \vdots & \vdots & \vdots \\
z_{n^{[l]}}^{(1)} & z_{n^{[l]}}^{(2)} & \dots & z_{n^{[l]}}^{(m)} \\
\end{bmatrix}
=\bm{W^{[l]}} \cdot \bm{A^{[l-1]}}+\bm{b^{[l]}}

\quad

\bm{A^{[l]}}=g(\bm{Z^{[l]}})
$$

* 注意：为了方便表示，我使用$d\bm{b^{[l]}} \ d\bm{Z^{[l]}} \ d\bm{A^{[l]}} \ d\bm{W^{[l]}}$来表示其中每一个元素的$\partial J/\partial?$所组成的矩阵
  * 注意：$*$ 表示的是将两个矩阵的对于元素进行相乘，而不是矩阵乘法
  * $d\bm{W^{[l]}}:(n^{[l]},n^{[l-1]}) \quad d\bm{Z^{[l]}}:(n^{[l]},m) \quad d\bm{A^{[l-1]}}:(n^{[l-1]},m)\Longrightarrow d\bm{A^{[l-1]}}^T:(m,n^{[l-1]})$
  * $d\bm{Z^{[l]}}\cdot \vec{e}$:表示的是将$d\bm{Z^{[l]}}$向量的每一行的值全部相加得到$d\bm{b^{[l]}}$的值，代码实现的方式是:np.sum(dZl,axis=1,keepdims=True)

$$
for \ l \in\{0,1,\dots,L\}:
\begin{cases}
\ d\bm{Z^{[l]}} &=d\bm{A^{[l]}}*g^{[l]'}(Z^{[l]}) \\
\ d\bm{W^{[l]}} &=d\bm{Z^{[l]}}\cdot \bm{A^{[l-1]}}^T \\
\ d\bm{b^{[l]}} &=d\bm{Z^{[l]}}\cdot \vec{e} \\
\ d\bm{A^{[l-1]}} &=\bm{W^{[l-1]}}^T \cdot \bm{Z^{[l]}} \\
\end{cases}
$$

* 注意：最后一次输出的结果$d\bm{A^{[L]}}$表示的是 $\hat{y}$ 的值，因此是一个常数

## 3. 核对矩阵的维数

* $\bm{b^{[l]}}:(n^{[l]},1) \quad \bm{Z^{[l]}}:(n^{[l]},m) \quad \bm{A^{[l]}}:(n^{[l]},m) \quad \bm{W^{[l]}}:(n^{[l]},n^{[l-1]})$ —— 向量/矩阵的维数
* 注意：$\bm{b^{[l]}}$也可以写成$(n^{[l]},m)$的形式

## 4. 为什么使用深层表示？

$\quad$在图像处理领域，深层神经网络随着层数由浅到深，神经网络提取的特征也是从边缘到局部特征到整体，由简单到复杂。如果隐藏层足够多，那么能够提取的特征就越丰富、越复杂，模型的准确率就会越高。

$\quad$在语音识别领域，浅层的神经元能够检测一些简单的音调，然后较深的神经元能够检测出基本的音素，更深的神经元就能够检测出单词信息。如果网络够深，还能对短语、句子进行检测。

$\quad$除了从提取特征复杂度的角度来说明深层网络的优势之外，深层网络还有另外一个优点，就是能够减少神经元个数，从而减少计算量。

## 5. 搭建神经网络块

<div align=center>
<img src="img/屏幕截图%202024-04-24%20203849.png">
</div>

## 6. 参数 vs. 超参数

* 参数：模型可以根据数据可以自动学习出的变量，应该就是参数。
  * 比如，深度学习的权重w，偏差b等。
* 超参数：就是用来确定模型的一些参数，超参数不同，模型是不同的，超参数一般就是根据经验确定的变量。
  * 在深度学习中，超参数有：学习速率，梯度下降的迭代次数，隐藏层数量，隐藏层单元数量以及激活函数选择等等。

* 进行多种组合，各种尝试，目的是选择效果最好的参数组合，在第二门课会介绍具体方式。

## 8. 深度学习和大脑的关联性

<div align=center>
<img src="img/屏幕截图%202024-04-24%20204524.png">
</div>