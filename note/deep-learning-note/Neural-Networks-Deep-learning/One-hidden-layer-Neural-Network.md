# One-hidden-layer-Neural-Network

## 1. Neural-Network Overview

<div align=center>
<img src="img/屏幕截图%202024-04-20%20225544.png" width=80%>
</div>

## 2. Neural-Network Representation

<div align=center>
<img src="img/屏幕截图%202024-04-21%20220605.png" width=80%>
</div>

## 3. Computing a Neural-Network's Output

* 每一个神经元经历了两步，先进行计算求$z$，再求$a$

<div align=center>
<img src="img/屏幕截图%202024-04-21%20220932.png" width=80%>
</div>

* 事实上，第一层输入的数据 $\vec{x}$ 可以看作 $\vec{a}^{[0]}$

<div align=center>
<img src="img/屏幕截图%202024-04-21%20221118.png" width=39.7%>
<img src="img/屏幕截图%202024-04-21%20221150.png" width=40.2%>
</div>

## 4. Vectorizing across multiple examples

* 注意：这里的 $z^{[j](i)}$ 表示的是第$i$个测试范例在第$j$层layer的$z$值

<div align=center>
<img src="img/屏幕截图%202024-04-21%20222129.png" width=45%>
<img src="img/屏幕截图%202024-04-21%20223059.png" width=45%>
</div>
<div align=center>
<img src="img/屏幕截图%202024-04-21%20223320.png" width=45%>
<img src="img/屏幕截图%202024-04-21%20223339.png" width=45%>
</div>

## 5. activation functions

* 具体图像在activation-functions.ipynb文件中

> * sigmoid function:
>   $$
>   a=\frac{1}{1+e^{-z}}
>   $$
> * tanh(z):
>   $$
>   \tanh(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}
>   $$

* tanh激活函数是sigmoid的平移伸缩结果，其效果在所有场合都优于sigmoid，tanh几乎适合所有场合。例外是二分类问题的输出层，想让结果介于0,1之间，所以使用
  sigmoid激活函数。

> * tanh、sigmoid两者的缺点：在特别大或者特别小$z$的情况下，导数的梯度或者函数的斜率会变得特别小，最后就会接近于0，导致降低梯度下降的速度。
> * ReLU function:
>   $$
>   a=\max(0,z)
>   $$
> * Leaky ReLU function:(实际中Leaky ReLu使用的并不多)(0.01只是演示结果，我们可以选取其他值)
>   $$
>   a=\max(0.01z,z)
>   $$
>
> <div align=center>
> <img src="img/屏幕截图%202024-04-21%20225054.png" width=45%>
> <img src="img/屏幕截图%202024-04-21%20225113.png" width=45%>
> </div>

### 激活函数的选择经验：

* 如果输出是0、1值（二分类问题），输出层选择sigmoid函数，其它所有单元都选择Relu函数；
* sigmoid函数需要进行浮点四则运算，在实践中，使用ReLu激活函数学习的更快；
* 隐藏层通常会使用Relu激活函数。有时，也会使用tanh激活函数，但Relu的一个缺点是：当是负值的时候，导数等于0；
* 另一个版本的Relu被称为Leaky Relu，当是负值时，这个函数的值不等于0，而是轻微的倾斜，这个函数通常比Relu激活函数效果要好，尽管在实际中Leaky
  ReLu使用的并不多。

## 6. 为什么需要非线性激活函数

线性隐藏层一点用也没有，因为线性函数的组合本身就是线性函数，所以除非你引入非线性，否则你无法计算出更有趣的函数，即使网络层数再多也不行。

* 不能在隐藏层用线性激活函数，可以用ReLU、tanh、leaky ReLU或者其他的非线性激活函数；
* 唯一可以用线性激活函数的通常就是输出层；在隐藏层使用线性激活函数非常少见。

## 7. Derivatives-of-activation-function —— 激活函数的导数

* 在Derivatives-of-activation-function.ipynb文件中有计算

$$
sigmoid-function:g'(z)=\frac{e^{-x}}{(1+e^{-x})^2}
$$

$$
tanh-function:g'(z)=1-\tanh^{2}(z)
$$

## 8.神经网络中的梯度下降

* shape=(n,m)，m在进行变换的过程中是不改变的，这个是表示的是示例的个数
* np.sum()的axis=1表示的是在这个维度上将全部相加，(n,m)的shape就是将m个数全部相加，表示的是将所有的示例遍历相加。对每一个特征进行说明
* 注意：keepdims=True的含义是保证输出的是$\left[n^{[i]},1\right]$格式的向量

<div align=center>
<img src="img/屏幕截图%202024-04-22%20225158.png" width=44%>
<img src="img/屏幕截图%202024-04-22%20225356.png" width=47%>
</div>

## 9. 反向传播

$$
z^{[1]}=W^{[1]}x+b^{[1]} \quad and \quad a^{[1]}=g(z^{[1]}) \quad and \quad z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}
$$

$$
\frac{\partial L(a^{[2]},y)}{\partial z^{[1]}_i}=\frac{\partial L(a^{[2]},y)}{\partial z^{[2]}}\frac{\partial z^{[2]
}}{\partial z^{[1]}_i}
$$

$$
\frac{\partial z^{[2]}}{\partial z^{[1]}_i}=w_i\cdot g'(z^{[1]})
$$

* 综合结果为:

$$
dz^{[1]}={W^{[2]}}^Tdz^{[1]}g'(z^{[1]})
$$

## 10. 随机初始化

$\quad$对于一个神经网络，如果你把权重或者参数都初始化为0，那么梯度下降将不会起作用，并且会存在神经单元的对称性问题，添加再多的神经单元也没有更好的效果。

<div align=center>
<img src="img/屏幕截图%202024-04-23%20232553.png" width=80%>
</div>

* 常数为什么是0.01，而不是100或者1000？因为如果w初始化很大的话，那么z就会很大，所以sigmoid/tanh激活函数值就会趋向平坦的地方，而sigmoid/tanh激活函数在很平坦的地方，学习非常慢。
* 当你训练一个非常非常深的神经网络，你可能要试试0.01以外的常数。
