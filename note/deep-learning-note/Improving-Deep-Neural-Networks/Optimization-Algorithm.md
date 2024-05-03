# Optimization-Algorithm

## 1. Mini-batch gradient descent —— 小批量梯度下降

在巨大的数据集上进行训练，速度非常慢，如何提高效率？

前面我们学过**向量化**可以较快的处理整个训练集的数据，但是如果样本非常的大，在进行下一次梯度下降之前，你必须完成前一次的梯度下降。如果我们能先处理一部分数据，算法速度会更快。

把训练集分割为小一点的子集(称之 mini-batch)训练，即 Mini-batch 梯度下降。

> 对比：
>
> * batch 梯度下降法：指的就是前面讲的梯度下降法，每次需要同时处理整个训练集
> * mini-batch梯度下降：每次处理的是单个的 mini-batch 训练子集

<div>
<img src="img/屏幕截图%202024-04-25%20163955.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20164353.png" width=45%>
</div>

## 2. 理解 mini-batch 梯度下降

mini-batch 梯度下降，每次迭代后 cost-function 不一定是下降的，因为每次迭代都在训练不同的样本子集，但总体趋势应该是下降的。

**mini-batch 的 size 大小**：

* 大小 = m，就是batch梯度下降法
* 大小 = 1，就是随机梯度下降

<div>
<img src="img/屏幕截图%202024-04-25%20165230.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20170520.png" width=45%>
</div>

## 3. 指数加权平均数

> $$
> v_t=\beta v_{t-1}+(1-\beta)\theta_t
> $$
>
> 其中 $v_t$ 代表到第$t$天的象征平均温度值，$\theta_t$ 代表第$t$天的温度值，$\beta$ 代表可调节的超参数值
>
> 也就是说，$v_t$ 代表的是指数加权平均数之后所得的一种平均值，由于 $v_t$ 的值越接近 $t$ 的温度所占的权重更高，所以有意义
>
> 由于 $\beta ^{t-m}(1-\beta) \theta_m$ 是参数，我们认为在 $t-m\approx (1-\beta)^{-1}$ 之后将没有多么大的意义，所以说 $v_t$ 可以看作之前 $(1-\beta)^{-1}$ 天的一个权重平均

* $\beta$ 值越大，则指数加权平均的天数越多，平均后的趋势线就越平缓，但是同时也会向右平移

<div>
<img src="img/屏幕截图%202024-04-25%20171648.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20174244.png" width=45%>
</div>

$$
\lim_{\varepsilon \to 1^-}(\beta)^{1/(1-\beta)}=\lim_{\varepsilon \to 0^+}(1-\varepsilon)^{1/\varepsilon}\approx \frac{1}{e}
$$

## 4. 理解指数加权平均数

$$
V_t=\sum_{m=1}^{t}\beta ^{t-m}(1-\beta) \theta_m+\beta^tV_0
$$

```python
# 代码实现
v_theta = 0
for (theta_t in theta)
    v_theta = beta ∗ v_theta + (1 − beta) * theta_t
```

## 5. 指数加权平均的偏差修正

当 **β=0.98** 时，指数加权平均结果如绿色曲线。但实际上真实曲线如紫色曲线
紫色曲线与绿色曲线的区别是，紫色曲线开始的时候相对较低一些。因为开始时设置 $V_0=0$ ，所以初始值会相对小一些，直到后面受前面的影响渐渐变小，趋于正常

修正这种问题的方法是进行**偏移校正(bias correction)**，即在每次计算完 $V_t$ 后，对 $V_t$ 进行下式处理:

$$
\frac{V_t}{1-\beta^t}
$$

刚开始的时候，**t**比较小，$1-\beta^t<1$，$V_t$ 被修正得更大一些，效果是把紫色曲线开始部分向上提升一些，与绿色曲线接近重合。随着**t**增大，$1-\beta^t\approx 1$，$V_t$ 基本不变，紫色曲线与绿色曲线依然重合。实现了简单的偏移校正，得到希望的绿色曲线

机器学习中，偏移校正并不是必须的。因为，在迭代一次次数后（**t**较大），$V_t$ 受初始值影响微乎其微，紫色曲线与绿色曲线基本重合。一般可以忽略初始迭代过程，等到一定迭代之后再取值就不需要进行偏移校正

<div align=center>
<img src="img/屏幕截图%202024-04-25%20180750.png" width=80%>
</div>

## 6. 动量梯度下降法（Gradient descent with Momentum ）

动量梯度下降算法速度比传统的梯度下降算法快很多。做法是在每次训练时，对梯度进行指数加权平均处理，然后用得到的梯度值更新权重 $W$ 和常数项 $b$

<div align=center>
<img src="img/屏幕截图%202024-04-25%20223538.png" width=46%>
<img src="img/屏幕截图%202024-04-25%20223636.png" width=45%>
</div>

原始的梯度下降算法如上图蓝色折线所示。在梯度下降过程中，梯度下降的振荡较大，尤其对于 $W$ 、$b$ 之间数值范围差别较大的情况。此时每一点处的梯度只与当前方向有关，产生类似折线的效果，前进缓慢。而如果对梯度进行指数加权平均，使当前梯度不仅与当前方向有关，还与之前的方向有关，在纵轴方向，平均过程中，正负数相互抵消，所以平均值接近于零。但在横轴方向，所有的微分都指向横轴方向，因此横轴方向的平均值仍然较大，用算法几次迭代后，最终纵轴方向的摆动变小了，横轴方向运动更快，因此算法走了一条更加直接的路径，在抵达最小值的路上减少了摆动，这样处理让梯度前进方向更加平滑，减少振荡，能够更快地到达最小值处

权重 $W$ 和常数项 $b$ 的指数加权平均表达式如下:

$$
\large\begin{cases}
\ V_{dW}&=β\cdot V_{dW}+(1−β)\cdot dW \\
\ V_{db}&=β\cdot V_{db}+(1−β)\cdot db \\
\end{cases}
$$

动量的角度来看，以权重 $W$ 为例，$V_{dW}$ 可以理解成速度 $V$，$dW$ 可以看成是加速度 $a$ 。指数加权平均实际上是计算当前的速度，当前速度由之前的速度和现在的加速度共同影响。而 $\beta<1$ 又能限制速度过大。即当前的速度是渐变的，而不是瞬变的，是动量的过程。保证了梯度下降的平稳性和准确性，减少振荡，较快地达到最小值

动量梯度下降算法的过程如下：

$On \ iteration \ t:$

$$
Compute \ dW,db \ on \ the \ current \ mini−batch
$$

$$
\large\begin{cases}
V_{dW}&=β\cdot V_{dW}+(1−β)\cdot dW \\
V_{db}&=β\cdot V_{db}+(1−β)\cdot db \\
W&=W−\alpha V_{dW} \\
b&=b−\alpha V_{dW}\\
\end{cases}
$$

初始时，令 $V_{dW}=0$，$V_{db}=0$。一般设置$\beta=0.9$，即指数加权平均前10次的数据，实际应用效果较好。

偏移校正可以不使用。因为经过10次迭代后，随着滑动平均的过程，偏移情况会逐渐消失

动量梯度下降法，并不是对所有情况都有效，它对碗状的优化效果较好介绍：与SDG结合使用的一种常用方法叫做Momentum。Momentum不仅会使用当前梯度，还会积累之前的梯度以确定走向。

## 7. RMSprop(root mean square prop)

RMSprop是另外一种优化梯度下降速度的算法。每次迭代训练过程中，其权重 $W$ 和常数项 $b$ 的更新表达式为:

$$
\large\begin{cases}
S_{dW}&=β\cdot S_{dW}+(1−β)\cdot dW^2 \\
S_{db}&=β\cdot S_{db}+(1−β)\cdot db^2 \\
\end{cases}
\implies
W=W−\alpha\frac{dW}{\sqrt{S_{dW}}} \quad
b=b−\alpha\frac{db}{\sqrt{S_{db}}}
$$

* 震荡越大，那么 $S$ 也就越大，那么就会让梯度下降不再震荡
* 为了避免RMSprop算法中分母为零，通常可以在分母增加一个极小的常数 $\varepsilon$ ($\varepsilon=10^{-8}$，或者其它较小值)

$$
\large{W\coloneqq W−\alpha\frac{dW}{\sqrt{S_{dW}}+\varepsilon} \quad
b\coloneqq b−\alpha\frac{db}{\sqrt{S_{db}}+\varepsilon}}
$$

<div align=center>
<img src="img/屏幕截图%202024-04-25%20230426.png" width=80%>
</div>

**RMSprop跟Momentum有很相似的一点**，可以**消除**梯度下降和mini-batch梯度下降中的**摆动**，并允许你使用一个更大的**学习率**，从而加快你的算法学习速度。

## 8. Adam 优化算法(Adam optimization algorithm)

Adam（Adaptive Moment Estimation）算法结合了动量梯度下降算法和RMSprop算法。其算法流程为:

$V_{dW}=0,S_{dW}=0,V_{db}=0,S_{db}=0$

$On \ iteration \ t:$

$$
compute \ dW,db
$$

$$
V_{dW}=β_1\cdot V_{dW}+(1−β_1)\cdot dW\quad V_{db}=β_1\cdot V_{db}+(1−β_1)\cdot db
$$

$$
S_{dW}=β_2\cdot S_{dW}+(1−β_2)\cdot dW^2 \quad S_{db}=β_2\cdot S_{db}+(1−β_2)\cdot db^2
$$

$$
V_{dW}^{corrected}=\frac{V_{dW}}{1-\beta_1^t} \quad
V_{db}^{corrected}=\frac{V_{db}}{1-\beta_1^t} \quad
S_{dW}^{corrected}=\frac{S_{dW}}{1-\beta_2^t} \quad
S_{db}^{corrected}=\frac{S_{db}}{1-\beta_2^t}
$$

$$
\large{W\coloneqq W−\alpha\frac{dV_{dW}^{corrected}}{\sqrt{S_{dW}^{corrected}}+\varepsilon} \quad
b\coloneqq b−\alpha\frac{dV_{db}^{corrected}}{\sqrt{S_{db}^{corrected}}+\varepsilon}}
$$

Adam算法包含了几个超参数，分别是: $\alpha,\beta_1,\beta_2,\varepsilon$
$\beta_1$ 通常设置为0.9，$\beta_2$ 通常设置为0.999，$\varepsilon$ 通常设置为$10^{-8}$。一般只需要对 $\beta_1$ 和 $\beta_2$ 进行调试

Adam算法结合了动量梯度下降和RMSprop各自的优点，使得神经网络训练速度大大提高

## 9. 学习率衰减(Learning rate decay)

减小学习因子 $\alpha$ 也能有效提高神经网络训练速度，这种方法被称为learning rate decay, Learning rate decay就是随着迭代次数增加，学习因子 $\alpha$ 逐渐减小

随着训练次数增加，$\alpha$ 逐渐减小，步进长度减小，使得能够在最优值处较小范围内微弱振荡，不断逼近最优值。相比较恒定的 $\alpha$ 来说，learning rate decay更接近最优值！！！！！！

<div align=center>
<img src="img/屏幕截图%202024-04-26%20013459.png" width=45%>
<img src="img/屏幕截图%202024-04-26%20013510.png" width=45%>
</div>

Learning rate decay中对 $\alpha$ 的公式：

$$
\alpha=\frac{1}{1+decay\_rate\cdot epoch}\cdot \alpha_0
$$

$deacy\_rate$ 是参数(可调)，$epoch$ 是迭代次数。随着 $epoch$ 增加，$\alpha$ 会不断变小

其它计算公式：

$$
\alpha=0.95^{epoch}\cdot \alpha_0
$$

$$
\alpha=\frac{k}{\sqrt{epoch}}\cdot \alpha_0 \quad or \quad \alpha=\frac{k}{\sqrt{t}}\cdot \alpha_0
$$

$k$ 为可调参数，$t$ 为mini-bach number

还可以设置 $\alpha$ 为关于 $t$ 的离散值，随着 $t$ 增加，$\alpha$ 呈阶梯式减小。也可以根据训练情况灵活调整当前的 $\alpha$ 值，但会比较耗时间

## 10. 局部最优的问题(The problem of local optima)

以前对局部最优解的理解是形如碗状的凹槽，如下图左边所示。但是在神经网络中，local optima的概念发生了变化。大部分梯度为零的“最优点”并不是这些凹槽处，而是形如右边所示的马鞍状，称为saddle point（鞍点）。即梯度为零并不能保证都是convex(极小值)，也有可能是concave(极大值)。特别是在神经网络中参数很多的情况下，所有参数梯度为零的点很可能都是右边所示的马鞍状的saddle point，而不是左边那样的local optimum

<div align=center>
<img src="img/屏幕截图%202024-04-26%20014236.png" width=80%>
</div>

类似马鞍状的plateaus(平稳端)会降低神经网络学习速度。Plateaus是梯度接近于零的平缓区域，在plateaus上梯度很小，前进缓慢，到达saddle point需要很长时间。到达saddle point后，由于随机扰动，梯度一般能够沿着图中红色箭头，离开saddle point，继续前进，只是在plateaus上花费了太多时间

<div align=center>
<img src="img/屏幕截图%202024-04-26%20014248.png" width=80%>
</div>

local optima的两点总结：

* 只要选择合理的强大的神经网络，一般不太可能陷入local optima
* Plateaus可能会使梯度下降变慢，降低学习速度

动量梯度下降，RMSprop，Adam算法都能有效解决plateaus下降过慢的问题，大大提高神经网络的学习速度
