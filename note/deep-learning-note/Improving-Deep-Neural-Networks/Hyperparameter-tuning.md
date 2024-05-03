# 1. 调试处理（Tuning process）

深度神经网络需要调试的超参数（Hyperparameters）包括：

* $\alpha$ ：学习因子
* $\beta$ ：动量梯度下降因子
* $\beta_1,\beta_2,\varepsilon$ ：Adam算法参数
* #layers：神经网络层数
* #hidden units：各隐藏层神经元个数
* learning rate decay：学习因子下降参数
* mini-batch size：批量训练样本包含的样本个数

学习因子 $\alpha$ 是最重要的超参数，也是需要重点调试的超参数。动量梯度下降因子 $\beta$ 、各隐藏层神经元个数#hidden units和mini-batch size的重要性仅次于,然后就是神经网络层数#layers和学习因子下降参数learning rate decay。最后，Adam算法的三个参数 $ \beta_1,\beta_2,\varepsilon$ 一般常设置为$0.9$，$0.999$和$10^{-8}$

传统的机器学习中，对每个参数等距离选取任意个数的点，分别使用不同点对应的参数组合进行训练，最后根据验证集上的表现好坏来选定最佳的参数。例如有两个待调试的参数，分别在每个参数上选取5个点，这样构成了5x5=25中参数组合：

<div align=center>
<img src="img/屏幕截图%202024-04-26%20173627.png" width=80%>
</div>

> 这种做法在参数比较少的时候效果较好

深度神经网络模型中是使用随机选择。随机选择25个点，作为待调试的超参数：

随机化选择参数是为了尽可能地得到更多种参数组合。如果使用均匀采样，每个参数只有5种情况；而使用随机采样的话，每个参数有25种可能的情况，更可能得到最佳的参数组合

另外一个好处是对重要性不同的参数之间的选择效果更好。假设hyperparameter1为 $\alpha$，hyperparameter2为 $\varepsilon$，显然二者的重要性是不一样的。如果使用第一种均匀采样的方法，$\varepsilon$ 的影响很小，相当于只选择了5个 $\alpha$ 值。而如果使用第二种随机采样的方法，$\varepsilon$ 和 $\alpha$ 都有可能选择25种不同值。这大大增加了 $\alpha$ 调试的个数，更有可能选择到最优值

在实际应用中完全不知道哪个参数更加重要的情况下，随机采样的方式能有效解决这一问题，但是均匀采样做不到这点

随机采样之后，可能得到某些区域模型的表现较好。为了得到更精确的最佳参数，继续对选定的区域进行由粗到细的采样（coarse to fine sampling scheme）。就是放大表现较好的区域，对此区域做更密集的随机采样

如对下图中右下角的方形区域再做25点的随机采样，以获得最佳参数：

<div align=center>
<img src="img/屏幕截图%202024-04-26%20173716.png" width=80%>
</div>

## 2. 为超参数选择合适的范围（Using an appropriate scale to pick hyperparameters）

随机取值并不是在有效范围内的随机均匀取值，而是选择合适的标尺，用于探究这些超参数

对于超参数#layers和#hidden units，都是正整数，是可以进行均匀随机采样的，即超参数每次变化的尺度都是一致

对于某些超参数，可能需要非均匀随机采样（即非均匀刻度尺）。例如超参数αα，待调范围是[0.0001, 1]。如果使用均匀随机采样，90%的采样点分布在[0.1, 1]之间，只有 $10\%$ 分布在 $[0.0001,0.1]$ 之间。而最佳的 $\alpha$ 值可能主要分布在 $[0.0001,0.1]$ 之间，因此更应在区间 $[0.0001,0.1]$ 内细分更多刻度

通常的做法是将linear scale转换为log scale，将均匀尺度转化为非均匀尺度，然后再在log scale下进行均匀采样。这样，[0.0001, 0.001]，[0.001, 0.01]，[0.01, 0.1]，[0.1, 1]各个区间内随机采样的超参数个数基本一致，扩大了之前[0.0001, 0.1]区间内采样值个数

如果线性区间为 $[a,b]$，令 $m=\log(a),n=\log(b)$，则对应的log区间为[m,n]。对log区间的[m,n]进行随机均匀采样，得到的采样值r，最后反推到线性区间，即$10^r$ 。$10^r$ 是最终采样的超参数。代码为:

```python
m = np.log10(a)
n = np.log10(b)
r = np.random.rand()
r = m + (n-m)*r
r = np.power(10,r)
```

动量梯度因子 $\beta$ 在超参数调试也需要进行非均匀采样。一般 $\beta$ 的取值范围在 $[0.9,0.999]$ 之间，$1-\beta$ 的取值范围在 $[0.001,0.1]$。那么直接对 $1-\beta$ 在 $[0.001, 0.1]$ 区间内进行 $\log$ 变换

为什么 $\beta$ 也需要向 $\alpha$ 那样做非均匀采样：

假设 $\beta$ 从0.9000变化为0.9005，那么 $\large\frac{1}{1-\beta}$ 基本没有变化。但假设 $\beta$ 从0.9990变化为0.9995，那么 $\large\frac{1}{1-\beta}$ 前后差别1000。$\beta$ 越接近1，指数加权平均的个数越多，变化越大。所以对 $\beta$ 接近1的区间，应该采集得更密集一些

## 3. 超参数训练的实践:Pandas VS Caviar(Hyperparameters tuning in practice:Pandas vs. Caviar)

在研究一个模型的时候，可以将其他的

经过调试选择完最佳的超参数不是一成不变的，一段时间之后(例如一个月)，需要根据新的数据和实际情况，再次调试超参数，以获得实时的最佳模型

<div align=center>
<img src="img/屏幕截图%202024-04-26%20175911.png" width=45%>
<img src="img/屏幕截图%202024-04-26%20175850.png" width=45%>
</div>

在训练深度神经网络时，一种情况是有庞大的数据组，但没有许多计算资源或足够的 CPU 和 GPU 的前提下，只能对一个模型进行训练，调试不同的超参数，使得这个模型有最佳的表现。称之为Babysitting one model。另外一种情况是可以对多个模型同时进行训练，每个模型上调试不同的超参数，根据表现情况，选择最佳的模型。称之为Training many models in parallel

第一种情况只使用一个模型，类比做Panda approach；第二种情况同时训练多个模型，类比做Caviar approach。使用哪种模型是由计算资源、计算能力所决定的。一般来说，对于非常复杂或者数据量很大的模型，使用Panda approach更多一些

## 4. 归一化网络的激活函数(Normalizing activations in a network)

在神经网络中，第 $l$ 层隐藏层的输入就是第 $l-1$ 层隐藏层的输出 $A^{[l−1]}$。对 $A^{[l−1]}$ 进行标准化处理，从原理上来说可以提高 $W^{[l]}$ 和 $b^{[l]}$ 的训练速度和准确度。这种对各隐藏层的标准化处理就是Batch Normalization。一般是对 $Z^{[l−1]}$ 进行标准化处理而不是 $A^{[l−1]}$

Batch Normalization对第 $l$ 层隐藏层的输入 $Z^{[l−1]}$ 做如下标准化处理，忽略上标 $[l−1]$:

$$
\mu=\frac{1}{m}\sum_{i}z^{(i)}
$$

$$
\sigma^2=\frac{1}{m}\sum_{i}(z^{(i)}-\mu)^2
$$

$$
z^{(i)}_{norm}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

$m$ 是单个mini-batch包含样本个数，$\varepsilon$ 是为了防止分母为零，可取值 $10^{-8}$。使得该隐藏层的所有输入 $z^{(i)}$ 均值为0，方差为1

大部分情况下并不希望所有的 $z^{(i)}$ 均值都为0，方差都为1，也不太合理。通常需要对 $z^{(i)}$ 进行进一步处理：

$$
\widetilde{z}^{(i)}=\gamma \cdot z^{(i)}_{norm}+\beta
$$

$\gamma$ 和 $\beta$ 是learnable parameters，可以通过梯度下降等算法求得。$\gamma$ 和 $\beta$ 是让 $\widetilde{z}^{(i)}$ 的均值和方差为任意值，只需调整其值。如：

$$
\gamma=\sqrt{\sigma^2+\varepsilon} \quad \beta=\mu
$$

则 $\widetilde{z}^{(i)}=z^{(i)}$，即identity function。设置 $\gamma$ 和 $\beta$ 为不同的值，可以得到任意的均值和方差

通过Batch Normalization，对隐藏层的各个 $z^{[l](i)}$ 进行标准化处理，得到 $\widetilde{z}^{[l](i)}$，替代 $z^{[l](i)}$

输入的标准化处理Normalizing inputs和隐藏层的标准化处理Batch Normalization是有区别的。Normalizing inputs使所有输入的均值为0，方差为1。而Batch Normalization可使各隐藏层输入的均值和方差为任意值。从激活函数的角度来说，如果各隐藏层的输入均值在靠近0的区域即处于激活函数的线性区域，这样不利于训练好的非线性神经网络，得到的模型效果也不会太好

## 5. 将Batch Norm拟合进神经网络(Fitting Batch Norm into a neural network)

前向传播的计算流程：

<div align=center>
<img src="img/屏幕截图%202024-04-26%20194205.png" width=45%>
<img src="img/屏幕截图%202024-04-26%20194246.png" width=45%>
</div>

**实现梯度下降：**

<div align=center>
<img src="img/屏幕截图%202024-04-26%20194259.png" width=80%>
</div>

$for \ t=1 \dots num$(这里num为Mini Batch的数量)：

* 在每一个 $X^t$ 上进行前向传播(forward prop)的计算：

  * 在每个隐藏层都用 Batch Norm 将 $z^{[l]}$ 替换为 $\widetilde{z}^{[l]}$
* 使用反向传播(Back prop)计算各个参数的梯度:$dw^{[l]},d\beta^{[l]},d\gamma^{[l]}$
* 更新参数：

  * $w^{[l]}=w^{[l]}-\alpha\cdot dw^{[l]}$
  * $\beta^{[l]}=\beta^{[l]}-\alpha\cdot d\beta^{[l]}$
  * $\gamma^{[l]}=\gamma^{[l]}-\alpha\cdot d\gamma^{[l]}$

Batch Norm对各隐藏层 $Z^{[l]}=W^{[l]}\cdot A^{[l-1]}+b^{[l]}$ 有去均值的操作，Batch Norm要做的就是将 $Z^{[l]}$ 归一化，结果成为均值为0，标准差为1的分布，再由 $\beta$ 和 $\gamma$ 进行重新的分布缩放，意味着无论 $b^{[l]}$ 值为多少，在这个过程中都会被减去，不会再起作用。所以常数项 $b^{[l]}$ 可以消去，其数值效果完全可以由 $\widetilde{z}^{[l]}$ 中的 $\beta$ 来实现。在使用Batch Norm的时候，可以忽略各隐藏层的常数项 $b^{[l]}$。在使用梯度下降算法时，分别对 $w^{[l]}\quad \beta^{[l]}\quad \gamma^{[l]}$ 进行迭代更新

除了传统的梯度下降算法之外，还可以使用动量梯度下降、RMSprop或者Adam等优化算法

### 理解：

* 首先超参数 $\beta,\gamma$ 是通过训练迭代出来的，那么就是说不管原来的输入的方差、标准值是多少，进行一次运算之后，方差定下来了。
* 这个会影响上一层对下一层的影响，代表了具体的数据变得没有意义，相对数据是我们考察的具体因素
* 对于测试集而言，方差过大的数据集不会对运行结果有过大影响，因为方差和均值的影响在算法中被消除了

## 6. Batch Norm 为什么奏效?(Why does Batch Norm work?)

Batch Norm 可以加速神经网络训练的原因:

* 和输入层的输入特征进行归一化，从而改变Cost function的形状，使得每一次梯度下降都可以更快的接近函数的最小值点，从而加速模型训练过程的原理有相同的道理，只是Batch Norm是将各个隐藏层的激活函数的激活值进行的归一化，并调整到另外的分布
* Batch Norm 可以使权重比网络更滞后或者更深层

#### 判别是否是猫的分类问题：

假设第一训练样本的集合中的猫均是黑猫，而第二个训练样本集合中的猫是各种颜色的猫。如果将第二个训练样本直接输入到用第一个训练样本集合训练出的模型进行分类判别，在很大程度上无法保证能够得到很好的判别结果

因为训练样本不具有一般性（即不是所有的猫都是黑猫），第一个训练集合中均是黑猫，而第二个训练集合中各色猫均有，虽然都是猫，但是很大程度上样本的分布情况是不同的，无法保证模型可以仅仅通过黑色猫的样本就可以完美的找到完整的决策边界

这种训练样本（黑猫）和测试样本（猫）分布的变化称之为covariate shift。如下图所示：

深度神经网络中，covariate shift会导致模型预测效果变差，重新训练的模型各隐藏层的 $W^{[l]}$ 和 $B^{[l]}$ 均产生偏移、变化。而Batch Norm的作用恰恰是减小covariate shift的影响，让模型变得更加健壮，鲁棒性更强

使用深层神经网络，使用Batch Norm，该模型对花猫的识别能力应该也是不错

<div align=center>
<img src="img/屏幕截图%202024-04-26%20201034.png" width=47%>
<img src="img/屏幕截图%202024-04-26%20201049.png" width=47%>
</div>

#### Batch Norm 解决 Covariate shift 的问题

网络的目的是通过不断的训练，最后输出一个更加接近于真实值的 $\hat{y}$，以第2个隐藏层为输入来看：

对于后面的神经网络，是以第二层隐层的输出值 $a^{[2]}$ 作为输入特征的，通过前向传播得到最终的 $\hat{y}$，但是网络还有前面两层，由于训练过程，参数 $w^{[1]},w^{[2]}$ 是不断变化的，对于后面的网络，$a^{[2]}$ 的值也是处于不断变化之中，所以就有了Covariate shift的问题

如果对 $z^{[2]}$ 使用了Batch Norm，即使其值不断的变化，其均值和方差却会保持。Batch Norm的作用是限制前层的参数更新导致对后面网络数值分布程度的影响，使得输入后层的数值变得更加稳定。Batch Norm减少了各层 $W^{[l]},B^{[l]}$ 之间的耦合性，让各层更加独立，实现自我训练学习的效果。如果输入发生covariate shift，Batch Norm的作用是对个隐藏层输出 $Z^{[l]}$ 进行均值和方差的归一化处理，让 $W^{[l]},B^{[l]}$ 更加稳定，使得原来的模型也有不错的表现

Batch Norm 削弱了前层参数与后层参数之间的联系，使得网络的每层都可以自己进行学习，相对其他层有一定的独立性，有助于加速整个网络的学习

#### Batch Norm 正则化效果

* 使用Mini-batch梯度下降，每次计算均值和偏差都是在一个Mini-batch上进行计算，而不是在整个数据样集上。这样在均值和偏差上带来一些比较小的噪声。那么用均值和偏差计算得到的 $\widetilde{z}^{[l]}$ 也将会加入一定的噪声
* 和Dropout相似，其在每个隐藏层的激活值上加入了一些噪声(Dropout以一定的概率给神经元乘上0或者1)。Batch Norm 也有轻微的正则化效果
* 如果使用Batch Norm ，使用大的Mini-batch如:256，相比使用小的Mini-batch如:64，会引入更少的噪声，会减少正则化的效果
* Batch Norm的正则化效果比较微弱，正则化不是Batch Norm的主要功能

## 7. 测试时的 Batch Norm(Batch Norm at test time)

训练过程中Batch Norm的主要过程:

$$
\mu=\frac{1}{m}\sum_{i}z^{(i)}
$$

$$
\sigma^2=\frac{1}{m}\sum_{i}(z^{(i)}-\mu)^2
$$

$$
z^{(i)}_{norm}=\frac{z^{(i)}-\mu}{\sqrt{\sigma^2+\varepsilon}}
$$

$$
\widetilde{z}^{(i)}=\gamma \cdot z^{(i)}_{norm}+\beta
$$

$\mu$ 和 $\sigma^2$ 是对单个mini-batch中所有 $m$ 个样本求得的。在测试过程中，如果只有一个样本，求其均值和方差是没有意义的，就需要对 $ \mu$ 和 $\sigma^2$ 进行估计。实际应用是使用指数加权平均(exponentially weighted average)的方法来预测测试过程单个样本的 $\mu$ 和 $\sigma^2$

对于第 $l$ 层隐藏层，在训练的过程中, 对于训练集的Mini-batch，考虑所有mini-batch在该隐藏层下的 $\mu^{[l]}$ 和 $\sigma^{[l]}$，使用指数加权平均，当训练结束的时候，得到指数加权平均后当前单个样本的 $\mu^{[l]}$ 和 $\sigma^{2[l]}$,这些值直接用于Batch Norm公式的计算，用以对测试样本进行预测，再利用训练过程得到的 $\gamma$ 和 $\beta$ 计算出各层的 $\widetilde{z}^{(i)}$ 值

### 说明：

* 先获取每个mini-batch的 $\mu,\sigma$ ，然后使用指数加权平均(exponentially weighted average)/流动平均 的方法来进行
  * 这种方式只需保留三个值，全局统计值、当前batch的统计值和衰减系数，消耗的存储资源少，在损失一定准确度的情况下，计算速度快，在训练阶段可同步完成总统计值的计算，不需额外的计算。实际运用中为了保证运行速度，采用这种方式
* 实际上，我们先对每个mini-batch进行训练，然后获取 $\mu,\sigma$ 之后，会进行重新训练第二次

## 8. Softmax 回归(Softmax regression)

Softmax回归，能在识别多种分类中的一个时做出预测，对于多分类问题，用 $C$ 表示种类个数，神经网络中输出层就有 $C$ 个神经元，即 $n^{[L]}=C$，每个神经元的输出依次对应属于该类的概率，即 $P(y=c∣x)$，处理多分类问题一般使用Softmax回归模型

<div align=center>
<img src="img/屏幕截图%202024-04-26%20233244.png" width=47%>
<img src="img/屏幕截图%202024-04-26%20233316.png" width=47%>
</div>

> 把猫做类 1，狗为类 2，小鸡是类 3，如果不属于以上任何一类，就分到“其它”或者“以上均不符合”这一类，叫做类 0

用大写 $C$ 表示输入会被分入的类别总个数,当有4个分类时，指示类别的数字，就是 $0\sim C−1$

Softmax回归模型输出层的激活函数:

$$
\large{z^{[L]}=W^{[L]}a^{[L−1]}+b^{[L]}}
$$

$$
\large{a_i^{[L]}=\frac{e^{z_i^{[L]}}}{\sum_{i=1}^{C}e^{z_i^{[L]}}}}
$$

输出层每个神经元的输出 $a_i^{[L]}$ 对应属于该类的概率，满足：

$$
\large{\sum_{i=1}^{C}a_i^{[L]}=1}
$$

所有的 $a_i^{[L]}$，即 $\hat{y}$，维度为 $(C,1)$

在没有隐藏隐藏层的时候，直接对Softmax层输入样本的特点，则在不同数量的类别下，Sotfmax层的作用：

<div align=center>
<img src="img/屏幕截图%202024-04-26%20233338.png" width=80%>
</div>

图中的颜色显示了 Softmax 分类器的输出的阈值，输入的着色是基于三种输出中概率最高的那种，任何两个分类之间的决策边界都是线性的

如果使用神经网络，特别是深层神经网络，可以得到更复杂、更精确的非线性模型

## 9. 训练一个 Softmax 分类器(Training a Softmax classifier)

$C=4$，某个样本的预测输出 $\hat{y}$ 和真实输出 $y$:

$$
\hat{y}=\begin{bmatrix} 0.3 \\ 0.2 \\ 0.1 \\ 0.4 \\ \end{bmatrix} \quad y=\begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ \end{bmatrix}
$$

从 $\hat{y}$ 值来看，$P(y=4∣x)=0.4$，概率最大，而真实样本属于第2类，该预测效果不佳

定义softmax classifier的loss function为:

$$
L(\hat{y},y)=-\sum_{j=1}^{4}y_j\cdot \log\hat{y}_j
$$

$L(\hat{y},y)$ 简化为:

$$
L(\hat{y},y)=-y_2\cdot \log\hat{y}_2=-\log\hat{y}_2
$$

让 $L(\hat{y},y)$ 更小，就应该让 $\hat{y}_2$ 越大越好。 $\hat{y}_2$ 反映的是概率

m个样本的cost function为:

$$
J=\frac{1}{m}\sum_{i=1}^{m}L(\hat{y},y)
$$

预测输出向量 $A^{[L]}$ 即 $\hat{Y}$ 的维度为 $(4,m)$

softmax classifier的反向传播过程:

$$
\Large\frac{\partial C}{\partial a_j}=\frac{\partial (-\sum_{j}y_j\cdot \log a_j)}{\partial a_j}=-\sum_{j}y_j\frac{1}{a_j}
$$

$$
\Large\frac{\partial a_i}{\partial z_j}=
\begin{cases}
\frac{\partial a_i}{\partial z_i}=\frac{\partial \frac{e^{z_i}}{\sum_C}}{\partial z_i}=\frac{e^{z_i}\cdot \sum_C-e^{z_i}\cdot e^{z_i}}{(\sum_C)^2}=a_i\cdot (1-a_i)& if \ i=j \\
\newline \\
\frac{\partial \frac{e^{z_i}}{\sum_C}}{\partial z_j}=\frac{-e^{z_i}\cdot e^{z_j}}{(\sum_C)^2}=-a_i\cdot a_j& if \ i\ne j \\
\end{cases}
$$

$$
\Large\Large\frac{\partial C}{\partial z_i}=(-\sum_{j}y_j\frac{1}{a_j})\frac{\partial a_j}{\partial z_i}=-\frac{y_i}{a_i}\cdot a_i\cdot (1-a_i)+(-\sum_{j\ne i}\frac{y_j}{a_j})(-a_i\cdot a_j)=-y_i+a_i\cdot\sum_{j}y_j=a_i-y_i
$$

所有 $m$ 个训练样本：

$$
\Large dZ^{[L]}=A^{[L]}−Y
$$

* 剩下的部分和之前讨论过的深度神经网络一样的反向传播方式！！！！

### 说明：

* 这里的cost function是这样来的:
  * 我们要求训练的结果和实际结果相符合，所谓的cost function实际上是为了检验是否符合实际的
  * 我们求的是实际情况在训练出来的概率下发生的概率，对于每一对象而言，发生的概率就是对应情况的概率
    但是为了方便对称性计算，我们发现 $x^0=1$，这样我们将其他的情况开0次方相乘也就没有影响了
  * 也就是说，我们需要让下式最大
    $$
    \prod_{i=1}^{m}\prod_{j=1}^{C}(\hat{y}_j^{(i)})^{y_j^{(i)}}
    $$
  * 我们将上式进行 $\log$，再乘以 -1 ，也就是我们所在前面得到的式子(因为我们要求的cost function求的是最小值，而上式是最大值！！！！)
