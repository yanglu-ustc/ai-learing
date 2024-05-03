# Setting-up-your-ML-application

## 1. Train/dev/test set

深度学习是一个典型的**迭代过程**，迭代的**效率**很关键，创建高质量的**训练数据集，验证集和测试集**有助于提高循环效率。

* 三者区别：
  **训练集(train set)** —— 用于模型拟合的数据样本。
  **验证集(development set)** ——
  是模型训练过程中单独留出的样本集，用于调整模型的超参数以及对模型的能力进行初步评估。通常用来在模型迭代训练时，用以验证当前模型泛化能力（准确率，召回率等），以决定是否停止继续训练。
  **测试集(test set)** —— 用来评估模最终模型的泛化能力。但不能作为调参、选择特征等算法相关的选择的依据。

$\newline$

* 切分标准：
  **小数据**量时代，常见做法是三七分，70%验证集，30%测试集；也可以 60%训练，20%验证和20%测试集来划分。**大数据**
  时代，数据量可能是百万级别，验证集和测试集占总量的比例会趋于变得更小。

$\newline$

* 数据来源：
  最好要确保 **验证集** 和 **测试集** 的数据来自同一分布，因为要用验证集来评估不同的模型，如果验证集和测试集来自同一个分布就会表现很好。

## 2. Bias and Variance:偏差 方差

* 高偏差 —— 欠拟合
* 高方差 —— 过拟合

**关键数据：**

* 训练集误差、验证集误差（基于人眼辨别错误率约为0情况下）
* 如果**人眼辨别的错误率**（贝叶斯误差，最优误差）非常高，比如15%。那么上面第二种分类器（训练误差15%，验证误差16%），15%的错误率对训练集来说也是非常合理的，偏差不高，方差也非常低。

<div align=center>
<img src="img/屏幕截图%202024-04-24%20212751.png" width=80%>
</div>

> **解决方案:(TODO)**
> 解决高方差：获得更多的训练样本；尝试减少特征的数量；尝试增加正则化程度$\lambda$。
> 解决高偏差：尝试获得更多的特征；尝试增加多项式特征；尝试减少正则化程度$\lambda$。

## 3. 机器学习基础

<div align=center>
<img src="img/屏幕截图%202024-04-24%20214136.png" width=80%>
</div>

## 4. Regularization —— 正则化

* 正则化有助于防止过拟合，降低方差。
* 范数(norm)几种范数的简单介绍、对于L1和L2正则化的不同解释

  * L1范数(表示非零元素的绝对值之和)
    $$
    \|X\|_{1}=\sum_{i=1}^{n}\left|x_{i}\right|
    $$
  * L2范数(表示元素的平方和再开方)
    $$
    \|X\|_ {2}=\sqrt {\sum_{i=1}^ {n} x_ {i}^ {2}}
    $$
  * 矩阵的L2范数叫做弗罗贝尼乌斯范数(所有元素的平方和)
    $$
    \|W\|_{F}^{2}
    $$
* 加上正则化的损失函数：

  * L1正则化

    $$
    J(\omega,b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\|\omega\|_{1}
    $$
  * L2正则化

    $$
    J(\omega,b)=\frac{1}{m}\sum_{i=1}^{m}\mathcal{L}(\hat{y}^{(i)},y^{(i)})+\frac{\lambda}{2m}\|\omega\|_{2}
    $$
* L1正则化 & L2正则化的特征：

  * L1 正则，权重 $\mathrm{w}$ 最终变得稀疏，多数变成0
  * L2 正则，使得权重衰减。
* 对于在神经网络 $W$ 矩阵中进行参数更新操作(权重不但减少了，还乘以了小于1的系数进行衰减)

$$
W^{[l]}:=\left(1-\frac{\alpha\lambda}{m}\right) * W^{[l]}-\alpha(\frac{\partial \mathcal{L}}{\partial W^{[l]}})
$$

## 5. 为什么正则化预防过拟合

<div align=center>
<img src="img/屏幕截图%202024-04-24%20221429.png" width=49%>
<img src="img/屏幕截图%202024-04-24%20221520.png" width=48.5%>
</div>

## 6. dropout（随机失活）正则化

* 实施dropout举例:最常用的方法—— inverted dropout(反向随机失活)，用一个三层网络举例
  注意：下面的a3和d3对应网络中所有的神经单元，实际实现过程中需要一层一层来实现。
* 注意：在测试的时候就不需要进行随机失活了，要保证测试的准确性

> * 定义向量d3，表示一个三层的dropout向量。它是一个概率矩阵，对应每个样本和每个隐藏单元。
>   * $\mathrm{d}3=\mathrm{np.random.rand}(\mathrm{a3.shape[0]},\mathrm{a3.shape[1]})$
>   * 定义keep-prob(表示保留某个隐藏单元的概率)，概率为(1-keep-prob)的元素对应为0，概率为keep-prob的元素对应为1
> * 获取激活函数 $\mathrm{a}3=\mathrm{np}.\mathrm{multiply}(\mathrm{a}3,\mathrm{d}3)$ ，使得d3中为0的元素把a3对应元素归零
>   * 向外扩展:a3/=keep-prob
> * 反向随机失活(inverted dropout)方法通过除以keep-prob，确保$a^{[3]}$的输出期望值不会被改变。
>   原因如下：虽然dropout忽略了某些隐藏层神经元，例如系数是0.8，则保留80%的神经元，则神经元输出A整体期望也为原来的80%。但是inverted
>   dropout的放缩操作，即对A除以80%。一缩一放过程，相当于输出期望没有改变。
>
> <div align=center>
> <img src="img/屏幕截图%202024-04-24%20223631.png" width=80%>
> </div>

## 7. 理解 dropout

* 其功能类似于 L2 正则化；
* 对于参数集多的层，可以使用较低的keep-prob值(不同的层，可以使用不同的值)，缺点是：需要交叉验证更多的参数；
* 注意：
  * dropout一大缺点就是：代价函数不再被明确定义，每次迭代，都会随机移除一些节点，想检查梯度下降的性能，实际上是很难进行复查的。
  * 在实际操作中，可以先关闭dropout，将keep-prob设置为1，确保J函数单调递减，然后再尝试使用dropout。
* 如果某一层容易发生过拟合，可以设置一个更小的keep-prob值
* 这个方法经常使用在计算机视觉中，因为数据量过于庞大

<div align=center>
<img src="img/屏幕截图%202024-04-24%20224706.png" width=80%>
</div>

## 8. 其他正则化

* 数据扩增，假如是图片数据，扩增数据代价高，我们可以：
  水平翻转；随意剪裁旋转放大（这种方式扩增数据，进而正则化数据集，减少过拟合成本很低）
  对于数字识别图片，我们可以进行旋转，扭曲来扩增数据集
* early stopping:在验证集误差变坏的时候，提早停止训练
* early stopping缺点:不能同时处理**过拟合**和**代价函数不够小**的问题

  * 提早停止，可能代价函数J不够小
  * 不提早结束，可能会过拟合
* 如果不使用 early stopping ，使用 L2 正则化的话，这样训练时间可能很长，参数搜索空间大，计算代价很高。
* early stopping优点:只运行一次梯度下降，可以找出 w 的较小值，中间值，较大值，无需尝试L2正则化超参数$\lambda$的很多值。

<div align=center>
<img src="img/屏幕截图%202024-04-25%20080637.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20081125.png" width=45%>
</div>

## 9. Normalization input —— 归一化输入

归一化输入，可以加速训练。它一般需要两个以下步骤：

* 零均值化（所有的数据减去均值)
* 归一化方差 (所有数据除以方差)

注意:$\mu$和$\sigma^{2}$是由训练集得到，然后用于其他所有数据集。

<div align=center>
<img src="img/屏幕截图%202024-04-25%20082004.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20082017.png" width=45%>
</div>

## 10. data vanishing & exploding gradients —— 梯度消失 & 梯度爆炸

**_在非常深的神经网络中，权重只要不等于1，激活函数将会呈指数级递增或者递减，导致训练难度上升，尤其是梯度与L相差指数级，梯度下降算法的步长会非常非常小，学习时间很长_**

<div align=center>
<img src="img/屏幕截图%202024-04-25%20082934.png" width=80%>
</div>

## 11. 神经网络权重初始化

* Xavier初始化:
  为了预防 _z_ 的值过大或过小，_n_ 越大时，你希望 $w_i$ 越小，因为 _z_ 是 $w_ix_i$ 的和，合理的方法是 $w_{i}=1 / n$，_n_ 是输入特征数量。

  $$
  W^{[l]}=np.random.randn(shape)\cdot np.\operatorname{sqrt}\left(\frac{1}{n^{[l-1]}}\right)
  $$

  其中$n^{[l-1]}$是给第 _l_ 层输入的特征数量。
* 如果使用ReLu激活函数(最常用)，则如下。被称为Kaiming初始化

  $$
  W^{[l]}=np.random.randn(shape)\cdot np.\operatorname{sqrt}\left(\frac{2}{n^{[l-1]}}\right)
  $$
* 如果使用tanh激活函数，则

  $$
  W^{[l]}=np.random.randn(shape)\cdot np.\operatorname{sqrt}\left(\frac{1}{n^{[l-1]}}\right)
  $$

  $$
  W^{[l]}=np.random.randn(shape)\cdot np.\operatorname{sqrt}\left(\frac{2}{n^{[l-1]}+n^{[l]}}\right)
  $$

> 这样设置的权重矩阵既不会增长过快，也不会太快下降到 0，从而训练出一个权重或梯度不会增长或消失过快的深度网络，这也是一个加快训练速度的技巧

## 12. 梯度的数值逼近

在反向传播时，有个测试叫做梯度检验。即计算误差时，我们需要使用双边误差，不使用单边误差，因为前者更准确。

$$
\left.f^{\prime}( \theta\right)=\frac{f(\theta+\varepsilon)-f(\theta-\varepsilon)}{2 \varepsilon}
$$

## 13. Gradient Checking —— 梯度检验

梯度检验帮助我们发现反向传播中的bug。

<div align=center>
<img src="img/屏幕截图%202024-04-25%20091143.png" width=45%>
<img src="img/屏幕截图%202024-04-25%20091158.png" width=45%>
</div>

## 14. 梯度检验的注意事项

1. 不要在训练中使用梯度检验，它只用于调试；
2. 如果算法的梯度检验失败，要检查所有项，检查每一项，并试着找出bug；
3. 如果使用了正则化，计算梯度的时候也要包括正则项；
4. 梯度检验不能与dropout 同时使用，可以关闭dropout，进行梯度检验，检验正确了，再打开dropout。

<div align=center>
<img src="img/屏幕截图%202024-04-25%20091403.png" width=80%>
</div>