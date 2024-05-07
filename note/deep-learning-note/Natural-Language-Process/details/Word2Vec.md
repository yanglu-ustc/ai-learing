[原文](https://zhuanlan.zhihu.com/p/114538417)

# Word2Vec的网络结构

Word2Vec是轻量级的神经网络，其模型仅仅包括输入层、隐藏层和输出层，模型框架根据输入输出的不同，主要包括CBOW和Skip-gram模型。 CBOW的方式是在知道词 $w_t$ 的上下文 $w_{t-2},w_{t-1},w_{t+1},w_{t+2}$ 的情况下预测当前词 $w_t$ .而Skip-gram是在知道了词 $w_t$ 的情况下,对词 $w_t$ 的上下文 $w_{t-2},w_{t-1},w_{t+1},w_{t+2}$ 进行预测，如下图所示:

![](https://pic3.zhimg.com/v2-7339e1444995c19f962c900cf8c67106_r.jpg)

## 1. CBOW

### 1.1 Simple CBOW Model

为了更好的了解模型深处的原理，我们先从Simple CBOW model(仅输入一个词，输出一个词)框架说起。

如上图所示：

input layer输入的X是单词的one-hot representation（考虑一个词表V，里面的每一个词 w_{i} 都有一个编号i∈{1,...,|V|}，那么词 w_{i} 的one-hot表示就是一个维度为|V|的向量，其中第i个元素值非零，其余元素全为0，例如： w_2=[0,1,0,...,0]^T ）；
输入层到隐藏层之间有一个权重矩阵W，隐藏层得到的值是由输入X乘上权重矩阵得到的（细心的人会发现，0-1向量乘上一个矩阵，就相当于选择了权重矩阵的某一行，如图：输入的向量X是[0，0，1，0，0，0]，W的转置乘上X就相当于从矩阵中选择第3行[2,1,3]作为隐藏层的值）;
隐藏层到输出层也有一个权重矩阵W'，因此，输出层向量y的每一个值，其实就是隐藏层的向量点乘权重向量W'的每一列，比如输出层的第一个数7，就是向量[2,1,3]和列向量[1，2，1]点乘之后的结果；
最终的输出需要经过softmax函数，将输出向量中的每一个元素归一化到0-1之间的概率，概率最大的，就是预测的词。
了解了Simple CBOW model的模型框架之后，我们来学习一下其目标函数。

\begin{aligned} L &= max \ log \ p(w|Context(w)) \\ &= max \log \ (y_{j}^{*}) \\ &= max \ log ( \frac{exp(u_{j}^{*})}{\sum {exp(u_{k}) } }) \end{aligned} \\输出层通过softmax归一化，u代表的是输出层的原始结果。通过下面公式，我们的目标函数可以转化为现在这个形式

a^{loga(N)} =N \\

max \ log ( \frac{exp(u_{j}^{*})}{\sum {exp(u_{k}) } }) = max \ u_{j}^{*} - log \sum_{k=1}^{V}{exp(u_{k})} \\
3.1.2 CBOW Multi-Word Context Model

了解了Simple CBOW model之后，扩展到CBOW就很容易了，只是把单个输入换成多个输入罢了（划红线部分）。

对比可以发现，和simple CBOW不同之处在于，输入由1个词变成了C个词，每个输入 X_{ik} 到达隐藏层都会经过相同的权重矩阵W，隐藏层h的值变成了多个词乘上权重矩阵之后加和求平均值。
3.2 Skip-gram Model

有了CBOW的介绍，对于Skip-gram model 的理解应该会更快一些。

如上图所示，Skip-gram model是通过输入一个词去预测多个词的概率。输入层到隐藏层的原理和simple CBOW一样，不同的是隐藏层到输出层，损失函数变成了C个词损失函数的总和，权重矩阵W'还是共享的。

一般神经网络语言模型在预测的时候，输出的是预测目标词的概率，也就是说我每一次预测都要基于全部的数据集进行计算，这无疑会带来很大的时间开销。不同于其他神经网络，Word2Vec提出两种加快训练速度的方式，一种是Hierarchical softmax，另一种是Negative Sampling。
4. 基于Hierarchical Softmax的模型

基于层次Softmax的模型主要包括输入层、投影层（隐藏层）和输出层，非常的类似神经网络结构。对于Word2Vec中基于层次Softmax的CBOW模型，我们需要最终优化的目标函数是 :

L=\sum \log p(w|Context(w)) \tag{6} \\ 其中 Context(w) 表示的是单词 w 的的上下文单词。而基于层次Softmax的Skip-gram模型的最终需要优化的目标函数是：L=\sum \log p(Context(w)|w) \tag{7}\\
4.1 基于Hierarchical Softmax的CBOW

4.1.1 CBOW模型网络结构

下图给出了基于层次Softmax的CBOW的整体结构，首先它包括输入层、投影层和输出层：

输入层：是指 Context(w) 中所包含的 2c 个词的词向量v(Context(w)_1),v( Context(w)_2), ..., v(Context(w)_{2c}) 。
投影层：指的是直接对 2c 个词向量进行累加，累加之后得到下式：X_w=\sum_{i=1}^{2c}v(Context(w)_i) \\
输出层：是一个Huffman树，其中叶子节点共N个，对应于N个单词，非叶子节点N-1个（对应上图中标成黄色的结点）。Word2Vec基于层次Softmax的方式主要的精华部分都集中在了哈夫曼树这部分，下面详细介绍。
4.1.2 CBOW的目标函数

为了便于下面的介绍和公式的推导，这里需要预先定义一些变量：

p^w ：从根节点出发，然后到达单词 w 对应叶子结点的路径。
l^w ：路径 p^w 中包含的结点的个数。
p^w_1, p^w_2, ..., p^w_{l^w} : 路径 p^w 中对应的各个结点，其中 p^w_1 代表根结点，而 p^w_{l^w} 代表的是单词 w 对应的结点。
d^w_2, d^w_3 ..., d^w_{l^w}\in \{0, 1 \} : 单词 w 对应的哈夫曼编码，一个词的哈夫曼编码是由 l^w-1 位构成的，d^w_j 表示路径 p^w 中的第j个单词对应的哈夫曼编码，根结点不参与对应的编码。
\theta^w_1, \theta^w_2, ..., \theta^w_{l^w-1}\in \Re^{m} : 路径 p^w 中非叶子节点对应的向量，\theta^w_j 表示路径 p^w 中第 j 个非叶子节点对应的向量。 这里之所以给非叶子节点定义词向量，是因为这里的非叶子节点的词向量会作为下面的一个辅助变量进行计算，在下面推导公式的时候就会发现它的作用了。 
既然已经引入了那么多符号，我们通过一个简单的例子把它们落到实处吧，我们考虑单词w="足球"的情形。下图中红色线路就是我们的单词走过的路径，整个路径上的5个结点就构成了路径 p^w ，其长度 l^w=5 ，然后 p^w_1, p^w_2,p^w_3,p^w_4,p^w_{5} 就是路径 p^w 上的五个结点，其中 p_{1}^{w} 对应根结点。 d^w_2,d^w_3,d^w_4,d^w_5 分别为1,0,0,1，即"足球"对应的哈夫曼编码就是1001。最后 \theta^w_1, \theta^w_2, \theta^w_3,\theta^w_4 就是路径 p^w 上的4个非叶子节点对应的向量。

下面我们需要开始考虑如何构建条件概率函数 p(w|Context(w)) ，以上面的w="足球"为例，从根节点到"足球"这个单词，经历了4次分支，也就是那4条红色的线，而对于这个哈夫曼树而言，每次分支相当于一个二分类。

既然是二分类，那么我们可以定义一个为正类，一个为负类。我们的"足球"的哈夫曼编码为1001，这个哈夫曼编码是不包含根节点的，因为根节点没法分为左还是右子树。那么根据哈夫曼编码，我们一般可以把正类就认为是哈夫曼编码里面的1，而负类认为是哈夫曼编码里面的0。不过这只是一个约定而已，因为哈夫曼编码和正类负类之间并没有什么明确要求对应的关系。事实上，Word2Vec将编码为1的认定为负类，而编码为0的认定为正类，也就是说如果分到了左子树，就是负类；分到了右子树，就是正类。那么我们可以定义一个正类和负类的公式： Label(p^w_i)=1-d^w_i, i=2, 3, 4, ..., l^w \\简而言之就是，将一个结点进行分类时，分到左边就是负类，分到右边就是正类。

在进行二分类的时候，这里选择了Sigmoid函数。那么，一个结点被分为正类的概率就是：

\sigma (x^T_w\theta)=\frac{1}{1+e^{-x^t_w\theta}} \\

被分为负类的概率就是 1-\sigma (x^T_w\theta) 。注意，公式里面包含的有 \theta ，这个就是非叶子对应的向量 \theta_{i}^{w} 。

对于从根节点出发到达“足球”这个叶子节点所经历的4次二分类，将每次分类的概率写出来就是：

第一次分类： p(d^w_2|x_w,\theta^w_1)=1-\sigma(x^T_w\theta^w_1)
第二次分类： p(d^w_3|x_w,\theta^w_2)=\sigma(x^T_w\theta^w_2)
第三次分类： p(d^w_4|x_w,\theta^w_3)=\sigma(x^T_w\theta^w_3)
第四次分类： p(d^w_5|x_w,\theta^w_4)=1-\sigma(x^T_w\theta^w_4) 
但是，我们要求的是 p(w|Context(w)) ，即 p(足球|Context(足球)) ，它跟这4个概率值有什么关系呢？关系就是：

p(足球|Context(足球))=\prod_{j=2}^{5}p(d^w_j|x_w,\theta^w_{j-1}) \\

至此，通过w="足球"的小例子，Hierarchical Softmax的基本思想就已经介绍完了。小结一下：对于词典中的任意一个单词 w ，哈夫曼树中肯定存在一条从根节点到单词 w 对应叶子结点的路径 p^w ，且这条路径是唯一的。路径 p^{w} 上存在 l^{w} -1 个分支，将每个分支看做一次二分类，每一次分类就产生一个概率，将这些概率乘起来，就是所需要的 p(w|Context(w)) 。

条件概率 p(w|Context(w)) 的一般公式可写为：

p(w|Context(w))=\prod_{j=2}^{l^w}p(d^w_j|x_w,\theta^w_{j-1}) \tag{8}\\ 其中：

p(d^w_j|x_w,\theta^w_{j-1})= \left\{ \begin{matrix} \sigma(x^T_w\theta^w_{j-1}), & d^w_j=0 \\ 1 - \sigma(x^T_w\theta^w_j-1), & d^w_j=1 \end{matrix} \right. \\

或者写成整体表达式：

p(d^w_j|x_w,\theta^w_{j-1})=[\sigma(x^T_w\theta^w_{j-1})]^{1-d^w_j}\cdot [1-\sigma(x^T_w\theta^w_{j-1})]^{d^w_j} \\

将公式(8)带入公式(6)中，得到

\begin{aligned} L &=\sum_{w \in C} \log \prod_{j=2}^{l^w}{ { [\sigma(x^T_w\theta^w_{j-1})]^{1-d^w_j}\cdot [1-\sigma(x^T_w\theta^w_{j-1})]^{d^w_j}} } \\ &= \sum_{w \in C} \sum_{j=2}^{l^w}{(1-d^w_j) \cdot \log [\sigma(x^T_w \theta ^w_{j-1})] + d^w_j \cdot \log [1-\sigma(x^T_w \theta ^w_{j-1})] } \end{aligned} \tag{9} \ \

为了梯度推导方便起见，将上式中双重求和符号下的内容简记为 L(w,j) ，即

L(w,j)=(1-d^w_j) \cdot \log [\sigma(x^T_w \theta ^w_{j-1})] + d^w_j \cdot \log [1-\sigma(x^T_w \theta ^w_{j-1})] \\

至此，已经推导出了CBOW模型的目标函数公式(9)，接下来就是讨论如何优化它，即如何将这个函数最大化。Word2Vec里面采用的是随机梯度上升法。而梯度类算法的关键是给出相应的梯度计算公式，进行反向传播。

4.1.3 参数更新

首先考虑L(w,j)关于 \theta^{w}_{j-1} 的梯度计算：

\begin {aligned} \frac{\Delta L(w,j)}{\Delta \theta ^w_{j-1}} &= \left\{ (1-d^w_j)[1- \sigma(x^T_w \theta ^w_{j-1})]x_w - d^w_j \sigma (x^T_w \theta^w_{j-1}) \right\}x_w \\ &= [1-d^w_j- \sigma(x^T_w \theta^w_{j-1})]x_w \end {aligned} \\

于是， \theta^{w}_{j-1}的更新公式可写为：

\theta ^ w_{j-1}= \theta^w_{j-1}+ \eta [1-d^w_j- \sigma(x^T_w \theta^w_{j-1})]x_w \\

接下来考虑L(w,j)关于 x_{w} 的梯度：

\frac{\Delta L(w,j)}{\Delta x_w} = [1-d^w_j- \sigma(x^T_w \theta^w_{j-1})] \theta^w_{j-1} \\

到了这里，我们已经求出来了 x_w 的梯度，但是我们想要的其实是每次进行运算的每个单词的梯度，而 x_w 是 Context(w) 中所有单词累加的结果，那么我们怎么使用 x_w 来对Context(w)中的每个单词 v(\widetilde{w}) 进行更新呢？这里原作者选择了一个简单粗暴的方式，直接使用 x_w 的梯度累加对v(\widetilde{w})进行更新：

v(\widetilde{w}) = v(\widetilde{w}) + \eta \sum^{l^w}_{j=2} \frac{\Delta L(w,j)}{\Delta x_w}, \ \ \widetilde{w} \in Context(w) \\
4.2 基于Hierarchical Softmax的Skip-gram

本小节介绍Word2Vec中的另一个模型-Skip-gram模型，由于推导过程与CBOW大同小异，因此会沿用上小节引入的记号。

4.2.1 Skip-gram模型网络结构

下图给出了Skip-gram模型的网络结构，同CBOW模型的网络结构一样，它也包括三层：输入层、投影层和输出层。下面以样本 (w, Context(w)) 为例，对这三层做简要说明。

输入层：只含当前样本的中心词 w 的词向量 v(w) \in \Re^{m} 。
投影层：这是个恒等投影，把 v(w) 投影到 v(w) 。因此，这个投影层其实是多余的，这里之所以保留投影层主要是方便和CBOW模型的网络结构做对比。
输出层：和CBOW模型一样，输出层也是一颗Huffman树。
4.2.2 Skip-gram的目标函数

对于Skip-gram模型，已知的是当前词 w ，需要对其上下文 Context(w) 中的词进行预测，因此目标函数应该形如公式（7），且关键是条件概率函数 p(Context(w)|w) 的构造，Skip-gram模型中将其定义为：

p(Context(w)|w)= \prod_{u \in Context(w)}p(u|w) \\ 上式中 p(u|w)可按照上小节介绍的Hierarchical Softmax思想，类似于公式（8），可写为：

p(u|w)= \prod^{l^u}_{j=2}{p(d^u_j|v(w), \theta^u_{j-1}) } \\

其中：

p(d^u_j|v(w), \theta^u_{j-1})=[\sigma(v(w)^T \theta^u_{j-1})]^{1-d^w_j} \cdot [1- \sigma(v(w)^T \theta^u_{j-1})]^{d^u_j} \tag{10} \\

将公式（10）依次代回，可得对数似然函数公式（7）的具体表达式：

\begin {aligned} L &= \sum_{w \in C} \log \prod_{u \in Context(w)} \prod_{j=2}^{l^u} { [\sigma(v(w)^T \theta^u_{j-1})]^{1-d^w_j} \cdot [1- \sigma(v(w)^T \theta^u_{j-1})]^{d^u_j} } \\ &= \sum_{w \in C} \sum_{u \in Context(w)} \sum_{j=2}^{l^u}{ (1-d^u_j) \cdot \log [\sigma(v(w)^T\theta^u_{j-1})] + d^u_j \log [1- \sigma(v(w)^T \theta^u_{j-1})] } \end {aligned} \tag{11} \\

同样，为了梯度推导方便，将三重求和符号里的内容简记为L(w,u,j) ，即：

L(w,u,j)=(1-d^u_j) \cdot \log [\sigma(v(w)^T\theta^u_{j-1})] + d^u_j \log [1- \sigma(v(w)^T \theta^u_{j-1})] \\

至此，已经推导出了Skip-gram模型的目标函数（公式11），接下来同样利用随机梯度上升法对其进行优化。而梯度类算法的关键是给出相应的梯度计算公式，进行反向传播。

4.2.3 参数更新

首先考虑L(w,u,j)关于 \theta^{u}_{j-1} 的梯度计算：

\begin {aligned} \frac{ \Delta L(w,u,j)}{\Delta \theta^u_{j-1}} &=\left\{ (1-d^u_j)(1- \sigma(v(w)^T \theta^u_{j-1}))v(w)-d^u_j \sigma(v(w)^T \theta^u_{j-1})x \right\} v(w) \\ &= [1-d^u_j-\sigma(v(w)^T \theta^u_{j-1}]v(w) \end {aligned} \\

于是， \theta^{u}_{j-1}的更新公式可写为：

\theta^u_{j-1}=\theta^u_{j-1} + \eta [1-d^u_j-\sigma(v(w)^T \theta^u_{j-1}]v(w) \\ 同理,根据对称性,可以很容易得到 L(w,u,j) 关于 v(w) 的梯度：

\begin {aligned} \frac{ \Delta L(w,u,j)}{\Delta v(w)} &= [1-d^u_j-\sigma(v(w)^T \theta^u_{j-1}] \theta^u_{j-1} \end {aligned} \\

我们也可以得到关于v(w)的更新公式:

v(w)=v(w)+ \eta \sum_{u \in Context(w)} \sum^{l^w}_{j=2} \frac{ \Delta L (w,u,j)}{\Delta v(w)} \\
5. 基于Negative Sampling的模型

本节将介绍基于Negative Sampling的CBOW和Skip-gram模型。Negative Sampling（简称为NEG）是Tomas Mikolov等人在论文《Distributed Representations of Words and Phrases and their Compositionality》中提出的，它是NCE（Noise Contrastive Estimation）的一个简化版，目的是用来提高训练速度并改善所得词向量的质量。与Hierarchical Softmax相比，NEG不再使用复杂的Huffman树，而是利用相对简单的随机负采样，能大幅度提高性能，因而可作为Hierarchical Softmax的一种替代。

NCE 的细节有点复杂，其本质是利用已知的概率密度函数来估计未知的概率密度函数。简单来说，假设未知的概率密度函数为X，已知的概率密度为Y，如果得到了X和Y的关系，那么X也就可以求出来了。具体可以参考论文《 Noise-contrastive estimation of unnormalized statistical models, with applications to natural image statistics》。

5.1 负采样算法简单介绍

顾名思义，在基于Negative Sampling的CBOW和Skip-gram模型中，负采样是个很重要的环节，对于一个给定的词w，如何生成NEG(w)呢？

词典D中的词在语料C中出现的次数有高有底，对于那些高频词，被选为负样本的概率就应该比较大，反之，对于那些低频词，其被选中的概率就应该比较小。这就是我们对采样过程的一个大致要求，本质上就是一个带权采样问题。

下面，先用一段通俗的描述来帮助读者理解带权采样的机理。设词典D中的每一个词w对应一个线段 len(w) ，长度为：

len(w)=\frac{counter(w)}{\sum_{u \in C}counter(u)} \tag{12} \\

这里 counter(.) 表示一个词在语料C中出现的次数（分母中的求和项用来做归一化）。现在将这些线段首尾相连地拼接在一起，形成一个长度为1的单位线段。如果随机地往这个单位线段上打点，则其中长度越长的线段（对应高频词）被打中的概率就越大。

接下来再谈谈Word2Vec中的具体做法。记 l_0=0,..., l_k=\sum^{k}_{j=1}len(w_j), \ \ \ k=1,2,...,N ，这里 w_j 表示词典中的第j个单词，则以 \left\{ l_i \right\}^N_{j=0} 为剖分结点可得到区间 [0,1] 上的一个非等距剖分， I_{i} = \left\{ l_{i-1}, l_{i} \right\},i = 1,2,...,N 为其N个剖分区间。进一步引入区间 [0,1] 上的一个等距离剖分，剖分结点为 \left\{ m_{j} \right\}_{j=0}^{M} ，其中 M>>N ，具体见下面给出的示意图。
图：Table(.)映射的建立示意图

将内部剖分结点 \left\{ m_{j} \right\}_{j=1}^{M-1} 投影到非等距剖分上，如上图中的红色虚线所示，则可建立\left\{ m_{j} \right\}_{j=1}^{M-1}与区间 \left\{ I_{j} \right\}_{j=1}^{N} （或者说 \left\{ w_{j} \right\}_{j=1}^{N} ）的映射关系。

Table(i)=w_k, \ \ where \ \ m_i \in I_{k}, \ \ i=1,2,...,M-1 \tag{13} \\

有了这个映射，采样就简单了：每次生成一个 [1, M-1] 间的随机整数r，Table(r)就是一个样本。当然，这里还有一个细节，当对 w_{i} 进行负采样时，如果碰巧选到 w_{i} 自己该怎么办？那就跳过去，Word2Vec的代码中也是这么处理的。

值得一提的是，Word2Vec源码中为词典D中的词设置权值时，不是直接用counter(w)，而是对其做了 \alpha 次幂，其中 \alpha = 0.75 ，即公式（12）变成了：

\begin {aligned} len(w) &= \frac{counter(w)^\alpha}{\sum_{u \in C}[counter(u)]^\alpha} \\ &= \frac{counter(w)^{0.75}}{\sum_{u \in C}[counter(u)]^{0.75}} \end {aligned} \\

此外，代码中取 M = 10^{8} ，源代码中是变量table_size。而映射公式（13）则是通过一个名为InitUnigramTable的函数来完成。

5.2 基于Negative Sampling的CBOW

上面已经介绍完了负采样算法，下面开始推导出基于Negative Sampling的CBOW的目标函数。首先我们先选好一个关于 Context(w) 的负样本子集 NEG(w) \ne \oslash ，对于 \forall \widetilde{w} \in D ，我们定义单词 \widetilde{w} 的标签为:

L^w(\widetilde{w})= \left\{ \begin{matrix} 1, \ \ \ \widetilde{w}=w \\ 0, \ \ \ \widetilde{w} \neq w \end{matrix} \right. \\

上式表示词 \widetilde{w} 的标签，即正样本的标签为1，负样本的标签为0。

对于一个给定的正样本 (Context(w),w)，我们希望最大化的目标函数是：

g(w)=\prod_{u \in {w} \cup NEG(w)} p(u|Context(w)) \tag{14} \\

其中

\begin {aligned} p(u|Context(w)) &= \left\{\begin{matrix} \sigma(x^T_w \theta^u), \ \ \ \ L^w(u)=1 \\ 1-\sigma(x^T_w \theta^u), \ \ \ \ L^w(u)=0 \end{matrix}\right. \\ &= [\sigma(x^T_w\theta^u)]^{L^w(u)} \cdot [1-\sigma(x^T_w)\theta^u]^{1-L^w(u)} \end {aligned} \tag{15} \\

这里 x_w 仍表示 Context(w) 中各个词的词向量之和，而 \theta^u \in R^m 在这里作为一个辅助向量，为待训练的参数。

为什么要最大化 g(w) 呢？让我们先来看看g(w)的表达式，将公式（15）带入公式（14），有：

g(w)=\sigma(x^T_w\theta^w) \prod_{u \in NEG(w)} [1- \sigma(x^T_w\theta^u)] \\

其中， \sigma(x^T_w \theta^w) 表示当上下文为 Context(w) 时，预测中心词为w的概率，而 \sigma(x^T_w\theta^u), \ u \in NEG(w) 则表示当上下文为Context(w) 时，预测中心词为u的概率，这里可以看一个二分类问题。从形式上看，最大化 g(w) ，相当于最大化 \sigma(x^T_w \theta^w) ，同时最小化所有的 \sigma(x^T_w\theta^u), \ u \in NEG(w) 。这不正是我们希望的吗？增大正样本的概率同时降低负样本的概率。于是，对于一个给定的语料库C，有函数：

G = \prod_{w \in C}g(w) \\

可以作为最终的整体优化目标。当然，这里为了求导方便，对G取对数，最终的目标函数就是：

\begin {aligned} L &= \log G = \sum_{w \in C} \log g(w) \\ &= \sum_{w \in C} \sum_{u \in {w} \cup NEG(w)} \log { [\sigma(x^T_w\theta^u)]^{L^w(u)} \cdot [1-\sigma(x^T_w)\theta^u]^{1-L^w(u)} } \\ &= \sum_{w \in C} \sum_{u \in {w} \cup NEG(w)} { L^w(u) \cdot \log[\sigma(x^T_w \theta^u) + [1-L^w(u)] \cdot \log [1-\sigma(x^T_w \theta^u)]] } \end {aligned}

同样，为了求导方便,我们还是取 L(w,u) ：

L(w,u) = L^w(u) \cdot \log[\sigma(x^T_w \theta^u) + [1-L^w(u)] \cdot \log [1-\sigma(x^T_w \theta^u)]]

接下来，利用随机梯度上升法求梯度。首先考虑L(w,u)关于 \theta^{u} 的梯度计算：

\begin {aligned} \frac{\Delta L(w,u)}{\Delta \theta^u} &= \left\{ L^w(u)[1- \sigma(x^T_w\theta^u)]x_w-[1-L^w(u)] \cdot \sigma(x^T_w \theta^u)\right\} x_w \\ &=[L^w(u)-\sigma(x^T_w \theta^u)]x_w \end {aligned} \\

那么 \theta^u 的更新公式可以写成：

\theta^u=\theta^u+\eta [L^w(u)-\sigma(x^T_w \theta^u)]x_w \\

同时根据对称性，可以得到 x_w 的梯度：

\begin {aligned} \frac{\Delta L(w,u)}{\Delta x_w} &=[L^w(u)-\sigma(x^T_w \theta^u)] \theta^u \end {aligned} \\

那么 v(w) 的更新公式可以写成：

v(\tilde w) =v(\tilde w)+ \eta \sum_{u \in {w} \cup NEG(w)} \frac{\Delta L(w,u)}{\Delta x_w}, \ \ \tilde w \in Context(w) \\

5.3 基于Ngative Sampling的Skip-gram

本小节介绍基于Negative Sampling的Skip-gram模型。它和上一小节介绍的CBOW模型所用的思想是一样的，因此，这里我们直接从目标函数出发，且沿用之前的记号。

对于一个给定的样本 (w,Context(w)) ， 我们希望最大化：

g(w)= \prod_{\tilde w \in Context(w)} \prod_{u \in {w} \cup NEU^{\tilde w}(w)}p(Context| \widetilde{w}) \\

其中：

p(u| \widetilde{w}) = \left\{ \begin{matrix} \sigma(v(\tilde w)^T \theta^u), \ \ \ L^w(u)=1 \\ 1-\sigma(v(\tilde w)^T \theta^u), \ \ \ L^w(u)=0 \end{matrix} \right. \\

或者写成整体表达式：

p(u| \widetilde{w})=[\sigma(v(\tilde w)^T]^{L^w(u)} \cdot [1-\sigma(v(\tilde w)^T]^{1-L^w(u)} \\

这里 NEG^{\widetilde{w}}(w) 表示处理词 \widetilde{w} 时生成的负样本子集。于是，对于一个给定的语料库C，函数：

G = \prod_{w \in C}g(w) \\

就可以作为整体优化的目标。同样，我们取G的对数，最终的目标函数就是：

\begin {aligned} L &= \log G = \sum_{w \in C} \log g(w) \\ &= \sum_{w\in C} \sum_{\tilde w \in Context(w)} \sum_{u \in {w} \cup NEU^{\tilde w}(w)} L^w(u)\log[\sigma(v(\tilde w)^T \theta^u)] + [1-L^w(u)]\log[1-\sigma(v(\tilde w)^T \theta^u)] \end {aligned}

为了梯度推导的方便，我们依旧将三重求和符号下的内容提取出来，记为 L(w, \tilde w, u) ，即：

L(w, \tilde w, u) = L^w(u)\log[\sigma(v(\tilde w)^T \theta^u)] + [1-L^w(u)]\log[1-\sigma(v(\tilde w)^T \theta^u)]

接下来，利用随机梯度上升法求梯度。首先考虑 L(w, \widetilde{w}, u) 关于 \theta^{u} 的梯度计算：

\begin {aligned} \frac{\Delta L(w, \tilde w, u)}{\Delta \theta^u} &= \left\{L^w(u)[1- \sigma(v(\tilde w)^T_w\theta^u)]v(\tilde w)-[1-L^w(u)] \cdot \sigma(v(\tilde w)_w \theta^u) \right\} v(\tilde w) \\ &=[L^w(u)-\sigma(v(\tilde w)^T \theta^u)]v(\tilde w) \end {aligned} \\

然后得到 \theta^u 的更新公式：

\theta^u = \theta^u + \eta =[L^w(u)-\sigma(v(\tilde w)^T \theta^u)]v(\tilde w) \\

同理根据对称性,得到：

\begin {aligned} \frac{\Delta L(w, \tilde w, u)}{\Delta v(\tilde w)} &=[L^w(u)-\sigma(v(\tilde w)^T \theta^u)]\theta^u \end {aligned} \\

然后得到 v(\tilde w) 的更新公式：

v(\tilde w) = v(\tilde w) + \eta \sum_{u \in {w} \cup NEU^{\tilde w}(w)} \frac{ \Delta L(w, \tilde w, u)}{\Delta v(\tilde w)}, \ \ \ \tilde w \in Context(w) \\
