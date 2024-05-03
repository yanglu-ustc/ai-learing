## Example

<div align=center>
<img src="images/屏幕截图%202024-04-18%20192846.png" width=45%>
<img src="images/屏幕截图%202024-04-18%20195417.png" width=45%>
</div>

* 描述state状态的变量：

$$
s = \begin{bmatrix} x\\ y\\ \theta \\ \vdots \\ \dot{x} \\ \dot{y} \\ \dot{\theta} \\ \vdots \\ \end{bmatrix}
$$

* 回放缓冲区(Replay Buffer)
  * 存放数据信息的区域

## 学习状态函数

* deep-Q-leaning(DQN)

<div align=center>
<img src="images/屏幕截图%202024-04-19%20072417.png" width=48%>
<img src="images/屏幕截图%202024-04-19%20072524.png" width=41%>
<img src="images/屏幕截图%202024-04-19%20072559.png" width=80%>
</div>

## Algorithm refinement:算法改进

### 改进的神经网络架构

<div align=center>
<img src="images/屏幕截图%202024-04-18%20210221.png" width=47%>
<img src="images/屏幕截图%202024-04-18%20210325.png" width=42%>
</div>

### $\varepsilon$-贪婪策略

* 我们所进行的选择不一定是最好的结果，我们在这个过程中分出一定的概率交给随机，以达到探索的目的

<div align=center>
<img src="images/屏幕截图%202024-04-19%20070206.png" width=90%>
</div>

### Mini-batch:小批量

* 当数据量极其大时，会只对其中的部分先进性拟合，拟合的结果再带回来，减少时间

<div align=center>
<img src="images/屏幕截图%202024-04-19%20071209.png" width=42%>
<img src="images/屏幕截图%202024-04-19%20071156.png" width=49%>
<img src="images/屏幕截图%202024-04-19%20071143.png" width=45%>
<img src="images/屏幕截图%202024-04-19%20071424.png" width=45%>
</div>

### soft updates

* 在进行更新的时候，按比例进行更新，设置一个参数

<div align=center>
<img src="images/屏幕截图%202024-04-19%20071757.png" width=90%>
</div>
