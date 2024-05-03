# random-stochastic-environment

* 在随机环境的状态下，我们关心的是$Return$的平均值，期望：

$$
Excepted \ Return=Average(\sum_{i=1}^{n}\gamma^{i-1}R_{i})=E\left[\sum_{i=1}^{n}\gamma^{i-1}R_{i}\right]
$$

$$
Q(s,a)=R(s)+\gamma E\left[\underset{a'}{\max}Q(s',a')\right]
$$

<div align=center>
<img src="images/屏幕截图%202024-04-18%20180300.png" width=40%>
<img src="images/屏幕截图%202024-04-18%20180308.png" width=46%>
</div>
